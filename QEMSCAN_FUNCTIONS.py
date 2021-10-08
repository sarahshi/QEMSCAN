# %% -*- coding: utf-8 -*-
""" Created on August 18, 2021 // @author: Sarah Shi """

# %%

import numpy as np
import pandas as pd
import seaborn as sns
import time
import itertools

import scipy as scipy
import scipy.stats as stats
import scipy.linalg as linalg
from scipy.ndimage import gaussian_filter
from sklearn import mixture
from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import skimage as ski
from skimage import morphology, measure, transform, io
from skimage.feature import canny

from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib import cm
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

rc('font',**{'family':'Avenir', 'size': 18})
plt.rcParams['pdf.fonttype'] = 42

# %% 

def datafilter(inputdf):

    filt_data = inputdf['Mg.1'] + inputdf['Fe.1'] + inputdf['Si.1']
    scan_int = inputdf[filt_data > 25]
    filt_data_int = scan_int['Mg.1'] + scan_int['Fe.1'] + scan_int['Si.1']
    scan_lim = scan_int[filt_data_int < 99]
    scan_cluster_elements = scan_lim[['Mg.1', 'Fe.1', 'Ca.1', 'Al.1', 'Si.1', 'O.1']]

    return scan_lim, scan_cluster_elements

def initialPCA(inputdf, filename): 

    pca = PCA(n_components = 13)
    pca_cols = ['Si.1', 'Ti.1', 'Al.1', 'Fe.1', 'Mn.1', 'Mg.1', 'Ca.1', 'Na.1', 'K.1', 'P.1', 'S.1', 'Cl.1', 'F.1']
    pca_input_df = pd.DataFrame(columns = pca_cols)
    pca_input_df['Si.1'] = inputdf['Si.1']
    pca_input_df['Ti.1'] = inputdf['Ti.1']
    pca_input_df['Al.1'] = inputdf['Al.1']
    pca_input_df['Fe.1'] = inputdf['Fe.1']
    pca_input_df['Mn.1'] = inputdf['Mn.1']
    pca_input_df['Mg.1'] = inputdf['Mg.1']
    pca_input_df['Ca.1'] = inputdf['Ca.1']
    pca_input_df['Na.1'] = inputdf['Na.1']
    pca_input_df['K.1'] = inputdf['K.1']
    pca_input_df['P.1'] = inputdf['P.1']
    pca_input_df['S.1'] = inputdf['S.1']
    pca_input_df['Cl.1'] = inputdf['Cl.1']
    pca_input_df['F.1'] = inputdf['F.1']
    pca_input_df = pca_input_df.fillna(0)

    principalcomponents = pca.fit_transform(pca_input_df)
    pca_df = pd.DataFrame(data = principalcomponents, columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11', 'PC12', 'PC13'])
    pca_df_lim = pca_df[['PC1', 'PC2', 'PC3', 'PC4']]

    plt.figure(figsize = (10, 6))
    plt.plot(pca.explained_variance_ratio_, 'k.-', linewidth = 2, markersize = 12)
    plt.title(filename+' PCA Explained Variance')
    plt.xlabel('PCA Component')
    plt.ylabel('Explained Variance Ratio')
    plt.savefig(filename+'_explainedvar.pdf', dpi = 350)
    plt.show()

    return pca, pca_input_df, pca_df_lim

def bicassess(pca_df_lim, filename): 

    lowest_bic = np.infty
    bic = []
    n_components_range = range(1, 10)
    cv_types = ['spherical', 'tied', 'diag', 'full']
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            gmm = GaussianMixture(n_components=n_components,
                                    covariance_type=cv_type, 
                                    init_params = 'kmeans', 
                                    random_state = 0)
            gmm.fit(pca_df_lim)
            bic.append(gmm.bic(pca_df_lim))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm

    bic = np.array(bic)
    color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue','darkorange'])
    clf = best_gmm
    bars = []

    # Plot the BIC scores
    plt.figure(figsize=(10, 6))
    spl = plt.subplot(1, 1, 1)
    for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
        xpos = np.array(n_components_range) + .2 * (i - 2)
        bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                    (i + 1) * len(n_components_range)],
                            width=.2, color=color))
    plt.xticks(n_components_range)
    plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
    plt.title('Model-Dependent BIC Score')
    xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
        .2 * np.floor(bic.argmin() / len(n_components_range))
    plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
    spl.set_xlabel('Number of components')
    spl.legend([b[0] for b in bars], cv_types)
    plt.savefig(filename+'_bicscore.pdf', dpi = 350)

    return gmm

def bicscoreplot(gmm, filename):

    n_components = 8
    colors = itertools.cycle(['navy', 'turquoise', 'cornflowerblue', 'darkorange', 'grey'])
    def make_ellipses(gmm, ax):
        for n, color in enumerate(colors):
            if gmm.covariance_type == 'full':
                covariances = gmm.covariances_[n][:2, :2]
            elif gmm.covariance_type == 'tied':
                covariances = gmm.covariances_[:2, :2]
            elif gmm.covariance_type == 'diag':
                covariances = np.diag(gmm.covariances_[n][:2])
            elif gmm.covariance_type == 'spherical':
                covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
            v, w = np.linalg.eigh(covariances)
            u = w[0] / np.linalg.norm(w[0])
            angle = np.arctan2(u[1], u[0])
            angle = 180 * angle / np.pi  # convert to degrees
            v = 2. * np.sqrt(2.) * np.sqrt(v)
            ell = patches.Ellipse(gmm.means_[n, :2], v[0], v[1], 180 + angle, color=color)
            ell.set_clip_box(ax.bbox)
            ell.set_alpha(0.5)
            ax.add_artist(ell)
            ax.set_aspect('equal', 'datalim')

    # Try GMMs using different types of covariances.
    estimators = {cov_type: GaussianMixture(n_components=n_components,
                covariance_type=cov_type, init_params = 'kmeans', random_state=0)
                for cov_type in ['spherical', 'diag', 'tied', 'full']}

    n_estimators = len(estimators)

    plt.figure(figsize=(3 * n_estimators // 2, 6))
    plt.subplots_adjust(bottom=.01, top=0.95, hspace=.15, wspace=.05, left=.01, right=.99)
    plt.savefig(filename+'_bic2.pdf', dpi = 350)

    return 

def gauss_filter(pc1, sigma): 
    V = pc1.copy()
    V[np.isnan(pc1)] = 0
    VV = gaussian_filter(V, sigma=sigma)
    W = 0 * pc1.copy() + 1
    W[np.isnan(pc1)] = 0
    WW = gaussian_filter(W, sigma=sigma)
    Z = VV/WW
    return Z 

# %% 

def elementmapper(scan_lim, element): 

    """ elementmapper takes in the raw data and indexes it into a usable 2D numpy array. 
        inputs: the imported CSV file, scan_lim, and the element of interest
        output: 2D numpy array of concentrations """

    col_index = (max(np.unique(scan_lim.X))) + 1
    row_index = (max(np.unique(scan_lim.Y))) + 1
    preallocate_element_map = np.empty([row_index, col_index])

    X = scan_lim.X.ravel()
    Y = scan_lim.Y.ravel()
    E = scan_lim[element].ravel()

    for i in range(len(X)-1): 
        preallocate_element_map[Y[i], X[i]] = E[i]

    preallocate_element_map[preallocate_element_map == 0] = np.nan

    return preallocate_element_map 

def phasemapper(scan_lim, label): 

    """ phasemapper takes in the raw data and labels output from the Gaussian Mixture Model and indexes these values into a usable 2D numpy array. 
        inputs: the imported CSV file, scan_lim, and the array of labels from Gaussian Mixture Modeling. 
        output: 2D numpy array of labels, indicating phase at a particular position """
    
    col_index = (max(np.unique(scan_lim.X))) + 1
    row_index = (max(np.unique(scan_lim.Y))) + 1
    preallocate_phase_map = np.empty([row_index, col_index])

    X = scan_lim.X.ravel()
    Y = scan_lim.Y.ravel()

    for i in range(len(X)-1): 
        preallocate_phase_map[Y[i], X[i]] = label[i] + 1

    preallocate_phase_map[preallocate_phase_map == 0] = np.nan

    return preallocate_phase_map 


# %% 

def mineralid(phase_map, mg_map, fe_map, na_map, ca_map, al_map, cr_map, si_map, o_map): 

    """ mineralid takes in the phase maps and elemental maps to identify the minerals. 
        this function is currently calibrated on basaltic minerals and utilizes the Mg/Si and Ca+Al+Si/Fe ratios
        to distinguish between various minerals. it also identifies the bad spectra that form the blobby 
        masses in the images, which are where there are glass signals coming through epoxy. 

        inputs: the phase map and elemental maps
        output: average composition of each phase, sorted labels corresponding to each phase 
        automated here to ID phases based on some expected mineral compositions."""

    uniqueindex = np.unique(phase_map)[~np.isnan(np.unique(phase_map))]
    min_dist_index = ['Mg.1', 'Fe.1', 'Na.1', 'Ca.1', 'Al.1', 'Cr.1', 'Si.1', 'O.1']

    uniqueindex_new = uniqueindex - 1
    mindf = pd.DataFrame(columns = min_dist_index, index = uniqueindex_new)

    for i in uniqueindex:
        min_phase_pixels = 15
        phase_mask = phase_map!=i
        mineral = np.ma.masked_array(phase_map, mask = phase_mask)
        mineral = mineral.filled(fill_value=0)

        mineral_mask_open = morphology.area_opening(mineral, min_phase_pixels)
        min_mask_morph = (mineral_mask_open==0)

        mg_mask = np.ma.masked_array(mg_map, mask = min_mask_morph)
        fe_mask = np.ma.masked_array(fe_map, mask = min_mask_morph)
        na_mask = np.ma.masked_array(na_map, mask = min_mask_morph)
        ca_mask = np.ma.masked_array(ca_map, mask = min_mask_morph)
        al_mask = np.ma.masked_array(al_map, mask = min_mask_morph)
        cr_mask = np.ma.masked_array(cr_map, mask = min_mask_morph)
        si_mask = np.ma.masked_array(si_map, mask = min_mask_morph)
        o_mask = np.ma.masked_array(o_map, mask = min_mask_morph)

        min_dist_mean = np.array([np.nanmean(mg_mask), np.nanmean(fe_mask), np.nanmean(na_mask), np.nanmean(ca_mask), np.nanmean(al_mask), np.nanmean(cr_mask), np.nanmean(si_mask), np.nanmean(o_mask)])
        mindf.iloc[int(i)-1] = min_dist_mean

    mindf['Mg.1/Si.1'] = mindf['Mg.1'] / mindf['Si.1']
    mindf['CaAlSi.1/Fe.1'] = (mindf['Ca.1']+mindf['Al.1']+mindf['Si.1']) / mindf['Fe.1']
    mindf = mindf.apply(pd.to_numeric)
    sorted = mindf['Mg.1/Si.1'].nlargest(2)
    
    badspecindex = mindf['Mg.1/Si.1'].argmin()
    olindex = mindf['Mg.1/Si.1'].argmax()
    cpxindex = int(sorted.index[1])
    plagindex = mindf['CaAlSi.1/Fe.1'].argmax()
    indices = np.array([badspecindex, olindex, cpxindex, plagindex])
    glassindex = int(np.setdiff1d(uniqueindex_new, indices))
    
    mineralorder = np.array(['Glass', 'Olivine', 'Plagioclase', 'Clinopyroxene', 'BadSpectra'])
    indexposition = np.array([glassindex, olindex, plagindex, cpxindex, badspecindex])
    mineralindex = pd.DataFrame(index=mineralorder)
    mineralindex['Label'] = indexposition + 1

    phase_map[phase_map == mineralindex.loc['BadSpectra'].Label] = np.nan
    mg_map[phase_map == mineralindex.loc['BadSpectra'].Label] = np.nan
    fe_map[phase_map == mineralindex.loc['BadSpectra'].Label] = np.nan
    na_map[phase_map == mineralindex.loc['BadSpectra'].Label] = np.nan
    ca_map[phase_map == mineralindex.loc['BadSpectra'].Label] = np.nan
    al_map[phase_map == mineralindex.loc['BadSpectra'].Label] = np.nan
    cr_map[phase_map == mineralindex.loc['BadSpectra'].Label] = np.nan
    si_map[phase_map == mineralindex.loc['BadSpectra'].Label] = np.nan
    o_map[phase_map == mineralindex.loc['BadSpectra'].Label] = np.nan
    # mineralindex = mineralindex.drop(index='BadSpectra')

    return mindf, mineralindex, phase_map, mg_map, fe_map, na_map, ca_map, al_map, cr_map, si_map, o_map


def mineralmasks(index, mineralindex, phase_map, mg_map, fe_map, na_map, ca_map, al_map, cr_map, si_map, o_map): 

    """ mineralmasks takes in the maps and mineral label indices to generate a masked 2D numpy array
    containing labels or chemical information for each individual mineral. 

        inputs: the phase maps, elemental maps, and label indices
        output: masked output for each mineral, chemical composition for mineral pixels,
        confidence intervals for each element"""

    phase_map[phase_map == mineralindex.loc['BadSpectra'].Label] = np.nan
    mg_map[phase_map == mineralindex.loc['BadSpectra'].Label] = np.nan
    fe_map[phase_map == mineralindex.loc['BadSpectra'].Label] = np.nan
    na_map[phase_map == mineralindex.loc['BadSpectra'].Label] = np.nan
    ca_map[phase_map == mineralindex.loc['BadSpectra'].Label] = np.nan
    al_map[phase_map == mineralindex.loc['BadSpectra'].Label] = np.nan
    cr_map[phase_map == mineralindex.loc['BadSpectra'].Label] = np.nan
    si_map[phase_map == mineralindex.loc['BadSpectra'].Label] = np.nan
    o_map[phase_map == mineralindex.loc['BadSpectra'].Label] = np.nan

    min_phase_pixels = 15
    phase_mask = phase_map!=index 
    mineral = np.ma.masked_array(phase_map, mask = phase_mask)
    mineral = mineral.filled(fill_value=0)

    mineral_mask_open = morphology.area_opening(mineral, min_phase_pixels)
    min_mask_morph = (mineral_mask_open == 0)
    # phase_mask_open[phase_mask_open==0] = np.nan

    mg_mask = np.ma.masked_array(mg_map, mask = min_mask_morph)
    fe_mask = np.ma.masked_array(fe_map, mask = min_mask_morph)
    na_mask = np.ma.masked_array(na_map, mask = min_mask_morph)
    ca_mask = np.ma.masked_array(ca_map, mask = min_mask_morph)
    al_mask = np.ma.masked_array(al_map, mask = min_mask_morph)
    cr_mask = np.ma.masked_array(cr_map, mask = min_mask_morph)
    si_mask = np.ma.masked_array(si_map, mask = min_mask_morph)
    o_mask = np.ma.masked_array(o_map, mask = min_mask_morph)

    mg_mean = np.mean(mg_mask.compressed())
    mg_ci = stats.t.interval(0.95, len(mg_mask.compressed())-1, loc=np.mean(mg_mask.compressed()), scale=stats.sem(mg_mask.compressed()))
    fe_mean = np.mean(fe_mask.compressed())
    fe_ci = stats.t.interval(0.95, len(fe_mask.compressed())-1, loc=np.mean(fe_mask.compressed()), scale=stats.sem(fe_mask.compressed()))
    fe_output = np.array([])
    na_mean = np.mean(na_mask.compressed())
    na_ci = stats.t.interval(0.95, len(na_mask.compressed())-1, loc=np.mean(na_mask.compressed()), scale=stats.sem(na_mask.compressed()))
    ca_mean = np.mean(ca_mask.compressed())
    ca_ci = stats.t.interval(0.95, len(ca_mask.compressed())-1, loc=np.mean(ca_mask.compressed()), scale=stats.sem(ca_mask.compressed()))
    al_mean = np.mean(al_mask.compressed())
    al_ci = stats.t.interval(0.95, len(al_mask.compressed())-1, loc=np.mean(al_mask.compressed()), scale=stats.sem(al_mask.compressed()))
    cr_mean = np.mean(cr_mask.compressed())
    cr_ci = stats.t.interval(0.95, len(cr_mask.compressed())-1, loc=np.mean(cr_mask.compressed()), scale=stats.sem(cr_mask.compressed()))
    si_mean = np.mean(si_mask.compressed())
    si_ci = stats.t.interval(0.95, len(si_mask.compressed())-1, loc=np.mean(si_mask.compressed()), scale=stats.sem(si_mask.compressed()))
    o_mean = np.mean(o_mask.compressed())
    o_ci = stats.t.interval(0.95, len(o_mask.compressed())-1, loc=np.mean(o_mask.compressed()), scale=stats.sem(o_mask.compressed()))
    ci_output = np.array([mg_mean, mg_ci[0], mg_ci[1], fe_mean, fe_ci[0], fe_ci[1], na_mean, na_ci[0], na_ci[1], ca_mean, ca_ci[0], ca_ci[1], 
    al_mean, al_ci[0], al_ci[1], si_mean, si_ci[0], si_ci[1], cr_mean, cr_ci[0], cr_ci[1], o_mean, o_ci[0], o_ci[1]])

    return mineral_mask_open, min_mask_morph, mg_mask, fe_mask, na_mask, ca_mask, al_mask, cr_mask, si_mask, o_mask, ci_output


def mineralplotter(mineralindex, phase_map, mg_map, fe_map, na_map, ca_map, al_map, cr_map, si_map, o_map, filename):

    """ mineralplotter takes in the mineral information and outputs figures. 

        inputs: the phase maps, elemental maps, and label indices
        output: figures!"""

    colors = ['#FEE0C0', '#D2D200', '#1EA0A0', '#0D6401', ''] # ordered glass, ol, plag, cpx, bad spec
    mineralindex['Color'] = colors
    mineralindex = mineralindex.sort_values(by=['Label'])
    mineralindex = mineralindex.drop(index='BadSpectra')
    cMap = ListedColormap(mineralindex.Color.ravel()) 
    colorredo = ['k', '#FFFFFF']
    cMapnew = ListedColormap(colorredo)

    min_dist_index = ['Mg', 'Fe', 'Na', 'Ca', 'Al', 'Cr', 'Si', 'O']

    patches = [mpatches.Patch(color = mineralindex.Color[i], label = mineralindex.index[i]) for i in range(len(mineralindex.index))]

    fig, ax = plt.subplots(figsize = (12.5, 17.5))
    plt.pcolormesh(phase_map, cmap = cMap, rasterized = True)
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')
    plt.legend(handles = patches, prop={'size': 16})
    # plt.colorbar()
    plt.tight_layout()
    plt.gca().invert_yaxis()
    plt.savefig(filename+'_phases.pdf', dpi = 350)
    plt.close('all')

    for i in mineralindex.Label.ravel(): 
        min_phase_pixels = 15
        phase_mask = phase_map!=i
        mineral = np.ma.masked_array(phase_map, mask = phase_mask)
        mineral = mineral.filled(fill_value=0)

        mineral_mask_open = morphology.area_opening(mineral, min_phase_pixels)
        min_mask_morph = (mineral_mask_open==0)

        mg_mask = np.ma.masked_array(mg_map, mask = min_mask_morph)
        fe_mask = np.ma.masked_array(fe_map, mask = min_mask_morph)
        na_mask = np.ma.masked_array(na_map, mask = min_mask_morph)
        ca_mask = np.ma.masked_array(ca_map, mask = min_mask_morph)
        al_mask = np.ma.masked_array(al_map, mask = min_mask_morph)
        cr_mask = np.ma.masked_array(cr_map, mask = min_mask_morph)
        si_mask = np.ma.masked_array(si_map, mask = min_mask_morph)
        o_mask = np.ma.masked_array(o_map, mask = min_mask_morph)

        min_dist_mean = np.array([np.nanmean(mg_mask.compressed()), np.nanmean(fe_mask.compressed()), np.nanmean(na_mask.compressed()), np.nanmean(ca_mask.compressed()), 
        np.nanmean(al_mask.compressed()), np.nanmean(cr_mask.compressed()), np.nanmean(si_mask.compressed()), np.nanmean(o_mask.compressed())])

        mg_95 = stats.t.interval(0.95, len(mg_mask.compressed())-1, loc=np.nanmean(mg_mask.compressed())) #, scale=stats.sem(mg_mask.compressed()))
        fe_95 = stats.t.interval(0.95, len(fe_mask.compressed())-1, loc=np.nanmean(fe_mask.compressed())) #, scale=stats.sem(fe_mask.compressed()))
        na_95 = stats.t.interval(0.95, len(na_mask.compressed())-1, loc=np.nanmean(na_mask.compressed())) #, scale=stats.sem(ca_mask.compressed()))
        ca_95 = stats.t.interval(0.95, len(ca_mask.compressed())-1, loc=np.nanmean(ca_mask.compressed())) #, scale=stats.sem(ca_mask.compressed()))
        al_95 = stats.t.interval(0.95, len(al_mask.compressed())-1, loc=np.nanmean(al_mask.compressed())) #, scale=stats.sem(al_mask.compressed()))
        cr_95 = stats.t.interval(0.95, len(al_mask.compressed())-1, loc=np.nanmean(cr_mask.compressed())) #, scale=stats.sem(al_mask.compressed()))
        si_95 = stats.t.interval(0.95, len(si_mask.compressed())-1, loc=np.nanmean(si_mask.compressed())) #, scale=stats.sem(si_mask.compressed()))
        o_95 = stats.t.interval(0.95, len(o_mask.compressed())-1, loc=np.nanmean(o_mask.compressed())) #, scale=stats.sem(o_mask.compressed()))

        fig, ax = plt.subplots(figsize = (8.5, 15))
        plt.subplot(5, 2, 1)
        plt.pcolormesh(min_mask_morph, cmap = cMapnew, rasterized = True)
        # plt.pcolormesh(phase_map, cmap = cMap, alpha = 0.2, rasterized = True)
        plt.title('Mineral = ' + str((mineralindex[mineralindex['Label'] == i]).index[0]))
        plt.gca().invert_yaxis()

        plt.subplot(5, 2, 2)
        sns.barplot(x = min_dist_index, y = min_dist_mean)
        plt.title('Concentration Average')
        plt.ylim([0, 40])

        plt.subplot(5, 2, 3)
        plt.scatter(mg_mask.compressed(), fe_mask.compressed(), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('Mg')
        plt.ylabel('Fe')

        plt.subplot(5, 2, 4)
        plt.scatter(na_mask.compressed(), ca_mask.compressed(), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('Na')
        plt.ylabel('Ca')

        plt.subplot(5, 2, 5)
        plt.scatter(mg_mask.compressed(), si_mask.compressed(), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('Mg')
        plt.ylabel('Si')

        plt.subplot(5, 2, 6)
        plt.scatter(mg_mask.compressed(), o_mask.compressed(), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('Mg')
        plt.ylabel('O')

        plt.subplot(5, 2, 7)
        plt.scatter(na_mask.compressed(), al_mask.compressed(), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('Na')
        plt.ylabel('Al')

        plt.subplot(5, 2, 8)
        plt.scatter(na_mask.compressed(), si_mask.compressed(), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('Na')
        plt.ylabel('Si')

        plt.subplot(5, 2, 9)
        plt.scatter(na_mask.compressed(), o_mask.compressed(), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('Na')
        plt.ylabel('O')

        plt.subplot(5, 2, 10)
        plt.scatter(si_mask.compressed(), o_mask.compressed(), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('Si')
        plt.ylabel('O')
        plt.tight_layout()

        plt.savefig(filename+'_'+str((mineralindex[mineralindex['Label'] == i]).index[0])+'.pdf', dpi = 350)
        plt.close('all')


def gmmqemscancomparison(pca_df_lim, pca_input_df, labels, mineralindex, filename):

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple'] # ordered glass, ol, plag, cpx, bad spec
    cMap = ListedColormap(colors) 

    labelssorted = mineralindex.Label.sort_values(ascending = True)
    index = labelssorted.index

    plt.figure(figsize = (8.5, 12))
    plt.subplot(3, 2, 1)
    scatter = plt.scatter(pca_df_lim.PC1[0::15], pca_df_lim.PC2[0::15], c = labels[0::15], cmap = cMap, rasterized = True, alpha = 0.75, lw = 0.25, edgecolor = 'k')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(handles = scatter.legend_elements()[0], labels = index.tolist(), prop={'size': 8})

    plt.subplot(3, 2, 2)
    scatter = plt.scatter(pca_df_lim.PC2[0::15], pca_df_lim.PC3[0::15], c = labels[0::15], cmap = cMap, rasterized = True, alpha = 0.75, lw = 0.25, edgecolor = 'k')
    plt.xlabel('PC2')
    plt.ylabel('PC3')
    plt.legend(handles = scatter.legend_elements()[0], labels = index.tolist(), prop={'size': 8})

    plt.subplot(3, 2, 3)
    scatter = plt.scatter(pca_df_lim.PC3[0::15], pca_df_lim.PC4[0::15], c = labels[0::15], cmap = cMap, rasterized = True, alpha = 0.75, lw = 0.25, edgecolor = 'k')
    plt.xlabel('PC3')
    plt.ylabel('PC4')
    plt.legend(handles = scatter.legend_elements()[0], labels = index.tolist(), prop={'size': 8})

    plt.subplot(3, 2, 4)
    scatter = plt.scatter(pca_df_lim.PC1[0::15], pca_df_lim.PC4[0::15], c = labels[0::15], cmap = cMap, rasterized = True, alpha = 0.75, lw = 0.25, edgecolor = 'k')
    plt.xlabel('PC1')
    plt.ylabel('PC4')
    plt.legend(handles = scatter.legend_elements()[0], labels = index.tolist(), prop={'size': 8})

    plt.subplot(3, 2, 5)
    scatter = plt.scatter(pca_input_df['Mg.1'][0::15], pca_input_df['Fe.1'][0::15], c = labels[0::15], cmap = cMap, rasterized = True, alpha = 0.75, lw = 0.25, edgecolor = 'k')
    plt.xlabel('Mg')
    plt.ylabel('Fe')
    plt.legend(handles = scatter.legend_elements()[0], labels = index.tolist(), prop={'size': 8})

    plt.subplot(3, 2, 6)
    scatter = plt.scatter(pca_input_df['Ca.1'][0::15], pca_input_df['Si.1'][0::15], c = labels[0::15], cmap = cMap, rasterized = True, alpha = 0.75, lw = 0.25, edgecolor = 'k')
    plt.xlabel('Ca')
    plt.ylabel('Si')
    plt.legend(handles = scatter.legend_elements()[0], labels = index.tolist(), prop={'size': 8})
    plt.tight_layout()
    plt.savefig(filename+'_GMMQEMSCAN_Comp.pdf')
    plt.show()
    return

def mineralgmmqemscancomparison(filtereddata, solutionelement_a, solutionelement_b, mineral, filename):

    if mineral == 'olivine': 
        fo = filtereddata[filtereddata['Mineral Name'] == 'Forsterite (A)']
        fa = filtereddata[filtereddata['Mineral Name'] == 'Fayalite (A)']
        fo90s = filtereddata[filtereddata['Mineral Name'] == 'Olivine Fo90 (S)']
        fo90  = filtereddata[filtereddata['Mineral Name'] == 'Olivine Fo90 (A)']
        fo80  = filtereddata[filtereddata['Mineral Name'] == 'Olivine Fo80 (A)']
        fo50  = filtereddata[filtereddata['Mineral Name'] == 'Olivine Fo50 (A)']
        fo90aug = filtereddata[filtereddata['Mineral Name'] == 'Fo90-Augite (75-25)']
        fo80by = filtereddata[filtereddata['Mineral Name'] == 'Fo80-Bytownite (75-25)']
        fo90an80 = filtereddata[filtereddata['Mineral Name'] == 'Fo90-An80 (75-25)']
        fo90ch = filtereddata[filtereddata['Mineral Name'] == 'Fo90-Chromite (75-25)']
        mgoolca = filtereddata[filtereddata['Mineral Name'] == 'MgOlivine-Calcite (75-25)']
        olca = filtereddata[filtereddata['Mineral Name'] == 'Olivine-Calcite (75-25)']

        plt.figure(figsize = (8.5, 4))
        plt.subplot(1, 2, 1)
        plt.scatter(fo['Mg.1'], fo['Fe.1'], lw = 0.25, edgecolor = 'k', label = 'Forsterite', rasterized = True)
        plt.scatter(fa['Mg.1'], fa['Fe.1'], lw = 0.25, edgecolor = 'k', label = 'Fayalite', rasterized = True)
        plt.scatter(fo90s['Mg.1'], fo90s['Fe.1'], lw = 0.25, edgecolor = 'k', label = 'Fo90_S', rasterized = True)
        plt.scatter(fo90['Mg.1'], fo90['Fe.1'], lw = 0.25, edgecolor = 'k', label = 'Fo90', rasterized = True)
        plt.scatter(fo80['Mg.1'], fo80['Fe.1'], lw = 0.25, edgecolor = 'k', label = 'Fo80', rasterized = True)
        plt.scatter(fo50['Mg.1'], fo50['Fe.1'], lw = 0.25, edgecolor = 'k', label = 'Fo50', rasterized = True)
        plt.scatter(fo90aug['Mg.1'], fo90aug['Fe.1'], lw = 0.25, edgecolor = 'k', label = 'Fo90-Augite', rasterized = True)
        plt.scatter(fo80by['Mg.1'], fo80by['Fe.1'], lw = 0.25, edgecolor = 'k', label = 'Fo80-Bytownite', rasterized = True)
        plt.scatter(fo90an80['Mg.1'], fo90an80['Fe.1'], lw = 0.25, edgecolor = 'k', label = 'Fo90-An80', rasterized = True)
        plt.scatter(fo90ch['Mg.1'], fo90ch['Fe.1'], lw = 0.25, edgecolor = 'k', label = 'Fo90-Chromite', rasterized = True)
        plt.scatter(mgoolca['Mg.1'], mgoolca['Fe.1'], lw = 0.25, edgecolor = 'k', label = 'MgO Ol-Calcite', rasterized = True)
        plt.scatter(olca['Mg.1'], olca['Fe.1'], lw = 0.25, edgecolor = 'k', label = 'Ol-Calcite', rasterized = True)
        plt.xlabel('Mg')
        plt.ylabel('Fe')
        plt.xlim([0, 50])
        plt.ylim([0, 90])
        plt.legend(prop={'size': 6})

        plt.subplot(1, 2, 2)
        plt.scatter(solutionelement_a[0::15], solutionelement_b[0::15], rasterized = True, lw = 0.25, edgecolor = 'k')
        plt.xlabel('Mg')
        plt.ylabel('Fe')
        plt.xlim([0, 50])
        plt.ylim([0, 90])
        plt.tight_layout()
        plt.savefig(filename+'_Ol_GMMQEMSCAN_Comp.pdf')
        plt.show()

    if mineral == 'plagioclase':  
        an = filtereddata[filtereddata['Mineral Name'] == 'Anorthite (A)']
        ans = filtereddata[filtereddata['Mineral Name'] == 'Anorthite (S)']
        plaggroundmass = filtereddata[filtereddata['Mineral Name'] == 'Plag-Groundmass boundary (R)']
        plagfeldspar  = filtereddata[filtereddata['Mineral Name'] == 'Plagioclase Feldspar']
        feldsparan  = filtereddata[filtereddata['Mineral Name'] == 'Feldspar_Anorthite']
        feldsparor  = filtereddata[filtereddata['Mineral Name'] == 'Feldspar-Orthoclase']
        an80fo90 = filtereddata[filtereddata['Mineral Name'] == 'An80-Fo90 (75-25)']
        kfeldspar = filtereddata[filtereddata['Mineral Name'] == 'K-Feldspar (A)']
        plagspinel = filtereddata[filtereddata['Mineral Name'] == 'Plagioclase-CrSpinel (75-25)']
        plagpyrite = filtereddata[filtereddata['Mineral Name'] == 'Plagioclase-Pyrite (75-25)']

        plt.figure(figsize = (8.5, 4))
        plt.subplot(1, 2, 1)
        plt.scatter(an['Ca.1'], an['Si.1'], lw = 0.25, edgecolor = 'k', label = 'Anorthite', rasterized = True)
        plt.scatter(ans['Ca.1'], ans['Si.1'], lw = 0.25, edgecolor = 'k', label = 'Anorthite_S', rasterized = True)
        plt.scatter(plaggroundmass['Ca.1'], plaggroundmass['Si.1'], lw = 0.25, edgecolor = 'k', label = 'Plagioclase Groundmass', rasterized = True)
        plt.scatter(plagfeldspar['Ca.1'], plagfeldspar['Si.1'], lw = 0.25, edgecolor = 'k', label = 'Plagioclase Feldspar', rasterized = True)
        plt.scatter(feldsparan['Ca.1'], feldsparan['Si.1'], lw = 0.25, edgecolor = 'k', label = 'Feldspar_An', rasterized = True)
        plt.scatter(feldsparor['Ca.1'], feldsparor['Si.1'], lw = 0.25, edgecolor = 'k', label = 'Feldspar_Or', rasterized = True)
        plt.scatter(an80fo90['Ca.1'], an80fo90['Si.1'], lw = 0.25, edgecolor = 'k', label = 'An80-Fo90', rasterized = True)
        plt.scatter(kfeldspar['Ca.1'], kfeldspar['Si.1'], lw = 0.25, edgecolor = 'k', label = 'K-Feldspar', rasterized = True)
        plt.scatter(plagspinel['Ca.1'], plagspinel['Si.1'], lw = 0.25, edgecolor = 'k', label = 'Plagioclase-CrSpinel', rasterized = True)
        plt.scatter(plagpyrite['Ca.1'], plagpyrite['Si.1'], lw = 0.25, edgecolor = 'k', label = 'Plagioclase-Pyrite', rasterized = True)
        plt.xlabel('Ca')
        plt.ylabel('Si')
        plt.xlim([0, 40])
        plt.ylim([0, 90])
        plt.legend(prop={'size': 6})

        plt.subplot(1, 2, 2)
        plt.scatter(solutionelement_a[0::15], solutionelement_b[0::15], rasterized = True, lw = 0.25, edgecolor = 'k')
        plt.xlabel('Ca')
        plt.ylabel('Si')
        plt.xlim([0, 40])
        plt.ylim([0, 90])
        plt.tight_layout()
        plt.savefig(filename+'_Plag_GMMQEMSCAN_Comp.pdf')
        plt.show()

    return



# %%


def mineralpca(mineral, plotting, scaling, mg_mask, fe_mask, na_mask, ca_mask, al_mask, cr_mask, si_mask, o_mask, samplename): 

    """ mineralpca takes in the mineral information and outputs PCA components and scores. 

        inputs: mineral, chemical masks, figure export lable
        output: pca, mineral pca, pca score dataframe"""

    if mineral == 'olivine': 
        pca = PCA(n_components = 4)
        ol_dist = ['Mg.1', 'Fe.1', 'Si.1', 'O.1']
        min_pca_df = pd.DataFrame(columns = ol_dist)
        min_pca_df['Mg.1'] = mg_mask.compressed()
        min_pca_df['Fe.1'] = fe_mask.compressed()
        min_pca_df['Si.1'] = si_mask.compressed()
        min_pca_df['O.1'] = o_mask.compressed()
        min_pca_df = min_pca_df.fillna(0)
        # ci = stats.t.interval(0.99, len(mg_mask.compressed())-1, loc=np.nanmean(mg_mask.compressed()))
        # min_pca_df = min_pca_df[(min_pca_df['Mg.1'] > ci[0]) & (min_pca_df['Mg.1'] < ci[1])]

        if scaling == 1: 
            scaled_min_pca_df = StandardScaler().fit_transform(min_pca_df.values)
            min_pca_df = pd.DataFrame(columns = ol_dist)
            min_pca_df['Mg.1'] = scaled_min_pca_df[:, 0]
            min_pca_df['Fe.1'] = scaled_min_pca_df[:, 1]
            min_pca_df['Si.1'] = scaled_min_pca_df[:, 2]
            min_pca_df['O.1'] = scaled_min_pca_df[:, 3]
            min_pca_df = min_pca_df.fillna(0)
        principalcomponents = pca.fit_transform(min_pca_df)
        pca_df = pd.DataFrame(data = principalcomponents, columns = ['PC1', 'PC2', 'PC3', 'PC4'])

        thinning = int(round(len(pca_df) / 10000))
        pca_df_thin = pca_df[0::thinning]
        min_pca_df_thin = min_pca_df[0::thinning]

        if plotting == 1: 
            plt.figure(figsize = (10, 4))
            plt.subplot(1, 2, 1)
            plt.pcolormesh(mg_mask, rasterized=True)
            plt.xlabel('X Pixel')
            plt.ylabel('Y Pixel')
            cbar = plt.colorbar()
            cbar.set_label('Mg', size = 10)
            plt.gca().invert_yaxis()
            plt.tight_layout()

            plt.subplot(1, 2, 2)
            plt.scatter(pca_df.columns, pca.explained_variance_ratio_, lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PCA')
            plt.ylabel('Explained Variance')
            plt.tight_layout()
            plt.savefig(samplename + '_Variance_Olivine.pdf', dpi = 350)
            plt.close('all')

            plt.figure(figsize = (8.5, 9))
            plt.subplot(3, 2, 1)
            plt.scatter(pca_df_thin['PC1'], pca_df_thin['PC2'], c = min_pca_df_thin['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)
            plt.tight_layout()

            plt.subplot(3, 2, 2)
            plt.scatter(pca_df_thin['PC1'], pca_df_thin['PC3'], c = min_pca_df_thin['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC1')
            plt.ylabel('PC3')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)
            plt.tight_layout()

            plt.subplot(3, 2, 3)
            plt.scatter(pca_df_thin['PC1'], pca_df_thin['PC4'], c = min_pca_df_thin['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC1')
            plt.ylabel('PC4')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)
            plt.tight_layout()

            plt.subplot(3, 2, 4)
            plt.scatter(pca_df_thin['PC2'], pca_df_thin['PC3'], c = min_pca_df_thin['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC2')
            plt.ylabel('PC3')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)
            plt.tight_layout()

            plt.subplot(3, 2, 5)
            plt.scatter(pca_df_thin['PC2'], pca_df_thin['PC4'], c = min_pca_df_thin['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC2')
            plt.ylabel('PC4')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)
            plt.tight_layout()

            plt.subplot(3, 2, 6)
            plt.scatter(pca_df_thin['PC3'], pca_df_thin['PC4'], c = min_pca_df_thin['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC3')
            plt.ylabel('PC4')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)
            plt.tight_layout()
            plt.savefig(samplename+'_PCA_Comparison_Olivine.pdf', dpi = 350)
            plt.close('all')

            plt.figure(figsize = (24, 22))
            plt.subplot(4, 4, 1)
            plt.scatter(pca_df_thin['PC1'], min_pca_df_thin['Mg.1'], c = min_pca_df_thin['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC1')
            plt.ylabel('Mg')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)
            plt.tight_layout()

            plt.subplot(4, 4, 2)
            plt.scatter(pca_df_thin['PC1'], min_pca_df_thin['Fe.1'], c = min_pca_df_thin['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC1')
            plt.ylabel('Fe')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)
            plt.tight_layout()

            plt.subplot(4, 4, 3)
            plt.scatter(pca_df_thin['PC1'], min_pca_df_thin['Si.1'], c = min_pca_df_thin['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC1')
            plt.ylabel('Si')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)
            plt.tight_layout()

            plt.subplot(4, 4, 4)
            plt.scatter(pca_df_thin['PC1'], min_pca_df_thin['O.1'], c = min_pca_df_thin['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC1')
            plt.ylabel('O')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)
            plt.tight_layout()

            plt.subplot(4, 4, 5)
            plt.scatter(pca_df_thin['PC2'], min_pca_df_thin['Mg.1'], c = min_pca_df_thin['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC2')
            plt.ylabel('Mg')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)
            plt.tight_layout()

            plt.subplot(4, 4, 6)
            plt.scatter(pca_df_thin['PC2'], min_pca_df_thin['Fe.1'], c = min_pca_df_thin['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC2')
            plt.ylabel('Fe')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)
            plt.tight_layout()

            plt.subplot(4, 4, 7)
            plt.scatter(pca_df_thin['PC2'], min_pca_df_thin['Si.1'], c = min_pca_df_thin['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC2')
            plt.ylabel('Si')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)
            plt.tight_layout()

            plt.subplot(4, 4, 8)
            plt.scatter(pca_df_thin['PC2'], min_pca_df_thin['O.1'], c = min_pca_df_thin['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC2')
            plt.ylabel('O')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)
            plt.tight_layout()

            plt.subplot(4, 4, 9)
            plt.scatter(pca_df_thin['PC3'], min_pca_df_thin['Mg.1'], c = min_pca_df_thin['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC3')
            plt.ylabel('Mg')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)
            plt.tight_layout()

            plt.subplot(4, 4, 10)
            plt.scatter(pca_df_thin['PC3'], min_pca_df_thin['Fe.1'], c = min_pca_df_thin['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC3')
            plt.ylabel('Fe')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)
            plt.tight_layout()

            plt.subplot(4, 4, 11)
            plt.scatter(pca_df_thin['PC3'], min_pca_df_thin['Si.1'], c = min_pca_df_thin['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC3')
            plt.ylabel('Si')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)
            plt.tight_layout()

            plt.subplot(4, 4, 12)
            plt.scatter(pca_df_thin['PC3'], min_pca_df_thin['O.1'], c = min_pca_df_thin['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC3')
            plt.ylabel('O')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)
            plt.tight_layout()

            plt.subplot(4, 4, 13)
            plt.scatter(pca_df_thin['PC4'], min_pca_df_thin['Mg.1'], c = min_pca_df_thin['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC4')
            plt.ylabel('Mg')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)
            plt.tight_layout()

            plt.subplot(4, 4, 14)
            plt.scatter(pca_df_thin['PC4'], min_pca_df_thin['Fe.1'], c = min_pca_df_thin['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC4')
            plt.ylabel('Fe')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)
            plt.tight_layout()

            plt.subplot(4, 4, 15)
            plt.scatter(pca_df_thin['PC4'], min_pca_df_thin['Si.1'], c = min_pca_df_thin['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC4')
            plt.ylabel('Si')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)
            plt.tight_layout()

            plt.subplot(4, 4, 16)
            plt.scatter(pca_df_thin['PC4'], min_pca_df_thin['O.1'], c = min_pca_df_thin['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC4')
            plt.ylabel('O')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)
            plt.tight_layout()
            plt.savefig(samplename+'_PCA_Elements_Olivine.pdf', dpi = 350)
            plt.close('all')

    elif mineral == 'plagioclase':
        pca = PCA(n_components = 5)
        plag_dist = ['Na.1' 'Ca.1', 'Al.1', 'Si.1', 'O.1']
        min_pca_df = pd.DataFrame(columns = plag_dist)
        min_pca_df['Na.1'] = na_mask.compressed()
        min_pca_df['Ca.1'] = ca_mask.compressed()
        min_pca_df['Al.1'] = al_mask.compressed()
        min_pca_df['Si.1'] = si_mask.compressed()
        min_pca_df['O.1'] = o_mask.compressed()
        min_pca_df = min_pca_df.fillna(0)
        # ci = stats.t.interval(0.99, len(ca_mask.compressed())-1, loc=np.nanmean(ca_mask.compressed()))
        # min_pca_df = min_pca_df[(min_pca_df['Ca.1'] > ci[0]) & (min_pca_df['Ca.1'] < ci[1])]

        if scaling == 1: 
            scaled_min_pca_df = StandardScaler().fit_transform(min_pca_df.values)
            min_pca_df = pd.DataFrame(columns = plag_dist)
            min_pca_df['Na.1'] = scaled_min_pca_df[:, 0]
            min_pca_df['Ca.1'] = scaled_min_pca_df[:, 1]
            min_pca_df['Al.1'] = scaled_min_pca_df[:, 2]
            min_pca_df['Si.1'] = scaled_min_pca_df[:, 3]
            min_pca_df = min_pca_df.fillna(0)
        principalcomponents = pca.fit_transform(min_pca_df)
        pca_df = pd.DataFrame(data = principalcomponents, columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5'])

        thinning = int(round(len(pca_df) / 10000))
        pca_df_thin = pca_df[0::thinning]
        min_pca_df_thin = min_pca_df[0::thinning]
        # min_pca_df_thin = min_pca_df_thin[(pca_df_thin.PC1 > np.nanpercentile(pca_df_thin.PC1, 5)) & (pca_df_thin.PC1 < np.nanpercentile(pca_df_thin.PC1, 95))]
        # pca_df_thin = pca_df_thin[(pca_df_thin.PC1 > np.nanpercentile(pca_df_thin.PC1, 5)) & (pca_df_thin.PC1 < np.nanpercentile(pca_df_thin.PC1, 95))]

        if plotting == 1: 
            plt.figure(figsize = (10, 4))
            plt.subplot(1, 2, 1)
            plt.pcolormesh(na_mask, rasterized = True)
            plt.xlabel('X Pixel')
            plt.ylabel('Y Pixel')
            cbar = plt.colorbar()
            cbar.set_label('Ca', size = 10)
            plt.gca().invert_yaxis()
            plt.tight_layout()

            plt.subplot(1, 2, 2)
            plt.scatter(pca_df.columns, pca.explained_variance_ratio_, lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PCA')
            plt.ylabel('Explained Variance')
            plt.tight_layout()
            plt.savefig(samplename+'_Variance_Plag.pdf', dpi = 350)
            plt.close('all')

            plt.figure(figsize = (8.5, 9))
            plt.subplot(3, 2, 1)
            plt.scatter(pca_df_thin['PC1'], pca_df_thin['PC2'], c = min_pca_df_thin['Ca.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            cbar = plt.colorbar()
            cbar.set_label('Ca', size = 10)
            plt.tight_layout()

            plt.subplot(3, 2, 2)
            plt.scatter(pca_df_thin['PC1'], pca_df_thin['PC3'], c = min_pca_df_thin['Ca.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC1')
            plt.ylabel('PC3')
            cbar = plt.colorbar()
            cbar.set_label('Ca', size = 10)
            plt.tight_layout()

            plt.subplot(3, 2, 3)
            plt.scatter(pca_df_thin['PC1'], pca_df_thin['PC4'], c = min_pca_df_thin['Ca.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC1')
            plt.ylabel('PC4')
            cbar = plt.colorbar()
            cbar.set_label('Ca', size = 10)
            plt.tight_layout()

            plt.subplot(3, 2, 4)
            plt.scatter(pca_df_thin['PC2'], pca_df_thin['PC3'], c = min_pca_df_thin['Ca.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC2')
            plt.ylabel('PC3')
            cbar = plt.colorbar()
            cbar.set_label('Ca', size = 10)
            plt.tight_layout()

            plt.subplot(3, 2, 5)
            plt.scatter(pca_df_thin['PC2'], pca_df_thin['PC4'], c = min_pca_df_thin['Ca.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC2')
            plt.ylabel('PC4')
            cbar = plt.colorbar()
            cbar.set_label('Ca', size = 10)
            plt.tight_layout()

            plt.subplot(3, 2, 6)
            plt.scatter(pca_df_thin['PC3'], pca_df_thin['PC4'], c = min_pca_df_thin['Ca.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC3')
            plt.ylabel('PC4')
            cbar = plt.colorbar()
            cbar.set_label('Ca', size = 10)
            plt.tight_layout()
            plt.savefig(samplename+'_PCA_Comparison_Plag.pdf', dpi = 350)
            plt.close('all')

            plt.figure(figsize = (24, 22))
            plt.subplot(5, 4, 1)
            plt.scatter(pca_df_thin['PC1'], min_pca_df_thin['Na.1'], c = min_pca_df_thin['Ca.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC1')
            plt.ylabel('Na')
            cbar = plt.colorbar()
            cbar.set_label('Ca', size = 10)
            plt.tight_layout()

            plt.subplot(5, 4, 2)
            plt.scatter(pca_df_thin['PC1'], min_pca_df_thin['Ca.1'], c = min_pca_df_thin['Ca.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC1')
            plt.ylabel('Ca')
            cbar = plt.colorbar()
            cbar.set_label('Ca', size = 10)
            plt.tight_layout()

            plt.subplot(5, 4, 3)
            plt.scatter(pca_df_thin['PC1'], min_pca_df_thin['Al.1'], c = min_pca_df_thin['Ca.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC1')
            plt.ylabel('Al')
            cbar = plt.colorbar()
            cbar.set_label('Ca', size = 10)
            plt.tight_layout()

            plt.subplot(5, 4, 4)
            plt.scatter(pca_df_thin['PC1'], min_pca_df_thin['Si.1'], c = min_pca_df_thin['Ca.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC1')
            plt.ylabel('Si')
            cbar = plt.colorbar()
            cbar.set_label('Ca', size = 10)
            plt.tight_layout()

            plt.subplot(5, 4, 5)
            plt.scatter(pca_df_thin['PC1'], min_pca_df_thin['O.1'], c = min_pca_df_thin['Ca.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC1')
            plt.ylabel('O')
            cbar = plt.colorbar()
            cbar.set_label('Ca', size = 10)
            plt.tight_layout()

            plt.subplot(5, 4, 6)
            plt.scatter(pca_df_thin['PC2'], min_pca_df_thin['Na.1'], c = min_pca_df_thin['Ca.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC2')
            plt.ylabel('Na')
            cbar = plt.colorbar()
            cbar.set_label('Ca', size = 10)
            plt.tight_layout()

            plt.subplot(5, 4, 7)
            plt.scatter(pca_df_thin['PC2'], min_pca_df_thin['Ca.1'], c = min_pca_df_thin['Ca.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC2')
            plt.ylabel('Ca')
            cbar = plt.colorbar()
            cbar.set_label('Ca', size = 10)
            plt.tight_layout()

            plt.subplot(5, 4, 8)
            plt.scatter(pca_df_thin['PC2'], min_pca_df_thin['Al.1'], c = min_pca_df_thin['Ca.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC2')
            plt.ylabel('Al')
            cbar = plt.colorbar()
            cbar.set_label('Ca', size = 10)

            plt.subplot(5, 4, 9)
            plt.scatter(pca_df_thin['PC2'], min_pca_df_thin['Si.1'], c = min_pca_df_thin['Ca.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC2')
            plt.ylabel('Si')
            cbar = plt.colorbar()
            cbar.set_label('Ca', size = 10)

            plt.subplot(5, 4, 10)
            plt.scatter(pca_df_thin['PC2'], min_pca_df_thin['O.1'], c = min_pca_df_thin['Ca.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC2')
            plt.ylabel('O')
            cbar = plt.colorbar()
            cbar.set_label('Ca', size = 10)

            plt.subplot(5, 4, 11)
            plt.scatter(pca_df_thin['PC3'], min_pca_df_thin['Na.1'], c = min_pca_df_thin['Ca.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC3')
            plt.ylabel('Na')
            cbar = plt.colorbar()
            cbar.set_label('Ca', size = 10)

            plt.subplot(5, 4, 12)
            plt.scatter(pca_df_thin['PC3'], min_pca_df_thin['Ca.1'], c = min_pca_df_thin['Ca.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC3')
            plt.ylabel('Ca')
            cbar = plt.colorbar()
            cbar.set_label('Ca', size = 10)

            plt.subplot(5, 4, 13)
            plt.scatter(pca_df_thin['PC3'], min_pca_df_thin['Al.1'], c = min_pca_df_thin['Ca.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC3')
            plt.ylabel('Al')
            cbar = plt.colorbar()
            cbar.set_label('Ca', size = 10)

            plt.subplot(5, 4, 14)
            plt.scatter(pca_df_thin['PC3'], min_pca_df_thin['Si.1'], c = min_pca_df_thin['Ca.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC3')
            plt.ylabel('Si')
            cbar = plt.colorbar()
            cbar.set_label('Ca', size = 10)

            plt.subplot(5, 4, 15)
            plt.scatter(pca_df_thin['PC3'], min_pca_df_thin['O.1'], c = min_pca_df_thin['Ca.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC3')
            plt.ylabel('O')
            cbar = plt.colorbar()
            cbar.set_label('Ca', size = 10)

            plt.subplot(5, 4, 16)
            plt.scatter(pca_df_thin['PC4'], min_pca_df_thin['Na.1'], c = min_pca_df_thin['Ca.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC4')
            plt.ylabel('Na')
            cbar = plt.colorbar()
            cbar.set_label('Ca', size = 10)

            plt.subplot(5, 4, 17)
            plt.scatter(pca_df_thin['PC4'], min_pca_df_thin['Ca.1'], c = min_pca_df_thin['Ca.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC4')
            plt.ylabel('Ca')
            cbar = plt.colorbar()
            cbar.set_label('Ca', size = 10)

            plt.subplot(5, 4, 18)
            plt.scatter(pca_df_thin['PC4'], min_pca_df_thin['Al.1'], c = min_pca_df_thin['Ca.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC4')
            plt.ylabel('Al')
            cbar = plt.colorbar()
            cbar.set_label('Ca', size = 10)

            plt.subplot(5, 4, 19)
            plt.scatter(pca_df_thin['PC4'], min_pca_df_thin['Si.1'], c = min_pca_df_thin['Ca.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC4')
            plt.ylabel('Si')
            cbar = plt.colorbar()
            cbar.set_label('Ca', size = 10)

            plt.subplot(5, 4, 20)
            plt.scatter(pca_df_thin['PC4'], min_pca_df_thin['O.1'], c = min_pca_df_thin['Ca.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC4')
            plt.ylabel('O')
            cbar = plt.colorbar()
            cbar.set_label('Ca', size = 10)
            plt.tight_layout()
            plt.savefig(samplename+'_PCA_Elements_Plag.pdf', dpi = 350)
            plt.close('all')

    elif mineral == 'clinopyroxene':
        pca = PCA(n_components = 4)

        cpx_dist = ['Mg.1', 'Fe.1', 'Ca.1', 'Si.1']
        min_pca_df = pd.DataFrame(columns = cpx_dist)
        min_pca_df['Mg.1'] = mg_mask.compressed()
        min_pca_df['Fe.1'] = fe_mask.compressed()
        min_pca_df['Ca.1'] = ca_mask.compressed()
        min_pca_df['Si.1'] = si_mask.compressed()
        min_pca_df = min_pca_df.fillna(0)
        # ci = stats.t.interval(0.99, len(fe_mask.compressed())-1, loc=np.nanmean(fe_mask.compressed()))
        # min_pca_df = min_pca_df[(min_pca_df['Fe.1'] > ci[0]) & (min_pca_df['Fe.1'] < ci[1])]

        if scaling == 1: 
            scaled_min_pca_df = StandardScaler().fit_transform(min_pca_df.values)
            min_pca_df = pd.DataFrame(columns = cpx_dist)
            min_pca_df['Mg.1'] = scaled_min_pca_df[:, 0]
            min_pca_df['Fe.1'] = scaled_min_pca_df[:, 1]
            min_pca_df['Ca.1'] = scaled_min_pca_df[:, 2]
            min_pca_df['Si.1'] = scaled_min_pca_df[:, 3]
            min_pca_df = min_pca_df.fillna(0)
        principalcomponents = pca.fit_transform(min_pca_df)
        pca_df = pd.DataFrame(data = principalcomponents, columns = ['PC1', 'PC2', 'PC3', 'PC4'])

        thinning = int(round(len(pca_df) / 10000))
        pca_df_thin = pca_df[0::thinning]
        min_pca_df_thin = min_pca_df[0::thinning]
        # min_pca_df_thin = min_pca_df_thin[(pca_df_thin.PC1 > np.nanpercentile(pca_df_thin.PC1, 5)) & (pca_df_thin.PC1 < np.nanpercentile(pca_df_thin.PC1, 95))]
        # pca_df_thin = pca_df_thin[(pca_df_thin.PC1 > np.nanpercentile(pca_df_thin.PC1, 5)) & (pca_df_thin.PC1 < np.nanpercentile(pca_df_thin.PC1, 95))]

        if plotting == 1: 
            plt.figure(figsize = (10, 4))
            plt.subplot(1, 2, 1)
            plt.pcolormesh(fe_mask, rasterized = True)
            plt.xlabel('X Pixel')
            plt.ylabel('Y Pixel')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)
            plt.gca().invert_yaxis()
            plt.tight_layout()

            plt.subplot(1, 2, 2)
            plt.scatter(pca_df.columns, pca.explained_variance_ratio_, lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PCA')
            plt.ylabel('Explained Variance')
            plt.tight_layout()
            plt.savefig(samplename+'_Variance_Cpx.pdf', dpi = 350)
            plt.close('all')

            plt.figure(figsize = (8.5, 9))
            plt.subplot(3, 2, 1)
            plt.scatter(pca_df_thin['PC1'], pca_df_thin['PC2'], c = min_pca_df_thin['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)
            plt.tight_layout()

            plt.subplot(3, 2, 2)
            plt.scatter(pca_df_thin['PC1'], pca_df_thin['PC3'], c = min_pca_df_thin['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC1')
            plt.ylabel('PC3')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)
            plt.tight_layout()

            plt.subplot(3, 2, 3)
            plt.scatter(pca_df_thin['PC1'], pca_df_thin['PC4'], c = min_pca_df_thin['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC1')
            plt.ylabel('PC4')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)
            plt.tight_layout()

            plt.subplot(3, 2, 4)
            plt.scatter(pca_df_thin['PC2'], pca_df_thin['PC3'], c = min_pca_df_thin['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC2')
            plt.ylabel('PC3')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)
            plt.tight_layout()

            plt.subplot(3, 2, 5)
            plt.scatter(pca_df_thin['PC2'], pca_df_thin['PC4'], c = min_pca_df_thin['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC2')
            plt.ylabel('PC4')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)
            plt.tight_layout()

            plt.subplot(3, 2, 6)
            plt.scatter(pca_df_thin['PC3'], pca_df_thin['PC4'], c = min_pca_df_thin['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC3')
            plt.ylabel('PC4')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)
            plt.tight_layout()
            plt.savefig(samplename+'_PCA_Comparison_Cpx.pdf')
            plt.close('all')

            plt.figure(figsize = (24, 22))
            plt.subplot(4, 4, 1)
            plt.scatter(pca_df_thin['PC1'], min_pca_df_thin['Mg.1'], c = min_pca_df_thin['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC1')
            plt.ylabel('Mg')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)
            plt.tight_layout()

            plt.subplot(4, 4, 2)
            plt.scatter(pca_df_thin['PC1'], min_pca_df_thin['Fe.1'], c = min_pca_df_thin['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC1')
            plt.ylabel('Fe')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)

            plt.subplot(4, 4, 3)
            plt.scatter(pca_df_thin['PC1'], min_pca_df_thin['Ca.1'], c = min_pca_df_thin['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC1')
            plt.ylabel('Ca')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)

            plt.subplot(4, 4, 4)
            plt.scatter(pca_df_thin['PC1'], min_pca_df_thin['Si.1'], c = min_pca_df_thin['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC1')
            plt.ylabel('Si')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)

            plt.subplot(4, 4, 5)
            plt.scatter(pca_df_thin['PC2'], min_pca_df_thin['Mg.1'], c = min_pca_df_thin['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC2')
            plt.ylabel('Mg')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)

            plt.subplot(4, 4, 6)
            plt.scatter(pca_df_thin['PC2'], min_pca_df_thin['Fe.1'], c = min_pca_df_thin['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC2')
            plt.ylabel('Fe')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)

            plt.subplot(4, 4, 7)
            plt.scatter(pca_df_thin['PC2'], min_pca_df_thin['Ca.1'], c = min_pca_df_thin['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC2')
            plt.ylabel('Ca')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)

            plt.subplot(4, 4, 8)
            plt.scatter(pca_df_thin['PC2'], min_pca_df_thin['Si.1'], c = min_pca_df_thin['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC2')
            plt.ylabel('Si')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)

            plt.subplot(4, 4, 9)
            plt.scatter(pca_df_thin['PC3'], min_pca_df_thin['Mg.1'], c = min_pca_df_thin['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC3')
            plt.ylabel('Mg')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)

            plt.subplot(4, 4, 10)
            plt.scatter(pca_df_thin['PC3'], min_pca_df_thin['Fe.1'], c = min_pca_df_thin['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC3')
            plt.ylabel('Fe')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)

            plt.subplot(4, 4, 11)
            plt.scatter(pca_df_thin['PC3'], min_pca_df_thin['Ca.1'], c = min_pca_df_thin['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC3')
            plt.ylabel('Ca')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)

            plt.subplot(4, 4, 12)
            plt.scatter(pca_df_thin['PC3'], min_pca_df_thin['Si.1'], c = min_pca_df_thin['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC3')
            plt.ylabel('Si')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)

            plt.subplot(4, 4, 13)
            plt.scatter(pca_df_thin['PC4'], min_pca_df_thin['Mg.1'], c = min_pca_df_thin['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC4')
            plt.ylabel('Mg')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)

            plt.subplot(4, 4, 14)
            plt.scatter(pca_df_thin['PC4'], min_pca_df_thin['Fe.1'], c = min_pca_df_thin['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC4')
            plt.ylabel('Fe')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)

            plt.subplot(4, 4, 15)
            plt.scatter(pca_df_thin['PC4'], min_pca_df_thin['Ca.1'], c = min_pca_df_thin['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC4')
            plt.ylabel('Ca')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)

            plt.subplot(4, 4, 16)
            plt.scatter(pca_df_thin['PC4'], min_pca_df_thin['Si.1'], c = min_pca_df_thin['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC4')
            plt.ylabel('Si')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)
            plt.tight_layout()
            plt.savefig(samplename+'_PCA_Elements_Cpx.pdf', dpi = 350)
            plt.close('all')

    elif mineral == 'glass':
        pca = PCA(n_components = 6)
        glass_dist = ['Mg.1', 'Fe.1', 'Ca.1', 'Al.1', 'Si.1', 'O.1']
        min_pca_df = pd.DataFrame(columns = glass_dist)
        min_pca_df['Mg.1'] = mg_mask.compressed()
        min_pca_df['Fe.1'] = fe_mask.compressed()
        min_pca_df['Ca.1'] = ca_mask.compressed()
        min_pca_df['Al.1'] = al_mask.compressed()
        min_pca_df['Si.1'] = si_mask.compressed()
        min_pca_df['O.1'] = o_mask.compressed()
        min_pca_df = min_pca_df.fillna(0)
        # ci = stats.t.interval(0.99, len(ca_mask.compressed())-1, loc=np.nanmean(ca_mask.compressed()))
        # min_pca_df = min_pca_df[(min_pca_df['Ca.1'] > ci[0]) & (min_pca_df['Ca.1'] < ci[1])]

        if scaling == 1: 
            scaled_min_pca_df = StandardScaler().fit_transform(min_pca_df.values)
            min_pca_df = pd.DataFrame(columns = glass_dist)
            min_pca_df['Mg.1'] = scaled_min_pca_df[:, 0]
            min_pca_df['Fe.1'] = scaled_min_pca_df[:, 1]
            min_pca_df['Ca.1'] = scaled_min_pca_df[:, 2]
            min_pca_df['Al.1'] = scaled_min_pca_df[:, 3]
            min_pca_df['Si.1'] = scaled_min_pca_df[:, 4]
            min_pca_df['O.1'] = scaled_min_pca_df[:, 5]
            min_pca_df = min_pca_df.fillna(0)
        principalcomponents = pca.fit_transform(min_pca_df)
        pca_df = pd.DataFrame(data = principalcomponents, columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6'])

        thinning = int(round(len(pca_df) / 10000))
        pca_df_thin = pca_df[0::thinning]
        min_pca_df_thin = min_pca_df[0::thinning]
        # min_pca_df_thin = min_pca_df_thin[(pca_df_thin.PC1 > np.nanpercentile(pca_df_thin.PC1, 5)) & (pca_df_thin.PC1 < np.nanpercentile(pca_df_thin.PC1, 95))]
        # pca_df_thin = pca_df_thin[(pca_df_thin.PC1 > np.nanpercentile(pca_df_thin.PC1, 5)) & (pca_df_thin.PC1 < np.nanpercentile(pca_df_thin.PC1, 95))]

        if plotting == 1: 
            plt.figure(figsize = (10, 4))
            plt.subplot(1, 2, 1)
            plt.pcolormesh(mg_mask, rasterized = True)
            plt.xlabel('X Pixel')
            plt.ylabel('Y Pixel')
            cbar = plt.colorbar()
            cbar.set_label('Mg', size = 10)
            plt.gca().invert_yaxis()
            plt.tight_layout()

            plt.subplot(1, 2, 2)
            plt.scatter(pca_df.columns, pca.explained_variance_ratio_, lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PCA')
            plt.ylabel('Explained Variance')
            plt.tight_layout()
            plt.savefig(samplename+'_Variance_Glass.pdf')
            plt.close('all')

            plt.figure(figsize = (8.5, 9))
            plt.subplot(3, 2, 1)
            plt.scatter(pca_df_thin['PC1'], pca_df_thin['PC2'], c = min_pca_df_thin['Mg.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            cbar = plt.colorbar()
            cbar.set_label('Mg', size = 10)
            plt.tight_layout()

            plt.subplot(3, 2, 2)
            plt.scatter(pca_df_thin['PC1'], pca_df_thin['PC3'], c = min_pca_df_thin['Mg.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC1')
            plt.ylabel('PC3')
            cbar = plt.colorbar()
            cbar.set_label('Mg', size = 10)
            plt.tight_layout()

            plt.subplot(3, 2, 3)
            plt.scatter(pca_df_thin['PC1'], pca_df_thin['PC4'], c = min_pca_df_thin['Mg.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC1')
            plt.ylabel('PC4')
            cbar = plt.colorbar()
            cbar.set_label('Mg', size = 10)
            plt.tight_layout()

            plt.subplot(3, 2, 4)
            plt.scatter(pca_df_thin['PC2'], pca_df_thin['PC3'], c = min_pca_df_thin['Mg.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC2')
            plt.ylabel('PC3')
            cbar = plt.colorbar()
            cbar.set_label('Mg', size = 10)
            plt.tight_layout()

            plt.subplot(3, 2, 5)
            plt.scatter(pca_df_thin['PC2'], pca_df_thin['PC4'], c = min_pca_df_thin['Mg.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC2')
            plt.ylabel('PC4')
            cbar = plt.colorbar()
            cbar.set_label('Mg', size = 10)
            plt.tight_layout()

            plt.subplot(3, 2, 6)
            plt.scatter(pca_df_thin['PC3'], pca_df_thin['PC4'], c = min_pca_df_thin['Mg.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC3')
            plt.ylabel('PC4')
            cbar = plt.colorbar()
            cbar.set_label('Mg', size = 10)
            plt.tight_layout()
            plt.savefig(samplename+'_PCA_Comparison_Glass.pdf', dpi = 200)
            plt.close('all')

            plt.figure(figsize = (24, 22))
            plt.subplot(4, 4, 1)
            plt.scatter(pca_df_thin['PC1'], min_pca_df_thin['Mg.1'], c = min_pca_df_thin['Mg.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC1')
            plt.ylabel('Mg')
            cbar = plt.colorbar()
            cbar.set_label('Mg', size = 10)
            plt.tight_layout()

            plt.subplot(4, 4, 2)
            plt.scatter(pca_df_thin['PC1'], min_pca_df_thin['Fe.1'], c = min_pca_df_thin['Mg.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC1')
            plt.ylabel('Fe')
            cbar = plt.colorbar()
            cbar.set_label('Mg', size = 10)

            plt.subplot(4, 4, 3)
            plt.scatter(pca_df_thin['PC1'], min_pca_df_thin['Ca.1'], c = min_pca_df_thin['Mg.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC1')
            plt.ylabel('Ca')
            cbar = plt.colorbar()
            cbar.set_label('Mg', size = 10)

            plt.subplot(4, 4, 4)
            plt.scatter(pca_df_thin['PC1'], min_pca_df_thin['Al.1'], c = min_pca_df_thin['Mg.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC1')
            plt.ylabel('Al')
            cbar = plt.colorbar()
            cbar.set_label('Mg', size = 10)

            plt.subplot(4, 4, 5)
            plt.scatter(pca_df_thin['PC1'], min_pca_df_thin['Si.1'], c = min_pca_df_thin['Mg.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC1')
            plt.ylabel('Si')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)

            plt.subplot(4, 4, 6)
            plt.scatter(pca_df_thin['PC1'], min_pca_df_thin['O.1'], c = min_pca_df_thin['Mg.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC1')
            plt.ylabel('O')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)

            plt.subplot(4, 4, 7)
            plt.scatter(pca_df_thin['PC2'], min_pca_df_thin['Mg.1'], c = min_pca_df_thin['Mg.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC2')
            plt.ylabel('Mg')
            cbar = plt.colorbar()
            cbar.set_label('Mg', size = 10)

            plt.subplot(4, 4, 8)
            plt.scatter(pca_df_thin['PC2'], min_pca_df_thin['Fe.1'], c = min_pca_df_thin['Mg.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC2')
            plt.ylabel('Fe')
            cbar = plt.colorbar()
            cbar.set_label('Mg', size = 10)

            plt.subplot(4, 4, 9)
            plt.scatter(pca_df_thin['PC2'], min_pca_df_thin['Ca.1'], c = min_pca_df_thin['Mg.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC2')
            plt.ylabel('Ca')
            cbar = plt.colorbar()
            cbar.set_label('Mg', size = 10)

            plt.subplot(4, 4, 10)
            plt.scatter(pca_df_thin['PC2'], min_pca_df_thin['Al.1'], c = min_pca_df_thin['Mg.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC2')
            plt.ylabel('Al')
            cbar = plt.colorbar()
            cbar.set_label('Mg', size = 10)

            plt.subplot(4, 4, 11)
            plt.scatter(pca_df_thin['PC2'], min_pca_df_thin['Si.1'], c = min_pca_df_thin['Mg.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC2')
            plt.ylabel('Si')
            cbar = plt.colorbar()
            cbar.set_label('Mg', size = 10)

            plt.subplot(4, 4, 12)
            plt.scatter(pca_df_thin['PC2'], min_pca_df_thin['O.1'], c = min_pca_df_thin['Mg.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC2')
            plt.ylabel('O')
            cbar = plt.colorbar()
            cbar.set_label('Mg', size = 10)

            plt.subplot(4, 4, 13)
            plt.scatter(pca_df_thin['PC3'], min_pca_df_thin['Mg.1'], c = min_pca_df_thin['Mg.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC3')
            plt.ylabel('Mg')
            cbar = plt.colorbar()
            cbar.set_label('Mg', size = 10)

            plt.subplot(4, 4, 14)
            plt.scatter(pca_df_thin['PC3'], min_pca_df_thin['Fe.1'], c = min_pca_df_thin['Mg.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC3')
            plt.ylabel('Fe')
            cbar = plt.colorbar()
            cbar.set_label('Mg', size = 10)

            plt.subplot(4, 4, 15)
            plt.scatter(pca_df_thin['PC3'], min_pca_df_thin['Ca.1'], c = min_pca_df_thin['Mg.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC3')
            plt.ylabel('Ca')
            cbar = plt.colorbar()
            cbar.set_label('Mg', size = 10)

            plt.subplot(4, 4, 16)
            plt.scatter(pca_df_thin['PC3'], min_pca_df_thin['Al.1'], c = min_pca_df_thin['Mg.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC3')
            plt.ylabel('Al')
            cbar = plt.colorbar()
            cbar.set_label('Mg', size = 10)

            plt.tight_layout()
            plt.savefig(samplename+'_PCA_Elements_Glass.pdf', dpi = 200)
            plt.close('all')
        
    return pca, min_pca_df, min_pca_df_thin, pca_df, pca_df_thin

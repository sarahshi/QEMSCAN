# %%

import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats

from sklearn.decomposition import PCA, NMF
from sklearn.mixture import GaussianMixture
import skimage as ski
from skimage import morphology, measure, transform, io
from matplotlib.colors import ListedColormap

from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib import cm
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

# %% ßß 

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

def mineralid(phase_map, mg_map, fe_map, ca_map, al_map, si_map, o_map): 

    """ mineralid takes in the phase maps and elemental maps to identify the minerals. 
        this function is currently calibrated on basaltic minerals and utilizes the Mg/Si and Ca+Al+Si/Fe ratios
        to distinguish between various minerals. it also identifies the bad spectra that form the blobby 
        masses in the images, which are where there are glass signals coming through epoxy. 

        inputs: the phase map and elemental maps
        output: average composition of each phase, sorted labels corresponding to each phase 
        automated here to ID phases based on some expected mineral compositions."""

    uniqueindex = np.unique(phase_map)[~np.isnan(np.unique(phase_map))]
    min_dist_index = ['Mg.1', 'Fe.1', 'Ca.1', 'Al.1', 'Si.1', 'O.1']

    uniqueindex_new = uniqueindex - 1
    mindf = pd.DataFrame(columns = min_dist_index, index = uniqueindex_new)

    for i in uniqueindex:
        min_phase_pixels = 5
        phase_mask = phase_map!=i
        mineral = np.ma.masked_array(phase_map, mask = phase_mask)
        mineral = mineral.filled(fill_value=0)

        mineral_mask_open = morphology.area_opening(mineral, min_phase_pixels)
        min_mask_morph = (mineral_mask_open==0)

        mg_mask = np.ma.masked_array(mg_map, mask = min_mask_morph)
        fe_mask = np.ma.masked_array(fe_map, mask = min_mask_morph)
        ca_mask = np.ma.masked_array(ca_map, mask = min_mask_morph)
        al_mask = np.ma.masked_array(al_map, mask = min_mask_morph)
        si_mask = np.ma.masked_array(si_map, mask = min_mask_morph)
        o_mask = np.ma.masked_array(o_map, mask = min_mask_morph)

        min_dist_mean = np.array([np.nanmean(mg_mask), np.nanmean(fe_mask), np.nanmean(ca_mask), np.nanmean(al_mask), np.nanmean(si_mask), np.nanmean(o_mask)])
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
    ca_map[phase_map == mineralindex.loc['BadSpectra'].Label] = np.nan
    al_map[phase_map == mineralindex.loc['BadSpectra'].Label] = np.nan
    si_map[phase_map == mineralindex.loc['BadSpectra'].Label] = np.nan
    o_map[phase_map == mineralindex.loc['BadSpectra'].Label] = np.nan
    # mineralindex = mineralindex.drop(index='BadSpectra')

    return mindf, mineralindex, phase_map, mg_map, fe_map, ca_map, al_map, si_map, o_map


def mineralmasks(index, mineralindex, phase_map, mg_map, fe_map, ca_map, al_map, si_map, o_map): 

    """ mineralmasks takes in the maps and mineral label indices to generate a masked 2D numpy array
    containing labels or chemical information for each individual mineral. 

        inputs: the phase maps, elemental maps, and label indices
        output: masked output for each mineral, chemical composition for mineral pixels,
        confidence intervals for each element"""

    phase_map[phase_map == mineralindex.loc['BadSpectra'].Label] = np.nan
    mg_map[phase_map == mineralindex.loc['BadSpectra'].Label] = np.nan
    fe_map[phase_map == mineralindex.loc['BadSpectra'].Label] = np.nan
    ca_map[phase_map == mineralindex.loc['BadSpectra'].Label] = np.nan
    al_map[phase_map == mineralindex.loc['BadSpectra'].Label] = np.nan
    si_map[phase_map == mineralindex.loc['BadSpectra'].Label] = np.nan
    o_map[phase_map == mineralindex.loc['BadSpectra'].Label] = np.nan

    min_phase_pixels = 5
    phase_mask = phase_map!=index 
    mineral = np.ma.masked_array(phase_map, mask = phase_mask)
    mineral = mineral.filled(fill_value=0)

    mineral_mask_open = morphology.area_opening(mineral, min_phase_pixels)
    min_mask_morph = (mineral_mask_open == 0)
    # phase_mask_open[phase_mask_open==0] = np.nan

    mg_mask = np.ma.masked_array(mg_map, mask = min_mask_morph)
    fe_mask = np.ma.masked_array(fe_map, mask = min_mask_morph)
    ca_mask = np.ma.masked_array(ca_map, mask = min_mask_morph)
    al_mask = np.ma.masked_array(al_map, mask = min_mask_morph)
    si_mask = np.ma.masked_array(si_map, mask = min_mask_morph)
    o_mask = np.ma.masked_array(o_map, mask = min_mask_morph)

    mg_mean = np.mean(mg_mask.compressed())
    mg_ci = stats.t.interval(0.95, len(mg_mask.compressed())-1, loc=np.mean(mg_mask.compressed()), scale=stats.sem(mg_mask.compressed()))
    fe_mean = np.mean(fe_mask.compressed())
    fe_ci = stats.t.interval(0.95, len(fe_mask.compressed())-1, loc=np.mean(fe_mask.compressed()), scale=stats.sem(fe_mask.compressed()))
    fe_output = np.array([])
    ca_mean = np.mean(ca_mask.compressed())
    ca_ci = stats.t.interval(0.95, len(ca_mask.compressed())-1, loc=np.mean(ca_mask.compressed()), scale=stats.sem(ca_mask.compressed()))
    al_mean = np.mean(al_mask.compressed())
    al_ci = stats.t.interval(0.95, len(al_mask.compressed())-1, loc=np.mean(al_mask.compressed()), scale=stats.sem(al_mask.compressed()))
    si_mean = np.mean(si_mask.compressed())
    si_ci = stats.t.interval(0.95, len(si_mask.compressed())-1, loc=np.mean(si_mask.compressed()), scale=stats.sem(si_mask.compressed()))
    o_mean = np.mean(o_mask.compressed())
    o_ci = stats.t.interval(0.95, len(o_mask.compressed())-1, loc=np.mean(o_mask.compressed()), scale=stats.sem(o_mask.compressed()))
    ci_output = np.array([mg_mean, mg_ci[0], mg_ci[1], fe_mean, fe_ci[0], fe_ci[1], ca_mean, ca_ci[0], ca_ci[1], 
    al_mean, al_ci[0], al_ci[1], si_mean, si_ci[0], si_ci[1], o_mean, o_ci[0], o_ci[1]])

    return mineral_mask_open, min_mask_morph, mg_mask, fe_mask, ca_mask, al_mask, si_mask, o_mask, ci_output


def mineralplotter(mineralindex, phase_map, mg_map, fe_map, ca_map, al_map, si_map, o_map, filename):

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

    min_dist_index = ['Mg.1', 'Fe.1', 'Ca.1', 'Al.1', 'Si.1', 'O.1']

    patches = [mpatches.Patch(color = mineralindex.Color[i], label = mineralindex.index[i]) for i in range(len(mineralindex.index))]

    # ratio = np.shape(phase_map)[1] / np.shape(phase_map)[0]

    fig, ax = plt.subplots(figsize = (15, 25))
    plt.pcolormesh(phase_map, cmap = cMap, rasterized = True)
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')
    plt.legend(handles = patches, prop={'size': 16})
    plt.colorbar()
    plt.tight_layout()
    plt.gca().invert_yaxis()
    plt.savefig(filename+'_phases.pdf', dpi = 300)
    plt.show()

    # mineralindex = mineralindex.drop(index='BadSpectra')

    for i in mineralindex.Label.ravel(): 
        min_phase_pixels = 5
        phase_mask = phase_map!=i
        mineral = np.ma.masked_array(phase_map, mask = phase_mask)
        mineral = mineral.filled(fill_value=0)

        mineral_mask_open = morphology.area_opening(mineral, min_phase_pixels)
        min_mask_morph = (mineral_mask_open==0)

        mg_mask = np.ma.masked_array(mg_map, mask = min_mask_morph)
        fe_mask = np.ma.masked_array(fe_map, mask = min_mask_morph)
        ca_mask = np.ma.masked_array(ca_map, mask = min_mask_morph)
        al_mask = np.ma.masked_array(al_map, mask = min_mask_morph)
        si_mask = np.ma.masked_array(si_map, mask = min_mask_morph)
        o_mask = np.ma.masked_array(o_map, mask = min_mask_morph)

        min_dist_mean = np.array([np.nanmean(mg_mask), np.nanmean(fe_mask), np.nanmean(ca_mask), np.nanmean(al_mask), np.nanmean(si_mask), np.nanmean(o_mask)])

        mg_95 = stats.t.interval(0.95, len(mg_mask.compressed())-1, loc=np.nanmean(mg_mask.compressed())) #, scale=stats.sem(mg_mask.compressed()))
        fe_95 = stats.t.interval(0.95, len(fe_mask.compressed())-1, loc=np.nanmean(fe_mask.compressed())) #, scale=stats.sem(fe_mask.compressed()))
        ca_95 = stats.t.interval(0.95, len(ca_mask.compressed())-1, loc=np.nanmean(ca_mask.compressed())) #, scale=stats.sem(ca_mask.compressed()))
        al_95 = stats.t.interval(0.95, len(al_mask.compressed())-1, loc=np.nanmean(al_mask.compressed())) #, scale=stats.sem(al_mask.compressed()))
        si_95 = stats.t.interval(0.95, len(si_mask.compressed())-1, loc=np.nanmean(si_mask.compressed())) #, scale=stats.sem(si_mask.compressed()))
        o_95 = stats.t.interval(0.95, len(o_mask.compressed())-1, loc=np.nanmean(o_mask.compressed())) #, scale=stats.sem(o_mask.compressed()))

        fig, ax = plt.subplots(figsize = (8.5, 12.5))
        plt.subplot(4, 2, 1)
        plt.pcolormesh(min_mask_morph, cmap = cMapnew, rasterized = True)
        plt.pcolormesh(phase_map, cmap = cMap, alpha = 0.2, rasterized = True)
        plt.title('Mineral = ' + str((mineralindex[mineralindex['Label'] == i]).index[0]))
        plt.gca().invert_yaxis()

        plt.subplot(4, 2, 2)
        sns.barplot(x = min_dist_index, y = min_dist_mean)
        plt.title('Concentration Average')
        plt.ylim([0, 40])

        plt.subplot(4, 2, 3)
        plt.pcolormesh(mg_mask, rasterized = True)
        plt.title('Mg 95% CI [' + str(round(mg_95[0], 3)) + '-' + str(round(mg_95[1], 3)) + ']')
        cbar = plt.colorbar(aspect=50, orientation='vertical') 
        plt.gca().invert_yaxis()

        plt.subplot(4, 2, 4)
        plt.pcolormesh(fe_mask, rasterized = True) 
        plt.title('Fe 95% CI [' + str(round(fe_95[0], 3)) + '-' + str(round(fe_95[1], 3)) + ']')
        plt.colorbar(aspect=50, orientation='vertical') 
        plt.gca().invert_yaxis()

        plt.subplot(4, 2, 5)
        plt.pcolormesh(ca_mask, rasterized = True)
        plt.title('Ca 95% CI [' + str(round(ca_95[0], 3)) + '-' + str(round(ca_95[1], 3)) + ']')
        plt.colorbar(aspect=50, orientation='vertical') 
        plt.gca().invert_yaxis()

        plt.subplot(4, 2, 6)
        plt.pcolormesh(al_mask, rasterized = True)
        plt.title('Al 95% CI [' + str(round(al_95[0], 3)) + '-' + str(round(al_95[1], 3)) + ']')
        plt.colorbar(aspect=50, orientation='vertical') 
        plt.gca().invert_yaxis()

        plt.subplot(4, 2, 7)
        plt.pcolormesh(si_mask, rasterized = True)
        plt.title('Si 95% CI [' + str(round(si_95[0], 3)) + '-' + str(round(si_95[1], 3)) + ']')
        plt.colorbar(aspect=50, orientation='vertical') 
        plt.gca().invert_yaxis()

        plt.subplot(4, 2, 8)
        plt.pcolormesh(o_mask, rasterized = True)
        plt.title('O 95% CI [' + str(round(o_95[0], 3)) + '-' + str(round(o_95[1], 3)) + ']')
        plt.colorbar(aspect=50, orientation='vertical') 
        plt.gca().invert_yaxis()

        fig.text(0.5, 0, 'X Pixel', ha='center')
        fig.text(0, 0.5, 'Y Pixel', va='center', rotation='vertical')
        plt.tight_layout()

        plt.savefig(filename+'_'+str((mineralindex[mineralindex['Label'] == i]).index[0])+'.pdf', dpi = 250)
        plt.show()


# %%


def mineralpca(mineral, plotting, mg_mask, fe_mask, ca_mask, al_mask, si_mask, o_mask, samplename): 

    """ mineralpca takes in the mineral information and outputs PCA components and scores. 

        inputs: mineral, chemical masks, figure export lable
        output: pca, mineral pca, pca score dataframe"""

    pca = PCA(n_components = 4)
    pca6 = PCA(n_components = 6)

    if mineral == 'olivine': 
        ol_dist = ['Mg.1', 'Fe.1', 'Si.1', 'O.1']
        min_pca_df = pd.DataFrame(columns = ol_dist)
        min_pca_df['Mg.1'] = mg_mask.compressed()
        min_pca_df['Fe.1'] = fe_mask.compressed()
        min_pca_df['Si.1'] = si_mask.compressed()
        min_pca_df['O.1'] = o_mask.compressed()
        min_pca_df = min_pca_df.fillna(0)
        principalcomponents = pca.fit_transform(min_pca_df)
        pca_df = pd.DataFrame(data = principalcomponents, columns = ['PC1', 'PC2', 'PC3', 'PC4'])

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
            plt.scatter(pca_df.columns, pca.explained_variance_ratio_, lw = 0.25, edgecolors = 'k', rasterized=True)
            plt.xlabel('PCA')
            plt.ylabel('Explained Variance')
            plt.tight_layout()
            plt.savefig(samplename + '_Variance_Olivine.pdf')

            plt.figure(figsize = (8.5, 9))
            plt.subplot(3, 2, 1)
            plt.scatter(pca_df['PC1'], pca_df['PC2'], c = min_pca_df['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)
            plt.tight_layout()

            plt.subplot(3, 2, 2)
            plt.scatter(pca_df['PC1'], pca_df['PC3'], c = min_pca_df['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC1')
            plt.ylabel('PC3')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)
            plt.tight_layout()

            plt.subplot(3, 2, 3)
            plt.scatter(pca_df['PC1'], pca_df['PC4'], c = min_pca_df['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC1')
            plt.ylabel('PC4')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)
            plt.tight_layout()

            plt.subplot(3, 2, 4)
            plt.scatter(pca_df['PC2'], pca_df['PC3'], c = min_pca_df['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC2')
            plt.ylabel('PC3')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)
            plt.tight_layout()

            plt.subplot(3, 2, 5)
            plt.scatter(pca_df['PC2'], pca_df['PC4'], c = min_pca_df['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC2')
            plt.ylabel('PC4')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)
            plt.tight_layout()

            plt.subplot(3, 2, 6)
            plt.scatter(pca_df['PC3'], pca_df['PC4'], c = min_pca_df['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC3')
            plt.ylabel('PC4')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)
            plt.tight_layout()
            plt.savefig(samplename+'_PCA_Comparison_Olivine.pdf', dpi = 250)


            plt.figure(figsize = (24, 22))
            plt.subplot(4, 4, 1)
            plt.scatter(pca_df['PC1'], min_pca_df['Mg.1'], c = min_pca_df['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC1')
            plt.ylabel('Mg')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)
            plt.tight_layout()

            plt.subplot(4, 4, 2)
            plt.scatter(pca_df['PC1'], min_pca_df['Fe.1'], c = min_pca_df['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC1')
            plt.ylabel('Fe')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)
            plt.tight_layout()

            plt.subplot(4, 4, 3)
            plt.scatter(pca_df['PC1'], min_pca_df['Si.1'], c = min_pca_df['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC1')
            plt.ylabel('Si')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)
            plt.tight_layout()

            plt.subplot(4, 4, 4)
            plt.scatter(pca_df['PC1'], min_pca_df['O.1'], c = min_pca_df['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC1')
            plt.ylabel('O')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)
            plt.tight_layout()

            plt.subplot(4, 4, 5)
            plt.scatter(pca_df['PC2'], min_pca_df['Mg.1'], c = min_pca_df['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC2')
            plt.ylabel('Mg')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)
            plt.tight_layout()

            plt.subplot(4, 4, 6)
            plt.scatter(pca_df['PC2'], min_pca_df['Fe.1'], c = min_pca_df['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC2')
            plt.ylabel('Fe')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)
            plt.tight_layout()

            plt.subplot(4, 4, 7)
            plt.scatter(pca_df['PC2'], min_pca_df['Si.1'], c = min_pca_df['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC2')
            plt.ylabel('Si')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)
            plt.tight_layout()

            plt.subplot(4, 4, 8)
            plt.scatter(pca_df['PC2'], min_pca_df['O.1'], c = min_pca_df['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC2')
            plt.ylabel('O')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)
            plt.tight_layout()

            plt.subplot(4, 4, 9)
            plt.scatter(pca_df['PC3'], min_pca_df['Mg.1'], c = min_pca_df['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC3')
            plt.ylabel('Mg')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)
            plt.tight_layout()

            plt.subplot(4, 4, 10)
            plt.scatter(pca_df['PC3'], min_pca_df['Fe.1'], c = min_pca_df['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC3')
            plt.ylabel('Fe')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)
            plt.tight_layout()

            plt.subplot(4, 4, 11)
            plt.scatter(pca_df['PC3'], min_pca_df['Si.1'], c = min_pca_df['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC3')
            plt.ylabel('Si')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)
            plt.tight_layout()

            plt.subplot(4, 4, 12)
            plt.scatter(pca_df['PC3'], min_pca_df['O.1'], c = min_pca_df['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC3')
            plt.ylabel('O')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)
            plt.tight_layout()

            plt.subplot(4, 4, 13)
            plt.scatter(pca_df['PC4'], min_pca_df['Mg.1'], c = min_pca_df['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC4')
            plt.ylabel('Mg')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)
            plt.tight_layout()

            plt.subplot(4, 4, 14)
            plt.scatter(pca_df['PC4'], min_pca_df['Fe.1'], c = min_pca_df['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC4')
            plt.ylabel('Fe')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)
            plt.tight_layout()

            plt.subplot(4, 4, 15)
            plt.scatter(pca_df['PC4'], min_pca_df['Si.1'], c = min_pca_df['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC4')
            plt.ylabel('Si')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)
            plt.tight_layout()

            plt.subplot(4, 4, 16)
            plt.scatter(pca_df['PC4'], min_pca_df['O.1'], c = min_pca_df['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC4')
            plt.ylabel('O')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)
            plt.tight_layout()
            plt.savefig(samplename+'_PCA_Elements_Olivine.pdf')
            plt.show()



    elif mineral == 'plagioclase':
        plag_dist = ['Ca.1', 'Al.1', 'Si.1', 'O.1']
        min_pca_df = pd.DataFrame(columns = plag_dist)
        min_pca_df['Ca.1'] = ca_mask.compressed()
        min_pca_df['Al.1'] = al_mask.compressed()
        min_pca_df['Si.1'] = si_mask.compressed()
        min_pca_df['O.1'] = o_mask.compressed()
        min_pca_df = min_pca_df.fillna(0)
        principalcomponents = pca.fit_transform(min_pca_df)
        pca_df = pd.DataFrame(data = principalcomponents, columns = ['PC1', 'PC2', 'PC3', 'PC4'])
        
        if plotting == 1: 
            plt.figure(figsize = (10, 4))
            plt.subplot(1, 2, 1)
            plt.pcolormesh(ca_mask, rasterized = True)
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
            plt.savefig(samplename+'_Variance_Plag.pdf')


            plt.figure(figsize = (8.5, 9))
            plt.subplot(3, 2, 1)
            plt.scatter(pca_df['PC1'], pca_df['PC2'], c = min_pca_df['Ca.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            cbar = plt.colorbar()
            cbar.set_label('Ca', size = 10)
            plt.tight_layout()

            plt.subplot(3, 2, 2)
            plt.scatter(pca_df['PC1'], pca_df['PC3'], c = min_pca_df['Ca.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC1')
            plt.ylabel('PC3')
            cbar = plt.colorbar()
            cbar.set_label('Ca', size = 10)
            plt.tight_layout()

            plt.subplot(3, 2, 3)
            plt.scatter(pca_df['PC1'], pca_df['PC4'], c = min_pca_df['Ca.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC1')
            plt.ylabel('PC4')
            cbar = plt.colorbar()
            cbar.set_label('Ca', size = 10)
            plt.tight_layout()

            plt.subplot(3, 2, 4)
            plt.scatter(pca_df['PC2'], pca_df['PC3'], c = min_pca_df['Ca.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC2')
            plt.ylabel('PC3')
            cbar = plt.colorbar()
            cbar.set_label('Ca', size = 10)
            plt.tight_layout()

            plt.subplot(3, 2, 5)
            plt.scatter(pca_df['PC2'], pca_df['PC4'], c = min_pca_df['Ca.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC2')
            plt.ylabel('PC4')
            cbar = plt.colorbar()
            cbar.set_label('Ca', size = 10)
            plt.tight_layout()

            plt.subplot(3, 2, 6)
            plt.scatter(pca_df['PC3'], pca_df['PC4'], c = min_pca_df['Ca.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC3')
            plt.ylabel('PC4')
            cbar = plt.colorbar()
            cbar.set_label('Ca', size = 10)
            plt.tight_layout()
            plt.savefig(samplename+'_PCA_Comparison_Plag.pdf')
            plt.show()

            plt.figure(figsize = (24, 22))
            plt.subplot(4, 4, 1)
            plt.scatter(pca_df['PC1'], min_pca_df['Ca.1'], c = min_pca_df['Ca.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC1')
            plt.ylabel('Ca')
            cbar = plt.colorbar()
            cbar.set_label('Ca', size = 10)
            plt.tight_layout()

            plt.subplot(4, 4, 2)
            plt.scatter(pca_df['PC1'], min_pca_df['Al.1'], c = min_pca_df['Ca.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC1')
            plt.ylabel('Al')
            cbar = plt.colorbar()
            cbar.set_label('Ca', size = 10)
            plt.tight_layout()

            plt.subplot(4, 4, 3)
            plt.scatter(pca_df['PC1'], min_pca_df['Si.1'], c = min_pca_df['Ca.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC1')
            plt.ylabel('Si')
            cbar = plt.colorbar()
            cbar.set_label('Ca', size = 10)
            plt.tight_layout()

            plt.subplot(4, 4, 4)
            plt.scatter(pca_df['PC1'], min_pca_df['O.1'], c = min_pca_df['Ca.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC1')
            plt.ylabel('O')
            cbar = plt.colorbar()
            cbar.set_label('Ca', size = 10)
            plt.tight_layout()

            plt.subplot(4, 4, 5)
            plt.scatter(pca_df['PC2'], min_pca_df['Ca.1'], c = min_pca_df['Ca.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC2')
            plt.ylabel('Ca')
            cbar = plt.colorbar()
            cbar.set_label('Ca', size = 10)
            plt.tight_layout()

            plt.subplot(4, 4, 6)
            plt.scatter(pca_df['PC2'], min_pca_df['Al.1'], c = min_pca_df['Ca.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC2')
            plt.ylabel('Al')
            cbar = plt.colorbar()
            cbar.set_label('Ca', size = 10)

            plt.subplot(4, 4, 7)
            plt.scatter(pca_df['PC2'], min_pca_df['Si.1'], c = min_pca_df['Ca.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC2')
            plt.ylabel('Si')
            cbar = plt.colorbar()
            cbar.set_label('Ca', size = 10)

            plt.subplot(4, 4, 8)
            plt.scatter(pca_df['PC2'], min_pca_df['O.1'], c = min_pca_df['Ca.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC2')
            plt.ylabel('O')
            cbar = plt.colorbar()
            cbar.set_label('Ca', size = 10)

            plt.subplot(4, 4, 9)
            plt.scatter(pca_df['PC3'], min_pca_df['Ca.1'], c = min_pca_df['Ca.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC3')
            plt.ylabel('Ca')
            cbar = plt.colorbar()
            cbar.set_label('Ca', size = 10)

            plt.subplot(4, 4, 10)
            plt.scatter(pca_df['PC3'], min_pca_df['Al.1'], c = min_pca_df['Ca.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC3')
            plt.ylabel('Al')
            cbar = plt.colorbar()
            cbar.set_label('Ca', size = 10)

            plt.subplot(4, 4, 11)
            plt.scatter(pca_df['PC3'], min_pca_df['Si.1'], c = min_pca_df['Ca.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC3')
            plt.ylabel('Si')
            cbar = plt.colorbar()
            cbar.set_label('Ca', size = 10)

            plt.subplot(4, 4, 12)
            plt.scatter(pca_df['PC3'], min_pca_df['O.1'], c = min_pca_df['Ca.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC3')
            plt.ylabel('O')
            cbar = plt.colorbar()
            cbar.set_label('Ca', size = 10)

            plt.subplot(4, 4, 13)
            plt.scatter(pca_df['PC4'], min_pca_df['Ca.1'], c = min_pca_df['Ca.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC4')
            plt.ylabel('Ca')
            cbar = plt.colorbar()
            cbar.set_label('Ca', size = 10)

            plt.subplot(4, 4, 14)
            plt.scatter(pca_df['PC4'], min_pca_df['Al.1'], c = min_pca_df['Ca.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC4')
            plt.ylabel('Al')
            cbar = plt.colorbar()
            cbar.set_label('Ca', size = 10)

            plt.subplot(4, 4, 15)
            plt.scatter(pca_df['PC4'], min_pca_df['Si.1'], c = min_pca_df['Ca.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC4')
            plt.ylabel('Si')
            cbar = plt.colorbar()
            cbar.set_label('Ca', size = 10)

            plt.subplot(4, 4, 16)
            plt.scatter(pca_df['PC4'], min_pca_df['O.1'], c = min_pca_df['Ca.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC4')
            plt.ylabel('O')
            cbar = plt.colorbar()
            cbar.set_label('Ca', size = 10)
            plt.tight_layout()
            plt.savefig(samplename+'_PCA_Elements_Plag.pdf')
            plt.show()

    elif mineral == 'clinopyroxene':
        cpx_dist = ['Mg.1', 'Fe.1', 'Ca.1', 'Si.1']
        min_pca_df = pd.DataFrame(columns = cpx_dist)
        min_pca_df['Mg.1'] = mg_mask.compressed()
        min_pca_df['Fe.1'] = fe_mask.compressed()
        min_pca_df['Ca.1'] = ca_mask.compressed()
        min_pca_df['Si.1'] = si_mask.compressed()
        min_pca_df = min_pca_df.fillna(0)
        principalcomponents = pca.fit_transform(min_pca_df)
        pca_df = pd.DataFrame(data = principalcomponents, columns = ['PC1', 'PC2', 'PC3', 'PC4'])
        
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
            plt.savefig(samplename+'_Variance_Cpx.pdf')


            plt.figure(figsize = (8.5, 9))
            plt.subplot(3, 2, 1)
            plt.scatter(pca_df['PC1'], pca_df['PC2'], c = min_pca_df['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)
            plt.tight_layout()

            plt.subplot(3, 2, 2)
            plt.scatter(pca_df['PC1'], pca_df['PC3'], c = min_pca_df['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC1')
            plt.ylabel('PC3')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)
            plt.tight_layout()

            plt.subplot(3, 2, 3)
            plt.scatter(pca_df['PC1'], pca_df['PC4'], c = min_pca_df['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC1')
            plt.ylabel('PC4')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)
            plt.tight_layout()

            plt.subplot(3, 2, 4)
            plt.scatter(pca_df['PC2'], pca_df['PC3'], c = min_pca_df['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC2')
            plt.ylabel('PC3')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)
            plt.tight_layout()

            plt.subplot(3, 2, 5)
            plt.scatter(pca_df['PC2'], pca_df['PC4'], c = min_pca_df['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC2')
            plt.ylabel('PC4')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)
            plt.tight_layout()

            plt.subplot(3, 2, 6)
            plt.scatter(pca_df['PC3'], pca_df['PC4'], c = min_pca_df['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC3')
            plt.ylabel('PC4')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)
            plt.tight_layout()
            plt.savefig(samplename+'_PCA_Comparison_Cpx.pdf')
            plt.show()

            plt.figure(figsize = (24, 22))
            plt.subplot(4, 4, 1)
            plt.scatter(pca_df['PC1'], min_pca_df['Mg.1'], c = min_pca_df['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC1')
            plt.ylabel('Mg')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)
            plt.tight_layout()

            plt.subplot(4, 4, 2)
            plt.scatter(pca_df['PC1'], min_pca_df['Fe.1'], c = min_pca_df['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC1')
            plt.ylabel('Fe')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)

            plt.subplot(4, 4, 3)
            plt.scatter(pca_df['PC1'], min_pca_df['Ca.1'], c = min_pca_df['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC1')
            plt.ylabel('Ca')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)

            plt.subplot(4, 4, 4)
            plt.scatter(pca_df['PC1'], min_pca_df['Si.1'], c = min_pca_df['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC1')
            plt.ylabel('Si')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)

            plt.subplot(4, 4, 5)
            plt.scatter(pca_df['PC2'], min_pca_df['Mg.1'], c = min_pca_df['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC2')
            plt.ylabel('Mg')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)

            plt.subplot(4, 4, 6)
            plt.scatter(pca_df['PC2'], min_pca_df['Fe.1'], c = min_pca_df['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC2')
            plt.ylabel('Fe')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)

            plt.subplot(4, 4, 7)
            plt.scatter(pca_df['PC2'], min_pca_df['Ca.1'], c = min_pca_df['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC2')
            plt.ylabel('Ca')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)

            plt.subplot(4, 4, 8)
            plt.scatter(pca_df['PC2'], min_pca_df['Si.1'], c = min_pca_df['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC2')
            plt.ylabel('Si')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)

            plt.subplot(4, 4, 9)
            plt.scatter(pca_df['PC3'], min_pca_df['Mg.1'], c = min_pca_df['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC3')
            plt.ylabel('Mg')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)

            plt.subplot(4, 4, 10)
            plt.scatter(pca_df['PC3'], min_pca_df['Fe.1'], c = min_pca_df['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC3')
            plt.ylabel('Fe')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)

            plt.subplot(4, 4, 11)
            plt.scatter(pca_df['PC3'], min_pca_df['Ca.1'], c = min_pca_df['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC3')
            plt.ylabel('Ca')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)

            plt.subplot(4, 4, 12)
            plt.scatter(pca_df['PC3'], min_pca_df['Si.1'], c = min_pca_df['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC3')
            plt.ylabel('Si')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)

            plt.subplot(4, 4, 13)
            plt.scatter(pca_df['PC4'], min_pca_df['Mg.1'], c = min_pca_df['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC4')
            plt.ylabel('Mg')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)

            plt.subplot(4, 4, 14)
            plt.scatter(pca_df['PC4'], min_pca_df['Fe.1'], c = min_pca_df['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC4')
            plt.ylabel('Fe')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)

            plt.subplot(4, 4, 15)
            plt.scatter(pca_df['PC4'], min_pca_df['Ca.1'], c = min_pca_df['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC4')
            plt.ylabel('Ca')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)

            plt.subplot(4, 4, 16)
            plt.scatter(pca_df['PC4'], min_pca_df['Si.1'], c = min_pca_df['Fe.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC4')
            plt.ylabel('Si')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)
            plt.tight_layout()
            plt.savefig(samplename+'_PCA_Elements_Cpx.pdf')
            plt.show()

    elif mineral == 'glass':
        glass_dist = ['Mg.1', 'Fe.1', 'Ca.1', 'Al.1', 'Si.1', 'O.1']
        min_pca_df = pd.DataFrame(columns = glass_dist)
        min_pca_df['Mg.1'] = mg_mask.compressed()
        min_pca_df['Fe.1'] = fe_mask.compressed()
        min_pca_df['Ca.1'] = ca_mask.compressed()
        min_pca_df['Al.1'] = al_mask.compressed()
        min_pca_df['Si.1'] = si_mask.compressed()
        min_pca_df['O.1'] = o_mask.compressed()
        min_pca_df = min_pca_df.fillna(0)
        principalcomponents = pca6.fit_transform(min_pca_df)
        pca_df = pd.DataFrame(data = principalcomponents, columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6'])
        
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
            plt.scatter(pca_df.columns, pca6.explained_variance_ratio_, lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PCA')
            plt.ylabel('Explained Variance')
            plt.tight_layout()
            plt.savefig(samplename+'_Variance_Glass.pdf')

            plt.figure(figsize = (8.5, 9))
            plt.subplot(3, 2, 1)
            plt.scatter(pca_df['PC1'], pca_df['PC2'], c = min_pca_df['Mg.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            cbar = plt.colorbar()
            cbar.set_label('Mg', size = 10)
            plt.tight_layout()

            plt.subplot(3, 2, 2)
            plt.scatter(pca_df['PC1'], pca_df['PC3'], c = min_pca_df['Mg.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC1')
            plt.ylabel('PC3')
            cbar = plt.colorbar()
            cbar.set_label('Mg', size = 10)
            plt.tight_layout()

            plt.subplot(3, 2, 3)
            plt.scatter(pca_df['PC1'], pca_df['PC4'], c = min_pca_df['Mg.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC1')
            plt.ylabel('PC4')
            cbar = plt.colorbar()
            cbar.set_label('Mg', size = 10)
            plt.tight_layout()

            plt.subplot(3, 2, 4)
            plt.scatter(pca_df['PC2'], pca_df['PC3'], c = min_pca_df['Mg.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC2')
            plt.ylabel('PC3')
            cbar = plt.colorbar()
            cbar.set_label('Mg', size = 10)
            plt.tight_layout()

            plt.subplot(3, 2, 5)
            plt.scatter(pca_df['PC2'], pca_df['PC4'], c = min_pca_df['Mg.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC2')
            plt.ylabel('PC4')
            cbar = plt.colorbar()
            cbar.set_label('Mg', size = 10)
            plt.tight_layout()

            plt.subplot(3, 2, 6)
            plt.scatter(pca_df['PC3'], pca_df['PC4'], c = min_pca_df['Mg.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC3')
            plt.ylabel('PC4')
            cbar = plt.colorbar()
            cbar.set_label('Mg', size = 10)
            plt.tight_layout()
            plt.savefig(samplename+'_PCA_Comparison_Glass.pdf')
            plt.show()

            

            plt.figure(figsize = (24, 33))
            plt.subplot(6, 4, 1)
            plt.scatter(pca_df['PC1'], min_pca_df['Mg.1'], c = min_pca_df['Mg.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC1')
            plt.ylabel('Mg')
            cbar = plt.colorbar()
            cbar.set_label('Mg', size = 10)
            plt.tight_layout()

            plt.subplot(6, 4, 2)
            plt.scatter(pca_df['PC1'], min_pca_df['Fe.1'], c = min_pca_df['Mg.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC1')
            plt.ylabel('Fe')
            cbar = plt.colorbar()
            cbar.set_label('Mg', size = 10)

            plt.subplot(6, 4, 3)
            plt.scatter(pca_df['PC1'], min_pca_df['Ca.1'], c = min_pca_df['Mg.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC1')
            plt.ylabel('Ca')
            cbar = plt.colorbar()
            cbar.set_label('Mg', size = 10)

            plt.subplot(6, 4, 4)
            plt.scatter(pca_df['PC1'], min_pca_df['Al.1'], c = min_pca_df['Mg.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC1')
            plt.ylabel('Al')
            cbar = plt.colorbar()
            cbar.set_label('Mg', size = 10)

            plt.subplot(6, 4, 5)
            plt.scatter(pca_df['PC1'], min_pca_df['Si.1'], c = min_pca_df['Mg.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC1')
            plt.ylabel('Si')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)

            plt.subplot(6, 4, 6)
            plt.scatter(pca_df['PC1'], min_pca_df['O.1'], c = min_pca_df['Mg.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC1')
            plt.ylabel('O')
            cbar = plt.colorbar()
            cbar.set_label('Fe', size = 10)

            plt.subplot(6, 4, 7)
            plt.scatter(pca_df['PC2'], min_pca_df['Mg.1'], c = min_pca_df['Mg.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC2')
            plt.ylabel('Mg')
            cbar = plt.colorbar()
            cbar.set_label('Mg', size = 10)

            plt.subplot(6, 4, 8)
            plt.scatter(pca_df['PC2'], min_pca_df['Fe.1'], c = min_pca_df['Mg.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC2')
            plt.ylabel('Fe')
            cbar = plt.colorbar()
            cbar.set_label('Mg', size = 10)

            plt.subplot(6, 4, 9)
            plt.scatter(pca_df['PC2'], min_pca_df['Ca.1'], c = min_pca_df['Mg.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC2')
            plt.ylabel('Ca')
            cbar = plt.colorbar()
            cbar.set_label('Mg', size = 10)

            plt.subplot(6, 4, 10)
            plt.scatter(pca_df['PC2'], min_pca_df['Al.1'], c = min_pca_df['Mg.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC2')
            plt.ylabel('Al')
            cbar = plt.colorbar()
            cbar.set_label('Mg', size = 10)

            plt.subplot(6, 4, 11)
            plt.scatter(pca_df['PC2'], min_pca_df['Si.1'], c = min_pca_df['Mg.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC2')
            plt.ylabel('Si')
            cbar = plt.colorbar()
            cbar.set_label('Mg', size = 10)

            plt.subplot(6, 4, 12)
            plt.scatter(pca_df['PC3'], min_pca_df['O.1'], c = min_pca_df['Mg.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC3')
            plt.ylabel('O')
            cbar = plt.colorbar()
            cbar.set_label('Mg', size = 10)

            plt.subplot(6, 4, 13)
            plt.scatter(pca_df['PC3'], min_pca_df['Mg.1'], c = min_pca_df['Mg.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC3')
            plt.ylabel('Mg')
            cbar = plt.colorbar()
            cbar.set_label('Mg', size = 10)

            plt.subplot(6, 4, 14)
            plt.scatter(pca_df['PC3'], min_pca_df['Fe.1'], c = min_pca_df['Mg.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC3')
            plt.ylabel('Fe')
            cbar = plt.colorbar()
            cbar.set_label('Mg', size = 10)

            plt.subplot(6, 4, 15)
            plt.scatter(pca_df['PC3'], min_pca_df['Ca.1'], c = min_pca_df['Mg.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC3')
            plt.ylabel('Ca')
            cbar = plt.colorbar()
            cbar.set_label('Mg', size = 10)

            plt.subplot(6, 4, 16)
            plt.scatter(pca_df['PC3'], min_pca_df['Al.1'], c = min_pca_df['Mg.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC3')
            plt.ylabel('Al')
            cbar = plt.colorbar()
            cbar.set_label('Mg', size = 10)

            plt.subplot(6, 4, 17)
            plt.scatter(pca_df['PC3'], min_pca_df['Si.1'], c = min_pca_df['Mg.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC3')
            plt.ylabel('Si')
            cbar = plt.colorbar()
            cbar.set_label('Mg', size = 10)

            plt.subplot(6, 4, 18)
            plt.scatter(pca_df['PC3'], min_pca_df['O.1'], c = min_pca_df['Mg.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC3')
            plt.ylabel('O')
            cbar = plt.colorbar()
            cbar.set_label('Mg', size = 10)

            plt.subplot(6, 4, 19)
            plt.scatter(pca_df['PC4'], min_pca_df['Mg.1'], c = min_pca_df['Mg.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC4')
            plt.ylabel('Mg')
            cbar = plt.colorbar()
            cbar.set_label('Mg', size = 10)

            plt.subplot(6, 4, 20)
            plt.scatter(pca_df['PC4'], min_pca_df['Fe.1'], c = min_pca_df['Mg.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC4')
            plt.ylabel('Fe')
            cbar = plt.colorbar()
            cbar.set_label('Mg', size = 10)

            plt.subplot(6, 4, 21)
            plt.scatter(pca_df['PC4'], min_pca_df['Ca.1'], c = min_pca_df['Mg.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC4')
            plt.ylabel('Ca')
            cbar = plt.colorbar()
            cbar.set_label('Mg', size = 10)

            plt.subplot(6, 4, 22)
            plt.scatter(pca_df['PC4'], min_pca_df['Al.1'], c = min_pca_df['Mg.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC4')
            plt.ylabel('Al')
            cbar = plt.colorbar()
            cbar.set_label('Mg', size = 10)

            plt.subplot(6, 4, 23)
            plt.scatter(pca_df['PC4'], min_pca_df['Si.1'], c = min_pca_df['Mg.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC4')
            plt.ylabel('Si')
            cbar = plt.colorbar()
            cbar.set_label('Mg', size = 10)

            plt.subplot(6, 4, 24)
            plt.scatter(pca_df['PC4'], min_pca_df['O.1'], c = min_pca_df['Mg.1'], lw = 0.25, edgecolors = 'k', rasterized = True)
            plt.xlabel('PC4')
            plt.ylabel('O')
            cbar = plt.colorbar()
            cbar.set_label('Mg', size = 10)
            plt.tight_layout()
            plt.savefig(samplename+'_PCA_Elements_Glass.pdf')
            plt.show()
        
    return pca, min_pca_df, pca_df


# %%


def mineralnmf(mineral, mg_mask, fe_mask, ca_mask, al_mask, si_mask, o_mask, samplename): 

    """ mineralnmf takes in the mineral information and outputs PCA components and scores. 

        inputs: mineral, chemical masks, figure export lable
        output: nmf, mineral nmf, nmf score dataframe"""

    if mineral == 'olivine': 
        ol_dist = ['Mg.1', 'Fe.1', 'Si.1', 'O.1']
        min_nmf_df = pd.DataFrame(columns = ol_dist)
        min_nmf_df['Mg.1'] = mg_mask.compressed()
        min_nmf_df['Fe.1'] = fe_mask.compressed()
        min_nmf_df['Si.1'] = si_mask.compressed()
        min_nmf_df['O.1'] = o_mask.compressed()

        nmf = NMF(n_components = 4, init = 'random', max_iter = 2000)
        nOL = nmf.fit_transform(min_nmf_df)
        nmf_df = pd.DataFrame(data = nOL, columns = ['NMF1', 'NMF2', 'NMF3', 'NMF4'])

        plt.figure(figsize = (8.5, 9))
        plt.subplot(3, 2, 1)
        plt.scatter(nmf_df['NMF1'], nmf_df['NMF2'], c = min_nmf_df['Mg.1']/(min_nmf_df['Mg.1']+min_nmf_df['Fe.1']), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('NMF1')
        plt.ylabel('NMF2')
        cbar = plt.colorbar()
        cbar.set_label('Fe', size = 10)
        plt.tight_layout()

        plt.subplot(3, 2, 2)
        plt.scatter(nmf_df['NMF1'], nmf_df['NMF3'], c = min_nmf_df['Mg.1']/(min_nmf_df['Mg.1']+min_nmf_df['Fe.1']), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('NMF1')
        plt.ylabel('NMF3')
        cbar = plt.colorbar()
        cbar.set_label('Fe', size = 10)
        plt.tight_layout()

        plt.subplot(3, 2, 3)
        plt.scatter(nmf_df['NMF1'], nmf_df['NMF4'], c = min_nmf_df['Mg.1']/(min_nmf_df['Mg.1']+min_nmf_df['Fe.1']), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('NMF1')
        plt.ylabel('NMF4')
        cbar = plt.colorbar()
        cbar.set_label('Fe', size = 10)
        plt.tight_layout()

        plt.subplot(3, 2, 4)
        plt.scatter(nmf_df['NMF2'], nmf_df['NMF3'], c = min_nmf_df['Mg.1']/(min_nmf_df['Mg.1']+min_nmf_df['Fe.1']), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('NMF2')
        plt.ylabel('NMF3')
        cbar = plt.colorbar()
        cbar.set_label('Fe', size = 10)
        plt.tight_layout()

        plt.subplot(3, 2, 5)
        plt.scatter(nmf_df['NMF2'], nmf_df['NMF4'], c = min_nmf_df['Mg.1']/(min_nmf_df['Mg.1']+min_nmf_df['Fe.1']), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('NMF2')
        plt.ylabel('NMF4')
        cbar = plt.colorbar()
        cbar.set_label('Fe', size = 10)
        plt.tight_layout()

        plt.subplot(3, 2, 6)
        plt.scatter(nmf_df['NMF3'], nmf_df['NMF4'], c = min_nmf_df['Mg.1']/(min_nmf_df['Mg.1']+min_nmf_df['Fe.1']), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('NMF3')
        plt.ylabel('NMF4')
        cbar = plt.colorbar()
        cbar.set_label('Fe', size = 10)
        plt.tight_layout()
        plt.savefig(samplename+'_NMF_Comparison_Olivine.pdf', dpi = 250)

        plt.figure(figsize = (24, 22))
        plt.subplot(4, 4, 1)
        plt.scatter(nmf_df['NMF1'], min_nmf_df['Mg.1'], c = min_nmf_df['Mg.1']/(min_nmf_df['Mg.1']+min_nmf_df['Fe.1']), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('NMF1')
        plt.ylabel('Mg')
        cbar = plt.colorbar()
        cbar.set_label('Fe', size = 10)
        plt.tight_layout()

        plt.subplot(4, 4, 2)
        plt.scatter(nmf_df['NMF1'], min_nmf_df['Fe.1'], c = min_nmf_df['Mg.1']/(min_nmf_df['Mg.1']+min_nmf_df['Fe.1']), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('NMF1')
        plt.ylabel('Fe')
        cbar = plt.colorbar()
        cbar.set_label('Fe', size = 10)
        plt.tight_layout()

        plt.subplot(4, 4, 3)
        plt.scatter(nmf_df['NMF1'], min_nmf_df['Si.1'], c = min_nmf_df['Mg.1']/(min_nmf_df['Mg.1']+min_nmf_df['Fe.1']), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('NMF1')
        plt.ylabel('Si')
        cbar = plt.colorbar()
        cbar.set_label('Fe', size = 10)
        plt.tight_layout()

        plt.subplot(4, 4, 4)
        plt.scatter(nmf_df['NMF1'], min_nmf_df['O.1'], c = min_nmf_df['Mg.1']/(min_nmf_df['Mg.1']+min_nmf_df['Fe.1']), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('NMF1')
        plt.ylabel('O')
        cbar = plt.colorbar()
        cbar.set_label('Fe', size = 10)
        plt.tight_layout()

        plt.subplot(4, 4, 5)
        plt.scatter(nmf_df['NMF2'], min_nmf_df['Mg.1'], c = min_nmf_df['Mg.1']/(min_nmf_df['Mg.1']+min_nmf_df['Fe.1']), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('NMF2')
        plt.ylabel('Mg')
        cbar = plt.colorbar()
        cbar.set_label('Fe', size = 10)
        plt.tight_layout()

        plt.subplot(4, 4, 6)
        plt.scatter(nmf_df['NMF2'], min_nmf_df['Fe.1'], c = min_nmf_df['Mg.1']/(min_nmf_df['Mg.1']+min_nmf_df['Fe.1']), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('NMF2')
        plt.ylabel('Fe')
        cbar = plt.colorbar()
        cbar.set_label('Fe', size = 10)
        plt.tight_layout()

        plt.subplot(4, 4, 7)
        plt.scatter(nmf_df['NMF2'], min_nmf_df['Si.1'], c = min_nmf_df['Mg.1']/(min_nmf_df['Mg.1']+min_nmf_df['Fe.1']), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('NMF2')
        plt.ylabel('Si')
        cbar = plt.colorbar()
        cbar.set_label('Fe', size = 10)
        plt.tight_layout()

        plt.subplot(4, 4, 8)
        plt.scatter(nmf_df['NMF2'], min_nmf_df['O.1'], c = min_nmf_df['Mg.1']/(min_nmf_df['Mg.1']+min_nmf_df['Fe.1']), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('NMF2')
        plt.ylabel('O')
        cbar = plt.colorbar()
        cbar.set_label('Fe', size = 10)
        plt.tight_layout()

        plt.subplot(4, 4, 9)
        plt.scatter(nmf_df['NMF3'], min_nmf_df['Mg.1'], c = min_nmf_df['Mg.1']/(min_nmf_df['Mg.1']+min_nmf_df['Fe.1']), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('NMF3')
        plt.ylabel('Mg')
        cbar = plt.colorbar()
        cbar.set_label('Fe', size = 10)
        plt.tight_layout()

        plt.subplot(4, 4, 10)
        plt.scatter(nmf_df['NMF3'], min_nmf_df['Fe.1'], c = min_nmf_df['Mg.1']/(min_nmf_df['Mg.1']+min_nmf_df['Fe.1']), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('NMF3')
        plt.ylabel('Fe')
        cbar = plt.colorbar()
        cbar.set_label('Fe', size = 10)
        plt.tight_layout()

        plt.subplot(4, 4, 11)
        plt.scatter(nmf_df['NMF3'], min_nmf_df['Si.1'], c = min_nmf_df['Mg.1']/(min_nmf_df['Mg.1']+min_nmf_df['Fe.1']), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('NMF3')
        plt.ylabel('Si')
        cbar = plt.colorbar()
        cbar.set_label('Fe', size = 10)
        plt.tight_layout()

        plt.subplot(4, 4, 12)
        plt.scatter(nmf_df['NMF3'], min_nmf_df['O.1'], c = min_nmf_df['Mg.1']/(min_nmf_df['Mg.1']+min_nmf_df['Fe.1']), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('NMF3')
        plt.ylabel('O')
        cbar = plt.colorbar()
        cbar.set_label('Fe', size = 10)
        plt.tight_layout()

        plt.subplot(4, 4, 13)
        plt.scatter(nmf_df['NMF4'], min_nmf_df['Mg.1'], c = min_nmf_df['Mg.1']/(min_nmf_df['Mg.1']+min_nmf_df['Fe.1']), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('NMF4')
        plt.ylabel('Mg')
        cbar = plt.colorbar()
        cbar.set_label('Fe', size = 10)
        plt.tight_layout()

        plt.subplot(4, 4, 14)
        plt.scatter(nmf_df['NMF4'], min_nmf_df['Fe.1'], c = min_nmf_df['Mg.1']/(min_nmf_df['Mg.1']+min_nmf_df['Fe.1']), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('NMF4')
        plt.ylabel('Fe')
        cbar = plt.colorbar()
        cbar.set_label('Fe', size = 10)
        plt.tight_layout()

        plt.subplot(4, 4, 15)
        plt.scatter(nmf_df['NMF4'], min_nmf_df['Si.1'], c = min_nmf_df['Mg.1']/(min_nmf_df['Mg.1']+min_nmf_df['Fe.1']), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('NMF4')
        plt.ylabel('Si')
        cbar = plt.colorbar()
        cbar.set_label('Fe', size = 10)
        plt.tight_layout()

        plt.subplot(4, 4, 16)
        plt.scatter(nmf_df['NMF4'], min_nmf_df['O.1'], c = min_nmf_df['Mg.1']/(min_nmf_df['Mg.1']+min_nmf_df['Fe.1']), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('NMF4')
        plt.ylabel('O')
        cbar = plt.colorbar()
        cbar.set_label('Fe', size = 10)
        plt.tight_layout()
        plt.savefig(samplename+'_NMF_Elements_Olivine.pdf')
        plt.show()

    elif mineral == 'plagioclase':
        plag_dist = ['Ca.1', 'Al.1', 'Si.1', 'O.1']
        min_nmf_df = pd.DataFrame(columns = plag_dist)
        min_nmf_df['Ca.1'] = ca_mask.compressed()
        min_nmf_df['Al.1'] = al_mask.compressed()
        min_nmf_df['Si.1'] = si_mask.compressed()
        min_nmf_df['O.1'] = o_mask.compressed()

        nmf = NMF(n_components = 4, init = 'random', max_iter = 2000)
        nPLAG = nmf.fit_transform(min_nmf_df)
        nmf_df = pd.DataFrame(data = nPLAG, columns = ['NMF1', 'NMF2', 'NMF3', 'NMF4'])

        plt.figure(figsize = (10, 4))
        plt.subplot(1, 2, 1)
        plt.pcolormesh(ca_mask, rasterized = True)
        plt.xlabel('X Pixel')
        plt.ylabel('Y Pixel')
        cbar = plt.colorbar()
        cbar.set_label('Ca', size = 10)
        plt.gca().invert_yaxis()
        plt.tight_layout()

        plt.subplot(1, 2, 2)
        plt.scatter(nmf_df.columns, nmf.explained_variance_ratio_, lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('NMF')
        plt.ylabel('Explained Variance')
        plt.tight_layout()
        plt.savefig(samplename+'Variance_Plag.pdf')

        plt.figure(figsize = (8.5, 9))
        plt.subplot(3, 2, 1)
        plt.scatter(nmf_df['NMF1'], nmf_df['NMF2'], c = min_nmf_df['Al.1']/(min_nmf_df['Al.1']+min_nmf_df['Ca.1']), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('NMF1')
        plt.ylabel('NMF2')
        cbar = plt.colorbar()
        cbar.set_label('Ca', size = 10)
        plt.tight_layout()

        plt.subplot(3, 2, 2)
        plt.scatter(nmf_df['NMF1'], nmf_df['NMF3'], c = min_nmf_df['Al.1']/(min_nmf_df['Al.1']+min_nmf_df['Ca.1']), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('NMF1')
        plt.ylabel('NMF3')
        cbar = plt.colorbar()
        cbar.set_label('Ca', size = 10)
        plt.tight_layout()

        plt.subplot(3, 2, 3)
        plt.scatter(nmf_df['NMF1'], nmf_df['NMF4'], c = min_nmf_df['Al.1']/(min_nmf_df['Al.1']+min_nmf_df['Ca.1']), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('NMF1')
        plt.ylabel('NMF4')
        cbar = plt.colorbar()
        cbar.set_label('Ca', size = 10)
        plt.tight_layout()

        plt.subplot(3, 2, 4)
        plt.scatter(nmf_df['NMF2'], nmf_df['NMF3'], c = min_nmf_df['Al.1']/(min_nmf_df['Al.1']+min_nmf_df['Ca.1']), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('NMF2')
        plt.ylabel('NMF3')
        cbar = plt.colorbar()
        cbar.set_label('Ca', size = 10)
        plt.tight_layout()

        plt.subplot(3, 2, 5)
        plt.scatter(nmf_df['NMF2'], nmf_df['NMF4'], c = min_nmf_df['Al.1']/(min_nmf_df['Al.1']+min_nmf_df['Ca.1']), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('NMF2')
        plt.ylabel('NMF4')
        cbar = plt.colorbar()
        cbar.set_label('Ca', size = 10)
        plt.tight_layout()

        plt.subplot(3, 2, 6)
        plt.scatter(nmf_df['NMF3'], nmf_df['NMF4'], c = min_nmf_df['Al.1']/(min_nmf_df['Al.1']+min_nmf_df['Ca.1']), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('NMF3')
        plt.ylabel('NMF4')
        cbar = plt.colorbar()
        cbar.set_label('Ca', size = 10)
        plt.tight_layout()
        plt.savefig(samplename+'NMF_Comparison_Plag.pdf')
        plt.show()

        plt.figure(figsize = (8.5, 9))
        plt.subplot(3, 2, 1)
        plt.scatter(nmf_df['NMF1'], min_nmf_df['Ca.1'], c = min_nmf_df['Al.1']/(min_nmf_df['Al.1']+min_nmf_df['Ca.1']), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('NMF1')
        plt.ylabel('Ca')
        cbar = plt.colorbar()
        cbar.set_label('Ca', size = 10)
        plt.tight_layout()

        plt.subplot(3, 2, 2)
        plt.scatter(nmf_df['NMF1'], min_nmf_df['Al.1'], c = min_nmf_df['Al.1']/(min_nmf_df['Al.1']+min_nmf_df['Ca.1']), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('NMF1')
        plt.ylabel('Al')
        cbar = plt.colorbar()
        cbar.set_label('Ca', size = 10)
        plt.tight_layout()

        plt.subplot(3, 2, 3)
        plt.scatter(nmf_df['NMF1'], min_nmf_df['Si.1'], c = min_nmf_df['Al.1']/(min_nmf_df['Al.1']+min_nmf_df['Ca.1']), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('NMF1')
        plt.ylabel('Si.1')
        cbar = plt.colorbar()
        cbar.set_label('Ca', size = 10)
        plt.tight_layout()

        plt.subplot(3, 2, 4)
        plt.scatter(nmf_df['NMF1'], min_nmf_df['O.1'], c = min_nmf_df['Al.1']/(min_nmf_df['Al.1']+min_nmf_df['Ca.1']), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('NMF1')
        plt.ylabel('O.1')
        cbar = plt.colorbar()
        cbar.set_label('Ca', size = 10)
        plt.tight_layout()

        plt.subplot(3, 2, 5)
        plt.scatter(nmf_df['NMF2'], min_nmf_df['Ca.1'], c = min_nmf_df['Al.1']/(min_nmf_df['Al.1']+min_nmf_df['Ca.1']), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('NMF2')
        plt.ylabel('Ca.1')
        cbar = plt.colorbar()
        cbar.set_label('Ca', size = 10)
        plt.tight_layout()

        plt.subplot(3, 2, 6)
        plt.scatter(nmf_df['NMF2'], min_nmf_df['Al.1'], c = min_nmf_df['Al.1']/(min_nmf_df['Al.1']+min_nmf_df['Ca.1']), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('NMF3')
        plt.ylabel('Al.1')
        cbar = plt.colorbar()
        cbar.set_label('Ca', size = 10)
        plt.tight_layout()
        plt.savefig(samplename+'_NMF_Elements_Plag.pdf')
        plt.show()

    return nmf, min_nmf_df, nmf_df

def mineralnmf(mineral, mg_mask, fe_mask, ca_mask, al_mask, si_mask, o_mask, samplename): 

    if mineral == 'olivine': 
        ol_dist = ['Mg.1', 'Fe.1', 'Si.1', 'O.1']
        min_nmf_df = pd.DataFrame(columns = ol_dist)
        min_nmf_df['Mg.1'] = mg_mask.compressed()
        min_nmf_df['Fe.1'] = fe_mask.compressed()
        min_nmf_df['Si.1'] = si_mask.compressed()
        min_nmf_df['O.1'] = o_mask.compressed()

        nmf = NMF(n_components = 4, init = 'random', max_iter = 2000)
        nOL = nmf.fit_transform(min_nmf_df)
        nmf_df = pd.DataFrame(data = nOL, columns = ['NMF1', 'NMF2', 'NMF3', 'NMF4'])

        plt.figure(figsize = (8.5, 9))
        plt.subplot(3, 2, 1)
        plt.scatter(nmf_df['NMF1'], nmf_df['NMF2'], c = min_nmf_df['Mg.1']/(min_nmf_df['Mg.1']+min_nmf_df['Fe.1']), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('NMF1')
        plt.ylabel('NMF2')
        cbar = plt.colorbar()
        cbar.set_label('Fe', size = 10)
        plt.tight_layout()

        plt.subplot(3, 2, 2)
        plt.scatter(nmf_df['NMF1'], nmf_df['NMF3'], c = min_nmf_df['Mg.1']/(min_nmf_df['Mg.1']+min_nmf_df['Fe.1']), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('NMF1')
        plt.ylabel('NMF3')
        cbar = plt.colorbar()
        cbar.set_label('Fe', size = 10)
        plt.tight_layout()

        plt.subplot(3, 2, 3)
        plt.scatter(nmf_df['NMF1'], nmf_df['NMF4'], c = min_nmf_df['Mg.1']/(min_nmf_df['Mg.1']+min_nmf_df['Fe.1']), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('NMF1')
        plt.ylabel('NMF4')
        cbar = plt.colorbar()
        cbar.set_label('Fe', size = 10)
        plt.tight_layout()

        plt.subplot(3, 2, 4)
        plt.scatter(nmf_df['NMF2'], nmf_df['NMF3'], c = min_nmf_df['Mg.1']/(min_nmf_df['Mg.1']+min_nmf_df['Fe.1']), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('NMF2')
        plt.ylabel('NMF3')
        cbar = plt.colorbar()
        cbar.set_label('Fe', size = 10)
        plt.tight_layout()

        plt.subplot(3, 2, 5)
        plt.scatter(nmf_df['NMF2'], nmf_df['NMF4'], c = min_nmf_df['Mg.1']/(min_nmf_df['Mg.1']+min_nmf_df['Fe.1']), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('NMF2')
        plt.ylabel('NMF4')
        cbar = plt.colorbar()
        cbar.set_label('Fe', size = 10)
        plt.tight_layout()

        plt.subplot(3, 2, 6)
        plt.scatter(nmf_df['NMF3'], nmf_df['NMF4'], c = min_nmf_df['Mg.1']/(min_nmf_df['Mg.1']+min_nmf_df['Fe.1']), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('NMF3')
        plt.ylabel('NMF4')
        cbar = plt.colorbar()
        cbar.set_label('Fe', size = 10)
        plt.tight_layout()
        plt.savefig(samplename+'_NMF_Comparison_Olivine.pdf', dpi = 250)

        plt.figure(figsize = (24, 22))
        plt.subplot(4, 4, 1)
        plt.scatter(nmf_df['NMF1'], min_nmf_df['Mg.1'], c = min_nmf_df['Mg.1']/(min_nmf_df['Mg.1']+min_nmf_df['Fe.1']), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('NMF1')
        plt.ylabel('Mg')
        cbar = plt.colorbar()
        cbar.set_label('Fe', size = 10)
        plt.tight_layout()

        plt.subplot(4, 4, 2)
        plt.scatter(nmf_df['NMF1'], min_nmf_df['Fe.1'], c = min_nmf_df['Mg.1']/(min_nmf_df['Mg.1']+min_nmf_df['Fe.1']), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('NMF1')
        plt.ylabel('Fe')
        cbar = plt.colorbar()
        cbar.set_label('Fe', size = 10)
        plt.tight_layout()

        plt.subplot(4, 4, 3)
        plt.scatter(nmf_df['NMF1'], min_nmf_df['Si.1'], c = min_nmf_df['Mg.1']/(min_nmf_df['Mg.1']+min_nmf_df['Fe.1']), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('NMF1')
        plt.ylabel('Si')
        cbar = plt.colorbar()
        cbar.set_label('Fe', size = 10)
        plt.tight_layout()

        plt.subplot(4, 4, 4)
        plt.scatter(nmf_df['NMF1'], min_nmf_df['O.1'], c = min_nmf_df['Mg.1']/(min_nmf_df['Mg.1']+min_nmf_df['Fe.1']), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('NMF1')
        plt.ylabel('O')
        cbar = plt.colorbar()
        cbar.set_label('Fe', size = 10)
        plt.tight_layout()

        plt.subplot(4, 4, 5)
        plt.scatter(nmf_df['NMF2'], min_nmf_df['Mg.1'], c = min_nmf_df['Mg.1']/(min_nmf_df['Mg.1']+min_nmf_df['Fe.1']), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('NMF2')
        plt.ylabel('Mg')
        cbar = plt.colorbar()
        cbar.set_label('Fe', size = 10)
        plt.tight_layout()

        plt.subplot(4, 4, 6)
        plt.scatter(nmf_df['NMF2'], min_nmf_df['Fe.1'], c = min_nmf_df['Mg.1']/(min_nmf_df['Mg.1']+min_nmf_df['Fe.1']), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('NMF2')
        plt.ylabel('Fe')
        cbar = plt.colorbar()
        cbar.set_label('Fe', size = 10)
        plt.tight_layout()

        plt.subplot(4, 4, 7)
        plt.scatter(nmf_df['NMF2'], min_nmf_df['Si.1'], c = min_nmf_df['Mg.1']/(min_nmf_df['Mg.1']+min_nmf_df['Fe.1']), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('NMF2')
        plt.ylabel('Si')
        cbar = plt.colorbar()
        cbar.set_label('Fe', size = 10)
        plt.tight_layout()

        plt.subplot(4, 4, 8)
        plt.scatter(nmf_df['NMF2'], min_nmf_df['O.1'], c = min_nmf_df['Mg.1']/(min_nmf_df['Mg.1']+min_nmf_df['Fe.1']), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('NMF2')
        plt.ylabel('O')
        cbar = plt.colorbar()
        cbar.set_label('Fe', size = 10)
        plt.tight_layout()

        plt.subplot(4, 4, 9)
        plt.scatter(nmf_df['NMF3'], min_nmf_df['Mg.1'], c = min_nmf_df['Mg.1']/(min_nmf_df['Mg.1']+min_nmf_df['Fe.1']), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('NMF3')
        plt.ylabel('Mg')
        cbar = plt.colorbar()
        cbar.set_label('Fe', size = 10)
        plt.tight_layout()

        plt.subplot(4, 4, 10)
        plt.scatter(nmf_df['NMF3'], min_nmf_df['Fe.1'], c = min_nmf_df['Mg.1']/(min_nmf_df['Mg.1']+min_nmf_df['Fe.1']), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('NMF3')
        plt.ylabel('Fe')
        cbar = plt.colorbar()
        cbar.set_label('Fe', size = 10)
        plt.tight_layout()

        plt.subplot(4, 4, 11)
        plt.scatter(nmf_df['NMF3'], min_nmf_df['Si.1'], c = min_nmf_df['Mg.1']/(min_nmf_df['Mg.1']+min_nmf_df['Fe.1']), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('NMF3')
        plt.ylabel('Si')
        cbar = plt.colorbar()
        cbar.set_label('Fe', size = 10)
        plt.tight_layout()

        plt.subplot(4, 4, 12)
        plt.scatter(nmf_df['NMF3'], min_nmf_df['O.1'], c = min_nmf_df['Mg.1']/(min_nmf_df['Mg.1']+min_nmf_df['Fe.1']), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('NMF3')
        plt.ylabel('O')
        cbar = plt.colorbar()
        cbar.set_label('Fe', size = 10)
        plt.tight_layout()

        plt.subplot(4, 4, 13)
        plt.scatter(nmf_df['NMF4'], min_nmf_df['Mg.1'], c = min_nmf_df['Mg.1']/(min_nmf_df['Mg.1']+min_nmf_df['Fe.1']), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('NMF4')
        plt.ylabel('Mg')
        cbar = plt.colorbar()
        cbar.set_label('Fe', size = 10)
        plt.tight_layout()

        plt.subplot(4, 4, 14)
        plt.scatter(nmf_df['NMF4'], min_nmf_df['Fe.1'], c = min_nmf_df['Mg.1']/(min_nmf_df['Mg.1']+min_nmf_df['Fe.1']), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('NMF4')
        plt.ylabel('Fe')
        cbar = plt.colorbar()
        cbar.set_label('Fe', size = 10)
        plt.tight_layout()

        plt.subplot(4, 4, 15)
        plt.scatter(nmf_df['NMF4'], min_nmf_df['Si.1'], c = min_nmf_df['Mg.1']/(min_nmf_df['Mg.1']+min_nmf_df['Fe.1']), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('NMF4')
        plt.ylabel('Si')
        cbar = plt.colorbar()
        cbar.set_label('Fe', size = 10)
        plt.tight_layout()

        plt.subplot(4, 4, 16)
        plt.scatter(nmf_df['NMF4'], min_nmf_df['O.1'], c = min_nmf_df['Mg.1']/(min_nmf_df['Mg.1']+min_nmf_df['Fe.1']), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('NMF4')
        plt.ylabel('O')
        cbar = plt.colorbar()
        cbar.set_label('Fe', size = 10)
        plt.tight_layout()
        plt.savefig(samplename+'NMF_Elements_Olivine.pdf')
        plt.show()

    elif mineral == 'plagioclase':
        plag_dist = ['Ca.1', 'Al.1', 'Si.1', 'O.1']
        min_nmf_df = pd.DataFrame(columns = plag_dist)
        min_nmf_df['Ca.1'] = ca_mask.compressed()
        min_nmf_df['Al.1'] = al_mask.compressed()
        min_nmf_df['Si.1'] = si_mask.compressed()
        min_nmf_df['O.1'] = o_mask.compressed()

        nmf = NMF(n_components = 4, init = 'random', max_iter = 2000)
        principalcomponents = nmf.fit_transform(min_nmf_df)
        nmf_df = pd.DataFrame(data = principalcomponents, columns = ['NMF1', 'NMF2', 'NMF3', 'NMF4'])

        plt.figure(figsize = (8.5, 9))
        plt.subplot(3, 2, 1)
        plt.scatter(nmf_df['NMF1'], nmf_df['NMF2'], c = min_nmf_df['Al.1']/(min_nmf_df['Al.1']+min_nmf_df['Ca.1']), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('NMF1')
        plt.ylabel('NMF2')
        cbar = plt.colorbar()
        cbar.set_label('Ca', size = 10)
        plt.tight_layout()

        plt.subplot(3, 2, 2)
        plt.scatter(nmf_df['NMF1'], nmf_df['NMF3'], c = min_nmf_df['Al.1']/(min_nmf_df['Al.1']+min_nmf_df['Ca.1']), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('NMF1')
        plt.ylabel('NMF3')
        cbar = plt.colorbar()
        cbar.set_label('Ca', size = 10)
        plt.tight_layout()

        plt.subplot(3, 2, 3)
        plt.scatter(nmf_df['NMF1'], nmf_df['NMF4'], c = min_nmf_df['Al.1']/(min_nmf_df['Al.1']+min_nmf_df['Ca.1']), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('NMF1')
        plt.ylabel('NMF4')
        cbar = plt.colorbar()
        cbar.set_label('Ca', size = 10)
        plt.tight_layout()

        plt.subplot(3, 2, 4)
        plt.scatter(nmf_df['NMF2'], nmf_df['NMF3'], c = min_nmf_df['Al.1']/(min_nmf_df['Al.1']+min_nmf_df['Ca.1']), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('NMF2')
        plt.ylabel('NMF3')
        cbar = plt.colorbar()
        cbar.set_label('Ca', size = 10)
        plt.tight_layout()

        plt.subplot(3, 2, 5)
        plt.scatter(nmf_df['NMF2'], nmf_df['NMF4'], c = min_nmf_df['Al.1']/(min_nmf_df['Al.1']+min_nmf_df['Ca.1']), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('NMF2')
        plt.ylabel('NMF4')
        cbar = plt.colorbar()
        cbar.set_label('Ca', size = 10)
        plt.tight_layout()

        plt.subplot(3, 2, 6)
        plt.scatter(nmf_df['NMF3'], nmf_df['NMF4'], c = min_nmf_df['Al.1']/(min_nmf_df['Al.1']+min_nmf_df['Ca.1']), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('NMF3')
        plt.ylabel('NMF4')
        cbar = plt.colorbar()
        cbar.set_label('Ca', size = 10)
        plt.tight_layout()
        plt.savefig(samplename+'_NMF_Comparison_Plag.pdf')
        plt.show()

        plt.figure(figsize = (8.5, 9))
        plt.subplot(3, 2, 1)
        plt.scatter(nmf_df['NMF1'], min_nmf_df['Ca.1'], c = min_nmf_df['Al.1']/(min_nmf_df['Al.1']+min_nmf_df['Ca.1']), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('NMF1')
        plt.ylabel('Ca')
        cbar = plt.colorbar()
        cbar.set_label('Ca', size = 10)
        plt.tight_layout()

        plt.subplot(3, 2, 2)
        plt.scatter(nmf_df['NMF1'], min_nmf_df['Al.1'], c = min_nmf_df['Al.1']/(min_nmf_df['Al.1']+min_nmf_df['Ca.1']), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('NMF1')
        plt.ylabel('Al')
        cbar = plt.colorbar()
        cbar.set_label('Ca', size = 10)
        plt.tight_layout()

        plt.subplot(3, 2, 3)
        plt.scatter(nmf_df['NMF1'], min_nmf_df['Si.1'], c = min_nmf_df['Al.1']/(min_nmf_df['Al.1']+min_nmf_df['Ca.1']), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('NMF1')
        plt.ylabel('Si.1')
        cbar = plt.colorbar()
        cbar.set_label('Ca', size = 10)
        plt.tight_layout()

        plt.subplot(3, 2, 4)
        plt.scatter(nmf_df['NMF1'], min_nmf_df['O.1'], c = min_nmf_df['Al.1']/(min_nmf_df['Al.1']+min_nmf_df['Ca.1']), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('NMF1')
        plt.ylabel('O.1')
        cbar = plt.colorbar()
        cbar.set_label('Ca', size = 10)
        plt.tight_layout()

        plt.subplot(3, 2, 5)
        plt.scatter(nmf_df['NMF2'], min_nmf_df['Ca.1'], c = min_nmf_df['Al.1']/(min_nmf_df['Al.1']+min_nmf_df['Ca.1']), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('NMF2')
        plt.ylabel('Ca.1')
        cbar = plt.colorbar()
        cbar.set_label('Ca', size = 10)
        plt.tight_layout()

        plt.subplot(3, 2, 6)
        plt.scatter(nmf_df['NMF2'], min_nmf_df['Al.1'], c = min_nmf_df['Al.1']/(min_nmf_df['Al.1']+min_nmf_df['Ca.1']), lw = 0.25, edgecolors = 'k', rasterized = True)
        plt.xlabel('NMF3')
        plt.ylabel('Al.1')
        cbar = plt.colorbar()
        cbar.set_label('Ca', size = 10)
        plt.tight_layout()
        plt.savefig(samplename+'_NMF_Elements_Plag.pdf')
        plt.show()

    return nmf, min_nmf_df, nmf_df

# %%

import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import time

import scipy as scipy
from sklearn.decomposition import PCA, NMF
from sklearn.mixture import GaussianMixture
from scipy.ndimage import gaussian_filter
import skimage as ski
from skimage import morphology, measure, transform, io

import QEMSCAN_FUNCTIONS as edsfunc

from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib import cm
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

%matplotlib inline
%config InlineBackend.figure_format = 'retina'
rc('font',**{'family':'Avenir', 'size': 18})
plt.rcParams['pdf.fonttype'] = 42


# %%

scan2 = pd.read_csv("20210321_TSII_5micron.csv", header = 1)
filt_data2 = scan2['Mg.1'] + scan2['Fe.1'] + scan2['Si.1']
scan_int2 = scan2[filt_data2 > 25]
filt_data_int2 = scan_int2['Mg.1'] + scan_int2['Fe.1'] + scan_int2['Si.1']
scan_lim2 = scan_int2[filt_data_int2 < 99]
filt_data_lim2 = scan_lim2['Mg.1'] + scan_lim2['Fe.1'] + scan_lim2['Si.1']


# %%
scan_cluster_withO2 = scan_lim2[['Mg.1', 'Fe.1', 'Ca.1', 'Al.1', 'Si.1', 'O.1']]

gmm_withO2 = GaussianMixture(n_components = 5)
gmm_withO2.fit(scan_cluster_withO2)
labels_withO2 = gmm_withO2.predict(scan_cluster_withO2)

# %%
# Use first function here to create 2D arrays of labels, or composition. 

imshow_map2 = edsfunc.phasemapper(scan_lim2, labels_withO2)
mg_map2 = edsfunc.elementmapper(scan_lim2, 'Mg.1')
fe_map2 = edsfunc.elementmapper(scan_lim2, 'Fe.1')
ca_map2 = edsfunc.elementmapper(scan_lim2, 'Ca.1')
al_map2 = edsfunc.elementmapper(scan_lim2, 'Al.1')
si_map2 = edsfunc.elementmapper(scan_lim2, 'Si.1')
o_map2 = edsfunc.elementmapper(scan_lim2, 'O.1')

# %%
# Mineral identification function used here, returns rewritten 2D arrays that remove the 'badspectra' showing up where glass signal comes through epoxy of sample. 

mindf2, mineralindex2, imshow_map2, mg_map2, fe_map2, ca_map2, al_map2, si_map2, o_map2 = edsfunc.mineralid(imshow_map2, mg_map2, fe_map2, ca_map2, al_map2, si_map2, o_map2)

# %% 
# Isolate mineral masks. 
glassmask_open2, glassmask2, glass_mg_mask2, glass_fe_mask2, glass_ca_mask2, glass_al_mask2, glass_si_mask2, glass_o_mask2, glass_ci_output2 = edsfunc.mineralmasks(mineralindex2.Label[0], mineralindex2, imshow_map2, mg_map2, fe_map2, ca_map2, al_map2, si_map2, o_map2)
olmask_open2, olmask2, ol_mg_mask2, ol_fe_mask2, ol_ca_mask2, ol_al_mask2, ol_si_mask2, ol_o_mask2, ol_ci_output2 = edsfunc.mineralmasks(mineralindex2.Label[1], mineralindex2, imshow_map2, mg_map2, fe_map2, ca_map2, al_map2, si_map2, o_map2)
plagmask_open2, plagmask2, plag_mg_mask2, plag_fe_mask2, plag_ca_mask2, plag_al_mask2, plag_si_mask2, plag_o_mask2, plag_ci_output2 = edsfunc.mineralmasks(mineralindex2.Label[2], mineralindex2, imshow_map2, mg_map2, fe_map2, ca_map2, al_map2, si_map2, o_map2)
cpxmask_open2, cpxmask2, cpx_mg_mask2, cpx_fe_mask2, cpx_ca_mask2, cpx_al_mask2, cpx_si_mask2, cpx_o_mask2, cpx_ci_output2 = edsfunc.mineralmasks(mineralindex2.Label[3], mineralindex2, imshow_map2, mg_map2, fe_map2, ca_map2, al_map2, si_map2, o_map2)

# %% 
# Plot phase map, each mineral mask and the related compositional variation. 
edsfunc.mineralplotter(mineralindex2, imshow_map2, mg_map2, fe_map2, ca_map2, al_map2, si_map2, o_map2, 'TSII')

# %%

# Perform PCA on each mineral, returns lots of figures of this all. 
ol_pca, ol_min_pca_df, ol_pca_df = edsfunc.mineralpca('olivine', 1, ol_mg_mask2, ol_fe_mask2, ol_ca_mask2, ol_al_mask2, ol_si_mask2, ol_o_mask2, 'TSII_OL_TEST') 
plag_pca, plag_min_pca_df, plag_pca_df = edsfunc.mineralpca('plagioclase', 1, plag_mg_mask2, plag_fe_mask2, plag_ca_mask2, plag_al_mask2, plag_si_mask2, plag_o_mask2, 'TSII_PLAG_TEST') 
cpx_pca, cpx_min_pca_df, cpx_pca_df = edsfunc.mineralpca('clinopyroxene', 1, cpx_mg_mask2, cpx_fe_mask2, cpx_ca_mask2, cpx_al_mask2, cpx_si_mask2, cpx_o_mask2, 'TSII_CPX_TEST')
glass_pca, glass_min_pca_df, glass_pca_df = edsfunc.mineralpca('glass', 1, glass_mg_mask2, glass_fe_mask2, glass_ca_mask2, glass_al_mask2, glass_si_mask2, glass_o_mask2, 'TSII_GLASS_TEST') 

# %%

ol_pca = np.empty_like(ol_mg_mask2)
ol_pca[:] = np.nan
plag_pca = np.empty_like(plag_fe_mask2)
plag_pca[:] = np.nan
cpx_pca = np.empty_like(cpx_mg_mask2)
cpx_pca[:] = np.nan
glass_pca = np.empty_like(glass_mg_mask2)
glass_pca[:] = np.nan


np.place(ol_pca, ~ol_mg_mask2.mask, ol_pca_df.PC1)
np.place(plag_pca, ~plag_fe_mask2.mask, plag_pca_df.PC1)
np.place(cpx_pca, ~cpx_mg_mask2.mask, cpx_pca_df.PC1)
np.place(glass_pca, ~glass_mg_mask2.mask, glass_pca_df.PC1)


plt.figure(figsize = (15, 25))
plt.pcolormesh(ol_pca/np.nanmax(ol_pca_df.PC1), cmap = 'Greens', rasterized = True)
plt.pcolormesh(plag_pca/np.nanmax(plag_pca_df.PC1), cmap = 'Blues', rasterized = True)
plt.pcolormesh(cpx_pca/np.nanmax(cpx_pca_df.PC1), cmap = 'Reds', rasterized = True)
plt.pcolormesh(glass_pca/np.nanmax(glass_pca_df.PC1), cmap = 'Greys', rasterized = True)
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.tight_layout()
plt.gca().invert_yaxis()
plt.savefig('TSII_Zonation.pdf', dpi = 200)

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

import QEMSCAN_FUNCTIONS as ef

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
# Read in data here and filter to get rid of bad points

ts1 = pd.read_csv("InputData/20210321_TSI_5micron.csv", header = 1)
ts1_filt, ts1_cluster_elements = ef.datafilter(ts1)
ts1_pca, ts1_pca_input_df, ts1_pca_df_lim = ef.initialPCA(ts1_filt, 'TS1')
ts1_gmm_scoring = ef.bicassess(ts1_pca_df_lim, 'TS1')
ef.bicscoreplot(ts1_gmm_scoring, 'TS1')

ts1_gmm = GaussianMixture(n_components = 5, covariance_type = 'full', max_iter = 100, init_params = 'kmeans', random_state = 0)
ts1_gmm.fit(ts1_pca_df_lim)
ts1_labels = ts1_gmm.predict(ts1_pca_df_lim)

ts1_imshow_map = ef.phasemapper(ts1_filt, ts1_labels)
ts1_mg_map = ef.elementmapper(ts1_filt, 'Mg.1')
ts1_fe_map = ef.elementmapper(ts1_filt, 'Fe.1')
ts1_na_map = ef.elementmapper(ts1_filt, 'Na.1')
ts1_ca_map = ef.elementmapper(ts1_filt, 'Ca.1')
ts1_al_map = ef.elementmapper(ts1_filt, 'Al.1')
ts1_cr_map = ef.elementmapper(ts1_filt, 'Cr.1')
ts1_si_map = ef.elementmapper(ts1_filt, 'Si.1')
ts1_o_map = ef.elementmapper(ts1_filt, 'O.1')

ts1_mindf, ts1_mineralindex, ts1_imshow_map, ts1_mg_map, ts1_fe_map, ts1_na_map, ts1_ca_map, ts1_al_map, ts1_cr_map, ts1_si_map, ts1_o_map = ef.mineralid(ts1_imshow_map, ts1_mg_map, ts1_fe_map, ts1_na_map, ts1_ca_map, ts1_al_map, ts1_cr_map, ts1_si_map, ts1_o_map)
ts1_glassmask_open, ts1_glassmask, ts1_glass_mg_mask, ts1_glass_fe_mask, ts1_glass_na_mask, ts1_glass_ca_mask, ts1_glass_al_mask, ts1_glass_cr_mask, ts1_glass_si_mask, ts1_glass_o_mask, ts1_glass_ci_output = ef.mineralmasks(ts1_mineralindex.Label[0], ts1_mineralindex, ts1_imshow_map, ts1_mg_map, ts1_fe_map, ts1_na_map, ts1_ca_map, ts1_al_map, ts1_cr_map, ts1_si_map, ts1_o_map)
ts1_olmask_open, ts1_olmask, ts1_ol_mg_mask, ts1_ol_fe_mask, ts1_ol_na_mask, ts1_ol_ca_mask, ts1_ol_al_mask, ts1_ol_cr_mask, ts1_ol_si_mask, ts1_ol_o_mask, ts1_ol_ci_output = ef.mineralmasks(ts1_mineralindex.Label[1], ts1_mineralindex, ts1_imshow_map, ts1_mg_map, ts1_fe_map, ts1_na_map, ts1_ca_map, ts1_al_map, ts1_cr_map,ts1_si_map, ts1_o_map)
ts1_plagmask_open, ts1_plagmask, ts1_plag_mg_mask, ts1_plag_fe_mask, ts1_plag_na_mask, ts1_plag_ca_mask, ts1_plag_al_mask, ts1_plag_cr_mask, ts1_plag_si_mask, ts1_plag_o_mask, ts1_plag_ci_output = ef.mineralmasks(ts1_mineralindex.Label[2], ts1_mineralindex, ts1_imshow_map, ts1_mg_map, ts1_fe_map, ts1_na_map, ts1_ca_map, ts1_al_map, ts1_cr_map,ts1_si_map, ts1_o_map)
ts1_cpxmask_open, ts1_cpxmask, ts1_cpx_mg_mask, ts1_cpx_fe_mask, ts1_cpx_na_mask, ts1_cpx_ca_mask, ts1_cpx_al_mask, ts1_cpx_cr_mask, ts1_cpx_si_mask, ts1_cpx_o_mask, ts1_cpx_ci_output = ef.mineralmasks(ts1_mineralindex.Label[3], ts1_mineralindex, ts1_imshow_map, ts1_mg_map, ts1_fe_map, ts1_na_map, ts1_ca_map, ts1_al_map, ts1_cr_map,ts1_si_map, ts1_o_map)

# %% 

ts1_ol_pca, ts1_ol_min_pca_df, ts1_ol_min_pca_thindf, ts1_ol_pca_df, ts1_ol_pca_thindf = ef.mineralpca('olivine', 1, 1, ts1_ol_mg_mask, ts1_ol_fe_mask, ts1_ol_na_mask, ts1_ol_ca_mask, ts1_ol_al_mask, ts1_ol_cr_mask, ts1_ol_si_mask, ts1_ol_o_mask, 'TSI_OL_TEST_Scaled') 
ts1_plag_pca, ts1_plag_min_pca_df, ts1_plag_min_pca_thindf, ts1_plag_pca_df, ts1_plag_pca_thindf = ef.mineralpca('plagioclase', 1, 1, ts1_plag_mg_mask, ts1_plag_fe_mask, ts1_plag_na_mask, ts1_plag_ca_mask, ts1_plag_al_mask, ts1_plag_cr_mask, ts1_plag_si_mask, ts1_plag_o_mask, 'TSI_PLAG_TEST_Scaled') 
ts1_cpx_pca, ts1_cpx_min_pca_df, ts1_cpx_min_pca_thindf, ts1_cpx_pca_df, ts1_cpx_pca_thindf = ef.mineralpca('clinopyroxene', 1, 1, ts1_cpx_mg_mask, ts1_cpx_fe_mask, ts1_cpx_na_mask, ts1_cpx_ca_mask, ts1_cpx_al_mask, ts1_cpx_cr_mask, ts1_cpx_si_mask, ts1_cpx_o_mask, 'TSI_CPX_TEST_Scaled')
ts1_glass_pca, ts1_glass_min_pca_df, ts1_glass_min_pca_thindf, ts1_glass_pca_df, ts1_glass_pca_thindf = ef.mineralpca('glass', 1, 1, ts1_glass_mg_mask, ts1_glass_fe_mask, ts1_glass_na_mask, ts1_glass_ca_mask, ts1_glass_al_mask, ts1_glass_cr_mask, ts1_glass_si_mask, ts1_glass_o_mask, 'TSI_GLASS_TEST_Scaled') 

ts1_ol_pca1plot = np.empty_like(ts1_ol_mg_mask)
ts1_ol_pca1plot[:] = np.nan
ts1_plag_pca1plot = np.empty_like(ts1_plag_fe_mask)
ts1_plag_pca1plot[:] = np.nan
ts1_cpx_pca1plot = np.empty_like(ts1_cpx_mg_mask)
ts1_cpx_pca1plot[:] = np.nan
ts1_glass_pca1plot = np.empty_like(ts1_glass_mg_mask)
ts1_glass_pca1plot[:] = np.nan

np.place(ts1_ol_pca1plot, ~ts1_ol_mg_mask.mask, ts1_ol_pca_df.PC1)
np.place(ts1_plag_pca1plot, ~ts1_plag_fe_mask.mask, ts1_plag_pca_df.PC1)
np.place(ts1_cpx_pca1plot, ~ts1_cpx_mg_mask.mask, ts1_cpx_pca_df.PC1)
np.place(ts1_glass_pca1plot, ~ts1_glass_mg_mask.mask, ts1_glass_pca_df.PC1)

# %% 

ef.gmmqemscancomparison(ts1_pca_df_lim, ts1_pca_input_df, ts1_labels, ts1_mineralindex, 'TS1')
ef.mineralgmmqemscancomparison(ts1_filt, ts1_ol_mg_mask, ts1_ol_fe_mask, 'olivine', 'TS1Ol')
ef.mineralgmmqemscancomparison(ts1_filt, ts1_plag_ca_mask, ts1_plag_si_mask, 'plagioclase', 'TS1Plag')


# %% Smoothing 

sigma=0.5

Z_ol = ef.gauss_filter(ts1_ol_pca1plot, sigma)
Z_plag = ef.gauss_filter(ts1_plag_pca1plot, sigma)
Z_cpx = ef.gauss_filter(ts1_cpx_pca1plot, sigma)
Z_glass = ef.gauss_filter(ts1_glass_pca1plot, sigma)

plt.figure(figsize = (12.5, 17.5))
plt.pcolormesh(Z_ol, cmap = 'Greens', rasterized = True)
plt.pcolormesh(Z_plag, cmap = 'Blues', rasterized = True)
plt.pcolormesh(Z_cpx, cmap = 'Reds', rasterized = True)
plt.pcolormesh(Z_glass, cmap = 'Greys', rasterized = True)
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.tight_layout()
plt.gca().invert_yaxis()
plt.savefig('TSI_Zonation_Gaussian_sigma05mu.pdf', dpi = 200, backend='pgf')

plt.figure(figsize = (12.5, 17.5))
plt.pcolormesh(ts1_ol_pca1plot, cmap = 'Greens', rasterized = True)
plt.pcolormesh(ts1_plag_pca1plot, cmap = 'Blues', rasterized = True)
plt.pcolormesh(ts1_cpx_pca1plot, cmap = 'Reds', rasterized = True)
plt.pcolormesh(ts1_glass_pca1plot, cmap = 'Greys', rasterized = True)
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.tight_layout()
plt.gca().invert_yaxis()
plt.savefig('TSI_Zonation_Gaussian_sigma00mu.pdf', dpi = 200, backend='pgf')


# %%
# %%

ts2 = pd.read_csv("InputData/20210321_TSII_5micron.csv", header = 1)
ts2_filt, ts2_cluster_elements = ef.datafilter(ts2)
ts2_pca, ts2_pca_input_df, ts2_pca_df_lim = ef.initialPCA(ts2_filt, 'TS2')
# ts2_gmm_scoring = ef.bicassess(ts2_pca_df_lim, 'TS2')
# ef.bicscoreplot(ts2_gmm_scoring, 'TS2')

ts2_gmm = GaussianMixture(n_components = 5, covariance_type = 'full', max_iter = 100, init_params = 'kmeans', random_state = 0)
ts2_gmm.fit(ts2_pca_df_lim)
ts2_labels = ts2_gmm.predict(ts2_pca_df_lim)

ts2_imshow_map = ef.phasemapper(ts2_filt, ts2_labels)
ts2_mg_map = ef.elementmapper(ts2_filt, 'Mg.1')
ts2_fe_map = ef.elementmapper(ts2_filt, 'Fe.1')
ts2_na_map = ef.elementmapper(ts2_filt, 'Na.1')
ts2_ca_map = ef.elementmapper(ts2_filt, 'Ca.1')
ts2_al_map = ef.elementmapper(ts2_filt, 'Al.1')
ts2_cr_map = ef.elementmapper(ts2_filt, 'Cr.1')
ts2_si_map = ef.elementmapper(ts2_filt, 'Si.1')
ts2_o_map = ef.elementmapper(ts2_filt, 'O.1')

ts2_mindf, ts2_mineralindex, ts2_imshow_map, ts2_mg_map, ts2_fe_map, ts2_na_map, ts2_ca_map, ts2_al_map, ts2_cr_map, ts2_si_map, ts2_o_map = ef.mineralid(ts2_imshow_map, ts2_mg_map, ts2_fe_map, ts2_na_map, ts2_ca_map, ts2_al_map, ts2_cr_map, ts2_si_map, ts2_o_map)
ts2_glassmask_open, ts2_glassmask, ts2_glass_mg_mask, ts2_glass_fe_mask, ts2_glass_na_mask, ts2_glass_ca_mask, ts2_glass_al_mask, ts2_glass_cr_mask, ts2_glass_si_mask, ts2_glass_o_mask, ts2_glass_ci_output = ef.mineralmasks(ts2_mineralindex.Label[0], ts2_mineralindex, ts2_imshow_map, ts2_mg_map, ts2_fe_map, ts2_na_map, ts2_ca_map, ts2_al_map, ts2_cr_map, ts2_si_map, ts2_o_map)
ts2_olmask_open, ts2_olmask, ts2_ol_mg_mask, ts2_ol_fe_mask, ts2_ol_na_mask, ts2_ol_ca_mask, ts2_ol_al_mask, ts2_ol_cr_mask, ts2_ol_si_mask, ts2_ol_o_mask, ts2_ol_ci_output = ef.mineralmasks(ts2_mineralindex.Label[1], ts2_mineralindex, ts2_imshow_map, ts2_mg_map, ts2_fe_map, ts2_na_map, ts2_ca_map, ts2_al_map, ts2_cr_map,ts2_si_map, ts2_o_map)
ts2_plagmask_open, ts2_plagmask, ts2_plag_mg_mask, ts2_plag_fe_mask, ts2_plag_na_mask, ts2_plag_ca_mask, ts2_plag_al_mask, ts2_plag_cr_mask, ts2_plag_si_mask, ts2_plag_o_mask, ts2_plag_ci_output = ef.mineralmasks(ts2_mineralindex.Label[2], ts2_mineralindex, ts2_imshow_map, ts2_mg_map, ts2_fe_map, ts2_na_map, ts2_ca_map, ts2_al_map, ts2_cr_map,ts2_si_map, ts2_o_map)
ts2_cpxmask_open, ts2_cpxmask, ts2_cpx_mg_mask, ts2_cpx_fe_mask, ts2_cpx_na_mask, ts2_cpx_ca_mask, ts2_cpx_al_mask, ts2_cpx_cr_mask, ts2_cpx_si_mask, ts2_cpx_o_mask, ts2_cpx_ci_output = ef.mineralmasks(ts2_mineralindex.Label[3], ts2_mineralindex, ts2_imshow_map, ts2_mg_map, ts2_fe_map, ts2_na_map, ts2_ca_map, ts2_al_map, ts2_cr_map,ts2_si_map, ts2_o_map)

# %% 

ts2_ol_pca, ts2_ol_min_pca_df, ts2_ol_min_pca_thindf, ts2_ol_pca_df, ts2_ol_pca_thindf = ef.mineralpca('olivine', 1, 1, ts2_ol_mg_mask, ts2_ol_fe_mask, ts2_ol_na_mask, ts2_ol_ca_mask, ts2_ol_al_mask, ts2_ol_cr_mask, ts2_ol_si_mask, ts2_ol_o_mask, 'TSII_OL_TEST_Scaled') 
ts2_plag_pca, ts2_plag_min_pca_df, ts2_plag_min_pca_thindf, ts2_plag_pca_df, ts2_plag_pca_thindf = ef.mineralpca('plagioclase', 1, 1, ts2_plag_mg_mask, ts2_plag_fe_mask, ts2_plag_na_mask, ts2_plag_ca_mask, ts2_plag_al_mask, ts2_plag_cr_mask, ts2_plag_si_mask, ts2_plag_o_mask, 'TSII_PLAG_TEST_Scaled') 
ts2_cpx_pca, ts2_cpx_min_pca_df, ts2_cpx_min_pca_thindf, ts2_cpx_pca_df, ts2_cpx_pca_thindf = ef.mineralpca('clinopyroxene', 1, 1, ts2_cpx_mg_mask, ts2_cpx_fe_mask, ts2_cpx_na_mask, ts2_cpx_ca_mask, ts2_cpx_al_mask, ts2_cpx_cr_mask, ts2_cpx_si_mask, ts2_cpx_o_mask, 'TSII_CPX_TEST_Scaled')
ts2_glass_pca, ts2_glass_min_pca_df, ts2_glass_min_pca_thindf, ts2_glass_pca_df, ts2_glass_pca_thindf = ef.mineralpca('glass', 1, 1, ts2_glass_mg_mask, ts2_glass_fe_mask, ts2_glass_na_mask, ts2_glass_ca_mask, ts2_glass_al_mask, ts2_glass_cr_mask, ts2_glass_si_mask, ts2_glass_o_mask, 'TSII_GLASS_TEST_Scaled') 

ts2_ol_pca1plot = np.empty_like(ts2_ol_mg_mask)
ts2_ol_pca1plot[:] = np.nan
ts2_plag_pca1plot = np.empty_like(ts2_plag_fe_mask)
ts2_plag_pca1plot[:] = np.nan
ts2_cpx_pca1plot = np.empty_like(ts2_cpx_mg_mask)
ts2_cpx_pca1plot[:] = np.nan
ts2_glass_pca1plot = np.empty_like(ts2_glass_mg_mask)
ts2_glass_pca1plot[:] = np.nan

np.place(ts2_ol_pca1plot, ~ts2_ol_mg_mask.mask, ts2_ol_pca_df.PC1)
np.place(ts2_plag_pca1plot, ~ts2_plag_fe_mask.mask, ts2_plag_pca_df.PC1)
np.place(ts2_cpx_pca1plot, ~ts2_cpx_mg_mask.mask, ts2_cpx_pca_df.PC1)
np.place(ts2_glass_pca1plot, ~ts2_glass_mg_mask.mask, ts2_glass_pca_df.PC1)

# %% 

ef.gmmqemscancomparison(ts2_pca_df_lim, ts2_pca_input_df, ts2_labels, ts2_mineralindex, 'TS2')
ef.mineralgmmqemscancomparison(ts2_filt, ts2_ol_mg_mask, ts2_ol_fe_mask, 'olivine', 'TS2Ol')
ef.mineralgmmqemscancomparison(ts2_filt, ts2_plag_ca_mask, ts2_plag_si_mask, 'plagioclase', 'TS2Plag')

# %% Smoothing 

sigma=0.5

Z_ol = ef.gauss_filter(ts2_ol_pca1plot, sigma)
Z_plag = ef.gauss_filter(ts2_plag_pca1plot, sigma)
Z_cpx = ef.gauss_filter(ts2_cpx_pca1plot, sigma)
Z_glass = ef.gauss_filter(ts2_glass_pca1plot, sigma)

plt.figure(figsize = (12.5, 17.5))
plt.pcolormesh(Z_ol, cmap = 'Greens', rasterized = True)
plt.pcolormesh(Z_plag, cmap = 'Blues', rasterized = True)
plt.pcolormesh(Z_cpx, cmap = 'Reds', rasterized = True)
plt.pcolormesh(Z_glass, cmap = 'Greys', rasterized = True)
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.tight_layout()
plt.gca().invert_yaxis()
plt.savefig('TSII_Zonation_Gaussian_sigma05mu.pdf', dpi = 200, backend='pgf')

plt.figure(figsize = (12.5, 17.5))
plt.pcolormesh(ts2_ol_pca1plot, cmap = 'Greens', rasterized = True)
plt.pcolormesh(ts2_plag_pca1plot, cmap = 'Blues', rasterized = True)
plt.pcolormesh(ts2_cpx_pca1plot, cmap = 'Reds', rasterized = True)
plt.pcolormesh(ts2_glass_pca1plot, cmap = 'Greys', rasterized = True)
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.tight_layout()
plt.gca().invert_yaxis()
plt.savefig('TSII_Zonation_Gaussian_sigma00mu.pdf', dpi = 200, backend='pgf')


# %%
# %% 
# %% 

ts3 = pd.read_csv("InputData/20210321_TSIII_5micron.csv", header = 1)
ts3_filt, ts3_cluster_elements = ef.datafilter(ts3)
ts3_pca, ts3_pca_input_df, ts3_pca_df_lim = ef.initialPCA(ts3_filt, 'TS3')
ts3_gmm_scoring = ef.bicassess(ts3_pca_df_lim, 'TS3')
ef.bicscoreplot(ts3_gmm_scoring, 'TS3')

ts3_gmm = GaussianMixture(n_components = 5, covariance_type = 'full', max_iter = 100, init_params = 'kmeans', random_state = 0)
ts3_gmm.fit(ts3_pca_df_lim)
ts3_labels = ts3_gmm.predict(ts3_pca_df_lim)

ts3_imshow_map = ef.phasemapper(ts3_filt, ts3_labels)
ts3_mg_map = ef.elementmapper(ts3_filt, 'Mg.1')
ts3_fe_map = ef.elementmapper(ts3_filt, 'Fe.1')
ts3_na_map = ef.elementmapper(ts3_filt, 'Na.1')
ts3_ca_map = ef.elementmapper(ts3_filt, 'Ca.1')
ts3_al_map = ef.elementmapper(ts3_filt, 'Al.1')
ts3_cr_map = ef.elementmapper(ts3_filt, 'Cr.1')
ts3_si_map = ef.elementmapper(ts3_filt, 'Si.1')
ts3_o_map = ef.elementmapper(ts3_filt, 'O.1')

ts3_mindf, ts3_mineralindex, ts3_imshow_map, ts3_mg_map, ts3_fe_map, ts3_na_map, ts3_ca_map, ts3_al_map, ts3_cr_map, ts3_si_map, ts3_o_map = ef.mineralid(ts3_imshow_map, ts3_mg_map, ts3_fe_map, ts3_na_map, ts3_ca_map, ts3_al_map, ts3_cr_map, ts3_si_map, ts3_o_map)
ts3_glassmask_open, ts3_glassmask, ts3_glass_mg_mask, ts3_glass_fe_mask, ts3_glass_na_mask, ts3_glass_ca_mask, ts3_glass_al_mask, ts3_glass_cr_mask, ts3_glass_si_mask, ts3_glass_o_mask, ts3_glass_ci_output = ef.mineralmasks(ts3_mineralindex.Label[0], ts3_mineralindex, ts3_imshow_map, ts3_mg_map, ts3_fe_map, ts3_na_map, ts3_ca_map, ts3_al_map, ts3_cr_map, ts3_si_map, ts3_o_map)
ts3_olmask_open, ts3_olmask, ts3_ol_mg_mask, ts3_ol_fe_mask, ts3_ol_na_mask, ts3_ol_ca_mask, ts3_ol_al_mask, ts3_ol_cr_mask, ts3_ol_si_mask, ts3_ol_o_mask, ts3_ol_ci_output = ef.mineralmasks(ts3_mineralindex.Label[1], ts3_mineralindex, ts3_imshow_map, ts3_mg_map, ts3_fe_map, ts3_na_map, ts3_ca_map, ts3_al_map, ts3_cr_map,ts3_si_map, ts3_o_map)
ts3_plagmask_open, ts3_plagmask, ts3_plag_mg_mask, ts3_plag_fe_mask, ts3_plag_na_mask, ts3_plag_ca_mask, ts3_plag_al_mask, ts3_plag_cr_mask, ts3_plag_si_mask, ts3_plag_o_mask, ts3_plag_ci_output = ef.mineralmasks(ts3_mineralindex.Label[2], ts3_mineralindex, ts3_imshow_map, ts3_mg_map, ts3_fe_map, ts3_na_map, ts3_ca_map, ts3_al_map, ts3_cr_map,ts3_si_map, ts3_o_map)
ts3_cpxmask_open, ts3_cpxmask, ts3_cpx_mg_mask, ts3_cpx_fe_mask, ts3_cpx_na_mask, ts3_cpx_ca_mask, ts3_cpx_al_mask, ts3_cpx_cr_mask, ts3_cpx_si_mask, ts3_cpx_o_mask, ts3_cpx_ci_output = ef.mineralmasks(ts3_mineralindex.Label[3], ts3_mineralindex, ts3_imshow_map, ts3_mg_map, ts3_fe_map, ts3_na_map, ts3_ca_map, ts3_al_map, ts3_cr_map,ts3_si_map, ts3_o_map)

# %% 

ts3_ol_pca, ts3_ol_min_pca_df, ts3_ol_min_pca_thindf, ts3_ol_pca_df, ts3_ol_pca_thindf = ef.mineralpca('olivine', 1, 1, ts3_ol_mg_mask, ts3_ol_fe_mask, ts3_ol_na_mask, ts3_ol_ca_mask, ts3_ol_al_mask, ts3_ol_cr_mask, ts3_ol_si_mask, ts3_ol_o_mask, 'TSIII_OL_TEST_Scaled') 
ts3_plag_pca, ts3_plag_min_pca_df, ts3_plag_min_pca_thindf, ts3_plag_pca_df, ts3_plag_pca_thindf = ef.mineralpca('plagioclase', 1, 1, ts3_plag_mg_mask, ts3_plag_fe_mask, ts3_plag_na_mask, ts3_plag_ca_mask, ts3_plag_al_mask, ts3_plag_cr_mask, ts3_plag_si_mask, ts3_plag_o_mask, 'TSIII_PLAG_TEST_Scaled') 
ts3_cpx_pca, ts3_cpx_min_pca_df, ts3_cpx_min_pca_thindf, ts3_cpx_pca_df, ts3_cpx_pca_thindf = ef.mineralpca('clinopyroxene', 1, 1, ts3_cpx_mg_mask, ts3_cpx_fe_mask, ts3_cpx_na_mask, ts3_cpx_ca_mask, ts3_cpx_al_mask, ts3_cpx_cr_mask, ts3_cpx_si_mask, ts3_cpx_o_mask, 'TSIII_CPX_TEST_Scaled')
ts3_glass_pca, ts3_glass_min_pca_df, ts3_glass_min_pca_thindf, ts3_glass_pca_df, ts3_glass_pca_thindf = ef.mineralpca('glass', 1, 1, ts3_glass_mg_mask, ts3_glass_fe_mask, ts3_glass_na_mask, ts3_glass_ca_mask, ts3_glass_al_mask, ts3_glass_cr_mask, ts3_glass_si_mask, ts3_glass_o_mask, 'TSIII_GLASS_TEST_Scaled') 

ts3_ol_pca1plot = np.empty_like(ts3_ol_mg_mask)
ts3_ol_pca1plot[:] = np.nan
ts3_plag_pca1plot = np.empty_like(ts3_plag_fe_mask)
ts3_plag_pca1plot[:] = np.nan
ts3_cpx_pca1plot = np.empty_like(ts3_cpx_mg_mask)
ts3_cpx_pca1plot[:] = np.nan
ts3_glass_pca1plot = np.empty_like(ts3_glass_mg_mask)
ts3_glass_pca1plot[:] = np.nan

np.place(ts3_ol_pca1plot, ~ts3_ol_mg_mask.mask, ts3_ol_pca_df.PC1)
np.place(ts3_plag_pca1plot, ~ts3_plag_fe_mask.mask, ts3_plag_pca_df.PC1)
np.place(ts3_cpx_pca1plot, ~ts3_cpx_mg_mask.mask, ts3_cpx_pca_df.PC1)
np.place(ts3_glass_pca1plot, ~ts3_glass_mg_mask.mask, ts3_glass_pca_df.PC1)

# %% 

ef.gmmqemscancomparison(ts3_pca_df_lim, ts3_pca_input_df, ts3_labels, ts3_mineralindex, 'TS3')
ef.mineralgmmqemscancomparison(ts3_filt, ts3_ol_mg_mask, ts3_ol_fe_mask, 'olivine', 'TS3Ol')
ef.mineralgmmqemscancomparison(ts3_filt, ts3_plag_ca_mask, ts3_plag_si_mask, 'plagioclase', 'TS3Plag')

# %% Smoothing 

sigma=0.5

Z_ol = ef.gauss_filter(ts3_ol_pca1plot, sigma)
Z_plag = ef.gauss_filter(ts3_plag_pca1plot, sigma)
Z_cpx = ef.gauss_filter(ts3_cpx_pca1plot, sigma)
Z_glass = ef.gauss_filter(ts3_glass_pca1plot, sigma)

plt.figure(figsize = (12.5, 17.5))
plt.pcolormesh(Z_ol, cmap = 'Greens', rasterized = True)
plt.pcolormesh(Z_plag, cmap = 'Blues', rasterized = True)
plt.pcolormesh(Z_cpx, cmap = 'Reds', rasterized = True)
plt.pcolormesh(Z_glass, cmap = 'Greys', rasterized = True)
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.tight_layout()
plt.gca().invert_yaxis()
plt.savefig('TSIII_Zonation_Gaussian_sigma05mu.pdf', dpi = 200, backend='pgf')

plt.figure(figsize = (12.5, 17.5))
plt.pcolormesh(ts3_ol_pca1plot, cmap = 'Greens', rasterized = True)
plt.pcolormesh(ts3_plag_pca1plot, cmap = 'Blues', rasterized = True)
plt.pcolormesh(ts3_cpx_pca1plot, cmap = 'Reds', rasterized = True)
plt.pcolormesh(ts3_glass_pca1plot, cmap = 'Greys', rasterized = True)
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.tight_layout()
plt.gca().invert_yaxis()
plt.savefig('TSIII_Zonation_Gaussian_sigma00mu.pdf', dpi = 200, backend='pgf')


# %%
# %%

k7 = pd.read_csv("InputData/K7_4mu_allphases.csv", header = 1)
k7_filt, k7_cluster_elements = ef.datafilter(k7)
k7_pca, k7_pca_input_df, k7_pca_df_lim = ef.initialPCA(k7_filt, 'K7')
k7_gmm_scoring = ef.bicassess(k7_pca_df_lim, 'K7')
ef.bicscoreplot(k7_gmm_scoring, 'K7')

k7_gmm = GaussianMixture(n_components = 5, covariance_type = 'full', max_iter = 100, init_params = 'kmeans', random_state = 0)
k7_gmm.fit(k7_pca_df_lim)
k7_labels = k7_gmm.predict(k7_pca_df_lim)

k7_imshow_map = ef.phasemapper(k7_filt, k7_labels)
k7_mg_map = ef.elementmapper(k7_filt, 'Mg.1')
k7_fe_map = ef.elementmapper(k7_filt, 'Fe.1')
k7_na_map = ef.elementmapper(k7_filt, 'Na.1')
k7_ca_map = ef.elementmapper(k7_filt, 'Ca.1')
k7_al_map = ef.elementmapper(k7_filt, 'Al.1')
k7_cr_map = ef.elementmapper(k7_filt, 'Cr.1')
k7_si_map = ef.elementmapper(k7_filt, 'Si.1')
k7_o_map = ef.elementmapper(k7_filt, 'O.1')


# %% 


phase_map = k7_imshow_map
mg_map = k7_mg_map
fe_map = k7_fe_map
na_map = k7_fe_map
ca_map = k7_ca_map
al_map = k7_al_map
cr_map = k7_cr_map
si_map = k7_si_map
o_map = k7_o_map

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

# %% 

k7_mindf, k7_mineralindex, k7_imshow_map, k7_mg_map, k7_fe_map, k7_na_map, k7_ca_map, k7_al_map, k7_cr_map, k7_si_map, k7_o_map = ef.mineralid(k7_imshow_map, k7_mg_map, k7_fe_map, k7_na_map, k7_ca_map, k7_al_map, k7_cr_map, k7_si_map, k7_o_map)
k7_glassmask_open, k7_glassmask, k7_glass_mg_mask, k7_glass_fe_mask, k7_glass_na_mask, k7_glass_ca_mask, k7_glass_al_mask, k7_glass_cr_mask, k7_glass_si_mask, k7_glass_o_mask, k7_glass_ci_output = ef.mineralmasks(k7_mineralindex.Label[0], k7_mineralindex, k7_imshow_map, k7_mg_map, k7_fe_map, k7_na_map, k7_ca_map, k7_al_map, k7_cr_map, k7_si_map, k7_o_map)
k7_olmask_open, k7_olmask, k7_ol_mg_mask, k7_ol_fe_mask, k7_ol_na_mask, k7_ol_ca_mask, k7_ol_al_mask, k7_ol_cr_mask, k7_ol_si_mask, k7_ol_o_mask, k7_ol_ci_output = ef.mineralmasks(k7_mineralindex.Label[1], k7_mineralindex, k7_imshow_map, k7_mg_map, k7_fe_map, k7_na_map, k7_ca_map, k7_al_map, k7_cr_map,k7_si_map, k7_o_map)
k7_plagmask_open, k7_plagmask, k7_plag_mg_mask, k7_plag_fe_mask, k7_plag_na_mask, k7_plag_ca_mask, k7_plag_al_mask, k7_plag_cr_mask, k7_plag_si_mask, k7_plag_o_mask, k7_plag_ci_output = ef.mineralmasks(k7_mineralindex.Label[2], k7_mineralindex, k7_imshow_map, k7_mg_map, k7_fe_map, k7_na_map, k7_ca_map, k7_al_map, k7_cr_map,k7_si_map, k7_o_map)
k7_cpxmask_open, k7_cpxmask, k7_cpx_mg_mask, k7_cpx_fe_mask, k7_cpx_na_mask, k7_cpx_ca_mask, k7_cpx_al_mask, k7_cpx_cr_mask, k7_cpx_si_mask, k7_cpx_o_mask, k7_cpx_ci_output = ef.mineralmasks(k7_mineralindex.Label[3], k7_mineralindex, k7_imshow_map, k7_mg_map, k7_fe_map, k7_na_map, k7_ca_map, k7_al_map, k7_cr_map,k7_si_map, k7_o_map)

# %% 

k7_ol_pca, k7_ol_min_pca_df, k7_ol_min_pca_thindf, k7_ol_pca_df, k7_ol_pca_thindf = ef.mineralpca('olivine', 1, 1, k7_ol_mg_mask, k7_ol_fe_mask, k7_ol_na_mask, k7_ol_ca_mask, k7_ol_al_mask, k7_ol_cr_mask, k7_ol_si_mask, k7_ol_o_mask, 'K7_OL_TEST_Scaled') 
k7_plag_pca, k7_plag_min_pca_df, k7_plag_min_pca_thindf, k7_plag_pca_df, k7_plag_pca_thindf = ef.mineralpca('plagioclase', 1, 1, k7_plag_mg_mask, k7_plag_fe_mask, k7_plag_na_mask, k7_plag_ca_mask, k7_plag_al_mask, k7_plag_cr_mask, k7_plag_si_mask, k7_plag_o_mask, 'K7_PLAG_TEST_Scaled') 
k7_cpx_pca, k7_cpx_min_pca_df, k7_cpx_min_pca_thindf, k7_cpx_pca_df, k7_cpx_pca_thindf = ef.mineralpca('clinopyroxene', 1, 1, k7_cpx_mg_mask, k7_cpx_fe_mask, k7_cpx_na_mask, k7_cpx_ca_mask, k7_cpx_al_mask, k7_cpx_cr_mask, k7_cpx_si_mask, k7_cpx_o_mask, 'K7_CPX_TEST_Scaled')
k7_glass_pca, k7_glass_min_pca_df, k7_glass_min_pca_thindf, k7_glass_pca_df, k7_glass_pca_thindf = ef.mineralpca('glass', 1, 1, k7_glass_mg_mask, k7_glass_fe_mask, k7_glass_na_mask, k7_glass_ca_mask, k7_glass_al_mask, k7_glass_cr_mask, k7_glass_si_mask, k7_glass_o_mask, 'K7_GLASS_TEST_Scaled') 

k7_ol_pca1plot = np.empty_like(k7_ol_mg_mask)
k7_ol_pca1plot[:] = np.nan
k7_plag_pca1plot = np.empty_like(k7_plag_fe_mask)
k7_plag_pca1plot[:] = np.nan
k7_cpx_pca1plot = np.empty_like(k7_cpx_mg_mask)
k7_cpx_pca1plot[:] = np.nan
k7_glass_pca1plot = np.empty_like(k7_glass_mg_mask)
k7_glass_pca1plot[:] = np.nan

np.place(k7_ol_pca1plot, ~k7_ol_mg_mask.mask, k7_ol_pca_df.PC1)
np.place(k7_plag_pca1plot, ~k7_plag_fe_mask.mask, k7_plag_pca_df.PC1)
np.place(k7_cpx_pca1plot, ~k7_cpx_mg_mask.mask, k7_cpx_pca_df.PC1)
np.place(k7_glass_pca1plot, ~k7_glass_mg_mask.mask, k7_glass_pca_df.PC1)

# %% 

ef.gmmqemscancomparison(k7_pca_df_lim, k7_pca_input_df, k7_labels, k7_mineralindex, 'K7')
ef.mineralgmmqemscancomparison(k7_filt, k7_ol_mg_mask, k7_ol_fe_mask, 'olivine', 'K7Ol')
ef.mineralgmmqemscancomparison(k7_filt, k7_plag_ca_mask, k7_plag_si_mask, 'plagioclase', 'K7Plag')

# %% Smoothing 

sigma=0.5

Z_ol = ef.gauss_filter(k7_ol_pca1plot, sigma)
Z_plag = ef.gauss_filter(k7_plag_pca1plot, sigma)
Z_cpx = ef.gauss_filter(k7_cpx_pca1plot, sigma)
Z_glass = ef.gauss_filter(k7_glass_pca1plot, sigma)

plt.figure(figsize = (12.5, 17.5))
plt.pcolormesh(Z_ol, cmap = 'Greens', rasterized = True)
plt.pcolormesh(Z_plag, cmap = 'Blues', rasterized = True)
plt.pcolormesh(Z_cpx, cmap = 'Reds', rasterized = True)
plt.pcolormesh(Z_glass, cmap = 'Greys', rasterized = True)
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.tight_layout()
plt.gca().invert_yaxis()
plt.savefig('K7_Zonation_Gaussian_sigma05mu.pdf', dpi = 200, backend='pgf')

plt.figure(figsize = (12.5, 17.5))
plt.pcolormesh(k7_ol_pca1plot, cmap = 'Greens', rasterized = True)
plt.pcolormesh(k7_plag_pca1plot, cmap = 'Blues', rasterized = True)
plt.pcolormesh(k7_cpx_pca1plot, cmap = 'Reds', rasterized = True)
plt.pcolormesh(k7_glass_pca1plot, cmap = 'Greys', rasterized = True)
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.tight_layout()
plt.gca().invert_yaxis()
plt.savefig('K7_Zonation_Gaussian_sigma00mu.pdf', dpi = 200, backend='pgf')


# %% 
# %%

k8 = pd.read_csv("InputData/K8_4mu_allphases.csv", header = 1)
k8_filt, k8_cluster_elements = ef.datafilter(k8)
k8_pca, k8_pca_input_df, k8_pca_df_lim = ef.initialPCA(k8_filt, 'K8')
k8_gmm_scoring = ef.bicassess(k8_pca_df_lim, 'K8')
ef.bicscoreplot(k8_gmm_scoring, 'K8')

k8_gmm = GaussianMixture(n_components = 5, covariance_type = 'full', max_iter = 100, init_params = 'kmeans', random_state = 0)
k8_gmm.fit(k8_pca_df_lim)
k8_labels = k8_gmm.predict(k8_pca_df_lim)

k8_imshow_map = ef.phasemapper(k8_filt, k8_labels)
k8_mg_map = ef.elementmapper(k8_filt, 'Mg.1')
k8_fe_map = ef.elementmapper(k8_filt, 'Fe.1')
k8_na_map = ef.elementmapper(k8_filt, 'Na.1')
k8_ca_map = ef.elementmapper(k8_filt, 'Ca.1')
k8_al_map = ef.elementmapper(k8_filt, 'Al.1')
k8_cr_map = ef.elementmapper(k8_filt, 'Cr.1')
k8_si_map = ef.elementmapper(k8_filt, 'Si.1')
k8_o_map = ef.elementmapper(k8_filt, 'O.1')

k8_mindf, k8_mineralindex, k8_imshow_map, k8_mg_map, k8_fe_map, k8_na_map, k8_ca_map, k8_al_map, k8_cr_map, k8_si_map, k8_o_map = ef.mineralid(k8_imshow_map, k8_mg_map, k8_fe_map, k8_na_map, k8_ca_map, k8_al_map, k8_cr_map, k8_si_map, k8_o_map)
k8_glassmask_open, k8_glassmask, k8_glass_mg_mask, k8_glass_fe_mask, k8_glass_na_mask, k8_glass_ca_mask, k8_glass_al_mask, k8_glass_cr_mask, k8_glass_si_mask, k8_glass_o_mask, k8_glass_ci_output = ef.mineralmasks(k8_mineralindex.Label[0], k8_mineralindex, k8_imshow_map, k8_mg_map, k8_fe_map, k8_na_map, k8_ca_map, k8_al_map, k8_cr_map, k8_si_map, k8_o_map)
k8_olmask_open, k8_olmask, k8_ol_mg_mask, k8_ol_fe_mask, k8_ol_na_mask, k8_ol_ca_mask, k8_ol_al_mask, k8_ol_cr_mask, k8_ol_si_mask, k8_ol_o_mask, k8_ol_ci_output = ef.mineralmasks(k8_mineralindex.Label[1], k8_mineralindex, k8_imshow_map, k8_mg_map, k8_fe_map, k8_na_map, k8_ca_map, k8_al_map, k8_cr_map,k8_si_map, k8_o_map)
k8_plagmask_open, k8_plagmask, k8_plag_mg_mask, k8_plag_fe_mask, k8_plag_na_mask, k8_plag_ca_mask, k8_plag_al_mask, k8_plag_cr_mask, k8_plag_si_mask, k8_plag_o_mask, k8_plag_ci_output = ef.mineralmasks(k8_mineralindex.Label[2], k8_mineralindex, k8_imshow_map, k8_mg_map, k8_fe_map, k8_na_map, k8_ca_map, k8_al_map, k8_cr_map,k8_si_map, k8_o_map)
k8_cpxmask_open, k8_cpxmask, k8_cpx_mg_mask, k8_cpx_fe_mask, k8_cpx_na_mask, k8_cpx_ca_mask, k8_cpx_al_mask, k8_cpx_cr_mask, k8_cpx_si_mask, k8_cpx_o_mask, k8_cpx_ci_output = ef.mineralmasks(k8_mineralindex.Label[3], k8_mineralindex, k8_imshow_map, k8_mg_map, k8_fe_map, k8_na_map, k8_ca_map, k8_al_map, k8_cr_map,k8_si_map, k8_o_map)

# %% 

k8_ol_pca, k8_ol_min_pca_df, k8_ol_min_pca_thindf, k8_ol_pca_df, k8_ol_pca_thindf = ef.mineralpca('olivine', 1, 1, k8_ol_mg_mask, k8_ol_fe_mask, k8_ol_na_mask, k8_ol_ca_mask, k8_ol_al_mask, k8_ol_cr_mask, k8_ol_si_mask, k8_ol_o_mask, 'K8_OL_TEST_Scaled') 
k8_plag_pca, k8_plag_min_pca_df, k8_plag_min_pca_thindf, k8_plag_pca_df, k8_plag_pca_thindf = ef.mineralpca('plagioclase', 1, 1, k8_plag_mg_mask, k8_plag_fe_mask, k8_plag_na_mask, k8_plag_ca_mask, k8_plag_al_mask, k8_plag_cr_mask, k8_plag_si_mask, k8_plag_o_mask, 'K8_PLAG_TEST_Scaled') 
k8_cpx_pca, k8_cpx_min_pca_df, k8_cpx_min_pca_thindf, k8_cpx_pca_df, k8_cpx_pca_thindf = ef.mineralpca('clinopyroxene', 1, 1, k8_cpx_mg_mask, k8_cpx_fe_mask, k8_cpx_na_mask, k8_cpx_ca_mask, k8_cpx_al_mask, k8_cpx_cr_mask, k8_cpx_si_mask, k8_cpx_o_mask, 'K8_CPX_TEST_Scaled')
k8_glass_pca, k8_glass_min_pca_df, k8_glass_min_pca_thindf, k8_glass_pca_df, k8_glass_pca_thindf = ef.mineralpca('glass', 1, 1, k8_glass_mg_mask, k8_glass_fe_mask, k8_glass_na_mask, k8_glass_ca_mask, k8_glass_al_mask, k8_glass_cr_mask, k8_glass_si_mask, k8_glass_o_mask, 'K8_GLASS_TEST_Scaled') 

k8_ol_pca1plot = np.empty_like(k8_ol_mg_mask)
k8_ol_pca1plot[:] = np.nan
k8_plag_pca1plot = np.empty_like(k8_plag_fe_mask)
k8_plag_pca1plot[:] = np.nan
k8_cpx_pca1plot = np.empty_like(k8_cpx_mg_mask)
k8_cpx_pca1plot[:] = np.nan
k8_glass_pca1plot = np.empty_like(k8_glass_mg_mask)
k8_glass_pca1plot[:] = np.nan

np.place(k8_ol_pca1plot, ~k8_ol_mg_mask.mask, k8_ol_pca_df.PC1)
np.place(k8_plag_pca1plot, ~k8_plag_fe_mask.mask, k8_plag_pca_df.PC1)
np.place(k8_cpx_pca1plot, ~k8_cpx_mg_mask.mask, k8_cpx_pca_df.PC1)
np.place(k8_glass_pca1plot, ~k8_glass_mg_mask.mask, k8_glass_pca_df.PC1)

# %% 

ef.gmmqemscancomparison(k8_pca_df_lim, k8_pca_input_df, k8_labels, k8_mineralindex, 'K8')
ef.mineralgmmqemscancomparison(k8_filt, k8_ol_mg_mask, k8_ol_fe_mask, 'olivine', 'K8Ol')
ef.mineralgmmqemscancomparison(k8_filt, k8_plag_ca_mask, k8_plag_si_mask, 'plagioclase', 'K8Plag')

# %% Smoothing 

sigma=0.5

Z_ol = ef.gauss_filter(k8_ol_pca1plot, sigma)
Z_plag = ef.gauss_filter(k8_plag_pca1plot, sigma)
Z_cpx = ef.gauss_filter(k8_cpx_pca1plot, sigma)
Z_glass = ef.gauss_filter(k8_glass_pca1plot, sigma)

plt.figure(figsize = (12.5, 17.5))
plt.pcolormesh(Z_ol, cmap = 'Greens', rasterized = True)
plt.pcolormesh(Z_plag, cmap = 'Blues', rasterized = True)
plt.pcolormesh(Z_cpx, cmap = 'Reds', rasterized = True)
plt.pcolormesh(Z_glass, cmap = 'Greys', rasterized = True)
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.tight_layout()
plt.gca().invert_yaxis()
plt.savefig('K8_Zonation_Gaussian_sigma05mu.pdf', dpi = 200, backend='pgf')

plt.figure(figsize = (12.5, 17.5))
plt.pcolormesh(k8_ol_pca1plot, cmap = 'Greens', rasterized = True)
plt.pcolormesh(k8_plag_pca1plot, cmap = 'Blues', rasterized = True)
plt.pcolormesh(k8_cpx_pca1plot, cmap = 'Reds', rasterized = True)
plt.pcolormesh(k8_glass_pca1plot, cmap = 'Greys', rasterized = True)
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.tight_layout()
plt.gca().invert_yaxis()
plt.savefig('K8_Zonation_Gaussian_sigma00mu.pdf', dpi = 200, backend='pgf')

# %%
# %% 

t7 = pd.read_csv("InputData/T7_4mu_allphases.csv", header = 1)
t7_filt, t7_cluster_elements = ef.datafilter(t7)
t7_pca, t7_pca_input_df, t7_pca_df_lim = ef.initialPCA(t7_filt, 'T7')
t7_gmm_scoring = ef.bicassess(t7_pca_df_lim, 'T7')
# ef.bicscoreplot(t7_gmm_scoring, 'T7')

t7_gmm = GaussianMixture(n_components = 5, covariance_type = 'full', max_iter = 100, init_params = 'kmeans', random_state = 0)
t7_gmm.fit(t7_pca_df_lim)
t7_labels = t7_gmm.predict(t7_pca_df_lim)

t7_imshow_map = ef.phasemapper(t7_filt, t7_labels)
t7_mg_map = ef.elementmapper(t7_filt, 'Mg.1')
t7_fe_map = ef.elementmapper(t7_filt, 'Fe.1')
t7_na_map = ef.elementmapper(t7_filt, 'Na.1')
t7_ca_map = ef.elementmapper(t7_filt, 'Ca.1')
t7_al_map = ef.elementmapper(t7_filt, 'Al.1')
t7_cr_map = ef.elementmapper(t7_filt, 'Cr.1')
t7_si_map = ef.elementmapper(t7_filt, 'Si.1')
t7_o_map = ef.elementmapper(t7_filt, 'O.1')

t7_mindf, t7_mineralindex, t7_imshow_map, t7_mg_map, t7_fe_map, t7_na_map, t7_ca_map, t7_al_map, t7_cr_map, t7_si_map, t7_o_map = ef.mineralid(t7_imshow_map, t7_mg_map, t7_fe_map, t7_na_map, t7_ca_map, t7_al_map, t7_cr_map, t7_si_map, t7_o_map)
t7_glassmask_open, t7_glassmask, t7_glass_mg_mask, t7_glass_fe_mask, t7_glass_na_mask, t7_glass_ca_mask, t7_glass_al_mask, t7_glass_cr_mask, t7_glass_si_mask, t7_glass_o_mask, t7_glass_ci_output = ef.mineralmasks(t7_mineralindex.Label[0], t7_mineralindex, t7_imshow_map, t7_mg_map, t7_fe_map, t7_na_map, t7_ca_map, t7_al_map, t7_cr_map, t7_si_map, t7_o_map)
t7_olmask_open, t7_olmask, t7_ol_mg_mask, t7_ol_fe_mask, t7_ol_na_mask, t7_ol_ca_mask, t7_ol_al_mask, t7_ol_cr_mask, t7_ol_si_mask, t7_ol_o_mask, t7_ol_ci_output = ef.mineralmasks(t7_mineralindex.Label[1], t7_mineralindex, t7_imshow_map, t7_mg_map, t7_fe_map, t7_na_map, t7_ca_map, t7_al_map, t7_cr_map,t7_si_map, t7_o_map)
t7_plagmask_open, t7_plagmask, t7_plag_mg_mask, t7_plag_fe_mask, t7_plag_na_mask, t7_plag_ca_mask, t7_plag_al_mask, t7_plag_cr_mask, t7_plag_si_mask, t7_plag_o_mask, t7_plag_ci_output = ef.mineralmasks(t7_mineralindex.Label[2], t7_mineralindex, t7_imshow_map, t7_mg_map, t7_fe_map, t7_na_map, t7_ca_map, t7_al_map, t7_cr_map,t7_si_map, t7_o_map)
t7_cpxmask_open, t7_cpxmask, t7_cpx_mg_mask, t7_cpx_fe_mask, t7_cpx_na_mask, t7_cpx_ca_mask, t7_cpx_al_mask, t7_cpx_cr_mask, t7_cpx_si_mask, t7_cpx_o_mask, t7_cpx_ci_output = ef.mineralmasks(t7_mineralindex.Label[3], t7_mineralindex, t7_imshow_map, t7_mg_map, t7_fe_map, t7_na_map, t7_ca_map, t7_al_map, t7_cr_map,t7_si_map, t7_o_map)

# %% 

t7_ol_pca, t7_ol_min_pca_df, t7_ol_min_pca_thindf, t7_ol_pca_df, t7_ol_pca_thindf = ef.mineralpca('olivine', 1, 1, t7_ol_mg_mask, t7_ol_fe_mask, t7_ol_na_mask, t7_ol_ca_mask, t7_ol_al_mask, t7_ol_cr_mask, t7_ol_si_mask, t7_ol_o_mask, 'T7_OL_TEST_Scaled') 
t7_plag_pca, t7_plag_min_pca_df, t7_plag_min_pca_thindf, t7_plag_pca_df, t7_plag_pca_thindf = ef.mineralpca('plagioclase', 1, 1, t7_plag_mg_mask, t7_plag_fe_mask, t7_plag_na_mask, t7_plag_ca_mask, t7_plag_al_mask, t7_plag_cr_mask, t7_plag_si_mask, t7_plag_o_mask, 'T7_PLAG_TEST_Scaled') 
t7_cpx_pca, t7_cpx_min_pca_df, t7_cpx_min_pca_thindf, t7_cpx_pca_df, t7_cpx_pca_thindf = ef.mineralpca('clinopyroxene', 1, 1, t7_cpx_mg_mask, t7_cpx_fe_mask, t7_cpx_na_mask, t7_cpx_ca_mask, t7_cpx_al_mask, t7_cpx_cr_mask, t7_cpx_si_mask, t7_cpx_o_mask, 'T7_CPX_TEST_Scaled')
t7_glass_pca, t7_glass_min_pca_df, t7_glass_min_pca_thindf, t7_glass_pca_df, t7_glass_pca_thindf = ef.mineralpca('glass', 1, 1, t7_glass_mg_mask, t7_glass_fe_mask, t7_glass_na_mask, t7_glass_ca_mask, t7_glass_al_mask, t7_glass_cr_mask, t7_glass_si_mask, t7_glass_o_mask, 'T7_GLASS_TEST_Scaled') 

t7_ol_pca1plot = np.empty_like(t7_ol_mg_mask)
t7_ol_pca1plot[:] = np.nan
t7_plag_pca1plot = np.empty_like(t7_plag_fe_mask)
t7_plag_pca1plot[:] = np.nan
t7_cpx_pca1plot = np.empty_like(t7_cpx_mg_mask)
t7_cpx_pca1plot[:] = np.nan
t7_glass_pca1plot = np.empty_like(t7_glass_mg_mask)
t7_glass_pca1plot[:] = np.nan

np.place(t7_ol_pca1plot, ~t7_ol_mg_mask.mask, t7_ol_pca_df.PC1)
np.place(t7_plag_pca1plot, ~t7_plag_fe_mask.mask, t7_plag_pca_df.PC1)
np.place(t7_cpx_pca1plot, ~t7_cpx_mg_mask.mask, t7_cpx_pca_df.PC1)
np.place(t7_glass_pca1plot, ~t7_glass_mg_mask.mask, t7_glass_pca_df.PC1)

# %% 

ef.gmmqemscancomparison(t7_pca_df_lim, t7_pca_input_df, t7_labels, t7_mineralindex, 'T7')
ef.mineralgmmqemscancomparison(t7_filt, t7_ol_mg_mask, t7_ol_fe_mask, 'olivine', 'T7Ol')
ef.mineralgmmqemscancomparison(t7_filt, t7_plag_ca_mask, t7_plag_si_mask, 'plagioclase', 'T7Plag')

# %% Smoothing 

sigma=0.5

Z_ol = ef.gauss_filter(t7_ol_pca1plot, sigma)
Z_plag = ef.gauss_filter(t7_plag_pca1plot, sigma)
Z_cpx = ef.gauss_filter(t7_cpx_pca1plot, sigma)
Z_glass = ef.gauss_filter(t7_glass_pca1plot, sigma)

plt.figure(figsize = (12.5, 17.5))
plt.pcolormesh(Z_ol, cmap = 'Greens', rasterized = True)
plt.pcolormesh(Z_plag, cmap = 'Blues', rasterized = True)
plt.pcolormesh(Z_cpx, cmap = 'Reds', rasterized = True)
plt.pcolormesh(Z_glass, cmap = 'Greys', rasterized = True)
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.tight_layout()
plt.gca().invert_yaxis()
plt.savefig('T7_Zonation_Gaussian_sigma05mu.pdf', dpi = 200, backend='pgf')

plt.figure(figsize = (12.5, 17.5))
plt.pcolormesh(t7_ol_pca1plot, cmap = 'Greens', rasterized = True)
plt.pcolormesh(t7_plag_pca1plot, cmap = 'Blues', rasterized = True)
plt.pcolormesh(t7_cpx_pca1plot, cmap = 'Reds', rasterized = True)
plt.pcolormesh(t7_glass_pca1plot, cmap = 'Greys', rasterized = True)
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.tight_layout()
plt.gca().invert_yaxis()
plt.savefig('T7_Zonation_Gaussian_sigma00mu.pdf', dpi = 200, backend='pgf')


# %%
# %% 

t8 = pd.read_csv("InputData/T8_4mu_allphases.csv", header = 1)
t8_filt, t8_cluster_elements = ef.datafilter(t8)
t8_pca, t8_pca_input_df, t8_pca_df_lim = ef.initialPCA(t8_filt, 'T8')
t8_gmm_scoring = ef.bicassess(t8_pca_df_lim, 'T8')
# ef.bicscoreplot(t8_gmm_scoring, 'T8')

t8_gmm = GaussianMixture(n_components = 5, covariance_type = 'full', max_iter = 100, init_params = 'kmeans', random_state = 0)
t8_gmm.fit(t8_pca_df_lim)
t8_labels = t8_gmm.predict(t8_pca_df_lim)

t8_imshow_map = ef.phasemapper(t8_filt, t8_labels)
t8_mg_map = ef.elementmapper(t8_filt, 'Mg.1')
t8_fe_map = ef.elementmapper(t8_filt, 'Fe.1')
t8_na_map = ef.elementmapper(t8_filt, 'Na.1')
t8_ca_map = ef.elementmapper(t8_filt, 'Ca.1')
t8_al_map = ef.elementmapper(t8_filt, 'Al.1')
t8_cr_map = ef.elementmapper(t8_filt, 'Cr.1')
t8_si_map = ef.elementmapper(t8_filt, 'Si.1')
t8_o_map = ef.elementmapper(t8_filt, 'O.1')

t8_mindf, t8_mineralindex, t8_imshow_map, t8_mg_map, t8_fe_map, t8_na_map, t8_ca_map, t8_al_map, t8_cr_map, t8_si_map, t8_o_map = ef.mineralid(t8_imshow_map, t8_mg_map, t8_fe_map, t8_na_map, t8_ca_map, t8_al_map, t8_cr_map, t8_si_map, t8_o_map)
t8_glassmask_open, t8_glassmask, t8_glass_mg_mask, t8_glass_fe_mask, t8_glass_na_mask, t8_glass_ca_mask, t8_glass_al_mask, t8_glass_cr_mask, t8_glass_si_mask, t8_glass_o_mask, t8_glass_ci_output = ef.mineralmasks(t8_mineralindex.Label[0], t8_mineralindex, t8_imshow_map, t8_mg_map, t8_fe_map, t8_na_map, t8_ca_map, t8_al_map, t8_cr_map, t8_si_map, t8_o_map)
t8_olmask_open, t8_olmask, t8_ol_mg_mask, t8_ol_fe_mask, t8_ol_na_mask, t8_ol_ca_mask, t8_ol_al_mask, t8_ol_cr_mask, t8_ol_si_mask, t8_ol_o_mask, t8_ol_ci_output = ef.mineralmasks(t8_mineralindex.Label[1], t8_mineralindex, t8_imshow_map, t8_mg_map, t8_fe_map, t8_na_map, t8_ca_map, t8_al_map, t8_cr_map,t8_si_map, t8_o_map)
t8_plagmask_open, t8_plagmask, t8_plag_mg_mask, t8_plag_fe_mask, t8_plag_na_mask, t8_plag_ca_mask, t8_plag_al_mask, t8_plag_cr_mask, t8_plag_si_mask, t8_plag_o_mask, t8_plag_ci_output = ef.mineralmasks(t8_mineralindex.Label[2], t8_mineralindex, t8_imshow_map, t8_mg_map, t8_fe_map, t8_na_map, t8_ca_map, t8_al_map, t8_cr_map,t8_si_map, t8_o_map)
t8_cpxmask_open, t8_cpxmask, t8_cpx_mg_mask, t8_cpx_fe_mask, t8_cpx_na_mask, t8_cpx_ca_mask, t8_cpx_al_mask, t8_cpx_cr_mask, t8_cpx_si_mask, t8_cpx_o_mask, t8_cpx_ci_output = ef.mineralmasks(t8_mineralindex.Label[3], t8_mineralindex, t8_imshow_map, t8_mg_map, t8_fe_map, t8_na_map, t8_ca_map, t8_al_map, t8_cr_map,t8_si_map, t8_o_map)

# %% 

t8_ol_pca, t8_ol_min_pca_df, t8_ol_min_pca_thindf, t8_ol_pca_df, t8_ol_pca_thindf = ef.mineralpca('olivine', 1, 1, t8_ol_mg_mask, t8_ol_fe_mask, t8_ol_na_mask, t8_ol_ca_mask, t8_ol_al_mask, t8_ol_cr_mask, t8_ol_si_mask, t8_ol_o_mask, 'T8_OL_TEST_Scaled') 
t8_plag_pca, t8_plag_min_pca_df, t8_plag_min_pca_thindf, t8_plag_pca_df, t8_plag_pca_thindf = ef.mineralpca('plagioclase', 1, 1, t8_plag_mg_mask, t8_plag_fe_mask, t8_plag_na_mask, t8_plag_ca_mask, t8_plag_al_mask, t8_plag_cr_mask, t8_plag_si_mask, t8_plag_o_mask, 'T8_PLAG_TEST_Scaled') 
t8_cpx_pca, t8_cpx_min_pca_df, t8_cpx_min_pca_thindf, t8_cpx_pca_df, t8_cpx_pca_thindf = ef.mineralpca('clinopyroxene', 1, 1, t8_cpx_mg_mask, t8_cpx_fe_mask, t8_cpx_na_mask, t8_cpx_ca_mask, t8_cpx_al_mask, t8_cpx_cr_mask, t8_cpx_si_mask, t8_cpx_o_mask, 'T8_CPX_TEST_Scaled')
t8_glass_pca, t8_glass_min_pca_df, t8_glass_min_pca_thindf, t8_glass_pca_df, t8_glass_pca_thindf = ef.mineralpca('glass', 1, 1, t8_glass_mg_mask, t8_glass_fe_mask, t8_glass_na_mask, t8_glass_ca_mask, t8_glass_al_mask, t8_glass_cr_mask, t8_glass_si_mask, t8_glass_o_mask, 'T8_GLASS_TEST_Scaled') 

t8_ol_pca1plot = np.empty_like(t8_ol_mg_mask)
t8_ol_pca1plot[:] = np.nan
t8_plag_pca1plot = np.empty_like(t8_plag_fe_mask)
t8_plag_pca1plot[:] = np.nan
t8_cpx_pca1plot = np.empty_like(t8_cpx_mg_mask)
t8_cpx_pca1plot[:] = np.nan
t8_glass_pca1plot = np.empty_like(t8_glass_mg_mask)
t8_glass_pca1plot[:] = np.nan

np.place(t8_ol_pca1plot, ~t8_ol_mg_mask.mask, t8_ol_pca_df.PC1)
np.place(t8_plag_pca1plot, ~t8_plag_fe_mask.mask, t8_plag_pca_df.PC1)
np.place(t8_cpx_pca1plot, ~t8_cpx_mg_mask.mask, t8_cpx_pca_df.PC1)
np.place(t8_glass_pca1plot, ~t8_glass_mg_mask.mask, t8_glass_pca_df.PC1)

# %% 

ef.gmmqemscancomparison(t8_pca_df_lim, t8_pca_input_df, t8_labels, t8_mineralindex, 'T8')
ef.mineralgmmqemscancomparison(t8_filt, t8_ol_mg_mask, t8_ol_fe_mask, 'olivine', 'T8Ol')
ef.mineralgmmqemscancomparison(t8_filt, t8_plag_ca_mask, t8_plag_si_mask, 'plagioclase', 'T8Plag')

# %% Smoothing 

sigma=0.5

Z_ol = ef.gauss_filter(t8_ol_pca1plot, sigma)
Z_plag = ef.gauss_filter(t8_plag_pca1plot, sigma)
Z_cpx = ef.gauss_filter(t8_cpx_pca1plot, sigma)
Z_glass = ef.gauss_filter(t8_glass_pca1plot, sigma)

plt.figure(figsize = (12.5, 17.5))
plt.pcolormesh(Z_ol, cmap = 'Greens', rasterized = True)
plt.pcolormesh(Z_plag, cmap = 'Blues', rasterized = True)
plt.pcolormesh(Z_cpx, cmap = 'Reds', rasterized = True)
plt.pcolormesh(Z_glass, cmap = 'Greys', rasterized = True)
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.tight_layout()
plt.gca().invert_yaxis()
plt.savefig('T8_Zonation_Gaussian_sigma05mu.pdf', dpi = 200, backend='pgf')

plt.figure(figsize = (12.5, 17.5))
plt.pcolormesh(t8_ol_pca1plot, cmap = 'Greens', rasterized = True)
plt.pcolormesh(t8_plag_pca1plot, cmap = 'Blues', rasterized = True)
plt.pcolormesh(t8_cpx_pca1plot, cmap = 'Reds', rasterized = True)
plt.pcolormesh(t8_glass_pca1plot, cmap = 'Greys', rasterized = True)
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.tight_layout()
plt.gca().invert_yaxis()
plt.savefig('T8_Zonation_Gaussian_sigma00mu.pdf', dpi = 200, backend='pgf')




# %%




# for index, (name, estimator) in enumerate(estimators.items()):
#     # do we want to begin this with some supervised learning?
#     # use qemscan mineral ID as initials?

#     estimator.means_init = np.array([X_train.mean(axis=0)
#                                     for i in range(n_classes)])

#     # Train the other parameters using the EM algorithm.
#     estimator.fit(X_train)

#     h = plt.subplot(2, n_estimators // 2, index + 1)
#     make_ellipses(estimator, h)

#     for n, color in enumerate(colors):
#         data = iris.data[iris.target == n]
#         plt.scatter(data[:, 0], data[:, 1], s=0.8, color=color,
#                     label=iris.target_names[n])
#     # Plot the test data with crosses
#     for n, color in enumerate(colors):
#         data = X_test[y_test == n]
#         plt.scatter(data[:, 0], data[:, 1], marker='x', color=color)

#     y_train_pred = estimator.predict(X_train)
#     train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
#     plt.text(0.05, 0.9, 'Train accuracy: %.1f' % train_accuracy,
#              transform=h.transAxes)

#     y_test_pred = estimator.predict(X_test)
#     test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
#     plt.text(0.05, 0.8, 'Test accuracy: %.1f' % test_accuracy,
#              transform=h.transAxes)

#     plt.xticks(())
#     plt.yticks(())
#     plt.title(name)

# plt.legend(scatterpoints=1, loc='lower right', prop=dict(size=12))

# plt.show()

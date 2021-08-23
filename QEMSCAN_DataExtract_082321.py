# %%

import numpy as np
import pandas as pd
import time
import seaborn as sns
import scipy.stats as stats
from sklearn.mixture import GaussianMixture

import matplotlib as mpl
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

start_time = time.time()
scan = pd.read_csv("test2.csv", header = 1)
elapsed_time = time.time() - start_time
print(elapsed_time)

# %% Filter data to remove bad data 

filt_data = scan['Mg.1'] + scan['Fe.1'] + scan['Si.1']
scan_int = scan[filt_data > 25]
filt_data_int = scan_int['Mg.1'] + scan_int['Fe.1'] + scan_int['Si.1']
scan_lim = scan_int[filt_data_int < 99]
filt_data_lim = scan_lim['Mg.1'] + scan_lim['Fe.1'] + scan_lim['Si.1']

# %%

# plt.figure(figsize = (8, 12))
# plt.scatter(scan.X, scan.Y, c = scan['Mg.1']/scan['Si.1'], s = 2.5, cmap = 'Blues', rasterized = True)
# plt.xlabel('X Pixel')
# plt.ylabel('Y Pixel')
# cbar = plt.colorbar()
# cbar.set_label('Mg')
# plt.gca().invert_yaxis()
# plt.tight_layout()

# %% Plot element maps

# plt.figure(figsize = (50, 45))
# plt.subplot(2, 3, 1)
# plt.scatter(scan_lim.X, scan_lim.Y, c = scan_lim['Si.1']/np.max(scan_lim['Si.1']), s = 2.5, cmap = 'Greys', rasterized = True)
# plt.xlabel('X Pixel')
# plt.ylabel('Y Pixel')
# cbar = plt.colorbar()
# cbar.set_label('Si')
# plt.gca().invert_yaxis()
# plt.tight_layout()

# plt.subplot(2, 3, 2)
# plt.scatter(scan_lim.X, scan_lim.Y, c = scan_lim['Mg.1'], s = 2.5, cmap = 'Blues', rasterized = True)
# plt.xlabel('X Pixel')
# plt.ylabel('Y Pixel')
# cbar = plt.colorbar()
# cbar.set_label('Mg')
# plt.gca().invert_yaxis()
# plt.tight_layout()

# plt.subplot(2, 3, 3)
# plt.scatter(scan_lim.X, scan_lim.Y, c = scan_lim['Fe.1'], s = 2.5, cmap = 'Reds', rasterized = True)
# plt.xlabel('X Pixel')
# plt.ylabel('Y Pixel')
# cbar = plt.colorbar()
# cbar.set_label('Fe')
# plt.gca().invert_yaxis()
# plt.tight_layout()

# plt.subplot(2, 3, 4)
# plt.scatter(scan_lim.X, scan_lim.Y, c = scan_lim['Ca.1'], s = 2.5, cmap = 'Purples', rasterized = True)
# plt.xlabel('X Pixel')
# plt.ylabel('Y Pixel')
# cbar = plt.colorbar()
# cbar.set_label('Ca')
# plt.gca().invert_yaxis()
# plt.tight_layout()

# plt.subplot(2, 3, 5)
# plt.scatter(scan_lim.X, scan_lim.Y, c = scan_lim['Al.1'], s = 2.5, cmap = 'Greens', rasterized = True)
# plt.xlabel('X Pixel')
# plt.ylabel('Y Pixel')
# cbar = plt.colorbar()
# cbar.set_label('Al')
# plt.gca().invert_yaxis()
# plt.tight_layout()

# plt.subplot(2, 3, 6)
# plt.scatter(scan_lim.X, scan_lim.Y, c = scan_lim['O.1'], s = 2.5, cmap = 'Oranges', rasterized = True)
# plt.xlabel('X Pixel')
# plt.ylabel('Y Pixel')
# cbar = plt.colorbar()
# cbar.set_label('O')
# plt.gca().invert_yaxis()
# plt.tight_layout()

# # plt.savefig('Mg_Scaled_All.pdf', dpi = 300) # transparent = False)

# %% Cluster Analysis -- Gaussian Mixture Modeling 

# scan_cluster_noO = scan_lim[['Mg.1', 'Fe.1', 'Ca.1', 'Al.1', 'Si.1']]
# n_components = np.arange(1, 21)

# start_time = time.time()
# models_noO = [GaussianMixture(n, covariance_type='full', random_state=0).fit(scan_cluster_noO) for n in n_components]
# elapsed_time = time.time() - start_time
# print(elapsed_time)

# # %% Plot to minimize AIC and BIC criterion, to optimize number of clusters. 
# # This optimization does nto work super well. 

# start_time = time.time()
# plt.plot(n_components, [m.bic(scan_cluster_noO) for m in models_noO], label='BIC')
# plt.plot(n_components, [m.aic(scan_cluster_noO) for m in models_noO], label='AIC')
# plt.legend(loc='best')
# plt.xlabel('n_components')
# elapsed_time = time.time() - start_time
# print(elapsed_time)

# # %% Predicted values. 

# start_time = time.time()

# gmm_noO = GaussianMixture(n_components = 5)
# gmm_noO.fit(scan_cluster_noO)

# labels_noO = gmm_noO.predict(scan_cluster_noO)
# elapsed_time = time.time() - start_time
# print(elapsed_time)

# # %%

# plt.figure(figsize = (10, 4))
# plt.subplot(1, 2, 1)
# plt.scatter(scan_cluster_noO['Mg.1'], scan_cluster_noO['Fe.1'], s = 5, c=labels_noO) # , cmap = 'viridis')
# plt.xlabel('Mg.1')
# plt.ylabel('Fe.1')
# plt.colorbar()

# plt.subplot(1, 2, 2)
# plt.scatter(scan_cluster_noO['Ca.1'], scan_cluster_noO['Al.1'], s = 5, c=labels_noO) # , cmap = 'viridis')
# plt.xlabel('Ca.1')
# plt.ylabel('Al.1')
# plt.colorbar()
# plt.show()

# # %%

# plt.figure(figsize = (8, 12))
# plt.scatter(scan_lim.X, scan_lim.Y, c = labels_noO, s = 2.5, cmap = 'viridis', rasterized = True)
# plt.xlabel('X Pixel')
# plt.ylabel('Y Pixel')
# plt.colorbar()
# plt.gca().invert_yaxis()
# plt.tight_layout()
# plt.savefig('Clustered_noSi.pdf', dpi = 300)

# # %%

# scan_remove = scan_lim[labels_noO != 3]
# labels_remove = labels_noO[labels_noO != 3]

# scan_remove_again = scan_remove[labels_remove != 4]
# labels_remove_again = labels_remove[labels_remove != 4]

# plt.figure(figsize = (8, 12))
# plt.scatter(scan_remove_again.X, scan_remove_again.Y, c = labels_remove_again, s = 1, cmap = 'viridis', rasterized = True)
# plt.xlabel('X Pixel')
# plt.ylabel('Y Pixel')
# plt.gca().invert_yaxis()
# plt.colorbar()
# plt.tight_layout()
# plt.savefig('Clustered_noSi_remove.pdf', dpi = 300)


# %%

# scan_cluster_withO = scan_lim[['Mg.1', 'Fe.1', 'Ca.1', 'Al.1', 'Si.1', 'O.1']]
# n_components = np.arange(1, 21)

# start_time = time.time()
# models_withO = [GaussianMixture(n, covariance_type='full', random_state=0).fit(scan_cluster_withO) for n in n_components]
# elapsed_time = time.time() - start_time
# print(elapsed_time)

# %% Plot to minimize AIC and BIC criterion, to optimize number of clusters. 
# This optimization does not work super well. 

# start_time = time.time()
# plt.plot(n_components, [m.bic(scan_cluster_withO) for m in models_withO], label='BIC')
# plt.plot(n_components, [m.aic(scan_cluster_withO) for m in models_withO], label='AIC')
# plt.legend(loc='best')
# plt.xlabel('n_components')
# elapsed_time = time.time() - start_time
# print(elapsed_time)

# %% Predicted values. 

scan_cluster_withO = pd.read_csv('scan_cluster_withO.csv', index_col = [0])

start_time = time.time()

gmm_withO = GaussianMixture(n_components = 5)
gmm_withO.fit(scan_cluster_withO)

labels_withO = gmm_withO.predict(scan_cluster_withO)
elapsed_time = time.time() - start_time
print(elapsed_time)

# %% 

col_index = (len(np.unique(scan_lim.X)))
row_index = (len(np.unique(scan_lim.Y)))
preallocate_phase_map = np.empty([row_index, col_index])
preallocate_mg_map = np.empty([row_index, col_index])
preallocate_fe_map = np.empty([row_index, col_index])
preallocate_ca_map = np.empty([row_index, col_index])
preallocate_al_map = np.empty([row_index, col_index])

def elementmapper(scan_lim, element, preallocate_element_map): 

    X = scan_lim.X.ravel()
    Y = scan_lim.Y.ravel()
    E = scan_lim[element].ravel()

    for i in range(len(X)-1): 
        preallocate_element_map[Y[i], X[i]] = E[i]

    return preallocate_element_map 


def phasemapper(scan_lim, label, preallocate_phase_map): 
    
    X = scan_lim.X.ravel()
    Y = scan_lim.Y.ravel()

    for i in range(len(X)-1): 
        preallocate_phase_map[Y[i], X[i]] = label[i] + 1

    return preallocate_phase_map 

# %%

imshow_map = phasemapper(scan_lim, labels_withO, preallocate_phase_map)
# imshow_map[imshow_map == 0] = np.nan
imshow_map[imshow_map == 4] = np.nan


# %% 

mg_map = elementmapper(scan_lim, 'Mg.1', preallocate_mg_map)
fe_map = elementmapper(scan_lim, 'Fe.1', preallocate_fe_map)
ca_map = elementmapper(scan_lim, 'Ca.1', preallocate_ca_map)
al_map = elementmapper(scan_lim, 'Al.1', preallocate_al_map)

# %%

color = ['#0D6401', '#1EA0A0', '#D2D200', '#FEE0C0']
cMap = ListedColormap(color)

values = ['Clinopyroxene', 'Plagioclase', 'Olivine', 'Glass']
patches = [mpatches.Patch(color=color[i], label=values[i]) for i in range(len(values)) ]
patches = [patches[3], patches[2], patches[1], patches[0]]

fig, ax = plt.subplots(figsize = (8, 12))
plt.pcolormesh(imshow_map, cmap = cMap, rasterized = True)
plt.xlabel('X Pixel')
plt.ylabel('Y Pixel')
plt.xlim([0, 4000])
plt.ylim([0, 7500])
plt.legend(handles = patches, loc = 'lower right', prop={'size': 12})
plt.gca().invert_yaxis()
plt.tight_layout()
ax.set_aspect('equal', 'box')
plt.savefig('pcolormesh_all.pdf', dpi=300)

ol_sect_plot = imshow_map[3300:3550, 900:1200]
fig, ax = plt.subplots(figsize = (8, 8))
plt.pcolormesh(ol_sect_plot, cmap = cMap, rasterized = True)
plt.xlabel('X Pixel')
plt.ylabel('Y Pixel')
plt.xlim([0, 300])
plt.ylim([0, 250])
plt.gca().invert_yaxis()
plt.legend(handles = patches, prop={'size': 12})
plt.tight_layout()
ax.set_aspect('equal', 'box')
plt.savefig('pcolormesh_ol.pdf', dpi=300)

# %%

imshow_map = phasemapper(scan_lim, labels_withO, preallocate_phase_map)
imshow_map[imshow_map == 0] = np.nan
imshow_map[imshow_map == 4] = np.nan


mg_map = elementmapper(scan_lim, 'Mg.1', preallocate_mg_map)
fe_map = elementmapper(scan_lim, 'Fe.1', preallocate_fe_map)
ca_map = elementmapper(scan_lim, 'Ca.1', preallocate_ca_map)
al_map = elementmapper(scan_lim, 'Al.1', preallocate_al_map)
si_map = elementmapper(scan_lim, 'Si.1', preallocate_al_map)
o_map = elementmapper(scan_lim, 'O.1', preallocate_al_map)


ol_sect_plot = imshow_map[3300:3550, 900:1200]
ol_mg_map = mg_map[3300:3550, 900:1200]
ol_fe_map = fe_map[3300:3550, 900:1200]
ol_ca_map = ca_map[3300:3550, 900:1200]
ol_al_map = al_map[3300:3550, 900:1200]
ol_si_map = si_map[3300:3550, 900:1200]
ol_o_map = o_map[3300:3550, 900:1200]

# mx = np.ma.masked_array(ol_mg_map, mask = ol_sect_plot!=2)

# %%

uniqueindex = np.unique(imshow_map)[~np.isnan(np.unique(imshow_map))]


colorredo = ['#FFFFFF', 'k']
cMapnew = ListedColormap(colorredo)
values_title = ['Clinopyroxene', 'Plagioclase', 'Olivine', '', 'Glass']

rc('font',**{'family':'Avenir', 'size': 14})


for i in uniqueindex:
    min_mask = ol_sect_plot!=i
    mineral = np.ma.masked_array(ol_sect_plot, mask = min_mask)
    mg_mask = np.ma.masked_array(ol_mg_map, mask = min_mask)
    fe_mask = np.ma.masked_array(ol_fe_map, mask = min_mask)
    ca_mask = np.ma.masked_array(ol_ca_map, mask = min_mask)
    al_mask = np.ma.masked_array(ol_al_map, mask = min_mask)
    si_mask = np.ma.masked_array(ol_si_map, mask = min_mask)
    o_mask = np.ma.masked_array(ol_o_map, mask = min_mask)

    min_dist_index = ['Mg.1', 'Fe.1', 'Ca.1', 'Al.1', 'Si.1', 'O.1']
    min_dist_mean = np.array([np.mean(mg_mask), np.mean(fe_mask), np.mean(ca_mask), np.mean(al_mask), np.mean(si_mask), np.mean(o_mask)])

    mg_95 = stats.t.interval(0.95, len(mg_mask.compressed())-1, loc=np.mean(mg_mask.compressed()), scale=stats.sem(mg_mask.compressed()))
    fe_95 = stats.t.interval(0.95, len(fe_mask.compressed())-1, loc=np.mean(fe_mask.compressed()), scale=stats.sem(fe_mask.compressed()))
    ca_95 = stats.t.interval(0.95, len(ca_mask.compressed())-1, loc=np.mean(ca_mask.compressed()), scale=stats.sem(ca_mask.compressed()))
    al_95 = stats.t.interval(0.95, len(al_mask.compressed())-1, loc=np.mean(al_mask.compressed()), scale=stats.sem(al_mask.compressed()))
    si_95 = stats.t.interval(0.95, len(si_mask.compressed())-1, loc=np.mean(si_mask.compressed()), scale=stats.sem(si_mask.compressed()))
    o_95 = stats.t.interval(0.95, len(o_mask.compressed())-1, loc=np.mean(o_mask.compressed()), scale=stats.sem(o_mask.compressed()))


    fig, ax = plt.subplots(figsize = (8.5, 12.5))

    plt.subplot(4, 2, 1)
    plt.pcolormesh(ol_sect_plot==i, cmap = cMapnew)
    plt.pcolormesh(ol_sect_plot, cmap = cMap, alpha = 0.1)
    plt.title('Mineral = ' + str(values_title[int(i-1)]))
    # plt.title("Pixels = "+ str(len(min)) + '/' + str(len(ol_sect_int)))
    plt.gca().invert_yaxis()

    plt.subplot(4, 2, 2)
    sns.barplot(x = min_dist_index, y = min_dist_mean)
    plt.title('Concentration Average')
    # plt.xlabel('Element')
    # plt.ylabel('Concentration')
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

    plt.savefig(str(values_title[int(i-1)])+'.pdf', dpi = 100)
    plt.show()

# %%








# %% 


ol_sect_int = scan_lim[scan_lim.X < 1200]
ol_sect_int1 = ol_sect_int[ol_sect_int.X > 900]
ol_sect_int2 = ol_sect_int1[ol_sect_int1.Y > 3250]
ol_sect_int3 = ol_sect_int2[ol_sect_int2.Y < 3550]

ol_sect = ol_sect_int3[['Mg.1', 'Fe.1', 'Ca.1', 'Al.1', 'Si.1']]
label_ol_sect = gmm.predict(ol_sect)

plt.figure(figsize = (30, 10))
plt.subplot(1, 3, 1)
plt.scatter(ol_sect_int3.X, ol_sect_int3.Y, c = ol_sect_int3['Mg.1']/(ol_sect_int3['Mg.1'] + ol_sect_int3['Fe.1']), s = 1.5, cmap = 'Blues', rasterized = True)
plt.xlabel('X Pixel')
plt.ylabel('Y Pixel')
plt.gca().invert_yaxis()
cbar = plt.colorbar()
cbar.set_label('Mg/(Mg+Fe)')
plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()

plt.subplot(1, 3, 2)
plt.scatter(ol_sect_int3.X, ol_sect_int3.Y, c = ol_sect_int3['Fe.1'], s = 1.5, cmap = 'Greens', rasterized = True)
plt.xlabel('X Pixel')
plt.ylabel('Y Pixel')
plt.gca().invert_yaxis()
cbar = plt.colorbar()
cbar.set_label('Fe')
plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()

plt.subplot(1, 3, 3)
plt.scatter(ol_sect_int3.X, ol_sect_int3.Y, c = label_ol_sect, s = 1.5, rasterized = True)
plt.xlabel('X Pixel')
plt.ylabel('Y Pixel')
plt.gca().invert_yaxis()
cbar = plt.colorbar()
cbar.set_label('Cluster')
plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()
# plt.savefig('Ol_Cluster.pdf', dpi = 300)


# %%

# %% 

ol_sect_int = scan_lim[scan_lim.X < 3800]
ol_sect_int1 = ol_sect_int[ol_sect_int.X > 3000]
ol_sect_int2 = ol_sect_int1[ol_sect_int1.Y > 1600]
ol_sect_int3 = ol_sect_int2[ol_sect_int2.Y < 2300]

plt.figure(figsize = (20, 10))
plt.subplot(1, 2, 1)
plt.scatter(ol_sect_int3.X, ol_sect_int3.Y, c = ol_sect_int3['Ca.1']/np.max(ol_sect_int3['Ca.1']), s = 2.5, cmap = 'Blues', rasterized = True)
plt.xlabel('X Pixel')
plt.ylabel('Y Pixel')
plt.gca().invert_yaxis()
cbar = plt.colorbar()
cbar.set_label('Mg/(Mg+Fe)')
plt.gca().set_aspect('equal', adjustable='box')
# plt.tight_layout()

plt.subplot(1, 2, 2)
plt.scatter(ol_sect_int3.X, ol_sect_int3.Y, c = ol_sect_int3['Fe.1'], s = 2.5, cmap = 'Greens', rasterized = True)
plt.xlabel('X Pixel')
plt.ylabel('Y Pixel')
cbar = plt.colorbar()
cbar.set_label('Fe')
plt.gca().invert_yaxis()
plt.gca().set_aspect('equal', adjustable='box')
# plt.tight_layout()

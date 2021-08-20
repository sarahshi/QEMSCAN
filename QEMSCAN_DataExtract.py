# %%

import numpy as np
import pandas as pd
import time
import seaborn as sns
from sklearn.mixture import GaussianMixture

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import rc
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

# %% Plot element maps

plt.figure(figsize = (50, 45))
plt.subplot(2, 3, 1)
plt.scatter(scan_lim.X, scan_lim.Y, c = scan_lim['Si.1']/np.max(scan_lim['Si.1']), s = 2.5, cmap = 'Greys', rasterized = True)
plt.xlabel('X Pixel')
plt.ylabel('Y Pixel')
cbar = plt.colorbar()
cbar.set_label('Si')
plt.gca().invert_yaxis()
plt.tight_layout()

plt.subplot(2, 3, 2)
plt.scatter(scan_lim.X, scan_lim.Y, c = scan_lim['Mg.1'], s = 2.5, cmap = 'Blues', rasterized = True)
plt.xlabel('X Pixel')
plt.ylabel('Y Pixel')
cbar = plt.colorbar()
cbar.set_label('Mg')
plt.gca().invert_yaxis()
plt.tight_layout()

plt.subplot(2, 3, 3)
plt.scatter(scan_lim.X, scan_lim.Y, c = scan_lim['Fe.1'], s = 2.5, cmap = 'Reds', rasterized = True)
plt.xlabel('X Pixel')
plt.ylabel('Y Pixel')
cbar = plt.colorbar()
cbar.set_label('Fe')
plt.gca().invert_yaxis()
plt.tight_layout()

plt.subplot(2, 3, 4)
plt.scatter(scan_lim.X, scan_lim.Y, c = scan_lim['Ca.1'], s = 2.5, cmap = 'Purples', rasterized = True)
plt.xlabel('X Pixel')
plt.ylabel('Y Pixel')
cbar = plt.colorbar()
cbar.set_label('Ca')
plt.gca().invert_yaxis()
plt.tight_layout()

plt.subplot(2, 3, 5)
plt.scatter(scan_lim.X, scan_lim.Y, c = scan_lim['Al.1'], s = 2.5, cmap = 'Greens', rasterized = True)
plt.xlabel('X Pixel')
plt.ylabel('Y Pixel')
cbar = plt.colorbar()
cbar.set_label('Al')
plt.gca().invert_yaxis()
plt.tight_layout()

plt.subplot(2, 3, 6)
plt.scatter(scan_lim.X, scan_lim.Y, c = scan_lim['O.1'], s = 2.5, cmap = 'Oranges', rasterized = True)
plt.xlabel('X Pixel')
plt.ylabel('Y Pixel')
cbar = plt.colorbar()
cbar.set_label('O')
plt.gca().invert_yaxis()
plt.tight_layout()

plt.savefig('Mg_Scaled_All.pdf', dpi = 300) # transparent = False)

# %%
# %% Cluster Analysis -- Gaussian Mixture Modeling 

scan_cluster = scan_lim[['Mg.1', 'Fe.1', 'Ca.1', 'Al.1', 'Si.1']]
n_components = np.arange(1, 21)

start_time = time.time()
models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(scan_cluster) for n in n_components]
elapsed_time = time.time() - start_time
print(elapsed_time)

# %% Plot to minimize AIC and BIC criterion, to optimize number of clusters. 
# This optimization does nto work super well. 

start_time = time.time()
plt.plot(n_components, [m.bic(scan_cluster) for m in models], label='BIC')
plt.plot(n_components, [m.aic(scan_cluster) for m in models], label='AIC')
plt.legend(loc='best')
plt.xlabel('n_components')
elapsed_time = time.time() - start_time
print(elapsed_time)

# %% Predicted values. 

start_time = time.time()

gmm = GaussianMixture(n_components = 5)
gmm.fit(scan_cluster)

labels = gmm.predict(scan_cluster)
elapsed_time = time.time() - start_time
print(elapsed_time)


# %%

plt.figure(figsize = (8, 8))
plt.scatter(scan_cluster['Mg.1'], scan_cluster['Fe.1'], s = 5, c=labels) # , cmap = 'viridis')
plt.xlabel('Mg.1')
plt.ylabel('Fe.1')
plt.legend()
plt.show()

plt.figure(figsize = (8, 8))
plt.scatter(scan_cluster['Ca.1'], scan_cluster['Al.1'], s = 5, c=labels) # , cmap = 'viridis')
plt.xlabel('Ca.1')
plt.ylabel('Al.1')
plt.legend()
plt.show()

# %%

plt.figure(figsize = (8, 12))
plt.scatter(scan_lim.X, scan_lim.Y, c = labels, s = 2.5, cmap = 'viridis', rasterized = True)
plt.xlabel('X Pixel')
plt.ylabel('Y Pixel')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('Clustered.pdf', dpi = 300)



# %%


# %%


# %% 

ol_sect_int = scan_lim[scan_lim.X < 1200]
ol_sect_int1 = ol_sect_int[ol_sect_int.X > 900]
ol_sect_int2 = ol_sect_int1[ol_sect_int1.Y > 3250]
ol_sect_int3 = ol_sect_int2[ol_sect_int2.Y < 3550]

plt.figure(figsize = (20, 10))
plt.subplot(1, 2, 1)
plt.scatter(ol_sect_int3.X, ol_sect_int3.Y, c = ol_sect_int3['Mg.1']/(ol_sect_int3['Mg.1'] + ol_sect_int3['Fe.1']), s = 2.5, cmap = 'Blues', rasterized = True)
plt.xlabel('X Pixel')
plt.ylabel('Y Pixel')
plt.gca().invert_yaxis()
cbar = plt.colorbar()
cbar.set_label('Mg/(Mg+Fe)')
plt.gca().set_aspect('equal', adjustable='box')
# plt.tight_layout()


ol_sect_int = scan_lim[scan_lim.X < 1200]
ol_sect_int1 = ol_sect_int[ol_sect_int.X > 900]
ol_sect_int2 = ol_sect_int1[ol_sect_int1.Y > 3250]
ol_sect_int3 = ol_sect_int2[ol_sect_int2.Y < 3550]

plt.subplot(1, 2, 2)
plt.scatter(ol_sect_int3.X, ol_sect_int3.Y, c = ol_sect_int3['Fe.1'], s = 2.5, cmap = 'Greens', rasterized = True)
plt.xlabel('X Pixel')
plt.ylabel('Y Pixel')
plt.gca().invert_yaxis()
cbar = plt.colorbar()
cbar.set_label('Fe')
plt.gca().set_aspect('equal', adjustable='box')



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


# %% 

# column_index = np.unique(scan.X)
# row_index = np.unique(scan.Y)

# working_df = pd.DataFrame(columns = column_index)
# working_df['#'] = row_index

# %%

# def elementdf(input_df, preallocate_df, element): 

#     for i in range(len(input_df)-1): 
#         index_val = input_df.iloc[i]
#         preallocate_df.iloc[index_val.Y, index_val.X] = index_val[element]

#     return preallocate_df 

# %%

# def elementdftest(input_df, preallocate_df, element): 
#     preallocate_df.iloc[input_df.Y, input_df.X] = input_df[element]
#     return preallocate_df 

# Mg_df_test = elementdftest(scan.iloc[1:5000], working_df, 'Mg')


# %% 

# start_time = time.time()
# Mg_df = elementdf(scan, working_df, 'Mg')
# elapsed_time = time.time() - start_time
# print(elapsed_time)

# # %% 
# Mg_df.to_csv('MgDF.csv')

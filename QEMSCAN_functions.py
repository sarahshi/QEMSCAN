#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 20:25:47 2021

@author: norbert
"""

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

def get_masks(label_array, values = None):
    """
    Creates a list of masks from a label array passed to it.

    Parameters
    ----------
    label_array : 2D numpy array
        Array containing phase labels assigned to each pixel.
    values : list of ints/labels, optional
        The labels to be masked from the label array. The default is None.

    Returns
    -------
    masks : list of 2D numpy arrays
        Colection of phase masks generated.

    """
    
    if values == None:
        values = np.unique(label_array)
    else:
        pass
    
    masks = []
    
    for label in values:
        temp_mask = np.zeros(label_array.shape)
        temp_mask[label_array == label] = 1
        masks.append(temp_mask)
        
    return masks
        
        

def build_conc_map(data, shape=None):
    """
    Build 3D numpy conc_map from pandas dataframe.
    
    Input
    -----------
    data (pd dataframe) - pandas dataframe to be transformed to numpy data matrix.
    shape (list) - (optional) desired shape of the resulting array; if not given
                        one will be generated using the data and shape of dataframe.
                        
    Return
    -----------
    conc_map (3D numpy array) - the resulting data matrix of shape either given or
                        calculated.
    """
    
    if isinstance(data, pd.DataFrame):
        pass
    else:
        raise ValueError("Data is not pandas dataframe.")
    
    if shape == None:
        x_max = data['X'].max() + 1
        y_max = data['Y'].max() + 1
        n_features = len(data.columns) - 2 #need to substract x,y
        shape = [x_max, y_max, n_features]
    else:
        pass
    
    conc_map = np.zeros(shape)
    k = 1
    length = len(data)
    
    for i in range(length):
        x = data.loc[i, 'X']
        y = data.loc[i, 'Y']
        conc_map[x,y] = data.iloc[i].to_numpy()

        if int((i/length)*10) == k:
            print(k*10)
            k += 1
        else:
            pass
    print(100)
    
    return conc_map



def flatten_3d(array, desired_shape = None):
    """
    Function to flatten 3d arrays into 2d to be used for further analyses.
    
    Input
    ---------------
    array (3D numpy array: [n1, n2, n3]) - array to be flattened.
    desired_shape (2 element list/array) - shape of new array, if not given it will
                                            be [n1*n2, n3].
    
    Return
    ---------------
    new_arr (2D numpy array of desired shape) - new array.
    shape (3 element list/array) - original shape of the array passed to the function
    """
    
    shape = array.shape
    if len(shape) == 3:
        pass
    else:
        raise ValueError("Input array needs to have 3 dimensions.")
    
    if desired_shape == None:
        desired_shape = [shape[0]*shape[1], shape[2]]
    else:
        pass
    
    new_arr = array.transpose()
    new_arr = new_arr.reshape(desired_shape[1], desired_shape[0])
    new_arr = new_arr.transpose()
    
    del array, desired_shape
    
    return new_arr, shape


def reshape_2d_to_3d(array, desired_shape = None):
    """
    Function to reshape 2d arrays into 3d as a post-processing step eg. after PCA/NMF.
    
    Input
    ---------------
    array (2D numpy array: [n1, n2]) - array to be flattened.
    desired_shape (3 element list/array) - shape of new array, if not given it will
                                            be approximately [sqrt(n1),sqrt(n1), n2].
    
    Return
    ---------------
    new_arr (3D numpy array of desired shape) - new array.
    shape (2 element list/array) - original shape of the array passed to the function
    """

    
    shape = array.shape
    if len(shape) == 2:
        pass
    else:
        raise ValueError("Input array needs to have 2 dimensions only.")
    
    if desired_shape == None:
        shape_sqrt = int(np.sqrt(shape[0]))
        desired_shape = [shape_sqrt, shape_sqrt + (shape[0]%shape_sqrt), shape[1]]
    else:
        pass
    
    new_arr = array.transpose()
    new_arr = new_arr.reshape(desired_shape[2], desired_shape[1], desired_shape[0])
    new_arr = new_arr.transpose()
    
    return new_arr, shape

def plot_cluster(labels, centers, plot_return = False, elements = None, shape = None):
    """
    Function to plot the results of the clustering function. It may be called directly or
    through the clustering function itself.
    
    Input
    -----------
    labels (2D numpy array) - array containing the labels assigned to every pixel.
    centers (2D numpy array) - list of cluster center compositions.
    plot_return (bool) - (optional) if True the matplotlib figure and axis objects
                            are returned; default is False
    elements (list) - (optional) x tick labels used for plotting cluster center
                            compositions.
    shape (list) - (optional) if label matrix passed is 1D, this is the shape used
                            to plot it on; default is to reshape as close to square
                            as possible (not recommended)
    
    Return (if set True)
    -----------
    fig - 2 element list of matplotlib figure objects; first element is for label image
                            and second for cluster composition plot.
    ax - 2 element list of matplotlib axis objects; first element is for label image
                            and second for cluster composition plot.
    """
    
    if len(labels.shape) == 1:
        if shape == None:
            shape_sqrt = int(np.sqrt(len(labels)))
            labels = labels.reshape(shape_sqrt, shape_sqrt + (len(labels)%shape_sqrt))
        else:
            labels = labels.reshape(shape)
    elif len(labels.shape) == 2:
        pass
    else:
        raise ValueError("Label array needs to have 1 or 2 dimensions.")
    
    fig, ax = plt.subplots(1,1, figsize = (12,12))
    ax.imshow(labels)
    fig.tight_layout()
    
    if elements == None:
        elements = [i for i in range(len(centers[:]))]
    else:
        pass
    
    fig2, ax2 = plt.subplots(int(len(centers)/2), len(centers)%2, figsize = (12,12))
    for i in range(len(centers)):
        ax2[int(i/2)][i%2].bar(range(len(elements)), centers[i], width = 0.5, tick_label = elements)
    fig2.tight_layout()
    
    if plot_return == True:
        return [fig,fig2], [ax, ax2]
    else:
        return None
    
def plot_decomp(scores, comps, plot_return = False, elements = None, shape = None):
    """
    Function to plot the results of the decomposition function. It may be called directly or
    through the clustering function itself.
    
    Input
    -----------
    scores (3D numpy array) - array containing the scores assigned to every pixel
                            with a separate image for every component.
    comps (2D numpy array) - list of component compositions ordered identical to
                            score maps.
    plot_return (bool) - (optional) if True the matplotlib figure and axis objects
                            are returned; default is False
    elements (list) - (optional) x tick labels used for plotting cluster center
                            compositions.
    shape (list) - (optional) if scores matrix passed is 2D, this is the shape used
                            to plot it on; default is to reshape first dimension as 
                            close to square as possible, keeping the second dimension
                            unchanged (not recommended)
    
    Return (if set True)
    -----------
    fig - matplotlib figure object for plot.
    ax - matplotlib axis object for plot.
    """
    
    if len(scores.shape) == 2:
        if shape == None:
            shape_sqrt = int(np.sqrt(len(scores)))
            scores = reshape_2d_to_3d(scores, [shape_sqrt, shape_sqrt + (len(scores)%shape_sqrt), len(comps)])
        else:
            scores = scores.reshape(shape)
    elif len(scores.shape) == 3:
        pass
    else:
        raise ValueError("Scores array needs to have 1 or 2 dimensions.")
    
    fig, ax = plt.subplots(len(comps), 2, figsize = (12,24))
    
    if elements == None:
        elements = [i for i in range(len(comps[:]))]
    else:
        pass

    for i in range(len(comps)):
        ax[i][0].imshow(scores[:,:,i])
        ax[i][1].bar(range(len(elements)), comps[i], width = 0.5, tick_label = elements)
        ax[i][1].plot([-0.5, len(elements)-0.5], [0, 0], 'k-')

    fig.tight_layout()
    
    if plot_return == True:
        return fig, ax
    else:
        return None

    
def cluster(data, n_clusters = 2, method = "k_means", shape = None,
            plot = False,plot_return = False, elements = None,
           df_shape = None):
    """
    Function to perform clustering on the dataset passed using the selected clustering
    algorithm.
    
    Input
    ------------
    data (either 2D or 3D numpy array) - the dataset to perform clustering on.
    n_clusters (int) - number of clusters to find, default is 2.
    method (str) - clustering algorithm to be used ["k_means", "gmm"]; default is k_means.
    shape (list) - shape of the output label array if data is a 2D array. Default is
                    approximately [sqrt(n1), sqrt(n1)].
    plot (bool) - Make True if results are to be plotted; default is false.
    plot_return (bool) - optional, if plot=true, make True to return fig and ax objects, default is false.
    elements (list/array) - optional, used when plotting results only, default is None.
    
    Return
    ------------
    labels (2D numpy array) - assigned labels for each cluster found within the passed dataset. 
                    Shape is the same as first two dimensions of data if it's 3D, otherwise it's
                    the shape parameter passed to the function.
    centers (2D numpy array of shape [n_clusters, n_features]) - list of the centres of clusters
                    found in the dataset.
    fig, ax (matplotlib objects (both of length 2)) - only if both plot and plot_return are set True.
    """
        
    if isinstance(data, pd.DataFrame):
        data = build_conc_map(data, df_shape)
    else:
        pass
    
    if len(data.shape) == 3:
        array, shape = flatten_3d(data)
    elif len(data.shape) == 2:
        array = data
        if shape == None:
            shape_sqrt = int(np.sqrt(data.shape[0]))
            shape = [shape_sqrt, shape_sqrt + (data.shape[0]%shape_sqrt)]
            del shape_sqrt
        else:
            pass
    else:
        raise ValueError("Input array needs to have 2 or 3 dimensions.")
            
    start = time.time()
    
    if method.lower() == "k_means":
        from sklearn.cluster import KMeans
        #perform k_means clustering
        kmeans = KMeans(n_clusters=n_clusters, init = 'k-means++').fit(array)
        labels = kmeans.labels_.reshape(shape[1],shape[0]).copy()
        labels = labels.transpose()
        centers = kmeans.cluster_centers_.copy()
        del array, shape, kmeans
        
    elif method.lower() == "gmm":
        from sklearn.mixture import GaussianMixture
        #perform GMM
        gmm = GaussianMixture(n_clusters)
        labels = gmm.fit_predict(array).reshape(shape[1],shape[0]).copy()
        labels = labels.transpose()
        centers = gmm.means_.copy()
        del array, shape, gmm
        
    else:
        raise ValueError("Method " + str(method) + " is not recognised.")
        
    process_time = time.time() - start    
    print("Processing time (s): " + str(process_time))
    
    if plot == True:
        fig, ax = plot_cluster(labels, centers, plot_return = True ,elements=elements)
        if plot_return == True:
            return labels, centers, fig, ax
        else:
            pass
    else:
        pass
    
    return labels, centers

def decompose(data, n_components = 2, method = "pca", tol = 0.05, shape = None;
              plot = False, plot_return = False, elements = None, 
             df_shape = None):
    """
    Function to perform the selected decomposition algorithm on the data passed. Ideal for use in
    the initial data exploration steps.
    
    Input
    ------------
    data (either 2D or 3D numpy array) - the dataset to be decomposed.
    n_components (int) - number of components to be kept after decomposition; default is 2.
    method (str) - choose one of ["pca", "nmf"] as the algorithm to be used; default is pca.
    tol (float) - tolerance value to be used for NMF decomposition; default is 0.05.
    plot (bool) - Make True if results are to be plotted; default is false.
    plot_return (bool) - optional, if plot=true, make True to return fig and ax objects; default is false.
    elements (list/array) - optional, used when plotting results only; default is None.
    
    Return
    ------------
    scores (3D numpy array) - scores relating each element in the original dataset to the 
                            components found. Third dimension equal to n_components.
    components (2D numpy array [n_components, n_features]) - Components found to describe the 
                            dataset; ordered by the fraction of dataset variance described by
                            each component.
    fig, ax (single matplotlib objects) - only if both plot and plot_return are set True.
    """
    from sklearn import decomposition
    
    if isinstance(data, pd.DataFrame):
        data = build_conc_map(data, df_shape)
    else:
        pass
    
    if len(data.shape) == 3:
        array, shape = flatten_3d(data)
    elif len(data.shape) == 2:
        array = data
        if shape == None:
            shape_sqrt = int(np.sqrt(data.shape[0]))
            shape = [shape_sqrt, shape_sqrt + (data.shape[0]%shape_sqrt)]
            del shape_sqrt
        else:
            pass
    else:
        raise ValueError("Input array needs to have 2 or 3 dimensions.")
        
    start = time.time()
    
    if method.lower() == "pca":
        #perform PCA decomposition
        pca = decomposition.PCA(n_components = n_components)
        scores = reshape_2d_to_3d(pca.fit_transform(array), [shape[0], shape[1], n_components])[0].copy()
        components = pca.components_.copy()
        
        del array, shape, pca
        
    elif method.lower() == "nmf":
        #perform NMF decomposition
        nmf = decomposition.NMF(n_components = n_components, tol = tol)
        scores = reshape_2d_to_3d(nmf.fit_transform(array), [shape[0], shape[1], n_components])[0].copy()
        components = nmf.components_.copy()
        
        del array, shape, nmf
        
    else:
        raise ValueError("Method " + str(method) + " is not recognised.")
    
    process_time = time.time() - start    
    print("Processing time (s): " + str(process_time))
    
    if plot == True:
        fig, ax = plot_decomp(scores, components, plot_return = True, elements=elements)
        if plot_return == True:
            return scores, components, fig, ax
        else:
            pass
    else:
        pass
    
    return scores, components


def complete_phaseMap(values, phases, cmaps):
    """
    Create a complete phase map of the results of previous analyses. Generates
    a large plot of all phases put on the same array as well as a collection of
    plots showing each phase/mask passed to the function separately with their
    respective colorbars.

    Parameters
    ----------
    values : list of 2D numpy arrays
        Values to be plotted on which the colour map intensity depends. It can
        be a simple binary phase mask (making all pixels of that phase the same
        colour) or sets of pca/nmf scores to show some compoisiton variation - or
        even a mix of the two for different phases!
        
    phases : list
            Phase labels to be used to annotate plots eg. ["Olivine, Plag, etc."] 
            in same order as the phases in the values input.
    cmaps : list
        List of colourmaps (see matplotlib documentation) for each phase; must be 
        in the same order as the above inputs to perfectly match each phase with
        its desired colourmap.
        
    Returns
    -------
    fig : list of two matplotlib figure object
        [fig, fig2] figure objects corresponding to the two different plots generated.
        Can be used to alter the plots at a later date.
    ax : list of matplotlib axis objects
        [ax, ax2] axis objects (where ax2 is a list of length n_phases) relating to the two
        plots generated - same as figure objects. Can be used to alter the axes in plots
        at a later date.
    """
    fig, ax = plt.subplots(1,1,figsize=(12,12))
    
    fig2 = plt.figure(figsize = (12,12))
    ax2 = [0*i for i in range(len(values))]
    
    nrows = int(len(values)/2)+1
    ncols = 2
    
    for i in range(len(values)):
        temp_map = values[i]
        
        if len(np.unique(temp_map)) <= 2:
            temp_vmin = temp_map.min()
            temp_vmax = temp_map.max()
        else:
            temp_vmin = temp_map[temp_map !=0].min()
            temp_vmax = temp_map.max()

        temp_alpha = np.zeros(temp_map.shape)
        temp_alpha[temp_map != 0] = 1
        
        ax.imshow(temp_map, cmap = cmaps[i], alpha = temp_alpha,
                 vmin = temp_vmin, vmax = temp_vmax)
        
        ax2[i] = fig2.add_subplot(nrows, ncols, i+1)
        separate = ax2[i].imshow(temp_map, cmap = cmaps[i],
                 vmin = temp_vmin, vmax = temp_vmax)
        fig2.colorbar(separate, ax = ax2[i],fraction=0.03, pad=0.04)
        ax2[i].title.set_text(phases[i])
    
    
    fig.tight_layout()
    fig2.tight_layout()
    
    return [fig, fig2], [ax, ax2]

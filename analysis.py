#!/usr/bin/env python

import os, sys
import numpy as np
from scipy import linalg
import pyrcca as cc
import nibabel as nib
import pandas as pd

CTX_ROIS = ['Q23_RAmy_RCT_maskq10_CUN.SOG.txt',
            'Q23_RAmy_RCT_maskq10_OLF.txt',
            'Q23_RAmy_LCT_maskq10_CUN.txt',
            'Q23_RAmy_LCT_maskq10_Ins.txt',
            'Q23_RAmy_LCT_maskq10_STG.txt']
DATADIR = 'data/'
ATLASDIR = 'atlases/'

class Vividict(dict):
    def __missing__(self, key):
        value = self[key] = type(self)() # retain local pointer to value
        return value                     # faster to return than dict lookup

def load_surface(filename):
    """Loads surface data as v,t matrix, stripping the 1st col (an index)"""
    return(np.genfromtxt(os.path.join(DATADIR, filename))[:, 1:])

def load_surface_roi(filename):
    return(np.genfromtxt(os.path.join(ATLASDIR, filename)))

def load_volume(filename):
    """Loads volume data as an x,y,z,t 4D array"""
    return(nib.load(os.path.join(DATADIR, filename)).get_data())

def mask_data(array, mask):
    """Returns roi,t matrix from nonzero elements of the mask."""
    assert array.shape[:-1] == mask.shape
    return(array[np.where(mask)])

def extract_ctx_data(ctx_roi_name, surf_l, surf_r):
    """
    Loads cortical ROI. Returns a roi,t matrix from nonzero elements of the
    mask from the appropriate hemisphere. Mask names must include either _RCT_
    or _LCT_ to distinguish between right cortex and left cortex.
    """
    ctx_roi = load_surface_roi(ctx_roi_name)
    if '_LCT_' in ctx_roi_name:
        ctx_data = mask_data(surf_l, ctx_roi)
    elif '_RCT_' in ctx_roi_name:
        ctx_data = mask_data(surf_r, ctx_roi)
    else:
        print('atlas {} has improper name (req. _LCT_ or _RCT_ to distinguish hemispheres)'.format(ctx_roi))
        sys.exit(1)

    return(ctx_data)

def pca_reduce(X, n=1):
    """Uses PCA to return the top n components of the data as a matrix."""

    mean = np.mean(X, axis=0)

    # calculate the covariance matrix from centered data
    X = X - np.mean(X)
    R = np.cov(X, rowvar=True)

    # calculate eigenvectors & eigenvalues of the covariance matrix
    evals, evecs = linalg.eigh(R)

    # sort by explained variance
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    evals = evals[idx]

    # reduce to n components
    evecs = evecs[:, :n]
    recon = np.dot(evecs.T, X)

    # sign flip to match mean if only taking one component
    if n == 1:
        recon = recon.flatten()
        corr = np.corrcoef(np.vstack((recon, mean)))[0,1]
        if corr < 0:
            recon = recon * -1

    return(recon)

def zscore(data):
    z = (data.T - np.mean(data.T, axis=0)) / np.std(data.T, axis=0)
    return(z.T)

def demean(data):
    d = (data.T - np.mean(data.T, axis=0))
    return(d.T)

def cancorr(ctx_data, amyg_l, amyg_r):
    """
    Uses regularized kernel canonical correlation analysis to extract the top
    component from each ROI, and correlates them.

    Pyrcca: regularized kernel canonical correlation analysis in Python and its
    applications to neuroimaging. Bilenko et al, 2015, arXiv.

    A depression network of functionally connected regions discovered via multi-
    attribute canonical correlation graphs. Kang et al, 2016. Neuroimage.
    """

    n_samps = ctx_data.T.shape[0]
    n_vox = min(amyg_l.shape[0], amyg_r.shape[0], ctx_data.shape[0])

    model = cc.CCA(kernelcca=True, ktype='linear', reg = 0.1, numCC=1, verbose=False)

    #ctx_train = ctx_data.T[:n_samps/2, :]
    #ctx_test = ctx_data.T[n_samps/2:, :]
    #amyg_l_train = amyg_l.T[:n_samps/2, :]
    #amyg_l_test = amyg_l.T[n_samps/2:, :]
    #amyg_r_train = amyg_r.T[:n_samps/2, :]
    #amyg_r_test = amyg_r.T[n_samps/2:, :]
    #model.train([ctx_train, amyg_l_train])
    #testcorrs = model.validate([ctx_test, amyg_l_test])

    # model.train([zscore(ctx_data).T, zscore(amyg_l).T])
    # might have made a mistake -- trying transpose method (using same number of
    # spatial locations for each ROI)
    model.train([zscore(ctx_data[:n_vox, :]), zscore(amyg_l[:n_vox, :])])
    corr_l = model.cancorrs

    #model.train([zscore(ctx_data).T, zscore(amyg_r).T])
    model.train([zscore(ctx_data[:n_vox, :]), zscore(amyg_r[:n_vox, :])])
    corr_r = model.cancorrs

    return(corr_l[0], corr_r[0])

def pca(ctx_data, amyg_l, amyg_r):
    """
    Uses PCA to extract the top timeseries component from each ROI, and
    correlates them.

    Determining functional connectivity using fMRI data with diffusion-based
    anatomical weighting. Bowman et al, 2012, Neuroimage.
    """
    ctx_data = pca_reduce(ctx_data)
    amyg_l = pca_reduce(amyg_l)
    amyg_r = pca_reduce(amyg_r)

    corr = np.corrcoef(np.vstack((ctx_data, amyg_l, amyg_r)))

    return(corr[0,1], corr[0,2])

def correlate(ctx_data, amyg_l, amyg_r):
    """Calculates the mean timeseries from all ROIs, and correlates them."""
    ctx_data = np.mean(ctx_data, axis=0)
    amyg_l = np.mean(amyg_l, axis=0)
    amyg_r = np.mean(amyg_r, axis=0)
    corr = np.corrcoef(np.vstack((ctx_data, amyg_l, amyg_r)))

    return(corr[0,1], corr[0,2])

def analyze_subject(files):
    """
    Accepts a list of files: 1 volume, 2 surfaces (l + r), and two amygdala
    parcellations (l + r). Since the surfaces are in standard space, the same
    cortcial ROIs are used for all subject (defined in CTX_ROIS).
    """
    results = Vividict()

    # read data from disk
    volume = load_volume(filter(lambda x: 'func_volsmooth' in x, files)[0])
    amyg_l = load_volume(filter(lambda x: 'LAMY_label' in x, files)[0])
    amyg_r = load_volume(filter(lambda x: 'RAMY_label' in x, files)[0])
    surf_l = load_surface(filter(lambda x: 'func_surface_L' in x, files)[0])
    surf_r = load_surface(filter(lambda x: 'func_surface_R' in x, files)[0])

    # extract ROIs from volume
    amyg_l = mask_data(volume, amyg_l)
    amyg_r = mask_data(volume, amyg_r)

    for ctx_roi in CTX_ROIS:
        # extract cortical ROIs from the correct surface file
        ctx_data = extract_ctx_data(ctx_roi, surf_l, surf_r)

        # calculate normal connectivity (simple correlation of mean timeseries)
        corr_l, corr_r = correlate(ctx_data, amyg_l, amyg_r)
        results[ctx_roi]['corr'] = [corr_l, corr_r]

        # calculate connectivity using the first PC of each ROI
        corr_l, corr_r = pca(ctx_data, amyg_l, amyg_r)
        results[ctx_roi]['pca'] = [corr_l, corr_r]

        # calculate cannonical correlation connectivity (using 1 component)
        corr_l, corr_r = cancorr(ctx_data, amyg_l, amyg_r)
        results[ctx_roi]['cc'] = [corr_l, corr_r]

    return(results)

def append(df, row, data):
    """
    Appends the data list to the specified row in df, increments row, and
    returns the incremented row value alongside the dataframe.
    """
    df.loc[row] = data
    row += 1

    return(df, row)


def main():
    """
    Get all of the available and complete data into a dictionary, and submit
    each to the analysis pipeline. Aggregate and plot the results.
    """
    # find all available data
    candidates = os.listdir(DATADIR)
    subjects = np.unique([l.split('_')[0] for l in candidates])
    data = {}

    # filter out subjects with missing data
    for subj in subjects:
        n = filter(lambda x: subj in x, candidates)
        if len(n) == 5:
            data[subj] = filter(lambda x: subj in x, candidates)
    print('{}/{} subjects have all inputs'.format(len(data), len(subjects)))

    # init outputs
    subjects = data.keys()
    outputs = pd.DataFrame(columns=['subj', 'ctx_roi', 'method', 'amyg', 'value'])

    row = 0
    for i, subj in enumerate(subjects):

        # run the three analysis on the data
        print('analyzing subject {}/{}'.format(i+1, len(data)))
        results = analyze_subject(data[subj])

        for j, ctx_roi in enumerate(CTX_ROIS):
            outputs, row = append(outputs, row, [subj, ctx_roi, 'cc',   'L', results[ctx_roi]['cc'][0]])
            outputs, row = append(outputs, row, [subj, ctx_roi, 'cc',   'R', results[ctx_roi]['cc'][1]])
            outputs, row = append(outputs, row, [subj, ctx_roi, 'corr', 'L', results[ctx_roi]['corr'][0]])
            outputs, row = append(outputs, row, [subj, ctx_roi, 'corr', 'R', results[ctx_roi]['corr'][1]])
            outputs, row = append(outputs, row, [subj, ctx_roi, 'pca',  'L', results[ctx_roi]['pca'][0]])
            outputs, row = append(outputs, row, [subj, ctx_roi, 'pca',  'R', results[ctx_roi]['pca'][1]])

    outputs.to_csv('results.csv')

if __name__ == '__main__':
    main()


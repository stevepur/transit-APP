import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astroquery.gaia import Gaia
from lightkurve import search_targetpixelfile
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from scipy.optimize import minimize
import pickle
import corner
import os
import requests
import subprocess
import pprint
from astropy.io import fits
from scipy import stats

import keplerDiffImage as kdi
import keplerAppReports
import keplerAppDetailedReports


class tpfDataClass:
    pass

class kdiDataClass:
    pass

# color-blind friendly colors
cbc = ['#377eb8', '#ff7f00', '#4daf4a',
    '#f781bf', '#a65628', '#984ea3',
    '#999999', '#e41a1c', '#dede00']


def background_binary_density(kepmag, galacticLatitude):
    # From Morton and Johnson 2011
    coeffs = [[-2.5038e-3, 0.12912, -2.4273, 19.980, -60.931],
        [3.0668e-3, -0.15902, 3.0365, -25.320, 82.605],
        [-1.5465e-5, 7.5396e-4, -1.2836e-2, 9.6434e-2, -0.27166],
        [2.7978e-7, -1.5572e-5, 3.1957e-4, -2.8543e-3, 9.3191e-3],
        [-6.4215e-6, 3.5358e-4, -7.1463e-3, 6.2522e-2, -0.19743]]

    coeffs = np.fliplr(coeffs); # columns in the table in Morton and Johnson are backwards
    kepmag = 15.719
    galLat = 9.1563768
    A = coeffs@[1, kepmag, kepmag**2, kepmag**3, kepmag**4]

    # Eqn 10 of Morton and Johnson gives # in a circle
    # of radius 2 arcsec.  We want # per square arcsec, so divide by pi*2^2
    blendRateIn2Arcsec = (A[2] + A[0]*np.exp(-galacticLatitude/A[1]))*(galacticLatitude*A[3] + A[4])
    blendRateIn1Arcsec = blendRateIn2Arcsec/(np.pi*4)
    
    return blendRateIn1Arcsec

def make_simulated_image(gaiaCatalog, prf, starIndex = None, depth = None):
    pixSim = np.zeros(prf.shape)
    for s in range(len(gaiaCatalog["col"])):
        pix = prf.evaluate(center_col=gaiaCatalog["col"][s], center_row=gaiaCatalog["row"][s])
        pix = pix/np.sum(np.sum(pix))
        if s == starIndex:
            pixSim = pixSim + (1-depth)*gaiaCatalog["flux"][s]*pix
        else:
            pixSim = pixSim + gaiaCatalog["flux"][s]*pix
    return pixSim

def draw_pix_image(pixImage, extent, gaiaCatalog, starCol=None, starRow=None):
    fig = plt.figure(figsize = (10,10));
    plt.imshow(pixImage, cmap="jet", extent=extent)
#    plt.colorbar()
    for s in range(len(gaiaCatalog["col"])):
        plt.scatter(gaiaCatalog["col"][s], gaiaCatalog["row"][s],
                    s=900, marker="*", color="w", edgecolor="k", linewidths=1)
        plt.text(gaiaCatalog["col"][s], gaiaCatalog["row"][s] + 0.2,
                 str(np.round(gaiaCatalog["phot_g_mean_mag"][s],1)), color="w", fontsize = 24,
                 path_effects=[pe.withStroke(linewidth=2,foreground='black')], clip_on=True)
    if starCol is not None:
        for s in range(len(starCol)):
            plt.scatter(starCol[s], starRow[s], s=300, marker="+", color="b", edgecolor="k", linewidths=0.5)
    plt.scatter(gaiaCatalog["targetCol"], gaiaCatalog["targetRow"],
                s=100, marker="o", color="r", edgecolor="k", linewidths=0.5, alpha = 0.1)
    plt.tick_params(axis='both', which='major', labelsize=24)
#    plt.grid()
    plt.xlim(extent[0], extent[1]);
    plt.ylim(extent[2], extent[3]);

def make_sim_diff_image(gaiaCatalog, prf, depth, starIndex):
    pixSim = make_simulated_image(gaiaCatalog, prf)
    pixSimInTransit = make_simulated_image(gaiaCatalog, prf, depth=depth, starIndex=starIndex)
    return pixSim - pixSimInTransit

def sim_observed_depth(depth, gaiaCatalog, prf, tpf, starIndex, observedDepth):
    pixSim = make_simulated_image(gaiaCatalog, prf)
    pixSimInTransit = make_simulated_image(gaiaCatalog, prf, depth=depth, starIndex=starIndex)
    simDepth = (np.sum(pixSim[tpf.pipeline_mask].ravel()) \
       -np.sum(pixSimInTransit[tpf.pipeline_mask].ravel())) \
      /np.sum(pixSim[tpf.pipeline_mask].ravel())
#     print(simDepth)
    return(np.abs(observedDepth - simDepth))

def make_prf_image(col, row, flux, prf):
    col = [col]
    row = [row]
    flux = [flux]
    pixSim = np.zeros(prf.shape)
    for s in range(len(col[0])):
        pix = prf.evaluate(center_col=col[0][s], center_row=row[0][s])
        pix = pix/np.sum(np.sum(pix))
        pixSim = pixSim + flux[0][s]*pix
    return pixSim

def nan_flatten(data, nanMask):
    flatData = data.flatten()
    return flatData[nanMask]


    
def extract_tpf_data(tpf):

    tpfData = tpfDataClass()
    for fieldName in dir(tpf):
        if (fieldName[0] != '_') & (fieldName != "hdu"):
            if not callable(getattr(tpf,fieldName)):
                setattr(tpfData, fieldName, getattr(tpf,fieldName))
    return tpfData

def extract_kdi_data(kdiObject):

    kdiData = {}
    for fieldName in dir(kdiObject):
        if (fieldName[0] != '_') & (fieldName != "tpf"):
            if not callable(getattr(kdiObject,fieldName)):
                kdiData[fieldName] = getattr(kdiObject,fieldName)
    return kdiData


    
def get_ouputDirBase(koiName, header = 'positionalProbability_'):
    return header + koiName
    


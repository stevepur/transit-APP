import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from lightkurve import search_targetpixelfile
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import pickle
import emcee
import corner
import os
import requests
import subprocess
import pprint


def make_app_report(koi, outputDirBase):

    f = open(outputDirBase + "/koiData.pickle", 'rb')
    koiData = pickle.load(f)
    f.close()

    f = open(outputDirBase + "/positionalProbabilityData.pickle", 'rb')
    quarterDataList = pickle.load(f)
    f.close()

    f = open(outputDirBase + "/final_result_ra_dec.pickle", 'rb')
    data = pickle.load(f)
    f.close()
    starList = data[0]
    gaiaCatalog = data[1]
    fullIntegral = data[2]
    if len(data) > 3:
        usedQuarters = data[3]
    else:
        usedQuarters = None
    
    separation = np.zeros(len(gaiaCatalog))
    for s in range(len(gaiaCatalog)):
        separation[s] = 3600*np.sqrt((gaiaCatalog[s]["ra"] - gaiaCatalog[0]["ra"])**2 * np.cos(gaiaCatalog[0]["dec"]*np.pi/180)**2 + (gaiaCatalog[s]["dec"] - gaiaCatalog[0]["dec"])**2)
    
    distance = np.round(1/(1e-3*gaiaCatalog[0]['parallax']),0) * u.pc
    projectedSeparation = np.round((distance * np.sin((separation[1] * u.arcsec).to(u.rad))).to(u.AU),0)

    relativeProbability = fullIntegral/np.sum(fullIntegral)

#    pprint.pprint(koiData)
    
    meanSimDepth = np.zeros(starList.shape)
    for si, star in enumerate(starList):
        meanDepth = 0
        meanDepthCount = 0
        for qDat in quarterDataList:
            for sDat in qDat["starData"]:
                if star == sDat["gaiaID"]:
                    meanDepth += sDat["simDepth"]
                    meanDepthCount += 1
        meanSimDepth[si] = meanDepth/meanDepthCount

#    print("meanSimDepth = " + str(meanSimDepth))

    tmpFilenameStub = 'report'
    texFilename = tmpFilenameStub + '.tex'
    pdfFilename = tmpFilenameStub +  '.pdf'
    outputFilename = koiData['koiNum'] + '_app_report.pdf'

    texFile = open(texFilename,"w")

    texFile.write('\\documentclass[10pt]{article}\n');
    texFile.write('\\usepackage{graphicx}\n');
    texFile.write('\\usepackage{lscape}\n');
    texFile.write('\\usepackage{fullpage}\n');

    texFile.write('\\begin{document}\n');

    texFile.write('\\title{ ' + koiData['koiNum'] + ', KepID ' + str(koiData['kepid']) + ' Positional Probability}\n');
    texFile.write('\\maketitle\n');

    texFile.write('Model-based probability analysis for ' + koiData['koiNum'] + ', KepID '
                  + str(koiData['kepid']) + '. Kepler magnitude: ' + str(koiData['kepmag'])
                  + ', period: ' + str(np.round(koiData['period'],5))
                  + ' days, Observed (fitted) Depth: ' + str(np.round(koiData['observedDepth'],5)) + '.\n');
    texFile.write('Modeling is based on the DR25 Q1-Q17 {\\it Kepler} pipeline run.\n');
    texFile.write(koiData['koiNum'] + ' is dispositioned as a ' + koiData['disposition'] + '.\n');
    texFile.write('The distance to the KOI is ' + str(distance) + ', and the projected separation from the nearest star on the sky is ' + str(projectedSeparation) + '.\n');
    
    texFile.write('\n');
    texFile.write('This report was generated using quarters ' + str(usedQuarters) + '.\n');


    texFile.write('\\vspace{5mm}\n');
    texFile.write('\n\n');

    texFile.write('\\begin{table}[htbp]\n');
    texFile.write('\\begin{center}\n');
    texFile.write('\\begin{tabular}{|r||c|c|c|c|}\n');
    texFile.write('\\hline\n');
    texFile.write('Star Index & gmag & separation & Relative Probability & Mean Modeled Depth \\\\  \n');
    texFile.write(' &  & [{\it arcsec}] &  & \\\\ \\hline \n');
    texFile.write('\\hline\n');
    for starIdx, star in enumerate(starList):
        if relativeProbability[starIdx] < 1e-4:
            if relativeProbability[starIdx] < 1e-16:
                probStr = '0'
            else:
                probStr = '{:0.2e}'.format(relativeProbability[starIdx])
        else:
            probStr = str(np.round(relativeProbability[starIdx], 5))
        texFile.write(str(starIdx)
                  + ' & ' + str(np.round(gaiaCatalog[starIdx]['phot_g_mean_mag'],2))
                  + ' & ' + str(np.round(separation[starIdx],2))
                  + ' & ' + probStr
                  + ' & ' + str(np.round(meanSimDepth[starIdx],5))
                  + '\\\\ \\hline \n');

    texFile.write('background & -- &  -- & ' + '{:0.2e}'.format(relativeProbability[-1])
                  + ' & -- \\\\ \\hline \n');
    texFile.write('\\end{tabular}\n');
    texFile.write('\\caption{Relative probability that the transit is on the indicated star or background.\n');
    texFile.write('The background probability is based on the Morton-Johnson background density.\n');
    texFile.write('The depths are modeled in each quarter to match the observed depth in the aperture, with the mean reported in the table.\n');
    texFile.write('}\n');
    texFile.write('\\end{center}\n');
    texFile.write('\\end{table}\n');


    texFile.write('\\begin{table}[htbp]\n');
    texFile.write('\\begin{center}\n');
    texFile.write('\\begin{tabular}{|r||c|c|c|c|}\n');
    texFile.write('\\hline\n');
    texFile.write('Star Index & Gaia ID & Parallax & RA Proper Motion & Dec Proper Motion \\\\ \n');
    texFile.write(' &  & [{\it mas}] & [{\it mas/year}] & [{\it mas/year}]\\\\ \\hline \n');
    texFile.write('\\hline\n');
    for starIdx, star in enumerate(starList):
        texFile.write(str(starIdx)
                  + ' & ' + str(gaiaCatalog[starIdx]['source_id'])
                  + ' & ' + str(np.round(gaiaCatalog[starIdx]['parallax'],3))
                  + ' & ' + str(np.round(gaiaCatalog[starIdx]['pmra'],3))
                  + ' & ' + str(np.round(gaiaCatalog[starIdx]['pmdec'],3))
                  + '\\\\ \\hline \n');

    texFile.write('\\end{tabular}\n');
    texFile.write('\\caption{Star key}\n');
    texFile.write('\\end{center}\n');
    texFile.write('\\end{table}\n');

    for starIdx, star in enumerate(starList):
        figFileName = outputDirBase + "/overlap_ra_dec_star_" + str(starIdx) + ".png"
        if not os.path.exists(figFileName):
            continue
        if (relativeProbability[starIdx] > 1e-3) or True:
            texFile.write('\\begin{figure}[hb]\n');
            texFile.write('\\centering\n');
            texFile.write('\\includegraphics[width=1\\linewidth]{'
                          + outputDirBase + "/overlap_ra_dec_star_" + str(starIdx) +".png" + '}\\\\\n');

            texFile.write('\\caption{Transit location distribution modeling the transit signal on star '
                          + str(starIdx) + '.\n');
            texFile.write('The orange points are the posterior of positions of the observed transit location,\n');
            texFile.write('with the blue contours showing the KDE version.\n');
            texFile.write('The green points are the posterior of positions of the transit location modeled\n');
            texFile.write('on star ' + str(starIdx) + ', with the gray contours showing the KDE version.\n');
            texFile.write('The figure on the left is centered on the target star.\n');
            texFile.write('The figure on the right is centered on star ' + str(starIdx) + '.\n');
            texFile.write('}\n');
            texFile.write('\\end{figure}\n');


    texFile.write('\\end{document}\n');

    texFile.close()

    os.system('pdflatex ' + texFilename);
    os.system('mv ' + pdfFilename + ' ' + outputDirBase + '/' + outputFilename );
    os.system('rm ' + texFilename);
    os.system('rm ' + tmpFilenameStub + '.log');
    os.system('rm ' + tmpFilenameStub + '.aux');

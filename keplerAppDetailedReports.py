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


def make_detailed_report(koi, outputDirBase):

    f = open(outputDirBase + "/koiData.pickle", 'rb')
    koiData = pickle.load(f)
    f.close()

    f = open(outputDirBase + "/positionalProbabilityData.pickle", 'rb')
    quarterDataList = pickle.load(f)
    f.close()

    f = open(outputDirBase + "/final_result_ra_dec.pickle", 'rb')
    data = pickle.load(f)
    starList = data[0]
    gaiaCatalog = data[1]
    fullIntegral = data[2]
    f.close()

    tmpFilenameStub = 'detailedReport'
    texFilename = tmpFilenameStub + '.tex'
    pdfFilename = tmpFilenameStub +  '.pdf'
    outputFilename = koiData['koiNum'] + '_detailed_report.pdf'

    texFile = open(texFilename,"w")

    texFile.write('\\documentclass[10pt]{article}\n');
    texFile.write('\\usepackage{graphicx}\n');
    texFile.write('\\usepackage{lscape}\n');
    texFile.write('\\usepackage{fullpage}\n');

    texFile.write('\\begin{document}\n');

    texFile.write('\\title{ ' + koiData['koiNum'] + ', KepID ' + str(koiData['kepid']) + ' APP Quarterly Figures}\n');
    texFile.write('\\maketitle\n');

    texFile.write('Model-based probability analysis figures for ' + koiData['koiNum'] + ', KepID '
                  + str(koiData['kepid']) + '. Kepler magnitude: ' + str(koiData['kepmag'])
                  + ', Observed (fitted) Depth: ' + str(np.round(koiData['observedDepth'],5)) + '.\n');
    texFile.write('Modeling is based on the DR25 Q1-Q17 {\\it Kepler} pipeline run.\n');
    texFile.write(koiData['koiNum'] + ' is dispositioned as a ' + koiData['disposition'] + '.\n');


    texFile.write('\\begin{table}[htbp]\n');
    texFile.write('\\begin{center}\n');
    texFile.write('\\begin{tabular}{|r||c|c|c|c|c|}\n');
    texFile.write('\\hline\n');
    texFile.write('Star Index & Gaia ID & gmag & Parallax & RA Proper Motion & Dec Proper Motion \\\\ \n');
    texFile.write(' &  &  & [{\it mas}] & [{\it mas/year}] & [{\it mas/year}]\\\\ \\hline \n');
    texFile.write('\\hline\n');
    for starIdx, star in enumerate(starList):
        texFile.write(str(starIdx)
                  + ' & ' + str(gaiaCatalog[starIdx]['source_id'])
                  + ' & ' + str(np.round(gaiaCatalog[starIdx]['phot_g_mean_mag'],2))
                  + ' & ' + str(np.round(gaiaCatalog[starIdx]['parallax'],3))
                  + ' & ' + str(np.round(gaiaCatalog[starIdx]['pmra'],3))
                  + ' & ' + str(np.round(gaiaCatalog[starIdx]['pmdec'],3))
                  + '\\\\ \\hline \n');

    texFile.write('\\end{tabular}\n');
    texFile.write('\\caption{Star key}\n');
    texFile.write('\\end{center}\n');
    texFile.write('\\end{table}\n');

    texFile.write('\\section{Pixel Data}\n');
    for qdt in quarterDataList:
        q = qdt["quarter"]
        quarterDirBase = outputDirBase + "/q" + str(q)
        if not os.path.exists(quarterDirBase):
            continue
        figFileName = quarterDirBase + "/difference_images_q" + str(q) + ".pdf"
        if not os.path.exists(figFileName):
            continue
        texFile.write('\\begin{figure}[hb]\n');
        texFile.write('\\centering\n');
        texFile.write('\\includegraphics[width=0.45\\linewidth]{'+ quarterDirBase + "/observed_OOT_image_q" + str(q) +".pdf" + '}\n');
        texFile.write('\\includegraphics[width=0.45\\linewidth]{'+ quarterDirBase + "/observed_diff_image_q" + str(q) +".pdf" + '}\\\\\n');

        texFile.write('\\caption{quarter ' + str(q) + '.}\n');
        texFile.write('\\end{figure}\n');
    texFile.write('\\clearpage\n');

    texFile.write('\\section{Diff Image Fit Corner}\n');
    qcount = 0
    for qdt in quarterDataList:
        q = qdt["quarter"]
        quarterDirBase = outputDirBase + "/q" + str(q)
        if not os.path.exists(quarterDirBase):
            continue
        figFileName = quarterDirBase + "/diffImage_fit_corner_q" + str(q) + ".pdf"
        if not os.path.exists(figFileName):
            continue

        if qcount%3 == 2:
            quarterList.append(q)
            texFile.write('\\includegraphics[width=0.3\\linewidth]{'
                          + quarterDirBase + "/diffImage_fit_corner_q" + str(q) +".pdf" + '}\\\\\n');
            texFile.write('\\caption{quarters ' + str(quarterList) + '.}\n');
            texFile.write('\\end{figure}\n');
        elif qcount%3 == 0:
            quarterList = [q]
            texFile.write('\\begin{figure}[hb]\n');
            texFile.write('\\centering\n');
            texFile.write('\\includegraphics[width=0.3\\linewidth]{'
                          + quarterDirBase + "/diffImage_fit_corner_q" + str(q) +".pdf" + '}\n');
        else:
            quarterList.append(q)
            texFile.write('\\includegraphics[width=0.3\\linewidth]{'
                          + quarterDirBase + "/diffImage_fit_corner_q" + str(q) +".pdf" + '}\n');

        qcount += 1

    if qcount%3 > 0: # qcount was incremented, so we're testing that it was last not == 2
        texFile.write('\\caption{quarters ' + str(quarterList) + '.}\n');
        texFile.write('\\end{figure}\n');
    texFile.write('\\clearpage\n');

    texFile.write('\\section{Diff Image Fit Chains}\n');
    for qdt in quarterDataList:
        q = qdt["quarter"]
        quarterDirBase = outputDirBase + "/q" + str(q)
        if not os.path.exists(quarterDirBase):
            continue
        figFileName = quarterDirBase + "/diffImage_fit_chains_q" + str(q) + ".png"
        if not os.path.exists(figFileName):
            continue
        texFile.write('\\begin{figure}[hb]\n');
        texFile.write('\\centering\n');
        texFile.write('\\includegraphics[width=0.9\\linewidth]{'
                      + quarterDirBase + "/diffImage_fit_chains_q" + str(q) +".png" + '}\\\\\n');

        texFile.write('\\caption{quarter ' + str(q) + '.}\n');
        texFile.write('\\end{figure}\n');
    texFile.write('\\clearpage\n');
    
    texFile.write('\\section{Diff Image Fit Posteriors}\n');
    qcount = 0
    for qdt in quarterDataList:
        q = qdt["quarter"]
        quarterDirBase = outputDirBase + "/q" + str(q)
        if not os.path.exists(quarterDirBase):
            continue
        figFileName = quarterDirBase + "/diffImage_posterior_q" + str(q) + ".png"
        if not os.path.exists(figFileName):
            continue

        if qcount%3 == 2:
            quarterList.append(q)
            texFile.write('\\includegraphics[width=0.3\\linewidth]{'
                          + quarterDirBase + "/diffImage_posterior_q" + str(q) + ".png" + '}\\\\\n');
            texFile.write('\\caption{quarters ' + str(quarterList) + '.}\n');
            texFile.write('\\end{figure}\n');
        elif qcount%3 == 0:
            quarterList = [q]
            texFile.write('\\begin{figure}[hb]\n');
            texFile.write('\\centering\n');
            texFile.write('\\includegraphics[width=0.3\\linewidth]{'
                          + quarterDirBase + "/diffImage_posterior_q" + str(q) + ".png" + '}\n');
        else:
            quarterList.append(q)
            texFile.write('\\includegraphics[width=0.3\\linewidth]{'
                          + quarterDirBase + "/diffImage_posterior_q" + str(q) + ".png" + '}\n');

        qcount += 1
        
    if qcount%3 > 0: # qcount was incremented, so we're testing that it was last not == 2
        texFile.write('\\caption{quarters ' + str(quarterList) + '.}\n');
        texFile.write('\\end{figure}\n');
    texFile.write('\\clearpage\n');

    texFile.write('\\section{Light Curves}\n');
    for qdt in quarterDataList:
        q = qdt["quarter"]
        quarterDirBase = outputDirBase + "/q" + str(q)
        if not os.path.exists(quarterDirBase):
            continue
        figFileName = quarterDirBase + "/light_curve_transits_q" + str(q) + ".pdf"
        if not os.path.exists(figFileName):
            continue
        texFile.write('\\begin{figure}[hb]\n');
        texFile.write('\\centering\n');
        texFile.write('\\includegraphics[width=0.9\\linewidth]{'
                      + quarterDirBase + "/light_curve_transits_q" + str(q) +".pdf" + '}\\\\\n');

        texFile.write('\\caption{quarter ' + str(q) + '.}\n');
        texFile.write('\\end{figure}\n');


    texFile.write('\\end{document}\n');

    texFile.close()

    os.system('pdflatex ' + texFilename);
    os.system('mv ' + pdfFilename + ' ' + outputDirBase + '/' + outputFilename );
    os.system('rm ' + texFilename);
    os.system('rm ' + tmpFilenameStub + '.log');
    os.system('rm ' + tmpFilenameStub + '.aux');

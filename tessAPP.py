from astropy.coordinates import SkyCoord
import astropy.table
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
import cloudpickle
import pickle
from tqdm import tqdm
import warnings
import pymc as pm
from scanf import scanf
import copy
import pandas as pd
import sys
import argparse
from IPython.display import display

import os

sys.path.insert(0, '../../TESS-plots/code')
import tessDiffImage
import tessprfmodel as tprf


class sectorData:
    def __init__(self, ticName, toiName, sector, outputDir = "./", closeupSize = None):
        self.sector = sector
        self.outputDir = outputDir
        self.closeupSize = closeupSize
        
        self.diffImageData = None
        self.catalogData = None
        self.targetData = None
        self.inTransitIndices = None
        self.outTransitIndices = None
        self.transitIndex = None
        self.planetData = None
        
    
        try:
            fName = self.outputDir + ticName + "/imageData_TOI_" + toiName + "_sector" + str(sector) + ".pickle"
#            print("loading " + fName)
            f = open(fName, 'rb')
            imageData = cloudpickle.load(f)
            f.close

        except:
            self.diffImageData = {}
            self.catalogData = {}
            return
            
        if len(imageData[0]) == 0:
            self.diffImageData = {}
            self.catalogData = {}
            return
        
        self.diffImageData = imageData[0]
        self.catalogData = imageData[1]
        self.targetData = imageData[2]
        self.inTransitIndices = imageData[3]
        self.outTransitIndices = imageData[4]
        self.transitIndex = imageData[5]
        self.planetData = imageData[6]

        self.ticFlux = self.catalogData["ticFlux"]
        self.ticRow = self.catalogData["ticRowPix"]
        self.ticCol = self.catalogData["ticColPix"]
        self.dCol = self.catalogData["dCol"]
        self.dRow = self.catalogData["dRow"]
        
        self.fitCoeff = None
        
        # we always need the full prf to normalize the flux
        self.prfLarge = tprf.SimpleTessPRF(shape=self.diffImageData["diffImageSigma"].shape,
                    sector = sector,
                    camera = self.targetData["camera"],
                    ccd = self.targetData["ccd"],
                    column=self.catalogData["extent"][0],
                    row=self.catalogData["extent"][2])
        self.prfExtentLarge = (self.prfLarge.column+0.5, self.prfLarge.column + self.prfLarge.shape[1]+0.5,
                     self.prfLarge.row+0.5, self.prfLarge.row + self.prfLarge.shape[0]+0.5)
        if closeupSize is None:
            self.prf = self.prfLarge
            self.prfExtent = self.prfExtentLarge
        else:
            self.prf = tprf.SimpleTessPRF(shape=(self.closeupSize,self.closeupSize),
                                    sector = sector,
                                    camera = self.targetData['camera'],
                                    ccd = self.targetData['ccd'],
                                    column=np.round(self.catalogData["targetColPix"][0])-(self.closeupSize-1)/2-0.5,
                                    row=np.round(self.catalogData["targetRowPix"][0])-(self.closeupSize-1)/2-0.5)
            self.prfExtent = (self.prf.column, self.prf.column + self.prf.shape[1],
                              self.prf.row, self.prf.row + self.prf.shape[0])

        self.centerColPix = np.round(self.catalogData["targetColPix"][0]) \
                                - self.catalogData["refColPix"] + self.prf.shape[1]/2
        self.centerRowPix = np.round(self.catalogData["targetRowPix"][0]) \
                                - self.catalogData["refRowPix"] + self.prf.shape[0]/2
        try:
            # match to observed data with model obs = a*sim + b
            # make sim image with a=1, b=0
            simImage = self.make_simulated_image(useLarge=True)
            # fit to the observed data
            X = np.ones((simImage.ravel().shape[0], 2))
            X[:,0] = simImage.ravel()
            obs = self.diffImageData["meanOutTransit"].ravel()
    #         print([X.shape, self.diffImageData["meanOutTransit"].ravel().shape])
            self.fitCoeff, r, rank, s=np.linalg.lstsq(X, obs, rcond=None)
            print("sector " + str(self.sector) + " normalization = " + str(self.fitCoeff))
        except:
            print("sector " + str(self.sector) + " normalization failed")
            self.fitCoeff = [-1, 0]

    def render_star(self, starIndex, depth=0, normalizeToObserved = False, useLarge=False):
        if useLarge:
            pix = self.prfLarge.evaluate(center_col=self.ticCol[starIndex] - self.dCol + 0.5,
                                    center_row=self.ticRow[starIndex] - self.dRow + 0.5)
        else:
            pix = self.prf.evaluate(center_col=self.ticCol[starIndex] - self.dCol + 0.5,
                                    center_row=self.ticRow[starIndex] - self.dRow + 0.5)
        pix = pix/np.sum(np.sum(pix))
        pix *= (1-depth)*self.ticFlux[starIndex]
        if normalizeToObserved:
            pix = self.fitCoeff[0]*pix + self.fitCoeff[1]
        return pix

    def make_simulated_image(self, starIndex = None, depth = None, normalizeToObserved = False, useLarge=False):
        if useLarge:
            pixSim = np.zeros(self.prfLarge.shape)
        else:
            pixSim = np.zeros(self.prf.shape)
        for s in range(len(self.ticCol)):
            if s == starIndex:
                thisDepth = depth
            else:
                thisDepth = 0.0
            
            pixSim = pixSim + self.render_star(s,
                                               depth = thisDepth,
                                               normalizeToObserved = False,
                                               useLarge = useLarge)
        if normalizeToObserved:
            pixSim = self.fitCoeff[0]*pixSim + self.fitCoeff[1]
        return pixSim
    
    def test_on_prf_pix(self, starIndex):
        return (self.ticCol[starIndex] < self.prfExtent[0]) | (self.ticCol[starIndex] > self.prfExtent[1]) \
                | (self.ticRow[starIndex] < self.prfExtent[2]) | (self.ticRow[starIndex] > self.prfExtent[3])

#####################################################################################################
#####################################################################################################
#####################################################################################################

class tessAPP:
    def __init__(self,
                 ticData,
                 cleanFiles = True,
                 maxPixelDistance = 3,
                 thinFactor = None,
                 usePyMC3 = False, # no longer used, thrown away
                 outputDir = "./",
                 spiceFileLocation = "../TESS-plots/",
                 qlpFlagsLocation = "../tessRobovetter/lightcurves/QLP_qflags/"):
        
        self.ticData = ticData
        self.ticName = "tic" + str(self.ticData["id"])
        self.toiName = str(self.ticData["planetData"][0]["TOI"])
        self.depthList = self.ticData["planetData"][0]['depth']
        
        self.outputDir = outputDir
        self.cleanFiles = cleanFiles
        self.maxPixelDistance = maxPixelDistance

        self.spiceFileLocation = spiceFileLocation
        self.qlpFlagsLocation = qlpFlagsLocation
        
        self.thinFactor = thinFactor
        
        self.sectorList = None
        self.transitingSectors = None
        self.sectorData = None
        
        self.tdi = None
        
        self.brightEnoughStars = None
        self.closeEnoughStars = None
        self.selectedStars = None
        
        self.firstTransitingSector = 0
        


      
    def make_difference_images(self):
        self.tdi = tessDiffImage.tessDiffImage(self.ticData,
                                                outputDir = self.outputDir,
                                               spiceFileLocation = self.spiceFileLocation,
                                               qlpFlagsLocation = self.qlpFlagsLocation,
                                               cleanFiles = self.cleanFiles)
        self.tdi.make_ffi_difference_image(thisPlanet = 0)

        self.sectorList = self.tdi.sectorList
        
    def load_sector(self, sector, closeupSize = None):
        self.sectorData = sectorData(self.ticName, self.toiName, sector, outputDir = self.outputDir, closeupSize=closeupSize)
            
    def find_first_sector(self):
        sectorIndex = 0
        self.load_sector(self.sectorList[sectorIndex])
        while len(self.sectorData.diffImageData) == 0:
            sectorIndex += 1
            self.load_sector(self.sectorList[sectorIndex])
        self.firstTransitingSector = self.sectorList[sectorIndex]
        print("firstTransitingSector = " + str(self.firstTransitingSector))
    
    def find_transiting_sectors(self):
        self.transitingSectors = []
        for sector in self.sectorList:
            self.load_sector(sector)

            # remove sectors with no difference image data
            if len(self.sectorData.diffImageData) == 0:
                print("no difference images")
                continue
            # remove sectors with negative normalization
            elif self.sectorData.fitCoeff[0] < 0:
                print("normalization is negative")
                continue
            # remove sectors where the target is too close to the edge of the CCD
            elif ((self.sectorData.catalogData["refColPix"] < 11) \
                    | (self.sectorData.catalogData["refColPix"] > 2048 - 11) \
                    | (self.sectorData.catalogData["refRowPix"] < 11) \
                    | (self.sectorData.catalogData["refRowPix"] > 2048 - 11)):
                print("target too close to the edge of the CCD")
                continue
            else:
                self.transitingSectors.append(sector)
        
        print("transiting sectors = " + str(self.transitingSectors))
            

    def find_bright_enough_stars(self):
        self.load_sector(self.transitingSectors[0])
        apCenter = [self.sectorData.centerColPix[0].astype(int),self.sectorData.centerRowPix[0].astype(int)]
        testAperture = np.zeros(self.sectorData.prf.shape)
        testAperture[apCenter[1]-1:apCenter[1]+2,apCenter[0]-1:apCenter[0]+2] = 1

        allStarSimImage = self.sectorData.make_simulated_image(normalizeToObserved = True)
        baseFlux = np.sum((allStarSimImage*testAperture).ravel())
#         print("baseFlux = " + str(baseFlux))

        minDepth = np.min(self.depthList)/2
#         print("minDepth = " + str(minDepth))
#         print("1 - minDepth = " + str(1 - minDepth))

        self.brightEnoughStars = []
        for s in tqdm(range(len(self.sectorData.catalogData["ticID"]))):
            if self.sectorData.test_on_prf_pix(s) | (self.sectorData.catalogData["separation"][s] > 27*self.maxPixelDistance):
                continue

            pix = self.sectorData.render_star(s, depth = 0, normalizeToObserved = True)

            simImage = allStarSimImage - pix
            starFlux = np.sum((simImage*testAperture).ravel())
#             if s < 20:
#                 print([s, starFlux/baseFlux, starFlux, 1-starFlux/baseFlux])
            if starFlux/baseFlux < 1.0 - minDepth:
                self.brightEnoughStars.append(s)

        print("there are " + str(len(self.brightEnoughStars)) + " bright enough stars")
        print(self.brightEnoughStars)

        
    def find_close_enough_stars(self):
        self.closeEnoughStars = []
        nCloseSectors = 0
        self.lastGoodSector = 0
        overlapFlux = np.zeros(len(self.brightEnoughStars))
        for sector in self.transitingSectors:
#             print("trying sector " + str(sector))
            self.load_sector(sector)

#             print([len(self.sectorData.diffImageData), len(self.sectorData.catalogData)])
            if len(self.sectorData.diffImageData) == 0:
                continue
            
            
#             print("loaded sector " + str(sector))

            overlapFluxIndex = np.array([])
            diffImage = self.sectorData.diffImageData["diffImage"].copy()
            closeStarsSector = []
            # normalize to max = 1 to compare hottest pixels
            diffImage /= np.max(np.max(diffImage))
            diffImage[diffImage<0] = 0.0
            sumDiffImage = np.sum(diffImage[diffImage > np.median(diffImage.flatten())])
            nCloseSectors += 1

            for si, s in enumerate(self.brightEnoughStars):
                if self.sectorData.catalogData["separation"][si] > 27*self.maxPixelDistance:
                    continue

                pix = self.sectorData.render_star(s, normalizeToObserved = False)
                pix /= np.max(np.max(pix))

                prod = np.abs(diffImage)*pix

                f = np.sum(prod.ravel())
                if not np.isnan(f):
                    overlapFlux[si] += f
                    overlapFluxIndex = np.append(overlapFluxIndex, s)

#         print("building closeEnoughStars")
        for si, s in enumerate(self.brightEnoughStars):
            # require star to be within 6 pixels to avoid edge effects
            if ((overlapFlux[si]/nCloseSectors>1)) \
                        | ((overlapFlux[si]/nCloseSectors>0.2) \
                        & (self.sectorData.catalogData["separation"][s] < 27*2)): # avoid duplicates of the target
                self.closeEnoughStars.append(s)

        print("there are " + str(len(self.closeEnoughStars)) + " close enough stars")
        print(np.array(self.closeEnoughStars))
        
    def find_possible_background_stars(self):
        if self.transitingSectors is None:
            self.find_transting_sectors()
        self.find_bright_enough_stars()
        self.find_close_enough_stars()
        
        self.selectedStars = self.closeEnoughStars.copy()
        if 0 not in self.selectedStars:
            self.selectedStars.insert(0,0)
            
        if (1 in self.selectedStars) & (self.sectorData.catalogData["separation"][1] == 0):
            print("removing duplicate target star")
            self.selectedStars.remove(1)

        if len(self.selectedStars) == 1:
            if self.sectorData.catalogData["separation"][1] > 0:
                self.selectedStars.append(1)
                self.selectedStars.append(2)
            else:
                self.selectedStars.append(2)
                self.selectedStars.append(3)

        maxRow = max(abs(self.sectorData.ticRow[self.selectedStars] - self.sectorData.catalogData["targetRowPix"]))
        maxCol = max(abs(self.sectorData.ticCol[self.selectedStars] - self.sectorData.catalogData["targetColPix"]))
        print([maxRow, maxCol])
        closeupSize = max([maxRow, maxCol])
        self.closeupSize = np.ceil(2*(closeupSize + 1)).astype(int)
        if self.closeupSize%2 == 0:
            self.closeupSize += 1
        if self.closeupSize > 19:
            self.closeupSize = 19
        if self.closeupSize < 5:
            self.closeupSize = 5
            
        print("there are " + str(len(self.selectedStars)) + " selected stars")
        print(np.array(self.selectedStars))
            
    def collect_sectors(self):
        self.allObs = np.array([])
        self.allObsImage = []
        self.allSigma = np.array([])
        self.allDiffImageFlat = {}
        self.allDiffImage = {}
        self.allObsNorm = []
        for i, s in enumerate(self.selectedStars):
            self.allDiffImageFlat[i] = np.array([])
            self.allDiffImage[i] = []


        for sector in self.transitingSectors:
            self.load_sector(sector, closeupSize = self.closeupSize)

            if len(self.sectorData.diffImageData) > 0:
                if np.any(np.isnan(self.sectorData.diffImageData["meanOutTransit"])):
                    continue;

                simImage = self.sectorData.make_simulated_image(normalizeToObserved = True)
                for i, s in enumerate(self.selectedStars):
                    transitImage = self.sectorData.make_simulated_image(starIndex = s, depth = 1e-3, normalizeToObserved = True)
                    diffImage = simImage-transitImage
                    self.allDiffImage[i].append(diffImage)
                    self.allDiffImageFlat[i] = np.append(self.allDiffImageFlat[i], diffImage.flatten())

                cCol = np.floor(self.sectorData.prfExtent[0]
                                -self.sectorData.catalogData["extent"][0]).astype(int)
                cRow = np.floor(self.sectorData.prfExtent[2]
                                -self.sectorData.catalogData["extent"][2]).astype(int)
#                 rowRange = cRow + np.array(range(self.closeupSize))
#                 colRange = cCol + np.array(range(self.closeupSize))
#                 obs = self.sectorData.diffImageData["diffImage"][rowRange, colRange] \
#                                 - np.nanmedian(self.sectorData.diffImageData["diffImage"][rowRange, colRange])
                rowRange = slice(cRow,cRow + self.closeupSize)
                colRange = slice(cCol,cCol + self.closeupSize)
                obs = self.sectorData.diffImageData["diffImage"][rowRange,colRange] \
                                - np.nanmedian(self.sectorData.diffImageData["diffImage"][rowRange,colRange])
                obs[obs<0] = 0
                self.allObsImage.append(obs)

                obs = obs.flatten()
                self.allObs = np.append(self.allObs, obs)
                sigma = self.sectorData.diffImageData["meanOutTransitSigma"][rowRange,colRange]
                sigma = sigma.flatten()
                self.allSigma = np.append(self.allSigma, sigma)
                self.allObsNorm.append(self.sectorData.fitCoeff)

        goodData = ~np.isnan(self.allObs) & (self.allSigma>0)
        self.allObs = self.allObs[goodData]
        self.allSigma = self.allSigma[goodData]
        for i, s in enumerate(self.selectedStars):
            self.allDiffImageFlat[i] = self.allDiffImageFlat[i][goodData]

    def compute_app(self):


        self.modelList = []
        self.traceDict = {}
        for i, s in enumerate(self.selectedStars):
            with pm.Model() as model:
                model.name = str(s) + ":" + str(self.sectorData.catalogData["ticID"][s])
                self.modelList.append(model)
                scale = pm.Normal('scale', mu=1, sigma=100)
                sigmaScale = pm.Normal('sigmaScale', mu=1, sigma=100)

                fobs = pm.Normal('fobs',
                                 mu=scale*self.allDiffImageFlat[i],
                                 sigma=sigmaScale*self.allSigma,
                                 observed=self.allObs)

                traceDiffFit = pm.sample(10000, tune=5000, chains=4, cores=1, step = pm.Metropolis(), idata_kwargs={"log_likelihood": True})
                if self.thinFactor is None:
                    self.traceDict[model] = traceDiffFit
                else:
                    self.traceDict[model] = traceDiffFit.sel(draw=slice(None,None,self.thinFactor))

        with pm.Model() as model:
            model.name = "noTransit"
            self.modelList.append(model)
            scale = pm.Normal('scale', mu=1, sigma=100)
            sigmaScale = pm.Normal('sigmaScale', mu=1, sigma=100)

            fobs = pm.Normal('fobs',
                             mu = scale*np.zeros(self.allDiffImageFlat[0].shape),
                             sigma = sigmaScale*self.allSigma,
                             observed=self.allObs)

            traceDiffFit = pm.sample(10000, tune=5000, chains=4, cores=1, step = pm.Metropolis(), idata_kwargs={"log_likelihood": True})
            if self.thinFactor is None:
                self.traceDict[model] = traceDiffFit
            else:
                self.traceDict[model] = traceDiffFit.sel(draw=slice(None,None,self.thinFactor))

        self.dfwaicBma = pm.compare(self.traceDict, ic='WAIC', method="BB-pseudo-BMA")
        self.dfwaicBma.index = [k.name for k,v in self.dfwaicBma.iterrows()]
        waicDispRange = slice(0,min([len(self.dfwaicBma),15]))
        display(self.dfwaicBma[waicDispRange])
        pm.plot_compare(self.dfwaicBma[waicDispRange], insample_dev=False);
        plt.xlabel("Log WAIC")
        plt.savefig(self.outputDir + self.ticName + "/" + "TOI_" + self.toiName + "_APP_waic.pdf", bbox_inches='tight')

        rank = []
        self.weight = []
        waic = []

        for i, s in enumerate(self.selectedStars):
            name = str(s) + ":" + str(self.sectorData.catalogData["ticID"][s])
            rank.append(self.dfwaicBma.loc[name]["rank"])
            self.weight.append(self.dfwaicBma.loc[name]["weight"])
            waic.append(self.dfwaicBma.loc[name]["p_waic"])
#         print(rank)
#         print(self.weight)
#         print(waic)
        waic = np.array(waic)
        mwaic= waic - np.mean(waic)

        print("selected stars:")
        print(np.array(self.selectedStars))
        sortIdx = np.flip(np.argsort(self.weight))
        self.sortedStars = np.array(self.selectedStars)[sortIdx]
        self.sortedWeights = np.array(self.weight)[sortIdx]
        print("sorted selected stars:")
        print(np.array(self.sortedStars))
        print("sorted weights:")
        print(np.array(self.sortedWeights))
        sortedNonTargetStars = self.sortedStars[self.sortedStars!=0]
        print("sorted Non-Target stars:")
        print(np.array(sortedNonTargetStars))
        if len(sortedNonTargetStars) > 1:
            self.starsToDisplay = [0, sortedNonTargetStars[0], sortedNonTargetStars[1]]
        else:
            self.starsToDisplay = [0, sortedNonTargetStars[0]]
        print("Stars to Display:")
        print(np.array(self.starsToDisplay))

        outputData = {"sortedStars": self.sortedStars,
                        "weights": self.weight,
                        "transitingSectors": self.transitingSectors,
                        "selectedStars": self.selectedStars}

        evalFilename = self.outputDir + self.ticName + "/" + "TOI_" + self.toiName + "_compareData" + ".pickle"
        f = open(evalFilename, 'wb')
        cloudpickle.dump([self.traceDict, self.modelList, self.dfwaicBma, outputData], f, pickle.HIGHEST_PROTOCOL)
        f.close()

        mostLikelyStar = self.sortedStars[0]
        print("TOI " + str(self.ticData["planetData"][0]["TOI"])  + ", TIC " + str(self.ticData['id'])\
              + ": most likely source star index " + str(mostLikelyStar)\
              + ": " + str(self.sectorData.catalogData["ticID"][mostLikelyStar]) \
              + " with probability " + str(np.round(self.sortedWeights[0], 2)) \
              + " at separation "\
              + str(np.round(self.sectorData.catalogData["separation"][self.sortedStars[0]],2)) + " arcsec"  \
              + " with delta magnitude "\
              + str(np.round(self.sectorData.catalogData["ticMag"][self.sortedStars[0]] - self.sectorData.catalogData["ticMag"][0],3)))

        print("catalog depth: " + str(self.ticData["planetData"][0]["depth"]))
        for s in range(min(15, len(self.modelList))):
            mName = self.modelList[s].name + "::scale"
            post = self.traceDict[self.modelList[s]].posterior[mName][0].values
            print(self.modelList[s].name + " scale: "
                  + str(np.round(np.mean(post),4)) + " +- "
                  + str(np.round(np.std(post),4)))
            print(self.modelList[s].name + " estimated depth: "
                  + str(np.round(1e-3*np.mean(post),4)) + " +- "
                  + str(np.round(1e-3*np.std(post),4)))

    def draw(self):
        sectorCount = 0
        for sectorIdx, sector in enumerate(self.transitingSectors):
            self.load_sector(sector, closeupSize = self.closeupSize)

            if len(self.sectorData.diffImageData) > 0:
                if np.any(np.isnan(self.sectorData.diffImageData["meanOutTransit"])):
                    print("sector " + str(sector) + " has nans")
                    continue;

                plt.figure(figsize=(15,15))
                ax = plt.subplot(2,2,1)
                self.tdi.draw_pix_catalog(self.sectorData.diffImageData["meanOutTransit"],
                                     self.sectorData.catalogData,
                                     extent=self.sectorData.catalogData["extent"],
                                     close=False,
                                     annotate=False,
                                     hiliteStar = self.selectedStars,
                                     dMagThreshold = 3, magColorBar=False, pixColorBar=False)
                plt.title("Sector " + str(sector) + " Mean Out Transit")

                ax = plt.subplot(2,2,2)
                self.tdi.draw_pix_catalog(self.sectorData.diffImageData["diffImage"],
                                     self.sectorData.catalogData,
                                     extent=self.sectorData.catalogData["extent"],
                                     close=False, annotate=False,
                                     dMagThreshold = 3, magColorBar=False, pixColorBar=False)
                plt.title("Sector " + str(sector) + " Difference Image")

                ax = plt.subplot(2,2,3)
                self.tdi.draw_pix_catalog(self.sectorData.make_simulated_image(useLarge=True,
                                                                          normalizeToObserved = True),
                                     self.sectorData.catalogData,
                                     extent=self.sectorData.catalogData["extent"],
                                     close=False,
                                     annotate=False,
                                     dMagThreshold = 3, magColorBar=False, pixColorBar=False)
                plt.title("Sector " + str(sector) + " Simulated Image")

                ax = plt.subplot(2,2,4)
                self.tdi.draw_pix_catalog(self.sectorData.diffImageData["diffSNRImage"],
                                     self.sectorData.catalogData,
                                     extent=self.sectorData.catalogData["extent"],
                                     close=False, annotate=False,
                                     dMagThreshold = 3, magColorBar=False, pixColorBar=False)
                plt.title("Sector " + str(sector) + " Difference SNR Image")

                plt.savefig(self.outputDir + self.ticName + "/" + "TOI_" + self.toiName + "_sector" + str(sector) + "_APP_overview.pdf", bbox_inches='tight')

                plt.figure(figsize=(15,15))
                ax = plt.subplot(2,2,1)
                self.tdi.draw_pix_catalog(self.allObsImage[sectorCount],
                                          self.sectorData.catalogData,
                                          extent=self.sectorData.prfExtent,
                                          close=True, annotate=True,
                                          dMagThreshold = 1, magColorBar=False,
                                          pixColorBar=False, ss=1600, fs=14, printMags=True,
                                          starsToAnnotate = self.starsToDisplay)
                plt.title("Sector " + str(sector) + " Difference Image")

                for i, s in enumerate(self.starsToDisplay):
                    ax = plt.subplot(2,2,2+i)
                    sIndex = self.selectedStars.index(s)
                    self.tdi.draw_pix_catalog(self.allDiffImage[sIndex][sectorCount],
                                              self.sectorData.catalogData,
                                              extent=self.sectorData.prfExtent,
                                              close=True, annotate=True, hiliteStar = s,
                                              dMagThreshold = 1, magColorBar=False,
                                              pixColorBar=False, ss=1600, fs=14, printMags=True,
                                              starsToAnnotate = self.starsToDisplay)
                    p = np.round(self.weight[sIndex], 2)
                    plt.title("Sector " + str(sector) + " Simulated Difference Image on star " + str(s) + ", p=" + str(p))

                plt.savefig(self.outputDir + self.ticName + "/" + "TOI_" + self.toiName + "_sector" + str(sector) + "_APP_detail.pdf", bbox_inches='tight')
                sectorCount += 1

    def draw_selected_stars(self, sector, brightEnoughStars = True):
        self.load_sector(sector)
        if brightEnoughStars:
            plt.figure(figsize=(10,10))
            self.tdi.draw_pix_catalog(self.sectorData.diffImageData["diffImage"],
                                 self.sectorData.catalogData,
                                 extent=self.sectorData.catalogData["extent"],
                                 close=False, annotate=False, dMagThreshold = 6,
                                 magColorBar=False, pixColorBar=False, hiliteStar = self.brightEnoughStars)

        plt.figure(figsize=(10,10))
        self.tdi.draw_pix_catalog(self.sectorData.diffImageData["diffImage"],
                             self.sectorData.catalogData,
                             extent=self.sectorData.catalogData["extent"],
                             close=False, annotate=False, dMagThreshold = 6,
                             magColorBar=False, pixColorBar=False, hiliteStar = self.closeEnoughStars)
                             
    def cleanup(self):
        if self.cleanFiles:
            os.system('rm ' + self.ticName + '/imageData_TOI_*.pickle')


def set_toi_data(toi, toiTable):
    bjdOffset = 2457000.0

    planetData = {}
    planetData['TOI'] = toi
    planetData['period'] = toiTable['Period (days)'].values[0]
    planetData['epoch'] = toiTable['Epoch (BJD)'].values[0] - bjdOffset
    planetData['durationHours'] = toiTable['Duration (hours)'].values[0]
    planetData['depth'] = toiTable['Depth (ppm)'].values[0]*1e-6
    planetData['rprs'] = np.sqrt(planetData['depth'])

    return planetData

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("toiListFilename", type=str,
                        help="file name of a single-column list of TOIs to be processed")
    parser.add_argument("--outputDir", type=str, default="./",
                        help="output directory")
    args = parser.parse_args()
    
    toisToRun = np.loadtxt(args.toiListFilename)
    outputDir = args.outputDir
    
    toiTable = pd.read_csv("toi_table_tfop.csv")

    for toiNum in toisToRun:
        print("TOI " + str(toiNum))
        thisToi = toiTable[toiTable.TOI == toiNum]
        
        if thisToi['TESS Mag'].values[0] < 7.5:
            print("this star is saturated")
            continue

        c = SkyCoord(thisToi['RA'].values[0], thisToi['Dec'].values[0], unit=(u.hourangle, u.deg))

        planet0 = {}
        ticCatalogData = {}
        ticCatalogData['id'] = thisToi['TIC ID'].values[0]
        ticCatalogData['raDegrees'] = c.ra.deg
        ticCatalogData['decDegrees'] = c.dec.deg
        ticCatalogData['sector'] = None
        ticCatalogData['cam'] = None
        ticCatalogData['ccd'] = None
        ticCatalogData['starRadius'] = thisToi['Stellar Radius (R_Sun)'].values[0]
        ticCatalogData['TMag'] = thisToi['TESS Mag'].values[0]

        planet0 = set_toi_data(toiNum, thisToi)
        if os.path.exists(outputDir + "tic" + str(ticCatalogData['id']) + "/TOI_" + str(toiNum) + "_compareData.pickle"):
            continue


        ticCatalogData["planetData"] = [planet0]

        # find other TOIs on this star
        otherToi = toiTable[(toiTable.TOI.astype(int) == int(toiNum)) & (toiTable.TOI != toiNum)]
        otherToi.TOI.values
        for t in otherToi.TOI.values:
            thisOtherToi = toiTable[toiTable.TOI == t]
            ticCatalogData["planetData"].append(set_toi_data(t, thisOtherToi))

        ticName = "tic" + str(ticCatalogData["id"])
        toiName = ticName + "/TOI_" + str(toiNum)
        display(ticCatalogData)

        ticData = copy.deepcopy(ticCatalogData)
        
        try:
            tapp = tessAPP(ticData,
                        outputDir = outputDir,
                        spiceFileLocation = "../TESS-plots/",
                        qlpFlagsLocation = "../QLP_qflags/",
                        cleanFiles = False)
            tapp.make_difference_images()
            tapp.find_transiting_sectors()
            if len(tapp.transitingSectors) == 0:
                print("no transiting sectors")
                continue
            tapp.find_possible_background_stars()
            if len(tapp.selectedStars) > 10:
                print("too many stars to check")
                continue
            tapp.collect_sectors()
            tapp.compute_app()
            tapp.draw()
        except KeyboardInterrupt:
            print('Interrupted')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)
        except:
            print("something went wrong, moving on")

        

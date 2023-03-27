from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy as sp
import warnings
import copy

from PIL import Image

import os
import pandas as pd
import lightkurve as lk
from IPython.display import display
import argparse

import sys
sys.path.insert(0, '../tess-point-master')
import tess_stars2px as trdp

import eleanor as el
import exoplanet as xo
import arviz as az
from wotan import flatten

class fitEphemeris():
    def __init__(self,
                 ticCatalogData,
                 reportLoc = "pdfReports/",
                 cleanFiles = True,
                 usePyMC3 = False,
                 spiceFileLocation = "../TESS-plots/",
                 qlpFlagsLocation = "../QLP_qflags/"):
        
        self.ticCatalogData = ticCatalogData
        self.ticName = "tic" + str(self.ticCatalogData["id"])
        self.toiName = str(self.ticCatalogData["planetData"][0]["TOI"])
        self.ticData = copy.deepcopy(self.ticCatalogData)
        
        self.reportLoc = reportLoc
        self.cleanFiles = cleanFiles
        self.usePyMC3 = usePyMC3
        
        self.spiceFileLocation = spiceFileLocation
        self.qlpFlagsLocation = qlpFlagsLocation
        
        self.sectorList = None
        self.transitingSectors = None
        
        self.windowSizeDays = 6*self.ticCatalogData["planetData"][0]['period']/24

        self.nSamples = 5000
        self.nTune = 2000
        self.nCores = 8
        self.nChains = 8
        
        print("processing TOI " + self.toiName)
        self.get_observed_sectors()
        if len(self.observedSectors) < 8:
            self.get_eleanor_data()
            self.make_sector_list()
            self.get_light_curves()
            self.get_bls_period()
            self.extract_masked_data(period = self.boxPeriod)
            self.plot_transit_mask(filename="catalogEphem.png",
                                    periodSource="catalog")
            self.plot_transit_mask(period = self.boxPeriod,
                                    filename="oldEphem.png",
                                    periodSource="box search")
            self.fit_period_epoch()
            self.plot_transit_mask(period=self.bestPeriod,
                                   epoch=self.bestEpoch,
                                   plotCatalogEphem=True,
                                   filename="newEphem.png",
                                   periodSource="final fit")
            self.plot_folded_lightcurve()
            self.make_report()
        else:
            print("skipping because there are too many sectors")
        
        if self.cleanFiles:
            os.system('rm -rf ./lightkurve-cache/*')
            os.system('rm -rf mastDownload/*')
        
    def get_eleanor_data(self):
        self.eleanorData = el.multi_sectors(tic=self.ticCatalogData['id'], sectors='all')

    def get_observed_sectors(self):
        outID, outEclipLong, outEclipLat, outSec, outCam, outCcd, \
                refColPix, refRowPix, scinfo = trdp.tess_stars2px_function_entry(
                            self.ticCatalogData["id"], self.ticCatalogData['raDegrees'], self.ticCatalogData['decDegrees'],
                            aberrate=True, trySector=None)

        self.observedSectors = list(outSec)
        
    def make_sector_list(self):
        outSec = list(self.observedSectors)
        self.eleanorSectorList = []
        for i in range(len(self.eleanorData)):
            self.eleanorSectorList.append(self.eleanorData[i].sector)

        outSec = list(set(outSec + self.eleanorSectorList))
        self.sectorList = np.sort(np.array(outSec))
        print(str(len(self.sectorList)) + " sectors")
        print("sector list:" + str(self.sectorList))
        
#        for i in range(len(self.sectorList)):
#            if self.eleanorData[i].sector != self.sectorList[i]:
#                print("self.sectorList = " + str(self.sectorList))
#                print("eleanor sector = " + str(self.eleanorSectorList))
#                raise ValueError('Eleanor and sectorList do not agree')
                
    def get_light_curves(self):
            
        ticStr = "TIC " + str(self.ticCatalogData["id"])
        print(ticStr)

        self.sectorData = {}
        self.sectorData["sectorNumber"] = []
        self.sectorData["startTime"] = []
        self.sectorData["endTime"] = []

        self.allTime = np.array([])
        self.allFlux = np.array([])
        self.allFluxErr = np.array([])
        self.allQuality = np.array([])
        for i in range(len(self.sectorList)):
        # for i in range(1):
            print(self.sectorList[i])
            # test to see if spoc data is available
#            print("trying spoc light curve")
            pixelfile = lk.search_targetpixelfile(ticStr, sector=self.sectorList[i], author="SPOC").download()
#            print("got spoc light curve")

            if pixelfile is not None:
                rawLightCurve = pixelfile.to_lightcurve().remove_outliers()
                q0 = (rawLightCurve.quality == 0) & ~np.isnan(rawLightCurve.flux.value)
                ft = rawLightCurve.time.value[q0]
                dtDays = np.min(np.diff(ft))
                print("spoc delta t = " + str(dtDays*24*60) + " minutes")
                ffMedian = np.median(rawLightCurve.flux.value[q0])
                ff = rawLightCurve.flux.value[q0]/ffMedian
                ffErr = rawLightCurve.flux_err.value[q0]/ffMedian

            elif i < len(self.eleanorSectorList):
#                print("trying eleanor light curve")
#                print("index of sector in eleanorData: " + str(self.eleanorSectorList.index(self.sectorList[i])))
                print(self.eleanorData[self.eleanorSectorList.index(self.sectorList[i])])
                data0 = el.TargetData(self.eleanorData[self.eleanorSectorList.index(self.sectorList[i])],
                                      height=15, width=15, bkg_size=31, do_psf=False, do_pca=False)
#                print("got eleanor light curve")
                q0 = (data0.quality == 0) & ~np.isnan(data0.corr_flux) & ~np.isnan(data0.flux_bkg)
            #     q0 = (data0.quality == 0)

                ft = data0.time[q0]
                dtDays = np.min(np.diff(ft))
                print("eleanor delta t = " + str(dtDays*24*60) + " minutes")
                ffMedian = np.median(data0.corr_flux[q0])
                ff = data0.corr_flux[q0]/ffMedian
                ffErr = data0.flux_err[q0]/ffMedian
            else:
                print("sector " + str(self.sectorList[i]) + " has no light curve data")
                continue
            q0 = q0[q0]

            fluxMad = np.nanmedian(np.abs(ff - np.nanmedian(ff)))
            fluxClip = 10*fluxMad + 1
#            print("fluxClip = " + str(fluxClip))
            goodIdx = ff < fluxClip
            ft = ft[goodIdx]
            ff = ff[goodIdx]
            ffErr = ffErr[goodIdx]
            q0 = q0[goodIdx]

            self.sectorData["sectorNumber"].append(self.sectorList[i])
            self.sectorData["startTime"].append(np.min(ft))
            self.sectorData["endTime"].append(np.max(ft))
#            print("calling flatten")
            windowSize = 8*self.ticCatalogData["planetData"][0]['durationHours']/24
#            print([windowSize, windowSize/self.ticCatalogData["planetData"][0]['period']])
            ffFlat, ffTrend = flatten(ft, ff, method="biweight",window_length=windowSize, edge_cutoff=0, break_tolerance=0.2, return_trend=True, cval=5.0)
#            print("returned from flatten")
            # bin to 10 minutes when the cadence is < 10 minutes
            if dtDays < 9/24/60:
        #     if False:
                # find something close to 10 minutes that is an integer multiple of actual cadence
                print("binning to 10 minutes")
                dt10MinDays = np.round(10/(dtDays*24*60), 0)*dtDays
                n10MinCadences = np.ceil((ft[-1] - ft[0])/dt10MinDays)
                bin10Min = ft[0] + dt10MinDays*np.array(list(range(n10MinCadences.astype(int)+1)))

                binnedFlux, binnedTimeEdges, binIndex = sp.stats.binned_statistic(ft, ffFlat, bins=bin10Min)
                binnedTime = (binnedTimeEdges[:-1]+binnedTimeEdges[1:])/2

                binnedQuality = np.zeros(binnedTime.shape, dtype=bool)
                binnedFluxErr = np.zeros(binnedTime.shape)
                # bin index is from 1 because it refers to the right hand boundary
                for bi in range(len(binIndex)):
                    binnedQuality[binIndex[bi]-1] = binnedQuality[binIndex[bi]-1] | ~q0[binIndex[bi]]
                    binnedFluxErr[binIndex[bi]-1] += ffErr[binIndex[bi]]**2
                binnedFluxErr = np.sqrt(binnedFluxErr)

                nanFlux = np.isnan(binnedFlux)
                binnedTime = binnedTime[~nanFlux]
                binnedFlux = binnedFlux[~nanFlux]
                binnedFluxErr = binnedFluxErr[~nanFlux]
                binnedQuality = binnedQuality[~nanFlux]

                binnedTime = binnedTime[~binnedQuality]
                binnedFlux = binnedFlux[~binnedQuality]
                binnedFluxErr = binnedFluxErr[~binnedQuality]
            else:
                binnedTime = ft
                binnedFlux = ffFlat
                binnedFluxErr = ffErr
                binnedQuality = ~q0

        #     plt.figure(figsize=(15,5))
        #     plt.plot(ft, ~q0, 'C0.')
        #     plt.plot(binnedTime, binnedQuality, 'C3.', alpha=0.01)

            plt.figure(figsize=(15,5))
            plt.plot(ft, ff, 'k.')
            plt.plot(ft, ffFlat, 'C1.')
            plt.plot(ft, ffTrend, 'C2.')
            plt.plot(binnedTime, binnedFlux, 'C3.')
        #     plt.plot([p1[i], p1[i]], plt.ylim(), 'r')
        #     plt.plot([tp1[i], tp1[i]], plt.ylim(), 'b')
            plt.title("Sector " + str(self.sectorList[i]))
            
            print("appending sector " + str(self.sectorList[i]))

            self.allTime = np.append(self.allTime, binnedTime)
            self.allFlux = np.append(self.allFlux, binnedFlux)
            self.allFluxErr = np.append(self.allFluxErr, binnedFluxErr)
            self.allQuality = np.append(self.allQuality, binnedQuality)
            
#            print("got light curve for sector " + str(self.sectorList[i]))

#        print("finished collecting light curves")

        
    def check_fit(self, period = None, epoch = None):
        if self.usePyMC3:
            import pymc3 as pm
        else:
            import pymc as pm

        with pm.Model() as periodModel:
            if period is None:
                usePeriod = self.ticCatalogData["planetData"][0]['period']
            else:
                usePeriod = period
            if epoch is None:
                useEpoch = self.ticCatalogData["planetData"][0]['epoch']
            else:
                useEpoch = epoch

            phase = (self.maskedTransitTime - useEpoch + 0.5 * usePeriod) % usePeriod - 0.5 * usePeriod

            pDepth = pm.TruncatedNormal('depth',
                               mu = self.ticData["planetData"][0]["depth"],
                               sigma = self.ticData["planetData"][0]["depth"]/5, lower=0)
            durationPhase = self.ticData["planetData"][0]["durationHours"]/24/usePeriod

            # box shape
            fvect = pm.math.switch(np.abs(phase) > durationPhase/2, 1.0, (1.-pDepth))

            fobs = pm.Normal('fobs',
                             mu=fvect,
                             sigma=self.maskedTransitFluxErr,
                             observed=self.maskedTransitFlux)

            periodTrace = pm.sample(self.nSamples,
                                    tune=self.nTune,
                                    cores=self.nCores,
                                    chains=self.nChains,
                                    step = pm.Metropolis())

        periodModel.name = "periodModel"

        return periodTrace

    def fit_period_var_depth(self, period = None, epoch = None):
        if self.usePyMC3:
            import pymc3 as pm
        else:
            import pymc as pm

        with pm.Model() as periodVarDepthModel:
            if period is None:
                usePeriod = self.ticCatalogData["planetData"][0]['period']
            else:
                usePeriod = period
            if epoch is None:
                useEpoch = self.ticCatalogData["planetData"][0]['epoch']
            else:
                useEpoch = epoch
                
            pPeriod = pm.Normal('period',
                                mu = usePeriod,
                                sigma = usePeriod/1000)
            pEpoch = pm.TruncatedNormal('epoch',
                               mu = self.ticData["planetData"][0]["epoch"],
                               sigma = usePeriod/100,
                                lower = useEpoch - 0.4*usePeriod,
                                upper = useEpoch + 0.4*usePeriod)

            depthList = []
            for s in self.sectorData["sectorNumber"]:
                depthList.append(pm.TruncatedNormal('depth_' + str(s),
                                        mu = self.ticData["planetData"][0]["depth"],
                                        sigma = self.ticData["planetData"][0]["depth"]/5,
                                        lower=0))

            phase = (self.maskedTransitTime - pEpoch + 0.5 * pPeriod) % pPeriod - 0.5 * pPeriod

            durationPhase = self.ticData["planetData"][0]["durationHours"]/24/pPeriod

            # box shape
            fobsList = []
            modelFlux = None
            for s in range(len(self.sectorData["sectorNumber"])):
                inSector = (self.maskedTransitTime >= self.sectorData["startTime"][s]) \
                                & (self.maskedTransitTime <= self.sectorData["endTime"][s])
                thisModelFlux = pm.math.switch(np.abs(phase[inSector]) > durationPhase/2, 1.0, (1.-depthList[s]))

                pm.Normal('fobs_' + str(s),
                          mu=thisModelFlux,
                          sigma=self.maskedTransitFluxErr[inSector],
                          observed=self.maskedTransitFlux[inSector])

            periodTrace = pm.sample(self.nSamples,
                                    tune=self.nTune,
                                    cores=self.nCores,
                                    chains=self.nChains,
                                    step = pm.Metropolis())
    #         periodTrace = pm.sample(100, tune=0, cores=4, chains=50, step = pm.Metropolis())

        periodVarDepthModel.name = "periodVarDepthModel"

        return periodTrace

    def process_trace(self, trace, variableDepth=False):
        if self.usePyMC3:
            periodPost = trace["period"]
            epochPost = trace["epoch"]
        else:
            periodPost = trace.posterior["period"][0].values
            epochPost = trace.posterior["epoch"][0].values
        periodMedian = np.nanmedian(periodPost)
        periodSigma = np.nanstd(periodPost)
        epochMedian = np.nanmedian(epochPost)
        epochSigma = np.nanstd(epochPost)
        
        if variableDepth:
            depthList = []
            depthSigmaList = []
            for s in self.sectorData["sectorNumber"]:
                if self.usePyMC3:
                    depthPost = trace['depth_' + str(s)]
                else:
                    depthPost = trace.posterior['depth::' + str(s)][0].values
                depthList.append(np.nanmedian(depthPost))
                depthSigmaList.append(np.nanstd(depthPost))
        else:
            if self.usePyMC3:
                depthPost = trace["depth"]
            else:
                depthPost = trace.posterior["depth"][0].values
            depthList = [np.nanmedian(depthPost)]
            depthSigmaList = [np.nanstd(depthPost)]

        print("Best period: " + str(np.round(periodMedian, 6)) + "+-" + str(np.round(periodSigma, 6)) + " days")
        print("Best epoch: " + str(np.round(epochMedian, 6)) + "+-" + str(np.round(epochSigma, 6)) + " days")
        if variableDepth:
            for i in range(len(self.sectorData["sectorNumber"])):
                print("Best depth sector " + str(self.sectorData["sectorNumber"][i])
                      + ": " + str(np.round(depthList[i], 5))
                      + "+-" + str(np.round(depthSigmaList[i], 5)))
        else:
            print("Best depth: " + str(np.round(depthList[0], 5))
                      + "+-" + str(np.round(depthSigmaList[0], 5)))

        return periodMedian, periodSigma, epochMedian, epochSigma, depthList, depthSigmaList

    def get_bls_period(self):
        catalogPeriod = self.ticCatalogData["planetData"][0]['period']
        minPeriod = np.max([0.8*catalogPeriod-1, 0.1])
        try:
            print("calling bls_estimator")
            pg = xo.estimators.bls_estimator(self.allTime,
                                             self.allFlux,
                                             self.allFluxErr,
                                             min_period=minPeriod,
                                             max_period=2*catalogPeriod+1)

            print("returned from bls_estimator")
            if pg["peak_info"] is not None:
                peak = pg["peak_info"]
                period_guess = peak["period"]
                t0_guess = peak["transit_time"]
                depth_guess = peak["depth"]

                print([period_guess, t0_guess])

                plt.plot(pg["bls"].period, pg["bls"].power, "k", linewidth=0.5)
                plt.xlabel("period [days]")
                plt.ylabel("bls power")
                plt.yticks([])
                _ = plt.xlim(pg["bls"].period.min(), pg["bls"].period.max())

                sortPower = np.flip(np.sort(pg["bls"].power))
                sortPowerIdx = np.flip(np.argsort(pg["bls"].power))
                sortPeriod = pg["bls"].period[sortPowerIdx]
                sortPeriodRem = sortPeriod%self.ticCatalogData["planetData"][0]['period']
                sortPeriodRemCloseIdx = sortPeriodRem < 0.01
                self.boxPeriod = sortPeriod[sortPeriodRemCloseIdx][0]
                self.boxFit = True
            else:
                print("bls_estimator peak is None, setting to catalog period")
                self.boxFit = False
                self.boxPeriod = self.ticCatalogData["planetData"][0]['period']
                
        except:
            print("bls_estimator failed, setting to catalog period")
            self.boxFit = False
            self.boxPeriod = self.ticCatalogData["planetData"][0]['period']

        print("box search period: " + str(self.boxPeriod))
        
    def extract_masked_data(self, period = None, epoch = None):
        if period is None:
            usePeriod = self.ticCatalogData["planetData"][0]['period']
        else:
            usePeriod = period
        if epoch is None:
            useEpoch = self.ticCatalogData["planetData"][0]['epoch']
        else:
            useEpoch = epoch
            
        transit_mask = (
            np.abs(
                (self.allTime - useEpoch + 0.5 * usePeriod) % usePeriod - 0.5 * usePeriod
            ) < self.windowSizeDays
        )
        self.maskedTransitTime = np.ascontiguousarray(self.allTime[transit_mask])
        self.maskedTransitFlux = np.ascontiguousarray(self.allFlux[transit_mask])
        self.maskedTransitFluxErr = np.ascontiguousarray(self.allFluxErr[transit_mask])
        self.maskedTransitQuality = np.ascontiguousarray(self.allQuality[transit_mask])

    def plot_transit_mask(self, period = None, epoch = None, plotCatalogEphem = False, filename=None, periodSource=""):
        if period is None:
            usePeriod = self.ticCatalogData["planetData"][0]['period']
        else:
            usePeriod = period
        if epoch is None:
            useEpoch = self.ticCatalogData["planetData"][0]['epoch']
        else:
            useEpoch = epoch
            
        plt.figure(figsize=(8, 4))
        t_fold = (
            self.maskedTransitTime - useEpoch + 0.5 * usePeriod
        ) % usePeriod - 0.5 * usePeriod
        f = self.maskedTransitFlux[self.maskedTransitQuality==0]
        plt.scatter(t_fold[self.maskedTransitQuality==0],
                    f,
                    c=self.maskedTransitTime[self.maskedTransitQuality==0],
                    s=3, cmap="jet", alpha=0.3)
                    
        gf = self.maskedTransitFlux[self.maskedTransitQuality==0]
        if len(gf) == 0:
            return
        yRange98Pct = np.array([np.percentile(gf[~np.isnan(gf)], 1),
                          np.percentile(gf[~np.isnan(gf)], 99)])
        if plotCatalogEphem:
            catalogPeriod = self.ticCatalogData["planetData"][0]['period']
            catalogEpoch = self.ticCatalogData["planetData"][0]['epoch']
            nOrbits = np.unique(np.round((self.allTime-catalogPeriod)/catalogPeriod,0))
            originalTransitTimes = catalogEpoch + nOrbits*catalogPeriod
            originalTransitFold = (originalTransitTimes - useEpoch + 0.5*usePeriod) % usePeriod - 0.5 * usePeriod
            for p in originalTransitFold:
                plt.plot([p,p], yRange98Pct, alpha=0.1)
        plt.ylim(2*(yRange98Pct-1) + 1)
        plt.xlabel("time since transit [days]")
        plt.ylabel("relative flux [ppt]")
        plt.colorbar(label="time [days]")
        plt.title("Using " + periodSource + " period = " + str(np.round(usePeriod, 6)) + ", epoch = " + str(np.round(useEpoch, 6)))
        if filename is not None:
            plt.savefig("figTemp/" + filename,bbox_inches='tight')


    def fit_period_epoch(self):
        if self.usePyMC3:
            import pymc3 as pm
        else:
            import pymc as pm

        import warnings
        warnings.filterwarnings("ignore")

        bestHalfWidth = None

        ###################################
        ###### use my box fitter for the fit
        ###################################

        #     periodTrace = fit_period(maskedTransitTime, maskedTransitFlux, maskedTransitFluxErr, ticData, periodMult=1)
        if self.boxFit:
            periodTraceVarDepth = self.fit_period_var_depth(period = self.boxPeriod)
            print("variable depth fit from box search init")
            bestPerVarDepth, bestPerVarDepthSigma, bestEpoVarDepth, \
                    bestEpoVarDepthSigma, bestDepthList, bestDepthSigmaList \
                    = self.process_trace(periodTraceVarDepth, variableDepth=True)
            periodVarDepthCheckTrace = self.check_fit(period = bestPerVarDepth, epoch = bestEpoVarDepth)

        periodTraceVarDepthCatalog = self.fit_period_var_depth()
        print("variable depth fit from catalog init")
        bestPerVarDepthCat, bestPerVarDepthSigmaCat, bestEpoVarDepthCat, \
                bestEpoVarDepthSigmaCat, bestDepthListCat, bestDepthSigmaListCat \
                = self.process_trace(periodTraceVarDepthCatalog, variableDepth=True)
        periodVarDepthCheckTraceCatalog = self.check_fit(period = bestPerVarDepthCat, epoch = bestEpoVarDepthCat)

        print("check catalog period")
        periodTraceCatalogPeriod = self.check_fit()

        if self.boxFit:
            traceDict = {"fitVarDepthFromBoxInit":periodVarDepthCheckTrace,
                                    "fitVarDepthFromCatalogInit":periodVarDepthCheckTraceCatalog,
                                    "checkCatalogPeriod":periodTraceCatalogPeriod}
        else:
            traceDict = {"fitVarDepthFromCatalogInit":periodVarDepthCheckTraceCatalog,
                                    "checkCatalogPeriod":periodTraceCatalogPeriod}
        self.periodCompare = pm.compare(traceDict, ic='WAIC', method="BB-pseudo-BMA")
        display(self.periodCompare)
        pm.plot_compare(self.periodCompare, insample_dev=False);
        plt.title("TOI" + self.toiName + " TIC " + self.ticName)
        plt.savefig("figTemp/mcmcCompare.png",bbox_inches='tight')

        print("final choice:")
        if self.periodCompare.index[0] == "fitVarDepthFromBoxInit":
            print("chose variable depth fit initialized from box search")
            self.bestPeriod = bestPerVarDepth
            self.bestEpoch = bestEpoVarDepth
            self.depthList = bestDepthList
            az.plot_trace(periodVarDepthCheckTrace);
            display(az.summary(periodVarDepthCheckTrace))
        elif self.periodCompare.index[0] == "fitVarDepthFromCatalogInit":
            print("chose variable depth fit initialized from catalog")
            self.bestPeriod = bestPerVarDepthCat
            self.bestEpoch = bestEpoVarDepthCat
            self.depthList = bestDepthListCat
            az.plot_trace(periodVarDepthCheckTraceCatalog);
            display(az.summary(periodVarDepthCheckTraceCatalog))
        elif self.periodCompare.index[0] == "checkCatalogPeriod":
            print("chose original catalog period")
            self.bestPeriod=self.ticCatalogData["planetData"][0]["period"].copy()
            self.bestEpoch=self.ticCatalogData["planetData"][0]["epoch"].copy()
            self.depthList = [self.ticCatalogData["planetData"][0]["depth"].copy()]
            az.plot_trace(periodTraceCatalogPeriod);
            display(az.summary(periodTraceCatalogPeriod))
        else:
            raise ValueError('Bad period compare name')


    def plot_folded_lightcurve(self, period = None, epoch = None, depthList = None):
        if period is None:
            usePeriod = self.bestPeriod
        else:
            usePeriod = period
        if epoch is None:
            useEpoch = self.bestEpoch
        else:
            useEpoch = epoch
        if depthList is None:
            useDepthList = self.depthList
        else:
            useDepthList = depthList
            
        t_fold = (
            self.maskedTransitTime - useEpoch + 0.5 * usePeriod
        ) % usePeriod - 0.5 * usePeriod
        phaseIdx = np.argsort(t_fold)
        t_fold_phaseOrder = t_fold[phaseIdx]/usePeriod
        flux_phaseOrder = self.maskedTransitFlux[phaseIdx]

        phaseBins = np.linspace(-0.5, 0.5, 300)
        binnedFlux, binnedPhaseEdges, _ = sp.stats.binned_statistic(t_fold_phaseOrder, flux_phaseOrder, bins=phaseBins)
        binnedPhase = (binnedPhaseEdges[:-1]+binnedPhaseEdges[1:])/2

        durationPhase = self.ticData["planetData"][0]["durationHours"]/24/usePeriod
        model = np.ones(t_fold_phaseOrder.shape)

        phase = t_fold_phaseOrder
        durationHalfPhase = durationPhase/2

        plt.figure(figsize=(8, 4))
        plt.scatter(t_fold_phaseOrder,
                    flux_phaseOrder,
                    s=3, cmap="jet", alpha=0.3)
        ylim = plt.ylim()
        plt.plot([-(durationPhase/2), -(durationPhase/2)], plt.ylim(), 'k', alpha=0.3)
        plt.plot([+(durationPhase/2), +(durationPhase/2)], plt.ylim(), 'k', alpha=0.3)
        plt.plot(binnedPhase, binnedFlux, 'r.')
        for depth in useDepthList:
            model = np.ones(t_fold_phaseOrder.shape)
            model[np.abs(phase) < durationHalfPhase] = 1.-depth
            plt.scatter(t_fold_phaseOrder, model, s=2, alpha=0.3)

        gf = flux_phaseOrder
        if len(gf) == 0:
            return
        yRange98Pct = np.array([np.percentile(gf[~np.isnan(gf)], 1),
                          np.percentile(gf[~np.isnan(gf)], 99)])
        plt.ylim(2*(yRange98Pct-1) + 1)
        plt.xlabel("phase")
        plt.ylabel("relative flux [ppt]");
        plt.title("TOI" + self.toiName + " TIC " + self.ticName)
        plt.savefig("figTemp/foldedLightcurve.png",bbox_inches='tight')
        
    def make_report(self):
        # get images
        img1 = Image.open('figTemp/oldEphem.png')
        img2 = Image.open('figTemp/newEphem.png')
        img3 = Image.open('figTemp/foldedLightcurve.png')
        img4 = Image.open('figTemp/mcmcCompare.png')
        img5 = Image.open('figTemp/catalogEphem.png')

        # get width and height
        w1, h1 = img1.size
        w2, h2 = img2.size
        w3, h3 = img3.size
        w4, h4 = img4.size
        w5, h5 = img4.size

        w = int(1.2*(max(w1,w3) + max(w2, w4)))
        h = int(1.2*(max(h1,h2) + max(h3,h5) + h4))
        vspace = int(0.1*h1)
        hspace = int(0.1*w1)
        
        new_image = Image.new('RGB', (w, h), color="white")
        new_image.paste(img1, (hspace, vspace))
        new_image.paste(img2, (w1 + 2*hspace, vspace))
        new_image.paste(img3, (hspace, h1 + 2*vspace))
        new_image.paste(img5, (w1 + 2*hspace, h1 + 2*vspace))
        new_image.paste(img4, (hspace, h1 + h3 + 3*vspace))

        new_image.save("ephemFits/" + self.toiName + "_ephemFit.png")

        ephemFit = {"period": self.bestPeriod, \
                    "epoch": self.bestEpoch, \
                    "boxPeriod": self.boxPeriod}
                    
        f = open("ephemFits/" + self.toiName + "_ephemFit.pickle", 'wb')
        pickle.dump([ephemFit, self.periodCompare], f, pickle.HIGHEST_PROTOCOL)
        f.close()
        
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
    args = parser.parse_args()
    
    toisToRun = np.loadtxt(args.toiListFilename)
    
    toiTable = pd.read_csv("toi_table_tfop.csv")

    for toiNum in toisToRun:
        if os.path.exists("ephemFits/" + str(toiNum) + "_ephemFit.png"):
            continue
                        
        thisToi = toiTable[toiTable.TOI == toiNum]
        if thisToi['TIC ID'].values[0] < 17000:
            print("strange tic number " + str(thisToi['TIC ID'].values[0]))
            continue
            
        if thisToi['TESS Mag'].values[0] < 7.5:
            print("this star is saturated, skipping")
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
        if planet0["period"] == 0:
            continue
            
        ticCatalogData["planetData"] = [planet0]

        display(ticCatalogData)
        
        try:
            fe = fitEphemeris(ticCatalogData, cleanFiles=False, usePyMC3 = True)
        except:
            print("something went wrong, moving on")



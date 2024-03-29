#!/usr/bin/env python
"""
author: J Cook, Feb 2020

Driver script for generating csv file containing DISORT BBA predictions along with
the input variable values used to generate that BBA. The purpose is to
generate a large training data set that can be used to train a neural network that
can predict BBA from the inpout parameters without running the full DISORT model. The
nerual network itself is trained using the TrainModel.py script (tensorflow)
or TrainModel_scikit.py (scikitlearn).

Uses BioPyDISORT, a Python wrapper to the DISORT library that allows SNICAR style input parameter
definitions, calculation of optical thicknesses for mixed ice/impurity layers and operation
over a wavelength range rather than monochrome calculations.

SNICAR inputs re used to define an optical thickness, ssa and gg for each vertical layer - these
are then fed into DISORT.

Module '_disort' is auto-generated with f2py (version:2).

RUN THIS SCRIPT FROM THE TERMINAL!

i.e
cd /home/joe/Code/AlbedoNeuralNet/DISORT_NN/
python GenerateTrainingData.py


This driver calls the external file "getParams.py" where gg, tau and ssa are calculated
Runs in Python 2.7 environment "pyDISORT"
(build from disort_env.yml)


"""

import disort
import numpy as np
import matplotlib.pyplot as plt
import gc
import xarray as xr
import numpy as np
import pandas as pd
from getParams import getParams_mie, getParams_GO
import time

###########################################
# 1) DEFINE FUNCTION FOR CALLING DISORT
###########################################

def run_disort(gg, w0, dTau, umu0, umu, phi0, phi, albedo, uTau, fbeam, prnt):

    [rfldir, rfldn, flup, dfdt, uavg, uu, albmed, trnmed] =\
                                    disort.run(dTau = dTau, w0=w0, iphas=iphas, gg=gg,
                                            umu0=umu0, phi0=phi0, albedo=albedo, fbeam=fbeam,
                                                utau=uTau, umu=umu, phi=phi, prnt=prnt, UsrAng=True,
                                                    onlyFl=False, Nstr=Nstr)
    alb = flup/(rfldir+rfldn) # calculate albedo from upwards and downards fluxes

    return alb


print_progess = True # toggle whether to print progress through total # runs to console
progress_step = 100 # how many runs between progress updates printed to console

###########################################
# 2) DEFINE PATHS AND VARIABLE VALUES
###########################################

save_path = '/home/joe/Code/Albedo_NeuralNet/DISORT_NN/' # path to save figures to
DIRECT = 1        # 1= Direct-beam incident flux, 0= Diffuse incident flux
dz = [0.001, 0.01, 0.01, 0.1, 0.1] # thickness of each vertical layer (unit = m)
nbr_lyr = len(dz)  # number of snow layers
R_sfc = 0.15 # reflectance of undrlying surface - set across all wavelengths
nbr_aer = 16  # Define total number of different LAPs/aerosols in model

# set filename stubs
stb1 = 'algae_geom_'  # %name stub 1
stb2 = '.nc'  # file extension
wrkdir2 = '/home/joe/Code/Albedo_NeuralNet/Data/Algal_Optical_Props/'  # working directory
snw_stb1 = 'snw_alg_'  # name stub for snow algae

# CHOOSE DIMENSIONS OF GLACIER ALGAE 1
algae_r = 6  # algae radius
algae_l = 120  # algae length
glacier_algae1 = str(wrkdir2 + stb1 + str(algae_r) + '_' + str(algae_l) + stb2)  # create filename string

# CHOOSE DIMENSIONS OF GLACIER ALGAE 2
algae2_r = 6  # algae radius
algae2_l = 20  # algae length
glacier_algae2 = str(wrkdir2 + stb1 + str(algae2_r) + '_' + str(algae2_l) + stb2)  # create filename string

# CHOOSE SNOW ALGAE DIAMETER
snw_algae_r = 1  # snow algae diameter
snw_alg = str(wrkdir2 + snw_stb1 + str(snw_algae_r) + stb2)  # create filename string

# SET FILE NAMES CONTAINING OPTICAL PARAMETERS FOR ALL IMPURITIES:

FILE_soot1 = 'mie_sot_ChC90_dns_1317.nc'
FILE_soot2 = 'miecot_slfsot_ChC90_dns_1317.nc'
FILE_dust1 = 'aer_dst_bln_20060904_01.nc'
FILE_dust2 = 'aer_dst_bln_20060904_02.nc'
FILE_dust3 = 'aer_dst_bln_20060904_03.nc'
FILE_dust4 = 'aer_dst_bln_20060904_04.nc'
FILE_ash1 = 'volc_ash_mtsthelens_20081011.nc'
FILE_GRISdust1 = 'dust_greenland_Cook_CENTRAL_20190911.nc'
FILE_GRISdust2 = 'dust_greenland_Cook_HIGH_20190911.nc'
FILE_GRISdust3 = 'dust_greenland_Cook_LOW_20190911.nc'
FILE_GRISdustP1 = 'dust_greenland_L_20150308.nc'
FILE_GRISdustP2 = 'dust_greenland_C_20150308.nc'
FILE_GRISdustP3 = 'dust_greenland_H_20150308.nc'
FILE_snw_alg = snw_alg  # snow algae (c nivalis)
FILE_glacier_algae1 = glacier_algae1  # Glacier algae
FILE_glacier_algae2 = glacier_algae2  # Glacier algae


#####################################
# 3) DEFINE ITERABLE VALUES
#####################################
#i.e. values for vars we want to loop through

density = [[300,300,300,300,300],[500,500,500,500,500],[700,700,700,700,700],[800,800,800,800,800],[900,900,900,900,900]]

rad = [[300,300,300,300,300],[500,500,500,500,500],[700,700,700,700,700],[900,900,900,900,900],[1100,1100,1100,1100,1100],
       [1300,1300,1300,1300,1300],[1500,1500,1500,1500,1500],[2000,2000,2000,2000,2000],[5000,5000,5000,5000,5000],
        [10000,10000,10000,10000,10000],[15000,15000,15000,15000,15000]]

dust = [0,1000,100000,250000,500000,1000000]

algae = [0,1000,100000,250000,500000,1000000]

coszen = [0.1,0.3,0.5,0.7,0.9]

soot = [0,100,1000,5000]

########################################
# 4) SETUP OUTPUT ARRAYS/DATAFRAME
########################################

outData = pd.DataFrame(columns=["coszen","rho","rds","dust","algae","soot","BBA"])
counter = 0
g_out = np.zeros(shape=(len(density) * len(rad) * len(coszen) * len(dust) * len(algae) * len(soot), 5, 470))
ssa_out = np.zeros(shape=(len(density) * len(rad) * len(coszen) * len(dust) * len(algae) * len(soot), 5, 470))
tau_out = np.zeros(shape=(len(density) * len(rad) * len(coszen) * len(dust) * len(algae) * len(soot), 5, 470))
alb_out = np.zeros(shape=(len(density) * len(rad) * len(coszen) * len(dust) * len(algae) * len(soot),470))
total_runs = (len(density) * len(rad) * len(coszen) * len(dust) * len(algae) * len(soot))

#########################################
# 5) BEGIN LOOPS
#########################################
start = time.time()

for rho_snw in density:
    for rds_snw in rad:
        for umu0 in coszen:
            for d in dust:
                for a in algae:
                    for BC in soot:

                        # SET IMPURITY MASS CONCENTRATIONS IN EACH LAYER
                        # units are ppb or ng/g i.e. 1e3 = 1 ppm or 1 ug/g, 1e6 = 1 ppt or 1 mg/g

                        mss_cnc_soot1 = [BC, 0, 0, 0, 0]  # uncoated black carbon
                        mss_cnc_soot2 = [0, 0, 0, 0, 0]  # coated black carbon
                        mss_cnc_dust1 = [0, 0, 0, 0, 0]  # global average dust 1
                        mss_cnc_dust2 = [0, 0, 0, 0, 0]  # global average dust 2
                        mss_cnc_dust3 = [0, 0, 0, 0, 0]  # global average dust 3
                        mss_cnc_dust4 = [0, 0, 0, 0, 0]  # global average dust 4
                        mss_cnc_ash1 = [0, 0, 0, 0, 0]  # volcanic ash species 1
                        mss_cnc_GRISdust1 = [d, 0, 0, 0, 0]  # GRIS dust 1 (Cook et al. 2019 "mean")
                        mss_cnc_GRISdust2 = [0, 0, 0, 0, 0]  # GRIS dust 2 (Cook et al. 2019 HIGH)
                        mss_cnc_GRISdust3 = [0, 0, 0, 0, 1]  # GRIS dust 3 (Cook et al. 2019 LOW)
                        mss_cnc_GRISdustP1 = [0, 0, 0, 0, 0]  # GRIS dust 1 (Polashenki2015: low hematite)
                        mss_cnc_GRISdustP2 = [0, 0, 0, 0, 0]  # GRIS dust 1 (Polashenki2015: median hematite)
                        mss_cnc_GRISdustP3 = [0, 0, 0, 0, 0]  # GRIS dust 1 (Polashenki2015: median hematite)
                        mss_cnc_snw_alg = [0, 0, 0, 0, 0]  # Snow Algae (spherical, C nivalis)
                        mss_cnc_glacier_algae1 = [a, 0, 0, 0, 0]  # glacier algae type1
                        mss_cnc_glacier_algae2 = [0, 0, 0, 0, 0]  # glacier algae type2

                        #if radius within Mie domain, use mie-generated ice single scattering optical props
                        if rds_snw[0] < 2000:
                            flx_slr, g_star, SSA_star, tau_star, wvl = getParams_mie(DIRECT,
                                R_sfc, dz, rho_snw, rds_snw, nbr_lyr, nbr_aer,mss_cnc_soot1, mss_cnc_soot2,
                                mss_cnc_dust1, mss_cnc_dust2, mss_cnc_dust3, mss_cnc_dust4, mss_cnc_ash1,
                                mss_cnc_GRISdust1, mss_cnc_GRISdust2, mss_cnc_GRISdust3, mss_cnc_GRISdustP1,
                                mss_cnc_GRISdustP2, mss_cnc_GRISdustP3, mss_cnc_snw_alg, mss_cnc_glacier_algae1,
                                mss_cnc_glacier_algae2, FILE_soot1, FILE_soot2, FILE_dust1, FILE_dust2, FILE_dust3, FILE_dust4,
                                FILE_ash1, FILE_GRISdust1, FILE_GRISdust2, FILE_GRISdust3, FILE_GRISdustP1, FILE_GRISdustP2,
                                FILE_GRISdustP3, FILE_snw_alg, FILE_glacier_algae1, FILE_glacier_algae2)

                        # if radius outside Mie domain, use GO-generated ice single scattering optical props

                        else:
                            side_length = rds_snw # in this version length/radius are equal
                            depth = rds_snw
                            flx_slr, g_star, SSA_star, tau_star, wvl = getParams_GO(DIRECT, R_sfc, dz, rho_snw,
                            side_length, depth, nbr_lyr, nbr_aer, mss_cnc_soot1, mss_cnc_soot2, mss_cnc_dust1, mss_cnc_dust2,
                            mss_cnc_dust3, mss_cnc_dust4, mss_cnc_ash1, mss_cnc_GRISdust1, mss_cnc_GRISdust2, mss_cnc_GRISdust3,
                            mss_cnc_GRISdustP1,mss_cnc_GRISdustP2, mss_cnc_GRISdustP3, mss_cnc_snw_alg, mss_cnc_glacier_algae1,
                            mss_cnc_glacier_algae2, FILE_soot1, FILE_soot2, FILE_dust1, FILE_dust2, FILE_dust3, FILE_dust4,
                            FILE_ash1, FILE_GRISdust1, FILE_GRISdust2, FILE_GRISdust3, FILE_GRISdustP1, FILE_GRISdustP2,
                            FILE_GRISdustP3, FILE_snw_alg, FILE_glacier_algae1, FILE_glacier_algae2)

                        # set DISORT-specific constants
                        prnt = np.array([False, False, False, False, False])  # determines what info to print to console
                        umu = 1.  # cosine of viewing zenith angle
                        phi0 = 0.  # solar azimuth angle
                        phi = 0.  # viewing azimuth angle
                        albedo = 0.2  # albedo of underlying surface
                        uTau = 0.  # optical thickness where fluxes are calculated
                        Nstr = 16  # number of streams to include in model
                        fbeam = flx_slr  # incoming irradiance (output by SNICAR)

                        # loop through wavelengths, isolate appropriate value from wavelength-resolved params
                        for i in range(len(wvl)):  # iterate over wavelength

                            dTau = tau_star[:, i]
                            w0 = SSA_star[:, i]
                            iphas = np.ones(len(dz), dtype='int') * 2
                            iphas[:] = 3
                            gg = g_star[:, i]
                            # CALL DISORT
                            alb = run_disort(gg, w0, dTau, umu0, umu, phi0, phi, albedo, uTau, fbeam, prnt)
                            alb_out[counter,i] = alb # append albedo at each wavelength to array = spectral albedo array

                        BBA = np.sum(alb_out[counter,:]*flx_slr)/np.sum(flx_slr) #calculate broadband albedo

                        # organise out data into dataframe
                        tempData = pd.DataFrame({"coszen": umu0, "rho":rho_snw[0], "rds":rds_snw[0], "dust":d, "algae":a, "soot":BC, "BBA":BBA},index=[counter])
                        outData = outData.append(tempData)

                        counter +=1

                        if print_progress:
                            if progress % progress_step = 0:
                                print("Progress = {} / {}".format(counter, total_runs))

outData.to_csv(str(save_path+'trainingData.csv'),index=None) # save out data to csv

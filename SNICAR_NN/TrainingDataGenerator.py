"""

#####################################################################
################# BioSNICAR_GO DRIVER SCRIPT ########################

This script is used to configure the 2-stream radiative transfer
model BioSNICAR_GO. Here variable values are defined, the model called
and the results plotted.

NB. Setting Mie = 1, GO = 0 and algal impurities = 0 is equivalent to
running the original SNICAR model of Flanner et al. (2007, 2009)

Author: Joseph Cook, October 2019

######################################################################
######################################################################


"""

###########################
# 1) Import SNICAR function

from snicar8d_mie import snicar8d_mie
from snicar8d_GO import snicar8d_GO
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 2) CHOOSE METHOD FOR DETERMINING OPTICAL PROPERTIES OF ICE GRAINS
# for small spheres choose Mie, for hexagonal plates or columns of any size
# choose GeometricOptics
Mie = True
GeometricOptics = False

######################################
## 2. RADIATIVE TRANSFER CONFIGURATION

DIRECT   = 1        # 1= Direct-beam incident flux, 0= Diffuse incident flux
APRX_TYP = 1        # 1= Eddington, 2= Quadrature, 3= Hemispheric Mean
DELTA    = 1        # 1= Apply Delta approximation, 0= No delta
dz = [0.001, 0.01, 0.01, 0.01, 0.01] # thickness of each vertical layer (unit = m)
nbr_lyr = len(dz)  # number of snow layers
R_sfc = 0.15 # reflectance of undrlying surface - set across all wavelengths
nbr_aer = 16 # Define total number of different LAPs/aerosols in model

# set filename stubs
stb1 = 'algae_geom_' # %name stub 1
stb2 = '.nc'  # file extension
wrkdir2 = '/home/joe/Code/BioSNICAR_GO_PY/Data/Algal_Optical_Props/' # working directory
snw_stb1 = 'snw_alg_' # name stub for snow algae

# CHOOSE DIMENSIONS OF GLACIER ALGAE 1
algae_r = 6 # algae radius
algae_l = 120 # algae length
glacier_algae1 = str(wrkdir2+stb1+str(algae_r)+'_'+str(algae_l)+stb2) # create filename string

# CHOOSE DIMENSIONS OF GLACIER ALGAE 2
algae2_r = 6 # algae radius
algae2_l = 20 # algae length
glacier_algae2 = str(wrkdir2+stb1+str(algae2_r)+'_'+str(algae2_l)+stb2) # create filename string

# CHOOSE SNOW ALGAE DIAMETER
snw_algae_r = 1 # snow algae diameter
snw_alg = str(wrkdir2+snw_stb1+str(snw_algae_r)+stb2) # create filename string

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

masterDF = pd.DataFrame(columns=['coszen','rho','rds','dust','algae','BBA'])
counter = 0

import time
start = time.time()
for coszen in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.57]:
    for rho_snw in [[300,300,300,300,300],[400,400,400,400,400],[500,500,500,500,500],[600,600,600,600,600],[700,700,700,700,700],[800,800,800,800,800]]:
        for rds_snw in [[200,200,200,200,200],[300,300,300,300,300,300],[500,500,500,500,500],[700,700,700,700,700],[1000,1000,1000,1000,1000],[1500,1500,1500,1500,1500]]:
            for dust in [0, 10000, 50000, 100000, 1000000, 15000000, 20000000]:
                for algae in [0, 10000, 50000, 100000, 1000000, 15000000, 20000000]:

                    mss_cnc_soot1 = [0,0,0,0,0]    # uncoated black carbon
                    mss_cnc_soot2 = [0,0,0,0,0]    # coated black carbon
                    mss_cnc_dust1 = [0,0,0,0,0]    # global average dust 1
                    mss_cnc_dust2 = [0,0,0,0,0]    # global average dust 2
                    mss_cnc_dust3 = [0,0,0,0,0]    # global average dust 3
                    mss_cnc_dust4 = [0,0,0,0,0]    # global average dust 4
                    mss_cnc_ash1 = [0,0,0,0,0]    # volcanic ash species 1
                    mss_cnc_GRISdust1 = [dust,0,0,0,0]    # GRIS dust 1 (Cook et al. 2019 "mean")
                    mss_cnc_GRISdust2 = [0,0,0,0,0]    # GRIS dust 2 (Cook et al. 2019 HIGH)
                    mss_cnc_GRISdust3 = [0,0,0,0,0]    # GRIS dust 3 (Cook et al. 2019 LOW)
                    mss_cnc_GRISdustP1 = [0,0,0,0,0]  # GRIS dust 1 (Polashenki2015: low hematite)
                    mss_cnc_GRISdustP2 = [0,0,0,0,0]  # GRIS dust 1 (Polashenki2015: median hematite)
                    mss_cnc_GRISdustP3 = [0,0,0,0,0]  # GRIS dust 1 (Polashenki2015: median hematite)
                    mss_cnc_snw_alg = [0,0,0,0,0]    # Snow Algae (spherical, C nivalis)
                    mss_cnc_glacier_algae1 = [algae,0,0,0,0]    # glacier algae type1
                    mss_cnc_glacier_algae2 = [0,0,0,0,0]    # glacier algae type2



                    [wvl, albedo, BBA, BBAVIS, BBANIR, abs_slr, abs_slr_tot, abs_vis_tot, heat_rt, total_insolation] = snicar8d_mie(DIRECT, APRX_TYP, DELTA, coszen, R_sfc, dz, rho_snw, rds_snw, nbr_lyr, nbr_aer, mss_cnc_soot1,
                    mss_cnc_soot2, mss_cnc_dust1, mss_cnc_dust2, mss_cnc_dust3, mss_cnc_dust4, mss_cnc_ash1, mss_cnc_GRISdust1,
                    mss_cnc_GRISdust2, mss_cnc_GRISdust3, mss_cnc_GRISdustP1, mss_cnc_GRISdustP2, mss_cnc_GRISdustP3,
                    mss_cnc_snw_alg, mss_cnc_glacier_algae1, mss_cnc_glacier_algae2, FILE_soot1, FILE_soot2, FILE_dust1, FILE_dust2,
                    FILE_dust3, FILE_dust4, FILE_ash1, FILE_GRISdust1, FILE_GRISdust2, FILE_GRISdust3, FILE_GRISdustP1, FILE_GRISdustP2,
                    FILE_GRISdustP3, FILE_snw_alg, FILE_glacier_algae1, FILE_glacier_algae2)

                    outDF = pd.DataFrame({"coszen": coszen,"rho":rho_snw[0],"rds":rds_snw[0],"dust":dust,"algae":algae,"BBA":BBA},index=[counter])
                    masterDF = masterDF.append(outDF)
                    counter+=1

#masterDF.to_csv('/home/joe/Desktop/NNtraining_data.csv')

elapsed_time = time.time() - start
print(elapsed_time)
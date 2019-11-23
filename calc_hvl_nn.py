import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, interp1d
from scipy.optimize import minimize
 


def calc_hvl(al_thick,kerma0, al_spline_mu_tot, energy, prob, first_layer):
    rho_al = 2.69890
    mu_tot_al_vec = np.exp(al_spline_mu_tot(np.log(energy)))

    exp_al = np.exp(-mu_tot_al_vec*rho_al*al_thick/10)
    #now we add the fluence
    y = prob*exp_al
    #and integrate over the energy
    Y = np.sum(y)
    #Y = np.sum(y*500)
    #now we normalize
    kerma = Y#/np.sum(prob*500)
    if first_layer == True:
        kerma0 = kerma0/2
    else:
        kerma0 = kerma0/4
        
   
    return(abs(kerma-kerma0))
    



def return_hvl(potential, kerma_vec, kerma0):

    energy = np.arange(1.25E3, potential*1000+250, 500)
    

    #read al coefficients
    df3 = pd.read_csv('spectra/at_elements/Al.txt', header=None, sep='\s+')
    mu_tot_al = df3[1]
    x_al = df3[0]*1E6
    al_spline_mu_tot = interp1d(np.log(x_al), np.log(mu_tot_al))
    rho_al = 2.69890
 


    

    #calc mean energy
    Y_mean = np.sum(kerma_vec*energy)/np.sum(kerma_vec)

    energy_mean = Y_mean#/norm_mean
    first_layer = True
    hvl_min = minimize(calc_hvl, 0.0, args=(kerma0, al_spline_mu_tot, energy, kerma_vec, first_layer), method='Nelder-Mead')
    first_layer_found = hvl_min.x
    first_layer = False
    hvl_min = minimize(calc_hvl, 0.0, args=(kerma0, al_spline_mu_tot, energy, kerma_vec, first_layer), method='Nelder-Mead')
    second_layer = hvl_min.x
    mean_energy = energy_mean


    return(np.mean(first_layer_found), mean_energy) 

    



             

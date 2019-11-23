import pandas as pd
from scipy.interpolate import interp1d
import numpy as np

def return_df_spec(anode, Filter, potential, thick_filter, pmma):
    df_spec = pd.read_table('spectra/anodes/'+anode+'/'+str(potential)+'.spec', header=None,
                            sep = '\t')
    e_spec = df_spec[0]*1000
    prob_spec = df_spec[1]
    if Filter!=None:
        df_filter = pd.read_table('spectra/at_elements/'+Filter+'.txt', header=None,
                            sep = ' ')
        df_rho = pd.read_table('spectra/at_elements/'+'density.txt', header=None,
                            sep = '\s+')
        rho_filter = df_rho[df_rho[0]==Filter]
        rho_filter = rho_filter[1].values
        interp_filter = interp1d(np.log(df_filter[0]*1E6), np.log(df_filter[1]))
        mu_filter = np.exp(interp_filter(np.log(e_spec)))
        at_filter = np.exp(-mu_filter*rho_filter*(thick_filter/10))

        if pmma>0:
            df_pmma = pd.read_table('spectra/at_elements/pmma.txt', header=None, sep = '\s+')
            density_pmma = 1.19
            interp_pmma = interp1d(np.log(df_pmma[0]*1E6), np.log(df_pmma[1]))
            mu_pmma = np.exp(interp_pmma(np.log(e_spec)))
            at_pmma = np.exp(-mu_pmma*density_pmma*(pmma/10))
        else:
            at_pmma = 1

    else:
        at_filter = 1
        at_pmma = 1

    df_spec.loc[:,1] = df_spec.loc[:,1]*at_filter*at_pmma
    df_spec.loc[:,1] = df_spec.loc[:,1]/np.sum(df_spec.loc[:,1])    
    return df_spec

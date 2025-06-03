import numpy as np
from scipy.interpolate import interp1d

Av_dict = { 
            'f275w': 2.02499,
            'f336w': 1.67536,
            'f435w': 1.33879,
            'f555w': 1.03065,
            'f814w': 0.59696,
    
            'f090w': 0.583,
            'f115w': 0.419,
            'f150w': 0.287,
            'f200w': 0.195,
    
            'f438w': 1.34148,
            'f606w': 0.90941,
            'f814w': 0.59845
          }

def IMF_Krp(m, ml=0.1, mint=0.5, mu=350.,a1=1.3,a2=2.3):

    h2 = (mu**(1.-a2)-mint**(1.-a2))/(1.-a2)
    h1 = (mint**(1.-a1)-ml**(1.-a1))/(1.-a1)

    c1 = 1./(h1+h2*mint**(a2-a1))
    c2 = c1*mint**(a2-a1)

    c = np.ones(len(m))
    c[np.where(m < mint)] = c1
    c[np.where(m >= mint)] = c2

    a = np.ones(len(m))
    a[np.where(m < mint)] = -a1
    a[np.where(m >= mint)] = -a2
    imf = c*m**a

    return imf
 
def sample_IMF(M_tot):
    
    m = np.logspace(np.log10(0.1), np.log10(350),10000)
    m_bins = 0.5*(m[1:] + m[:-1])
    dm = np.diff(m)

    # PDF = IMF
    imf = IMF_Krp(m_bins)

    # PDF to CDF
    cdf = np.cumsum(imf*dm)
    cdf /= cdf[-1]
    
    cdf = interp1d(cdf, m_bins, bounds_error=False, fill_value=0.1)

    # Uniformly sampling inverse of CDF.
    ms = []

    # Initial guess of total number of stars for a given M_tot
    N = np.round(M_tot/0.6,0).astype(int)
    ms = cdf(np.random.uniform(0,1,N))
    
    ms = list(ms)
    
    while np.sum(ms)<=M_tot:
        m = cdf(np.random.uniform(0,1))
        ms.append(m)
    ms = np.array(ms)

    return ms
    
def sample_iso(mass=1e7, df_cmd=None, age=10.0, met=0.002, DM=29.81, Av=0.19, mag_det=29,
              filt1='F115W', filt2='F150W', filt3='F200W'):
    
    df_test = df_cmd[(np.round(df_cmd['logAge'],1)==age) 
                                 & (df_cmd['Zini']==met) 
                                 &  (df_cmd['label']<9) ].copy()
    
    df_test[filt1 + 'mag'] = df_test[filt1 + 'mag'] + DM  + Av_dict[filt1.lower()]*Av
    df_test[filt2 + 'mag'] = df_test[filt2 + 'mag'] + DM  + Av_dict[filt2.lower()]*Av
    df_test[filt3 + 'mag'] = df_test[filt3 + 'mag'] + DM  + Av_dict[filt3.lower()]*Av
    
    df_test = df_test[(df_test[filt1 + 'mag']<=mag_det) 
                    & (df_test[filt2 + 'mag']<=mag_det) 
                    & (df_test[filt3 + 'mag']<=mag_det)]    

    mini = df_test['Mini'].values
    mfin = df_test['Mass'].values
    
    m_lim = mini.min()
    l = df_test['label']

    if mass>1e7:
        iters = np.round(mass/1e7).astype(int)
        sampled_masses = []
        for i in range(iters):
            masses = sample_IMF(1e7)
            masses = masses[masses>=m_lim]
            sampled_masses.append(masses)
        
        sampled_masses = np.array(sampled_masses)
    else:
        sampled_masses = sample_IMF(mass)
        sampled_masses = sampled_masses[sampled_masses>=m_lim]
    
    filt1_mag = []
    filt2_mag = []
    filt3_mag = []
    
    for i in np.unique(l): 
        l_ind = l==i
        
        m1 = mini[l_ind]
        m2 = mfin[l_ind]
        
        m_up = m1.max()
        m_low = m1.min()
        
        m_ind = np.argsort(m1)
        m1    = m1[m_ind]
        
        filt1_iso = df_test[filt1 + 'mag'].values[l_ind][m_ind]
        filt2_iso = df_test[filt2 + 'mag'].values[l_ind][m_ind]
        filt3_iso = df_test[filt3 + 'mag'].values[l_ind][m_ind]
        
        filt1_func = interp1d(m1,filt1_iso)
        filt2_func = interp1d(m1,filt2_iso)
        filt3_func = interp1d(m1,filt3_iso)
        
        ind_mass = (sampled_masses>=m_low) &  (sampled_masses<=m_up) 

        sampled_masses_sub = sampled_masses[ind_mass]
        sampled_masses = sampled_masses[~ind_mass]
        
        filt1_mag += list(filt1_func(sampled_masses_sub))
        filt2_mag += list(filt2_func(sampled_masses_sub))
        filt3_mag += list(filt3_func(sampled_masses_sub))
        
    return np.array(filt1_mag), np.array(filt2_mag), np.array(filt3_mag)
        
    return np.array(filt1_mag), np.array(filt2_mag), np.array(filt3_mag)

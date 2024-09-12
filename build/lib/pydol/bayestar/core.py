from numpy import append, array, exp, \
                linspace, loadtxt, log10, pi, meshgrid, \
                savetxt, sqrt, where, ones, percentile,trapz, all
from scipy import special
import matplotlib.pyplot as plt
import time, glob, os, sys
import multiprocessing as mp
from pathlib import Path

import stan

code = """

    functions{
        real P(int N1, int N2, vector v, matrix M) {
            vector[N1] Mj;
            vector[N1] ln_Mj;

            Mj= M*v;
            for (j in 1:N1){
                if (Mj[j]<=0.)
                    Mj[j] = 1.;
            }
            ln_Mj = log(Mj);
            return sum(ln_Mj);
        }
    }

    data {
        int<lower=0> Nj; // number of data
        int<lower=0> Ni; // number of isochrones
        matrix[Nj,Ni] Pij; // Probability matrix
        matrix[Nj,Ni] Cij; // Normalization matrix
    }

    parameters {
        simplex[Ni] a;
    }

    model {
        target += dirichlet_lpdf(a | rep_vector(1., Ni));
        target += P(Nj,Ni,a,Pij);
        target += -1.*P(Nj,Ni,a,Cij);
    }

    """

class Base():

    def Normal_MGk(self, fw_dat,fw_err,Iso_sig):    # Error model apparent maginute
        sig2 = fw_err*fw_err + Iso_sig*Iso_sig          # And normal Isochrone 
        
        return  lambda fw_iso : exp( -0.5*(fw_dat-fw_iso)**2 / sig2 ) / sqrt(2.*pi*sig2)

    def Phi_MGk(self,fwj2, sig_fwj2, fwklim, sig_i2):
        b = sig_i2*sig_i2 + sig_fwj2*sig_fwj2
        b1 = sig_i2*sig_i2/b
        b2 = sig_fwj2*sig_fwj2/b
        b3 = sig_i2*sig_i2/sqrt(b)
        return  lambda fw_i2 : special.ndtr( ( fwklim - b1*fwj2 - b2*fw_i2 ) / b3 )

    def IMF_Krp(self, m, ml=0.1, mint=0.5, mu=350.,a1=1.3,a2=2.3):

        h2 = (mu**(1.-a2)-mint**(1.-a2))/(1.-a2)
        h1 = (mint**(1.-a1)-ml**(1.-a1))/(1.-a1)

        c1 = 1./(h1+h2*mint**(a2-a1))
        c2 = c1*mint**(a2-a1)

        c = ones(len(m))
        c[where(m < mint)] = c1
        c[where(m >= mint)] = c2

        a = ones(len(m))
        a[where(m < mint)] = -a1
        a[where(m >= mint)] = -a2
        imf = c*m**a

        return(imf)


    def P_ij_map(self, IDp):
        fw1_lim = self.fw_lims[0]
        fw2_lim = self.fw_lims[1]
        fw3_lim = self.fw_lims[2]

        sig_i = self.sig_fw[0]

        if not os.path.exists('pij_cij_results'):
            os.mkdir('pij_cij_results')

        filename_p = '%s_Pij_Data_LimMag%.2lf_%srows_%siso_IsoModel_sig%s_IMF_%s_Simple.txt' % \
            (IDp,fw2_lim,str(self.N_dat),str(self.NIso),str(sig_i).replace('.','p'), self.IMF)    ## Opening file
        fp = open(os.path.join("pij_cij_results",filename_p),'a')
        args = []

         ## Pij is calcutated row by row, i.e. fix j-th dat and run each i-th isochrone.

        for j in range(self.N_dat):                       
    #                    0   1     2    3     4         5        6      7       8        9     10
            args.append([j, self.dat, self.NIso, self.Iso, 
                         fw1_lim, fw2_lim, fw3_lim, self.N_dat, filename_p, sig_i, self.IMF])
        with mp.Pool(mp.cpu_count()-1) as p:         ## Pooling Pij rows using all the abailable CPUs (Parallel computation)
            results = p.map(self.P_ij_row_map, args)
            Pij_out=[]
            for [j,wr] in results:
                fp.write('{}'.format(' '.join(wr))+'\n')
                Pij_out.append(array(wr, dtype=float))
        fp.close()
        
        return([Pij_out, filename_p])

    def P_ij_row_map(self, args):

        j = args[0]
        dat = args[1]
        Niso = args[2]
        Iso = args[3]

        fw1_lim = args[4]
        fw2_lim = args[5]
        fw3_lim = args[6]

        Ndat = args[7]
        filename_p = args[8]
        sig_i = args[9]
        imf = args[10]
        
        P_fw1 = self.Normal_MGk(self.dat[2][j],self.dat[3][j],sig_i)
        P_fw2 = self.Normal_MGk(self.dat[4][j],self.dat[5][j],sig_i)
        P_fw3 = self.Normal_MGk(self.dat[6][j],self.dat[7][j],sig_i)

        Phi_fw1 = self.Phi_MGk(self.dat[2][j], self.dat[3][j], fw1_lim, sig_i)
        Phi_fw2 = self.Phi_MGk(self.dat[4][j], self.dat[5][j], fw2_lim, sig_i)
        Phi_fw3 = self.Phi_MGk(self.dat[6][j], self.dat[7][j], fw3_lim, sig_i)

        wr=[]
        for i in range(self.NIso):                    ## Isochrone loop

            if self.IMF == "Krp":
                imf_p = self.IMF_Krp(self.Iso[i][1])
            elif self.IMF == "Slp":
                print("Not available at the moment")
                #imf_p = self.IMF_Salp(Iso[i][1])
            else:
                # Default
                imf_p = self.IMF_Krp(self.Iso[i][1])

            Ps = P_fw1(self.Iso[i][2])*P_fw2(self.Iso[i][3])*P_fw3(self.Iso[i][4])
            
            Phis = Phi_fw1(self.Iso[i][2])*Phi_fw2(self.Iso[i][3])*Phi_fw3(self.Iso[i][4])
            Intg = imf_p*Ps*Phis

            ## Interand
            p = trapz(Intg,self.Iso[i][1])

            wr.append(str(p))

        return ([j,wr])

    def C_ij_map(self, IDc):

        fw1_lim= self.fw_lims[0]
        fw2_lim= self.fw_lims[1]
        fw3_lim= self.fw_lims[2]
        
        sig_i = self.sig_fw[0]

        filename_c = '%s_Cij_Data_LimMag%.2lf_%srows_%siso_IsoModel_sig%s_IMF_%s_Simple.txt' % \
            (IDc,fw2_lim,str(self.N_dat),str(self.NIso),str(self.sig_fw[0]).replace('.','p'), self.IMF)
        fp = open(os.path.join("pij_cij_results",filename_c),'a')   ## output matrix
        args = []

        # Cij is calcutated row by row, i.e. fix j-th dat and run each i-th isochrone.

        for j in range(self.N_dat):    
            args.append([j, self.dat, self.NIso, self.Iso, fw1_lim, 
                         fw2_lim, fw3_lim, self.N_dat, filename_c, sig_i, self.IMF])

        with mp.Pool(mp.cpu_count()-1) as p:
            results = p.map(self.C_ij_row_map, args)
            Cij_out=[]
            for [j,wr] in results:
                fp.write('{}'.format(' '.join(wr))+'\n')
                Cij_out.append(array(wr, dtype=float))
        fp.close()
        
        return(Cij_out)

    def C_ij_row_map(self,args):

        j = args[0]
        dat = args[1]
        Niso = args[2]
        Iso = args[3]

        fw1_lim = args[4]
        fw2_lim = args[5]
        fw3_lim = args[6]

        Ndat = args[7]
        filename_c = args[8]
        sig_i = args[9]
        imf = args[10]

        phi_fw1 = self.Phi_MGk(dat[2][j], dat[3][j], fw1_lim, sig_i)
        phi_fw2 = self.Phi_MGk(dat[4][j], dat[5][j], fw2_lim, sig_i)
        phi_fw3 = self.Phi_MGk(dat[6][j], dat[7][j], fw2_lim, sig_i)

        wr = []
        for i in range(Niso):

            if imf == "Krp":
                imf_c = self.IMF_Krp(Iso[i][1])
            elif imf=="Slp":
                print("Not available at the moment")
                #imf_c = self.IMF_Salp(Iso[i][1])
            else:
                imf_c = self.IMF_Krp(Iso[i][1])

            intg_c = imf_c*phi_fw1(Iso[i][2])*phi_fw2(Iso[i][3])*phi_fw3(Iso[i][4])
            p_c = trapz(intg_c,Iso[i][1])

            wr.append(str(p_c))

        return ([j,wr])

    def ai_samp(self, ID, Name):


        ### Data for STAN ###
        dats = {'Nj' : self.N_dat,
                'Ni' : self.NIso,
                'Pij': self.P_ij,
                'Cij': self.C_ij  }

        ############ Running pystan ############

        sm = stan.build(code, data=dats, random_seed=1234)
        fit = sm.sample(num_samples=self.N_smp, num_chains=self.N_wlk, num_warmup=200)
        self.fit = fit
        a_sp = fit["a"].T

        ######### Saving the MCMC sample #########

        N_iso = len(a_sp[0])

        a_perc = array([ percentile(ai,[10,50,90]) for ai in a_sp.T])       ##  10th, 50th, 90th percentiles

        sfh=array([self.Z_age_isos[:,0], self.Z_age_isos[:,1], a_perc[:,0], a_perc[:,1], a_perc[:,2] ]).T

        ##
        hd='Z,Log_age,p10,p50,p90'
        filename = ID+"_ai"+Name+"_Niter"+str(len(a_sp))+".txt"
        savetxt(filename, sfh, header=hd, fmt="%.6f", delimiter=",",comments='')
        
        return filename

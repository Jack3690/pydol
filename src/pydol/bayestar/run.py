from .core import *
data_dir = Path(__file__).parent.joinpath('data')
import pandas as pd

class SFH(Base):
    def __init__(self,df=None,N_wlk=20, N_smp=500, fw1_lim=30.,fw2_lim=30., fw3_lim=30.0, 
                 A_fw1=0, A_fw2=0, A_fw3=0, sig_fw1=0.1,sig_fw2=0.1, sig_fw3=0.1, 
                 dismod=29.67,isofiles='', isodir=None, ph_sup=100,m_inf=0.1,
                 IMF='Krp',parallel=True):
        
        self.N_wlk, self.N_smp = N_wlk, N_smp
        
        self.A_fw1, self.A_fw2, self.A_fw3 = A_fw1, A_fw2, A_fw3
        
        self.fw_lims = array([fw1_lim,fw2_lim,fw3_lim])
        
        self.sig_fw = array([sig_fw1,sig_fw2,sig_fw3])
        
        if isodir is None:
            isodir = f'{data_dir}/test_files/Isochrone.test'
        
        ########### Reading Isochrones ###########
        if (isofiles != ''):
            filelist = []
            f = open(isofiles, "r")
            lines = f.readlines()
            for line in lines:
                filelist.append(line.replace('\n',''))
        
            self.filelist = filelist
            f.close()
        else:
            filelist = glob.glob(os.path.join(isodir, "*.isoc"))
            self.filelist = filelist
        
        #               F435W  F555W  F814W
        #   ph   mass   mag1   mag2   mag3  Z  log_age
        iso = array([ loadtxt(k) for k in sorted(filelist) ], dtype=object)
        
        self.ages = [float(i.split('Myr')[0].split('AGE')[1]) for i in self.filelist]
        
        self.Z_age_isos = array([ loadtxt(k)[0][-2:] for k in sorted(self.filelist) ], dtype=object)
        
        ph_sup = 100
        m_inf = 0.1
        
        for l in range(len(iso)):
            iso[l] = iso[l][where( (iso[l].T[1]>=m_inf) & (iso[l].T[0]<=ph_sup))]       ## mass Truncation & stellar phase Truc
            iso[l] = iso[l].T
        
        self.Iso = iso
        ################################### DATA #######################################
        
        #  0      1       2        3         4          5         6          7
        #  RA    DEC     fw1   fw1_error    fw2     fw2_error    fw3     fw3_error
        if df is None:
            df = pd.read_fwf(f"{data_dir}/test_files/data.test", sep=' ')
            col_dict = {
                        'F435Wmag'        : 'fw1',
                        'F435Wmag_err'    : 'fw1_error',
                        'F555Wmag'        : 'fw2',
                        'F555Wmag_err'    : 'fw2_error',
                        'F814Wmag'        : 'fw3',
                        'F814Wmag_err'    : 'fw3_error'}

            df = df.rename(columns=col_dict)
        keys = ['RA','DEC','fw1','fw1_error','fw2','fw2_error','fw3','fw3_error']
        try:
            dt = df[keys]
        except:
            print("Data Frame input keys: ", keys)
            raise Exception("Input data keys don't match!")
        
        step = int(1)
        dat = dt.values.astype(float)
        msg = "from %d... (FW1 <= %.2lf) (FW2 <= %.2lf) (FW3 <= %.2lf)" % (len(dat), fw1_lim, fw2_lim, fw3_lim)
        
        # Completeness Filtering
        dat = dat[where((dat[:,2] < fw1_lim) & (dat[:,4]  < fw2_lim) & (dat[:,6]  < fw3_lim) )]        # Truncate by apparent magnitude
        
        #dat = dat[where((dat[:,4]  < fw2_lim))]  
        w_dat = dat[:,0]
        
        # Adding Extinction
        dat[:,2] -= A_fw1+dismod
        dat[:,4] -= A_fw2+dismod
        dat[:,6] -= A_fw3+dismod
        
        print ("Selecting %d %s" % (len(dat), msg))
        dat = dat[::step]
        dat = dat
        self.dat = dat.T
        
        self.N_dat = len(self.dat[0])
        self.NIso = len(self.Iso)
        
        self.parallel = parallel
        self.IMF = IMF

    def __call__(self):
        ########################### Execution Routines #################################
        
        start = time.time()
        print ("Starting Pij, Cij computation")
        if (self.parallel):
            print ("\tParallel mode...")
        
            ID = str(int(time.time()))
        
            self.Pij_reslt = self.P_ij_map(ID)
            self.P_ij, Pij_name = self.Pij_reslt[0], self.Pij_reslt[1]
            
            Name = Pij_name[Pij_name.find("_Pij")+4:Pij_name.find(".txt")]
        
            self.C_ij = self.C_ij_map(ID)
        
        else:
            print ("\tSequential mode... Not available for the moment")
           # P_ij(dat, N_dat, r_int, iso, N_iso)
        print ("Finished                                ")
        
        
        end = time.time()
        
        elapsed = end - start
        
        print ("Elapsed time: %02d:%02d:%02d" % (int(elapsed / 3600.), int((elapsed % 3600)/ 60.), elapsed % 60))
        
        ###########################################
        filename = self.ai_samp(ID, Name)
        print("Completed!!!")
        return filename

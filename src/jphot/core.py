from .dolphot import Base
from .data import jwst_data, hst_data

class pydol(Base):
  def __init__(self, in_dir='.', regions={}, filters=['F115W'],
               inputs= ['JWST', 'HST'], 
               jwst_ext='i2d.fits', hst_ext='drc.fits', show_filters=False):
    
    if len(regions.keys())>1:
      self.regions = regions
    else:
      raise Exception('Regions dict not provided!')

    # Data
    dict_images = {}
    for i in inputs[:2]:
      if i == 'JWST':
        jwst_dict = jwst_data(in_dir, jwst_ext, show_filters)
        dict_images.update(jwst_dict)

      elif i == 'HST':
        hst_dict = hst_data(in_dir, hst_ext, show_filters)
        dict_images.update(hst_dict)
        
    super.__init__(dict_images=dict_images, regions=regions)

            
    self._nircam_filters = ['F115W','F150W', 'F200W']
    self._acs_filters = ['F435W', 'F555W', 'F814W']
    self._wfc3_filters = ['F275W', 'F336W']
                    self._nircam_filters = list(dict_images['NRCB3'].keys())
    self._acs_filters = list(dict_images['WFC'].keys())
    self._wfc3_filters = list(dict_images['UVIS'].keys())
  
    self.nircam = []
    self.acs = []
    self.wfc3 = []
  
    for filter in filters:
      if filter in self._nircam_filters and filter in list(dict_images['NRCB3'].keys()):
        self.nircam.append(filter)
        
      elif filter in self._acs_filters and filter in list(dict_images['WFC'].keys()) :
        self.acs.append(filter)
        
      elif filter in self._wfc3_filters and filter in list(dict_images['UVIS'].keys()):
        self.wfc3.append(filter)
  
      else:
        print(f'{filter} not supported!!!')
    
    def __call__(self,  out_dir='.', d=50, comp=False, skip_phot=False):
      self.out_dir = '.'
      for region in self.regions:
        for filter in self._nircam_filters:
          self.nircam_phot(out_dir, region, filter, d,
                      skip_phot=skip_phot, comp=comp) 

        for filter in self._acs_filters:
          self.acs_phot(out_dir, region, filter, d,
                      skip_phot=skip_phot, comp=comp) 

        for filter in self._wfc3_filters:
          self.wfc3_phot(out_dir, region, filter, d,
                      skip_phot=skip_phot, comp=comp) 
          
    def generate_colors(self):
      if len(self._nircam_filters)>1:
        generate_jwst_color(self.out_dir)

      if len(self._acs_filters)>1:
        generate_acs_color(self.out_dir)

      if len(self._wfc3_filters)>1:
        generate_wfc3_color(self.out_dir)
        
    

import os
import glob
from astropy.io import fits

def jwst_data(images_dir='.', ext='i2d.fits', show_filters=False):
  dict_images_jwst = {}
  
  images = sorted(glob.glob(os.path.join(images_dir, f"*/*{ext}")))
  
  for image in images:
  
      im = fits.open(image)
      f = im[0].header['FILTER']
      d = im[0].header['DETECTOR']
  
      if d == 'MULTIPLE':
          d = 'NRCB3'
  
      # Image
      if d not in dict_images_jwst.keys():
          dict_images_jwst[d] = {}
          
      if f not in dict_images_jwst[d].keys():
          dict_images_jwst[d][f] =  {'images': [image]}
      else:
          dict_images_jwst[d][f]['images'].append(image)
      
  if show_filters:
    print('Available JWST Detectors and Filters\n-------------------------------')
          
    for i in dict_images_jwst.keys():
        print(f'{i} :', list(dict_images_jwst[i].keys()))

  return dict_images_jwst

def hst_data(images_dir='.', ext='drc.fits', show_filters=False):
  
  images = sorted(glob.glob(os.path.join(images_dir, f"*/*{ext}")))
  
  dict_images_hst = {}
  
  for image in images:
      im = fits.open(image)
      if 'FILTER2' in im[0].header.keys():
          f = im[0].header['FILTER2']
      elif 'FILTER' in im[0].header.keys():
          f = im[0].header['FILTER']
      
      if 'CLEAR' in f:
          f = im[0].header['FILTER1']
          
      d = im[0].header['DETECTOR']
  
      # Image
      if d not in dict_images_hst.keys():
          dict_images_hst[d] = {}
          
      if f not in dict_images_hst[d].keys():
          dict_images_hst[d][f] =  {'images': [image]}
      else:
          dict_images_hst[d][f]['images'].append(image)    
        
  if show_filters: 
    print('Available HST Detectors and Filters\n-------------------------------')
          
    for i in dict_images_hst.keys():
        print(f'{i} :', list(dict_images_hst[i].keys()))
    
  return dict_images_hst

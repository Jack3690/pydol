import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval, LinearStretch, SqrtStretch, SquaredStretch, LogStretch, PowerStretch, PowerDistStretch, SinhStretch, AsinhStretch, ManualInterval

from skimage import color as ski_color
import colorsys
from skimage.exposure import rescale_intensity


scaling_fns={'linear':LinearStretch, 'sqrt':SqrtStretch, 'squared':SquaredStretch, 'log':LogStretch, 'power':PowerDistStretch, 'sinh':SinhStretch, 'asinh':AsinhStretch}

def adjust_gamma(array_in,gamma):
    """
    Replacement function for skimage.exposure.adjust_gamma, so that NaNs don't throw errors
    
    Parameters
    ----------
    array_in : array
        Input image array
    gamma : float  
        Gamma correction value
    
    Returns
    -------
    array
        Gamma-adjusted image values
    """
    return array_in**(float(gamma))


def greyRGBize_image(datin,rescalefn='linear',scaletype='abs',min_max=[None,None], gamma=2.2, checkscale=False):
    """
    ### Takes an image and returns 3-frame [R,G,B] (vals from 0...1)
    
    Parameters
    ----------
    datin : array 
        Input 2D image data array
    rescalefn : func 
        Function to use for rescaling intensity.  imscale.linear/sqrt/squared/log/power/sinh/asinh
    scaletype : str 
        'abs' for absolute values, 'perc' for percentiles
    min_max : list
        [min,max] vals to use in rescale.  if scaletype='perc', list the percentiles to use, e.g. [1.,95.]
    gamma : float 
        Value for gamma correction.  For combining colorized frames, use default gamma=2.2.  For inverse, use gamma=(1./2.2)
    checkscale : bool  
        True to bring up plot to check the new image scale.
    
    Returns
    -------
    array
        Greyscale RGB image, shape=[ypixels,xpixels,3]
    """
    if 'per' in scaletype.lower(): 
        if min_max==[None,None]: min_max=[0.,100.]
        minval,maxval=np.percentile(np.ma.masked_invalid(datin).compressed(),min_max)
    else: 
        minval=[np.nanmin(datin) if min_max[0] is None else min_max[0]][0]
        maxval=[np.nanmax(datin) if min_max[1] is None else min_max[1]][0]
    #Used specified rescaling function
    datscaled=(scaling_fns[rescalefn](a=10) + ManualInterval(vmin=minval,vmax=maxval))(datin)
    #datscaled=rescalefn(datin,vmin=minval,vmax=maxval)
    if gamma!=1: datscaled=adjust_gamma(datscaled,gamma)
    #Need to scale image between -1 and 1 if data type is float...
    datlinear=LinearStretch()(datscaled)
    #datlinear=imscale.linear(np.nan_to_num(datscaled))
    #Convert to RGB
    dat_greyRGB=ski_color.gray2rgb(datlinear)
    
    if checkscale : 
        plt.clf(); plt.close('all')
        fig0=plt.figure(0); 
        ax1=fig0.add_subplot(121); 
        plt.imshow(datin,interpolation='nearest',origin='lower',cmap='gist_gray'); 
        plt.title('Input Image')
        ax2=fig0.add_subplot(122, sharex=ax1, sharey=ax1); 
        plt.imshow(dat_greyRGB**(1./gamma),interpolation='nearest',origin='lower'); 
        plt.title('Scaled Image')
        plt.show(); #plt.clf(); plt.close('all')

    return dat_greyRGB

def rgb_to_hex(rgb): 
    """
    Converts RGB tuple to a hexadecimal string
    
    Parameters
    ----------
    rgb : tuple
        RGB tuple such as (256,256,256)
    Returns
    -------
    str
        Hexadecimal string such as '#FFFFFF'
    """    
    return '#%02x%02x%02x'%rgb 

def hex_to_rgb(hexstring):
    """
    Converts a hexadecimal string to RGB tuple 
    
    Parameters
    ----------
    hexstring : str
        Hexadecimal string such as '#FFFFFF'
    Returns
    -------
    tuple
        RGB tuple such as (256,256,256)
    """
    #From http://stackoverflow.com/a/214657
    hexstring = hexstring.lstrip('#')
    lv = len(hexstring)
    return tuple(int(hexstring[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

def hex_to_hsv(hexstring):
    """
    Convert a hexadecimal string to HSV tuple
    
    Parameters
    ----------
    hexstring : str
        Hexadecimal string such as '#3300FF'
    Returns
    -------
    tuple
        HSV tuple such as (0.7,1.,1.)
    """
    #See Wikipedia article for details -- https://en.wikipedia.org/wiki/HSL_and_HSV#From_HSV  .  Modified from colorsys.py,
    # HSV: Hue, Saturation, Value
    # H = position in the spectrum, S = color saturation ("purity"), V = color brightness
    r,g,b=np.array(hex_to_rgb(hexstring))/255. #Convert Hex to RGB fracs (i.e., 0..1 instead of 0..255)
    maxc = max(r,g,b); minc = min(r,g,b); 
    s = (maxc-minc) / maxc
    v = maxc
    if minc == maxc: return 0.0, 0.0, v
    rc = (maxc-r) / (maxc-minc);   gc = (maxc-g) / (maxc-minc);   bc = (maxc-b) / (maxc-minc)
    if r == maxc: h = bc-gc
    elif g == maxc: h = 2.0+rc-bc
    else: h = 4.0+gc-rc
    h = (h/6.0) % 1.0
    return h, s, v #All in fractions in range [0...1]

def colorize_image(image, colorvals, colorintype='hsv',dtype=np.float64,gammacorr_color=1):
    """
    ### Add color of the given hue to an RGB greyscale image.
    
    Parameters
    ----------
    image : array
        Greyscale RGB image -- as would be output from greyRGBize_image()
    colorvals : str or list or tuple 
        color values to apply to image.  e.g., '#FF0000' if colorintype='hex'
    colorintype : str 
        'hsv' for [0..1,0..1,0..1],  'rgb' for [0..255,0..255,0..255], or 'hex' for '#XXXXXX'
    dtype : dtype 
        Defaults to standard numpy float, but may want to lower to e.g. float32 for large images (>~1000x1000)
    gammacorr_color : float
        To use color as-is, leave as 1 (default).  To gamma-correct color at this step (e.g., to match gamma for checking a scaled image), specify a factor
    
    Returns
    -------
    array
        Colorized RGB image, shape=[ypixels,xpixels,3]
    """
    if colorintype not in ['hsv', 'hsv_dict', 'rgb', 'hex']: raise Exception("  colorintype must be 'hsv', 'hsv_dict', 'rgb', or 'hex'")
    hsv = ski_color.rgb2hsv(image).astype(dtype)
    if colorintype.lower()=='rgb': colorvals=np.array(hex_to_hsv(rgb_to_hex(colorvals))).astype(dtype)
    elif colorintype.lower()=='hex': colorvals=np.array(hex_to_hsv(colorvals)).astype(dtype) #from custom_colormaps.py
    if colorintype.lower()=='hsv_dict': hue,saturation,v=colorvals['hue'],colorvals['sat'],colorvals['v'],
    else: hue,saturation,v=colorvals
    if gammacorr_color!=1: 
        hue,saturation,v = colorsys.rgb_to_hsv( *np.array( colorsys.hsv_to_rgb(hue,saturation,v) )**gammacorr_color )
    hsv[:, :, 2] *= v 
    hsv[:, :, 1] = saturation
    hsv[:, :, 0] = hue
    return ski_color.hsv2rgb(hsv).astype(dtype)

def combine_multicolor(im_list_colorized,gamma=2.2,inverse=False):
    """
    Combines input colorized RGB images [:,:,3] into one intensity-rescaled RGB image
    
    Parameters
    ----------
    im_list_colorized : list 
        List of colorized RGB images.  e.g., [ halpha_purple, co21_orange, sio54_teal ]
    gamma : float 
        Value used for gamma correction ^1/gamma.  Default=2.2.  
    inverse : bool  
        True will invert the scale so that white is the background
    
    Returns
    -------
    array
        Colorized RGB image (combined), shape=[ypixels,xpixels,3]
    """
    combined_RGB=LinearStretch()(np.nansum(im_list_colorized,axis=0))
    if inverse==True: RGB_maxints=tuple(1.-np.nanmax(combined_RGB[:,:,i]) for i in [0,1,2])
    else: RGB_maxints=tuple(np.nanmax(combined_RGB[:,:,i]) for i in [0,1,2])
    for i in [0,1,2]: 
        combined_RGB[:,:,i]=np.nan_to_num(rescale_intensity(combined_RGB[:,:,i], out_range=(0, combined_RGB[:,:,i].max()/np.max(RGB_maxints) )));
    combined_RGB=LinearStretch()(combined_RGB**(1./gamma)) #gamma correction
    if inverse==True: combined_RGB=1.-combined_RGB #gamma correction
    return combined_RGB
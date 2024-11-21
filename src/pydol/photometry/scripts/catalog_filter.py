import numpy as np
import pandas as pd
from astropy.table import Table
import astropy.units as u
from astropy.coordinates import SkyCoord, AltAz, SkyOffsetFrame
import matplotlib.pyplot as plt
from astropy.coordinates import Angle

def box(catalog_data,ra_column, dec_column, ra_center, dec_center,
        width_in=24/3600, height_in=24/3600, 
        width_out=24/3600, height_out=24/3600, 
        angle=245.00492):
    
    # Convert catalog RA and Dec to SkyCoord object
    coords = SkyCoord(ra=catalog_data[ra_column].value * u.deg,
                      dec=catalog_data[dec_column].value * u.deg, frame='icrs')

    # Convert the center to a SkyCoord object
    center = SkyCoord(ra=ra_center * u.deg, dec=dec_center * u.deg, frame='icrs')

    # Project the coordinates to the tangent plane centered on 'center'
    tan_plane = SkyOffsetFrame(origin=center)
    proj_coords = coords.transform_to(tan_plane)

    # Calculate offsets in the tangent plane
    offset_ra = proj_coords.lon.to(u.deg).value  # Offset RA in degrees
    offset_dec = proj_coords.lat.to(u.deg).value # Offset Dec in degrees

    # Calculate rotation matrix
    theta = np.deg2rad(angle)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotation_matrix = np.array([[cos_theta, -sin_theta],
                                [sin_theta, cos_theta]])

    # Apply rotation to offsets
    rotated_offsets = np.dot(rotation_matrix, np.vstack((offset_ra, offset_dec)))

    # Define half-widths for filtering
    half_width_in = width_in / 2
    half_height_in = height_in / 2
    half_width_out = width_out / 2
    half_height_out = height_out / 2

    # Filter points within the rotated rectangle
    mask = (((np.abs(rotated_offsets[0]) >= half_width_in)  |
            (np.abs(rotated_offsets[1]) >= half_height_in)) &
            (np.abs(rotated_offsets[0]) <= half_width_out) &
            (np.abs(rotated_offsets[1]) <= half_height_out))

    # Extract the filtered data
    filtered_catalog = catalog_data[mask]

    # Print the number of selected objects
    print(f"Number of objects in the selected region: {len(filtered_catalog)}")
    
    return filtered_catalog

def ellipse(catalog_data, ra_column, dec_column, ra_center, dec_center, angle=0, a1=1, b1=0,
           a2=1,b2=1):

    # Convert catalog RA and Dec to SkyCoord object
    coords = SkyCoord(ra=catalog_data[ra_column].value * u.deg,
                      dec=catalog_data[dec_column].value * u.deg, frame='icrs')

    # Convert the center to a SkyCoord object
    center = SkyCoord(ra=ra_center * u.deg, dec=dec_center * u.deg, frame='icrs')

    # Project the coordinates to the tangent plane centered on 'center'
    tan_plane = SkyOffsetFrame(origin=center)
    proj_coords = coords.transform_to(tan_plane)

    # Calculate offsets in the tangent plane
    offset_ra = proj_coords.lon.to(u.deg).value  # Offset RA in degrees
    offset_dec = proj_coords.lat.to(u.deg).value # Offset Dec in degrees
    
    # Calculate rotation matrix
    theta = np.deg2rad(angle)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotation_matrix = np.array([[cos_theta, -sin_theta],
                                [sin_theta, cos_theta]])

    # Apply rotation to offsets
    rotated_offsets = np.dot(rotation_matrix, np.vstack((offset_ra, offset_dec)))

    # Filter points within the rotated ellipse
    mask = (rotated_offsets[0]**2/a1**2 + rotated_offsets[1]**2/b1**2  >= 1)
    mask = mask &  (rotated_offsets[0]**2/a2**2 + rotated_offsets[1]**2/b2**2  <= 1)

    # Extract the filtered data
    filtered_catalog = catalog_data[mask]
    
    # Print the number of selected objects
    print(f"Number of objects in the selected region: {len(filtered_catalog)}")
    
    return filtered_catalog

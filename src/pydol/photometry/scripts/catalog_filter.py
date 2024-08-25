import numpy as np
import pandas as pd
from astropy.table import Table
import astropy.units as u
from astropy.coordinates import SkyCoord, AltAz, SkyOffsetFrame
import matplotlib.pyplot as plt
from astropy.coordinates import Angle

# Load the local catalog (replace 'local_catalog.csv' with your actual file)

def box(catalog_data,ra_column, dec_column, ra_center, dec_center,
        width=24/3600, height=24/3600, angle=245.00492):
    
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
    half_width = width / 2
    half_height = height / 2

    # Filter points within the rotated rectangle
    mask = ((np.abs(rotated_offsets[0]) <= half_width) &
            (np.abs(rotated_offsets[1]) <= half_height))

    # Extract the filtered data
    filtered_catalog = catalog_data[mask]

    # Print the number of selected objects
    print(f"Number of objects in the selected region: {len(filtered_catalog)}")
    
    return filtered_catalog


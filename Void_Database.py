#!/usr/bin/env python
# coding: utf-8

# Code explanation:

# In[1]:


from sklearn.cluster import DBSCAN
from scipy import ndimage
import gzip
import os
from skimage.feature import canny # NEW as of July 15
from skimage.segmentation import find_boundaries # NEW

import glob
import shutil
import gzip
import pandas as pd
import netCDF4 as nc
from skimage.feature import canny
from skimage.segmentation import find_boundaries
from skimage.draw import circle_perimeter
from skimage.transform import hough_circle, hough_circle_peaks
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops, find_contours
import numpy as np
from matplotlib.patches import Circle

from skimage.filters import sobel
from skimage.filters import rank
from skimage.morphology import disk 
from skimage.util import img_as_ubyte
from skimage.filters.rank import median, gradient
from skimage.morphology import local_minima
from scipy.ndimage import label as ndi_label

from skimage.measure import regionprops
from scipy.ndimage import minimum_filter
from skimage.segmentation import watershed
import cv2
from skimage.morphology import local_minima
from sklearn.cluster import DBSCAN

from sklearn.cluster import KMeans
import matplotlib

import re
import xarray as xr
import datetime
from datetime import datetime


# In[2]:


# Regex pattern for extracting day of year from the filename
pattern = re.compile(r'.*_(\d{4})-(\d{3})_.*')

# Iterate over pairs of files
last_doy=-1


# In[ ]:





# In[2]:


os.environ['PROJ_LIB'] = '/mnt/home/anaise.gaillard/.conda/envs/aim/share/proj'

def gunzip_shutil(source_filepath, dest_filepath, block_size=1024*1024):#parameters source_filepath (path to compressed file,
    # dest_filepath (path to save uncompressed file), block_size (size of blocks to read at a time--defauls 1 MB)
#open source file in read-binary mode ('rb') and destination file in write-binary mode ('wb') the with statement ensures that both files are properly closed after the operation
    with gzip.open(source_filepath, 'rb') as s_file, \
        open(dest_filepath, 'wb') as d_file:
        shutil.copyfileobj(s_file, d_file, block_size) # copy the content from the source file to the destination file in chunks of 'block-size' bytes



# In[3]:


def detect_regions_dbscan(cloud_albedo, eps=0.35, min_samples=15, min_size=100):
    albedo_flat = cloud_albedo.flatten()
    coordinates = np.array([(i, j) for i in range(cloud_albedo.shape[0]) for j in range(cloud_albedo.shape[1])])
    
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coordinates, sample_weight=albedo_flat)
    labels = clustering.labels_.reshape(cloud_albedo.shape)
    
    # Label the regions
    labeled_regions = label(labels == -1)
    
    # Filter regions by size
    regions_map = np.zeros_like(labels, dtype=int)
    props = regionprops(labeled_regions)
    
    for prop in props:
        if prop.area >= min_size:
            regions_map[labeled_regions == prop.label] = 1
    
    return regions_map


# In[4]:


def detect_voids_threshold(cloud_albedo, threshold=2, min_size=100):
    binary_albedo = np.where(cloud_albedo > threshold, 0, 1)
    dilated_albedo = ndimage.binary_dilation(binary_albedo, structure=np.ones((3,3)))
    border = dilated_albedo - binary_albedo
    surrounded_regions = dilated_albedo - border
    labeled_regions, num_regions = ndimage.label(surrounded_regions)
    region_sizes = ndimage.sum(surrounded_regions, labeled_regions, range(1, num_regions+1))
    region_mask = np.isin(labeled_regions, np.where((region_sizes > min_size))[0] + 1)
    
    return region_mask


# In[5]:


def detect_regions_kmeans(cloud_albedo, n_clusters=4):
    albedo_flat = cloud_albedo.flatten()
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(albedo_flat.reshape(-1, 1))
    cluster_labels = kmeans.labels_ + 1  # Add 1 to labels to avoid 0 as background
    cluster_labels_2d = cluster_labels.reshape(cloud_albedo.shape)
    
    return cluster_labels_2d


# In[6]:


def detect_voids_blob_detector(cloud_albedo):
    from skimage.feature import blob_dog
    from skimage.draw import circle_perimeter
    
    blobs = blob_dog(cloud_albedo, max_sigma=30, threshold=0.1)
    regions_map = np.zeros_like(cloud_albedo, dtype=int)
    for blob in blobs:
        y, x, r = blob
        rr, cc = circle_perimeter(int(y), int(x), int(r))
        regions_map[rr, cc] = 1
    
    return regions_map


# In[7]:


def detect_watershed_albedo_(cloud_albedo, latitude, min_size=100):
    # Normalize cloud albedo by latitude
    #cloud_albedo_norm = normalize_by_latitude(cloud_albedo, latitude)
    
    # normalize between 0 and 1
    cloud_albedo_norm = min_max_normalize(cloud_albedo)
    
    gradient = sobel(cloud_albedo)
                     
    #try local minima of cloud albedo, and then of norm cloud
    local_min = local_minima(cloud_albedo) # could put gradient here, or normalized albedo

    # Label the markers
    markers = label(local_min) #, background = -1000) # label nans as 0 in the labeling process

    # Apply the watershed algorithm using the gradient image and the markers
    labels = watershed(cloud_albedo, markers, watershed_line = False) # try without markers?

    # Filter regions based on size
    valid_labels = []
    for region in regionprops(labels):
        if region.area >= min_size:
            valid_labels.append(region.label)

    # Create a mask for the valid regions
    region_mask = np.isin(labels, valid_labels) # use this instead of direct labels bc this filters out really small and really big regions


    return labels


# In[8]:


def calculate_region_properties(regions_map, cloud_albedo, min_size=100):
    labeled_regions, num = label(regions_map, return_num=True)
    properties = regionprops(labeled_regions, intensity_image=cloud_albedo)
    
    region_info = []
    for prop in properties:
        if prop.area < min_size:
            continue
        # Circularity: 4*pi*Area / Perimeter^2
        if prop.perimeter > 0:
            circularity = (4 * np.pi * prop.area) / (prop.perimeter ** 2)
        else:
            circularity = 0
        region_info.append({
            'region_label': prop.label,
            'size': prop.area,
            'center': prop.centroid,
            'circularity': circularity,
            'min_albedo': prop.intensity_min,
            'max_albedo': prop.intensity_max
        })
    return pd.DataFrame(region_info)


# In[9]:


# save images of each orbit on each method
def plot_albedo_with_regions_contours(cloud_albedo, regions_map, title='', min_size=100, output_dir='./output', orbit='', method_name=''):
    plt.figure(figsize=(14, 8))
    
    # Original albedo plot
    plt.subplot(3, 1, 1)
    plt.imshow(cloud_albedo, cmap='jet')
    plt.colorbar(label='Cloud Albedo')
    plt.title('Cloud Albedo')
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')
    
    # Albedo plot with regions
    plt.subplot(3, 1, 2)
    plt.imshow(cloud_albedo, cmap='gray')
    plt.colorbar(label='Cloud Albedo')
    plt.title('Detected Regions')
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')
    
    # Labeled regions
    labeled_regions = label(regions_map)
    props = regionprops(labeled_regions)
    
    colors = [
            '#FF00FF',  # Magenta
            '#00FFFF',  # Cyan
            '#FFFF00',  # Yellow
            '#00FF00',  # Lime
            '#FF0000',  # Red
            '#0000FF',  # Blue
            '#FF6600',  # Orange
            '#CCFF00',  # Electric Lime
            '#FF33CC',  # Pink Flamingo
            '#33FF33',  # Neon Green
            '#6633FF',  # Electric Indigo
            '#FF3399',  # Neon Pink
        ]
        #for i, prop in enumerate(props):
        #    if prop.area >= min_size:
        #        color = colors[i % len(colors)]
        #        coords = np.array(prop.coords)
        #        plt.plot(coords[:, 1], coords[:, 0], color=color)
                
    for i, prop in enumerate(props):
        if prop.area >= min_size:
            color = colors[i % len(colors)]
            contours = find_contours(labeled_regions == prop.label, 0.5)
            for contour in contours:
                plt.plot(contour[:, 1], contour[:, 0], color=color)
    
    ### Contours and Hough Transform
    plt.subplot(3, 1, 3)
    plt.imshow(cloud_albedo, cmap='gray')
    plt.colorbar(label='Cloud Albedo')
    plt.title('Detected Regions with Hough Transform Circles')
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')
    
    edges = np.zeros_like(labeled_regions, dtype=np.uint8)
    
    for i, prop in enumerate(props):
        if prop.area >= min_size:
            # Extract the region of interest from the labeled regions
            region_mask = (labeled_regions == prop.label)
            
            # Find contours
            contours = find_contours(region_mask, level=0.5) # try region_map instead of region_mask
            
            for contour in contours:
                contour = np.round(contour).astype(int)
                edges[contour[:, 0], contour[:, 1]] = 1
                

            
    if np.any(edges):       
                # Adaptive number of peaks based on region area
        total_num_peaks = 5 #int(min(10, max(1, area // 100)))  # Ensure this is an integer

                # Hough Transform for circles
        hough_radii = np.arange(20, 300)
        hough_res = hough_circle(edges.astype(np.uint8), hough_radii) 
                
        accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, min_xdistance = 20, min_ydistance = 20,total_num_peaks = 5)
    
        for center_y, center_x, radius in zip(cy, cx, radii): 
            circle = Circle((center_x, center_y), radius=radius, fill=False, color='red', linewidth=2)
            plt.gca().add_patch(circle)  # Add circle to the current axis (3rd subplot)
    
    
    plt.suptitle(title)
    plt.tight_layout()  # Adjust layout to prevent overlap
    
     # Save the plot
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_path = os.path.join(output_dir, f"{orbit}_{method_name}.png")
    plt.savefig(file_path)
    plt.close()
    
    return file_path


# In[10]:


def create_hough_circle_dataframe(edges, min_radius=20, max_radius=300, step=1):
    if np.any(edges):  # Only if there are edges detected
        hough_radii = np.arange(min_radius, max_radius, step)  # Define a range of radii to search for circles
        hough_res = hough_circle(edges.astype(np.uint8), hough_radii)  # Apply Hough Transform
        accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, min_xdistance=20, min_ydistance=20, total_num_peaks=5)  # Find peaks in the Hough space
        
        circle_data = {
            'center_x': cx,
            'center_y': cy,
            'radius': radii,
            'size_Area': np.pi * radii**2
        }
    else:
        circle_data = {
            'center_x': [None],
            'center_y': [None],
            'radius': [None],
            'size_Area': [None]
        }
    
    return pd.DataFrame(circle_data)



# In[11]:


def process_orbit(orbit, catfile, cldfile, method_func, methods, min_size=100):
    
    # Extract the year and day of year from the filename
    match = pattern.match(catfile)
    year, day_of_year = map(int, match.groups())

    # Check if the day of year is in the desired range
    if not (151 <= day_of_year <= 239):
        return None  # Skip processing if not in the desired range

    
    # Load cloud data and latitude data
    gunzip_shutil(catfile, 'geolocation.nc')
    gunzip_shutil(cldfile, 'cloud.nc')

    cloud_data = nc.Dataset('cloud.nc')
    geo_data = nc.Dataset('geolocation.nc')
    
    latitude = geo_data['LATITUDE'][:]
    cloud_albedo = cloud_data['CLD_ALBEDO'][:]
    #cloud_max = np.nanmax(cloud_albedo)
    cloud_albedo_nan = np.nan_to_num(cloud_albedo, nan= -1000) # try 0 for nans
     # Normalize cloud albedo data
    albedo_min = np.nanmin(cloud_albedo_nan)
    albedo_max = np.nanmax(cloud_albedo_nan)
    cloud_albedo_normalized = (cloud_albedo_nan - albedo_min) / (albedo_max - albedo_min)
    
    # Get the date from the geolocation file
    ut_date_str = geo_data['UT_DATE'][:].tobytes().decode().strip()[:8]  # Take only the first 8 characters and remove any trailing characters
    date = datetime.strptime(ut_date_str, '%Y%m%d').date()  # parse string to date object
    
    
    
    method_data = {}
        # Apply segmentation method
    regions_map = method_func(cloud_albedo_nan, min_size=min_size)
    region_properties_df = calculate_region_properties(regions_map, cloud_albedo_nan)
    # Labeled regions
    labeled_regions = label(regions_map)
    props = regionprops(labeled_regions)
   
    edges = np.zeros_like(labeled_regions, dtype=np.uint8)
        # Create edge map for Hough transform
    for i, prop in enumerate(props):
        if prop.area >= min_size:
            # Extract the region of interest from the labeled regions
            region_mask = (labeled_regions == prop.label)
            
            # Find contours
            contours = find_contours(region_mask) # try find_boundaries
            
            for contour in contours:
                contour = np.round(contour).astype(int)
                edges[contour[:, 0], contour[:, 1]] = 1
                
        # region properties dataframe
        region_properties_df = calculate_region_properties(regions_map, cloud_albedo_nan)
        
        # Run Hough transform and create DataFrame
        hough_dataframe = create_hough_circle_dataframe(edges)
        
        # Add to method data
        method_data[prop.label] = hough_dataframe 
        
    # Add region properties DataFrame to method data
    method_data['region_properties'] = region_properties_df
    
    image_paths = []
    for method_name, method_func in methods.items():
        file_path = plot_albedo_with_regions_contours(cloud_albedo, regions_map, 
        title=f'Orbit {orbit} -Method {method_name}', min_size=min_size, output_dir='./output',
        orbit=orbit, method_name=method_name)
        image_paths.append({'orbit': orbit, 'method': method_name, 'file_path': file_path, 'date': date})
    
    images_df = pd.DataFrame(image_paths)
    method_data['images'] = images_df
    
    return method_data



# In[1]:


def main():
    
    dir_path = '/mnt/home/anaise.gaillard/interim_archive/cips/data/PMC/north_2007/level_2/ver_05.20/rev_05' # honga tonga

    # Collect all .gz files in the directory
    all_files = glob.glob(os.path.join(dir_path, '*.gz'))

    # Sort files into cat_files and cld_files
    cat_files = [f for f in all_files if '_cat.nc.gz' in f]
    cld_files = [f for f in all_files if '_cld.nc.gz' in f]

    # Sort both lists to make sure paired files are in the same order
    cat_files.sort()
    cld_files.sort()

    # Assert that there is an equal number of cat_files and cld_files
    assert len(cat_files) == len(cld_files), 'Unequal number of cat_files and cld_files!'

    # Create a dictionary of orbits dynamically
    orbits_dic = {f'orbit_{i:05d}': (cat_files[i], cld_files[i]) for i in range(len(cat_files))}

    methods = {
        "DBSCAN": detect_regions_dbscan,
        "KMeans": detect_regions_kmeans,
        "Threshold": detect_voids_threshold,
        "Watershed": detect_watershed_albedo_
    }

    all_data_season = {}
    
    for method_name, method_func in methods.items():
        method_data = {}
        for orbit, (catfile, cldfile) in orbits_dic.items():
            result = process_orbit(orbit, catfile, cldfile, method_func, methods)
            if result is not None:
                method_data[orbit] = result
        
        all_data_season[method_name] = method_data
    
    # Convert all data to a single nested dictionary structure
    return all_data_season

# Run the main function
all_data_season = main()



# In[ ]:


def save_database_to_netcdf(database, filename):
    with nc.Dataset(filename, 'w', format='NETCDF4') as dataset:
        for method_name, method_data in database.items():
            method_group = dataset.createGroup(method_name)
            
            for orbit, orbit_data in method_data.items():
                orbit_group = method_group.createGroup(orbit)
                
                # Save region properties
                region_properties = orbit_data['region_properties']
                labels = region_properties["region_label"].values
                sizes = region_properties["size"].values
                locations = region_properties["location"].apply(lambda x: list(x)).values
                circularities = region_properties["circularity"].values
                min_albedos = region_properties["min_albedo"].values
                max_albedos = region_properties["max_albedo"].values
                
                orbit_group.createDimension('regions', len(labels))
                
                region_labels = orbit_group.createVariable('Region_Label', 'i4', ('regions',))
                region_sizes = orbit_group.createVariable('Size_pixels', 'i4', ('regions',))
                region_locations = orbit_group.createVariable('Location', 'f4', ('regions', 2))
                region_circularities = orbit_group.createVariable('Circularity', 'f4', ('regions',))
                region_min_albedos = orbit_group.createVariable('Min_Albedo', 'f4', ('regions',))
                region_max_albedos = orbit_group.createVariable('Max_Albedo', 'f4', ('regions',))
                
                region_labels[:] = labels
                region_sizes[:] = sizes
                region_locations[:, :] = np.array(list(locations))
                region_circularities[:] = circularities
                region_min_albedos[:] = min_albedos
                region_max_albedos[:] = max_albedos

                # Save images paths
                images_df = orbit_data['images']
                image_paths = images_df['file_path'].values
                image_orbits = images_df['orbit'].values
                image_methods = images_df['method'].values
                
                orbit_group.createDimension('images', len(image_paths))
                
                img_paths = orbit_group.createVariable('Image_Path', str, ('images',))
                img_orbits = orbit_group.createVariable('Image_Orbit', str, ('images',))
                img_methods = orbit_group.createVariable('Image_Method', str, ('images',))
                
                img_paths[:] = image_paths
                img_orbits[:] = image_orbits
                img_methods[:] = image_methods

methods = {
    "DBSCAN": detect_regions_dbscan,
    "Threshold": detect_voids_threshold,
    "Watershed": detect_watershed_albedo_,
    "KMeans": detect_regions_kmeans
}

img_segmentation_database = main()
save_database_to_netcdf(img_segmentation_database, 'cloud_regions_database.nc')


# In[ ]:





# In[ ]:
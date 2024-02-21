import cv2
import numpy as np
import pandas as pd
import scipy
import napari
import matplotlib.pyplot as plt
import pyclesperanto_prototype as cle
import skimage.io
import argparse
import os

from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage import filters, morphology, measure, segmentation

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--volTmpSeq', type=str, required=True)
    parser.add_argument('--coordsFile',type=str)
    args= parser.parse_args()
    
    # Selecting CPU/GPU to run analysis on
    device = cle.select_device("Intel(R)")
    
    # Loading data - reconstructed volume sequence
    matfile = scipy.io.loadmat(args.volTmpSeq)
    data = matfile['volTmpSeq_']

    # Taking standard deviation of data across time axis
    stds = np.std(data,axis=0,dtype=np.float32)
    #stds = np.squeeze(data)
    [A,B,C] = stds.shape
    
    # Cropping out edges
    stds = stds[:,3*13+1:B-3*13,3*13+1:C-3*13]

    voxel_size_x = 1
    voxel_size_y = 1
    voxel_size_z = 1

    input_gpu = cle.push(stds)
    
    resampled = cle.create([int(input_gpu.shape[0] * voxel_size_z), int(input_gpu.shape[1] * voxel_size_y), int(input_gpu.shape[2] * voxel_size_x)])
    cle.scale(input_gpu, resampled, factor_x=voxel_size_x, factor_y=voxel_size_y, factor_z=voxel_size_z, centered=False)
    
    
    # Equalizing intensity
    equalized_intensities_stack = cle.create_like(resampled)
    a_slice = cle.create([resampled.shape[1], resampled.shape[2]])

    num_slices = resampled.shape[0]
    mean_intensity_stack = cle.mean_of_all_pixels(resampled)
    corrected_slice = None
    for z in range(0, num_slices):
        # get a single slice out of the stack
        cle.copy_slice(resampled, a_slice, z)
        # measure its intensity
        mean_intensity_slice = cle.mean_of_all_pixels(a_slice)
        # correct the intensity
        correction_factor = mean_intensity_slice/mean_intensity_stack
        corrected_slice = cle.multiply_image_and_scalar(a_slice, corrected_slice, correction_factor)
        # copy slice back in a stack
        cle.copy_slice(corrected_slice, equalized_intensities_stack, z)
    
    no_bg = cle.top_hat_box(equalized_intensities_stack, radius_x=5, radius_y=5, radius_z=5)

    # Voronoi otsu labeling
    # spot_sigma/outline_sigma determines how many neurons are found 
    segmented = cle.voronoi_otsu_labeling(no_bg, spot_sigma=5, outline_sigma=1) 

    # Getting location of neurons
    coordinates = []

    info_table = pd.DataFrame(
        measure.regionprops_table(
            segmented,
            intensity_image=stds,
            properties=['label', 'slice', 'coords'],
        )
    ).set_index('label')

    num_neurons = len(info_table['slice'])

    for i in range(1,num_neurons+1):
        temp_coord = []
        for j in range(0,3):
            temp_xyz = [getattr(info_table['slice'][i][j],'start'),getattr(info_table['slice'][i][j],'stop')]
            temp_coord.append(temp_xyz)
        print(temp_coord)
        
        coordinates.append(temp_coord)
    print(coordinates)

    # Optional - saving coordinate locations
    #np.save('coordinates.npy',coordinates)

    # ------------------------------------------------------------------------------
    # Creating synthetic volumes 
    [A,B,C,D] = data.shape
    E = len(coordinates)
    total_vol = np.zeros((E,C,D,B))

    # If coordinates are in a separate file
    # coords = np.load(args.coords_file)
    factor = -3+3*13
    for i in range(0,3):
        z = coordinates[i][0]
        y = coordinates[i][1]
        x = coordinates[i][2]
        
        temp = data[:,z[0]:z[1],y[0]+factor:y[1]+factor,x[0]+factor:x[1]+factor]
        temp = np.mean(temp,axis=0,dtype=np.float32)
        vol = np.zeros([B,C,D])
        vol[z[0]:z[1],y[0]+factor:y[1]+factor,x[0]+factor:x[1]+factor] = temp
        vol = np.transpose(vol,[1,2,0])
        vol = vol.astype(np.uint8)
        total_vol[i,:,:,:] = vol
    
    #np.save('synth_LFvol.npy',total_vol)

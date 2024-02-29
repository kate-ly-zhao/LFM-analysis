import cv2
import os
import glob
import torch
import scipy
import random
import scipy.io
import argparse
import skimage.io
import hdf5storage
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyclesperanto_prototype as cle
import torch.optim as optim
import torch.backends.cudnn as cudnn


from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage import filters, morphology, measure, segmentation
from pickle import dump,load
from torch import nn
from tqdm import tqdm
from random import randint

from utils import AverageMeter, calc_psnr
from modelsAdv import *

np.random.seed(123)
torch.manual_seed(123)
random.seed(123)

# Previously:
# volTmpSeq.mat -> neuronFinding.py -> synthLFvol.npy
# synthLFvol.npy, epochFl.pth -> vol2footprint.py -> lfSeqAll.npy
# lfSeqAll.npy, LF.mat -> timeSeqFinder.py -> nice little plot of time sequence

# Inputs: volTmpSeq, epochFl.pth, LF.mat, size of sigma (?)
# Outputs: synthLFvol.npy, lfSeqAll.npy, matrix of time sequences (?), png plot of time sequences

# python3 entireSystem_v3.py --weights-fileFl "./epochFl_24.pth" --weights-fileG "./epochG_40.pth" --fileLF "./LF_S2A3.mat"

if __name__ == '__main__':
    
    # Reading in arguments

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-fileFl', type=str, required=True)
    parser.add_argument('--weights-fileG', type=str, required=True)
    parser.add_argument('--fileLF', type=str, required=True)
    
    parser.add_argument('--volTmpSeq', type=str)
    parser.add_argument('--coordsFile',type=str)
    parser.add_argument('--spotSigma',type=int,default=5)
    parser.add_argument('--outlineSigma',type=int,default=1)
    args= parser.parse_args()

    # ---------- taking stdev of LF first ----------
    # To try: using mean instead of stdev, not normalizing

    print('Taking standard deviation of raw light field data')

    matfile = scipy.io.loadmat(args.fileLF)
    LF = matfile['lfTrainTmp'] # lfTrainTmpd1 for S1A2
    
    """stdevs = np.std(LF,axis=0,dtype=np.float32)

    # Normalize range values and type before saving
    #mn = stdevs.min()
    mx = stdevs.max()
    #mx -= mn
    stdevs = stdevs / mx * 255 #stdevs = ((stdevs - mn)/mx) * 255
    stdevs = stdevs.round().astype(np.uint8) #stdevs = stdevs.astype(np.uint8)"""

    S = LF

    # this is the value reported by the camera pixels in the dark
    drkct = 140.

    # filter each pixel in the time dimension
    Sfilt = ndi.uniform_filter(S, size=6, axes=(0))

    # get the minimum of each pixel
    smin = Sfilt.min(axis=0)

    # calculate the baseline fluorescence of each pixel (min-dark)
    baseline = smin - drkct

    # allow only positive values
    baseline[baseline<=0] = 0.01

    # subtract the filtered from the raw time series to isolate the shot noise, then divide by baseline
    N = (S - Sfilt)/baseline

    # Calculate dF/F
    dff = (Sfilt - smin)/baseline

    # Calculate the peak (max) dF/F in each pixel
    dff_max = dff.max(axis=0)

    # Calculate the baseline-normalized noise in each pixel
    noise = np.std(N,axis=0)

    # Calculate the temporal SNR
    snr = dff_max / noise

    # Rescale and cast to uint8
    snr8 = (snr / snr.max() * 255.).round().astype(np.uint8)
    #stdevs = snr8

    # Expanding an extra dimension to make reconstruction easy
    #stdevs = np.expand_dims(stdevs, 0)

    # ---------- reconstructing stdev LF ----------
    # To-do: add a check to see if a volume is already given

    print('Reconstructing standard deviation LF image')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Parameters
    s=3
    N_=19
    nDepths=53
    V=51
    nLayers=6
    G=InvrsModel(nIter=nLayers,nDepths=nDepths,s=s,V=V,NxN=N_*N_).to(device) # Loading trained model
    state_dict=G.state_dict()
    for n, p in torch.load(args.weights_fileG, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)

    # Loading data
    #lfTrainFTmp=np.array(stdevs).astype(np.uint8)
    #lfTrainFTmp=torch.from_numpy(stdevs).to(device)
    lfTrainFTmp=torch.from_numpy(snr8).to(device)
    sz0=lfTrainFTmp.shape[0] # Dimensions of data tensor
    sz1=lfTrainFTmp.shape[1]
    sz2=lfTrainFTmp.shape[2]
    sz3=lfTrainFTmp.shape[3]
    
    volTmpSeq_=torch.zeros((sz0,nDepths,s*sz2,s*sz3),dtype=torch.uint8).to(device)# uint8 to save memory
    maxVal=torch.zeros((1,1),dtype=torch.uint8).to(device) # initialize maxVal to 0
    
    #Compute maximun value saving memory - for normalizing 
    for j in range(len(lfTrainFTmp)):
        lfTmp=lfTrainFTmp[j,:,:,:].float()
        volTmp=G(lfTmp[None,:,:,:])
        tmpMax=volTmp.max()
        if tmpMax>maxVal:
            maxVal=tmpMax

    #Save temporal stack
    for j in range(len(lfTrainFTmp)):
        lfTmp=lfTrainFTmp[j,:,:,:].float()
        volTmp=torch.nn.functional.relu(G(lfTmp[None,:,:,:],test=1)) # Applied ReLU: max(0,x) - getting rid of negative values?
        volTmp[volTmp>maxVal]=maxVal # Capping range of values?
        volTmp=255*volTmp/maxVal #normalization to match uint8
        volTmpSeq_[j]=volTmp
            
    #scipy.io.savemat('volTmpSeq_stdev.mat', mdict={'volTmpSeq_':volTmpSeq_.cpu().numpy()})
    volTmpSeq_ = volTmpSeq_.cpu().numpy()

    # ---------- neuron finding/segmentation ----------
    # To-do: check to see if volume sequence is already given??

    print('Performing segmentation on reconstructed volume')

    device = cle.select_device("Intel(R)")

    std_recon_vol = np.squeeze(volTmpSeq_)
    [A,B,C] = std_recon_vol.shape

    std_recon_vol = std_recon_vol[:,3*13+1:B-3*13,3*13+1:C-3*13]

    voxel_size_x = 1
    voxel_size_y = 1
    voxel_size_z = 1

    input_gpu = cle.push(std_recon_vol)
    
    resampled = cle.create([int(input_gpu.shape[0] * voxel_size_z), int(input_gpu.shape[1] * voxel_size_y), int(input_gpu.shape[2] * voxel_size_x)])
    cle.scale(input_gpu, resampled, factor_x=voxel_size_x, factor_y=voxel_size_y, factor_z=voxel_size_z, centered=False)
    
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

    segmented = cle.voronoi_otsu_labeling(no_bg, spot_sigma=args.spotSigma, outline_sigma=args.outlineSigma) # spot = 3, outline = 1

    coordinates = []

    info_table = pd.DataFrame(
        measure.regionprops_table(
            segmented,
            intensity_image=std_recon_vol,
            properties=['label', 'slice', 'coords'],
        )
    ).set_index('label')

    num_neurons = len(info_table['slice'])

    for i in range(1,num_neurons+1):
        temp_coord = []
        for j in range(0,3):
            temp_xyz = [getattr(info_table['slice'][i][j],'start'),getattr(info_table['slice'][i][j],'stop')]
            temp_coord.append(temp_xyz)        
        coordinates.append(temp_coord)
    print(coordinates)

    #np.save('coordinates_stdev.npy',coordinates)

    # ---------- creating synthetic volumes ----------

    print('Creating synthetic volumes and passing through forward models to get LF footprints')

    [A,B,C,D] = volTmpSeq_.shape
    E = len(coordinates)
    total_vol = np.zeros((E,C,D,B))

    # Getting coordinates
    # coords = np.load(args.coords_file)
    factor = -3+3*13
    for i in range(0,3):
        z = coordinates[i][0]
        y = coordinates[i][1]
        x = coordinates[i][2]
        
        temp = volTmpSeq_[:,z[0]:z[1],y[0]+factor:y[1]+factor,x[0]+factor:x[1]+factor]
        temp = np.mean(temp,axis=0,dtype=np.float32)
        vol = np.zeros([B,C,D])
        vol[z[0]:z[1],y[0]+factor:y[1]+factor,x[0]+factor:x[1]+factor] = temp
        vol = np.transpose(vol,[1,2,0])
        vol = vol.astype(np.uint8)
        total_vol[i,:,:,:] = vol
    
    #np.save('synth_LFvol_stdev.npy',total_vol)

    # ---------- passing synthetic volumes through forward model to make LF footprints
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Parameters
    s=3
    N_=19
    nDepths=53
    V=51
    nLayers=6
    centDpt=nDepths//2
    L2=17
    haarL=8
    nLFs=28
    l=3
    c=400
    
    Fl=multConvFModel(nDepths=nDepths,s=s,V=V,NxN=N_*N_,haarL=haarL,l=l,c=c).to(device)
    state_dict=Fl.state_dict()
    for n, p in torch.load(args.weights_fileFl, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)

    # Loading data - already a numpy array
    lfTrainFTmp = total_vol.astype(np.uint8)
    [A,sz0,sz1,sz2] = lfTrainFTmp.shape
    
    LFfootprints = np.zeros((A,361,107,107))
                            
    for i in range(0,A):
        print('Neuron ', i)
        
        lfTrain = total_vol[i,:,:,:]
        lfTrain=torch.from_numpy(lfTrain).to(device)

        lfTmp_=torch.zeros((361,107,107),dtype=torch.uint8).to(device)# uint8 to save memory
        maxVal=torch.zeros((1,1),dtype=torch.uint8).to(device) # initialize maxVal to 0

        #Compute maximun value saving memory - for normalizing 
        lfTmp = lfTrain[:,:,:].float()
        lfTmp = torch.permute(lfTmp,(2,0,1))
        volTmp = Fl(lfTmp[None,:,:,:])
        tmpMax = volTmp.max()
        if tmpMax>maxVal:
            maxVal = tmpMax

        # Getting LF
        lfTmp = lfTrain[:,:,:].float()
        lfTmp = torch.permute(lfTmp,(2,0,1))
        volTmp = torch.nn.functional.relu(Fl(torch.nn.functional.pad(lfTmp[None,:,:,:],((L2//2)*s,(L2//2)*s,(L2//2)*s,(L2//2)*s),'reflect')))
        volTmp[volTmp>maxVal] = maxVal # Capping range of values?
        volTmp = 255*volTmp/maxVal #normalization to match uint8
        lfTmp_ = volTmp.detach().cpu().numpy()
        lfTmp_ = np.squeeze(lfTmp_)        
        LFfootprints[i,:,:,:] = lfTmp_
        
    #np.save('lfSeq_all_stdev.npy', LFfootprints)

    # ---------- finding time sequence ----------
    # SVD to remove background?

    print('Finding time sequences')
    D = 19
    E = 107
    
    LFout = np.reshape(LF,(500,D,D,E,E),order='F')
    LFout = np.transpose(LFout,(0,1,3,2,4))
    LFout = np.reshape(LFout,(500,2033,2033),order='F')
    LFout = np.transpose(LFout,(1,2,0))

    A,B,C = LFout.shape
    LFout = np.reshape(LFout,(A*B,C),order='F')
    
    # SVD to remove the background
    U, S, Vh = np.linalg.svd(LFout,full_matrices=False)
    del LFout

    bg_ratio = 0.9
    S[0] = (1-bg_ratio)*S[0]
    LF_new = np.dot(U * S, Vh)

    # Getting the neurons now
    #neurs = np.load(args.fileLFSeq)
    neurs = LFfootprints
    
    N = np.array([])
    X = len(neurs)

    for i in range(0,X):
        neur = neurs[i]

        neur = np.reshape(neur,(D,D,E,E),order='F')
        neur = np.transpose(neur,(0,2,1,3))
        neur = np.reshape(neur,(D*D*E*E,1),order='F')

        N = np.hstack([N, neur]) if N.size else neur
        
    N = np.where(np.isfinite(N), N, 0)
    N_pinv = np.linalg.pinv(N)
    T = np.matmul(N_pinv,LFout)

    X,Y = T.shape
    #for i in range(0,X):
    #    plt.plot(range(0,Y),T[i,:])
    #plt.savefig('TimeSeq.png',bbox_inches='tight')
    #np.save('timeSeq.npy',T)

    # ---------- comparing with fully reconstructed time sequence/"ground truth" ----------
    # aka plotting pretty things
    timeDerivLF = np.zeros(X)
    for i in range(0,X):
        #tempT = scipy.signal.detrend(T[i,:],type='linear')
        tempT = np.gradient(T[i,:])
        #plt.plot(range(0,Y),tempT,'g-',linewidth=0.5)
        timeDerivLF[i] = np.max(tempT)-np.min(tempT)
    timeDeriv_idx = np.argsort(-timeDerivLF)


    # --------------------Pretty plots now

    [A,B,C,D] = volTmpSeq_.shape
    vol = volTmpSeq_[:,:,3*13+1:C-3*13,3*13+1:D-3*13]

    fig, ax = plt.subplots(10,1,figsize=(6,20))
    for i in range(0,10):
        idx = timeDeriv_idx[i]
        coords_tmp = coordinates[idx]
        coords_tmp = coords_tmp.flatten()
        ts_tmp = np.mean(vol[:,coords_tmp[0]:coords_tmp[1],coords_tmp[2]:coords_tmp[3],coords_tmp[4]:coords_tmp[5]],axis=(1,2,3))
        ts_tmp = (ts_tmp-min(ts_tmp))/(max(ts_tmp)-min(ts_tmp))
        ts_mtrx = T[idx,:]
        ts_mtrx = (ts_mtrx-min(ts_mtrx))/(max(ts_mtrx)-min(ts_mtrx))
                
        ax[i].plot(range(0,Y),ts_tmp,'b-',linewidth=0.65,label='ROI-based')
        ax[i].plot(range(0,Y),ts_mtrx,'r-',linewidth=0.65,label='Matrix-based')
        ax[i].set_xlabel('Time Index',fontsize=10)
        ax[i].tick_params(axis='both', which='major', labelsize=10)

    ax[0].legend(loc='upper left') 
    fig.tight_layout()

    plt.savefig('S2A3_TimeSeq_.png',bbox_inches='tight')
        
    

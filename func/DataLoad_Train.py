# -*- coding: utf-8 -*-
"""
Load training data set

Created on Feb 2018

@author: fangshuyang (yfs2016@hit.edu.cn)

"""


import numpy as np
import torch
from skimage.measure import block_reduce
import skimage
import scipy.io
from IPython.core.debugger import set_trace

def DataLoad_Train(train_size,train_data_dir,data_dim,in_channels,model_dim,data_dsp_blk,label_dsp_blk,start,datafilename,dataname,truthfilename,truthname):
    for i in range(start,start+train_size):
        
        filename_seis = train_data_dir+'georec_train/'+datafilename+str(i)
        print(filename_seis)
        # Load .mat data
        data1_set = scipy.io.loadmat(filename_seis)
        data1_set = np.float32(data1_set[str(dataname)].reshape([data_dim[0],data_dim[1],in_channels]))
        # Change the dimention [h, w, c] --> [c, h, w]
        for k in range (0,in_channels):
            data11_set     = np.float32(data1_set[:,:,k])
            data11_set     = np.float32(data11_set)
            # Data downsampling
            # note that the len(data11_set.shape)=len(block_size.shape)=2
            data11_set     = block_reduce(data11_set,block_size=data_dsp_blk,func=decimate)
            data_dsp_dim   = data11_set.shape
            data11_set     = data11_set.reshape(1,data_dsp_dim[0]*data_dsp_dim[1])
            if k==0:
                train1_set = data11_set
            else:
                train1_set = np.append(train1_set,data11_set,axis=0)
        filename_label     = train_data_dir+'vmodel_train/'+truthfilename+str(i)
        data2_set          = scipy.io.loadmat(filename_label)
        data2_set          = np.float32(data2_set[str(truthname)].reshape(model_dim))
        # Label downsampling
        data2_set          = block_reduce(data2_set,block_size=label_dsp_blk,func=np.max)
        label_dsp_dim      = data2_set.shape
        data2_set          = data2_set.reshape(1,label_dsp_dim[0]*label_dsp_dim[1])
        data2_set          = np.float32(data2_set)
        if i==start:
            train_set      = train1_set
            label_set      = data2_set
        else:
            train_set      = np.append(train_set,train1_set,axis=0)
            label_set      = np.append(label_set,data2_set,axis=0)
            
    train_set = train_set.reshape((train_size,in_channels,data_dsp_dim[0]*data_dsp_dim[1]))
    label_set = label_set.reshape((train_size,1,label_dsp_dim[0]*label_dsp_dim[1]))
    
    return train_set, label_set, data_dsp_dim, label_dsp_dim

# downsampling function by taking the middle value
def decimate(a,axis):
    idx = np.round((np.array(a.shape)[np.array(axis).reshape(1,-1)]+1.0)/2.0-1).reshape(-1)
    downa = np.array(a)[:,:,idx[0].astype(int),idx[1].astype(int)]
    return downa

def get_petrobras_loader(main_dir):
    dim_A, dim_B =  (4,128,64), (3,128,64)
    train_set, label_set = get_petrobras_data(main_dir)
    loader = data_generator(train_set,label_set,dim_A, dim_B )
    return loader, dim_A, dim_B

def get_petrobras_data(main_dir):

    path= main_dir +"../PHD/data/synthetic_dataset/"
    near = normalize_data(np.load(path+"NEAR.npy"))
    mid = normalize_data(np.load(path+"MID.npy"))
    far = normalize_data(np.load(path+"FAR.npy"))
    ufar = normalize_data(np.load(path+"UFAR.npy"))
    s_data = np.stack((near, mid, far, ufar))
    s_data = np.swapaxes(s_data, 1, 3)
    train_slices = [i for i in range(np.shape(s_data)[2]) if (i < 80 or i > 120)]
    s_data = s_data[:,:,train_slices]


    v_data = np.swapaxes(np.load(path+"vp.npy"), 0, 2)
    vs_data = np.swapaxes(np.load(path+"vs.npy"), 0, 2)
    rho_data = np.swapaxes(np.load(path+"rho.npy"), 0, 2)
    v_data = normalize_data(v_data)
    vs_data = normalize_data(vs_data)
    rho_data = normalize_data(rho_data)
    v_data = np.stack((v_data, vs_data, rho_data))[:,:,train_slices]
    return s_data, v_data


def normalize_data(data, mean=None, std=None):
    print(f"Normalizing with mean {np.mean(data)} and std {np.std(data)}")
    if not mean:
        mean = np.mean(data)
    if not std:
        std = np.std(data)
    data = (data - mean) / std
    return data
def data_generator(data_A, data_B, dim_A, dim_B, matching_pairs=True, batch_size=1):
    print(np.shape(data_A), np.shape(data_B))
    def get_indx():

        ix = np.random.randint(0, data_shape[0] - dim_A[1], batch_size)
        iy = np.random.randint(0, data_shape[1], batch_size)
        iz = np.random.randint(0, data_shape[2] - dim_A[2], batch_size)
        return ix, iy, iz

    while True:
        # choose random instances

        data_shape = np.shape(data_A)[1:]
        ix, iy, iz = get_indx()
        # retrieve selected images
        A = np.asarray([data_A[:, x:x + dim_A[1], y, z: z + dim_A[2]] for x, y, z in zip(ix, iy, iz)])

        if not matching_pairs:

            data_shape = np.shape(data_B)[1:]
            ix, iy, iz = get_indx()

        B = np.asarray([data_B[:, x:x + dim_B[1], y, z: z + dim_B[2]] for x, y, z in zip(ix, iy, iz)])

        yield torch.Tensor(A), torch.Tensor(B)

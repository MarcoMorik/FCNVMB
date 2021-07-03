# -*- coding: utf-8 -*-
"""
Parameters setting

Created on Feb 2018

@author: fangshuyang (yfs2016@hit.edu.cn)

"""


####################################################
####             MAIN PARAMETERS                ####
####################################################
SimulateData  = True          # If False denotes training the CNN with SEGSaltData
ReUse         = False         # If False always re-train a network 
DataDim = [128,64]
data_dsp_blk  = (1,1)         # Downsampling ratio of input
ModelDim      = [128,64]     # Dimension of one velocity model
label_dsp_blk = (1,1)         # Downsampling ratio of output
dh            = 1            # Space interval 

#DataDim       = [2000,301]    # Dimension of original one-shot seismic data
#data_dsp_blk  = (5,1)         # Downsampling ratio of input
#ModelDim      = [201,301]     # Dimension of one velocity model
#label_dsp_blk = (1,1)         # Downsampling ratio of output
#dh            = 10            # Space interval 


####################################################
####             NETWORK PARAMETERS             ####
####################################################
if SimulateData:
    Epochs        = 100       # Number of epoch
    TrainSize     = 1600      # Number of training set
    TestSize      = 100       # Number of testing set
    TestBatchSize = 10
else:
    Epochs        = 50
    TrainSize     = 130      
    TestSize      = 10       
    TestBatchSize = 1
    
BatchSize         = 10        # Number of batch size
LearnRate         = 1e-3      # Learning rate
Nclasses          = 3        # Number of output channels
Inchannels        = 4        # Number of input channels, i.e. the number of shots
SaveEpoch         = 20        
DisplayStep       = 1000         # Number of steps till outputting stats

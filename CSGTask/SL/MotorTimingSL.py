# Implement Supervised Learning RNN as by David Sussillo

# Libraries
import warnings
import gym
import neurogym as ngym
import tensorflow as tf
import numpy as np
from neurogym.utils import plotting
from neurogym.wrappers import monitor
import matplotlib.pyplot as plt
import csv
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
warnings.filterwarnings('ignore')

# Define Parameters
inputs = True
train = False

task = 'MotorTiming-v0'
numSteps = 3500 # Number of Steps
conditions = 10 # Number of Conditions

# Define Hyperparameters
Input_cycle = 1 # Number of Cycles for Input Generation
Train_cycle = 100 # Number of Cycles for Training

LR = 5*(10**-5) # Learning Rate
InputNoise = 0 # Context Cue Noise
TargetThreshold = 0.01 # Allowed Production Deviation  

ModelDir='IN{}%_TT{}%/{}Cycles_{}LR'.format(InputNoise*100, TargetThreshold*100, Train_cycle, LR)

# Generate Input Arrays
if inputs:
    # Define Env
    kwargs = {'training': True, 'InputNoise': InputNoise, 'TargetThreshold': TargetThreshold}
    envInput = gym.make(task, **kwargs)
    
    # Run Env & Save Observations
    for i_trial in range(Input_cycle*conditions):
        observationList=[]
        observation = envInput.reset()
        for t in range(numSteps):
            action = envInput.action_space.sample()
            observation, reward, done, info = envInput.step(action)
            observationList.append(observation)
        np.savetxt('CSGTask/Models/CSG_SL/Inputs/'+'inputs_{}.csv'.format(i_trial), observationList, delimiter=",")
        
    envInput.close()

# Read Input:
# IN = np.loadtxt('CSGTask/Models/CSG_SL/Inputs/inputs_0.csv', delimiter=',')

# Train Env
if train:
    print("Train Environment")

    # Define Env
    kwargs = {'training': True, 'InputNoise': InputNoise, 'TargetThreshold': TargetThreshold}
    envTrain = gym.make(task, **kwargs)
    envTrain.reset()
    envTrain = monitor.Monitor(envTrain, folder='CSGTask/Plots/CSG_SL/Train/', 
                                sv_per=numSteps, verbose=False, sv_fig=True, num_stps_sv_fig=numSteps)

    # Train
    print("Start Training")
        # Train
    print("Done")

    envTrain.close()
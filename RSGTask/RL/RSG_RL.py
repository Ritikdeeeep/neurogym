# Libraries
import warnings
import gym
import neurogym as ngym
import tensorflow as tf
import numpy as np
from neurogym.utils import plotting
import matplotlib
import matplotlib.pyplot as plt
from neurogym.wrappers import monitor
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
warnings.filterwarnings('ignore')

# Define Parameters
plot = True
run = False

task = 'MotorTiming-v0'
numSteps = 3500 # Number of Steps
conditions = 10 # Number of Conditions

# Define Hyperparameters
numTrials = 10 # Number of Trials for Plot
Run_cycle = 1 # Number of Cycles for Run
Train_cycle = 30 # Number of Cycles for Training
    # TotalSteps= 3500 * 10 * Train_cycle

LR = 5*(10**-5) # Learning Rate
#LR = 'Linear_5e-6'
InputNoise = 0
TargetThreshold = 0.01
ThresholdDelay = 20
TargetRamp = True

ModelDir='IN{}%_TT{}%/{}Cycles_{}LR'.format(InputNoise*100, TargetThreshold*100, Train_cycle, LR)

# Plot Env
if plot:
    print("Plot Environment")

    # Define Kwargs
    kwargs = {'params': (True, InputNoise, 0.03, ThresholdDelay, TargetRamp)}

    ContextCue = np.empty((numSteps,conditions))
    ContextCue[:] = np.NaN
    ReadySetCue = np.empty((numSteps,conditions))
    ReadySetCue[:] = np.NaN
    Target = np.empty((numSteps,conditions))
    Target[:] = np.NaN

    # Generate Data
    for i in range(conditions):
        envPlot = gym.make(task, **kwargs)
        trial, ob, gt = envPlot._new_trial(scenario=i)
        CC = ob[:,0]
        RSC = ob[:,1]
        T = gt[:,0]
        ContextCue[0:0+CC.shape[0], i] = CC
        ReadySetCue[0:0+RSC.shape[0], i] = RSC
        Target[0:0+T.shape[0], i] = T
        envPlot.close()

    # Define Figure
    fig, axs = plt.subplots(3, sharex=True, sharey=True)
    fig.suptitle('RSG Input')
    plt.xlabel("Time")
    colors = matplotlib.cm.get_cmap('coolwarm', conditions)

    for i in range(conditions):
        # Context Cue
        axs[0].plot(ContextCue[:,i], color=colors(i/conditions))
        axs[0].set_title('Context Cue')
        axs[0].set_ylabel('Context Magnitude')

            # Set Cue
        axs[1].plot(ReadySetCue[:,i], color=colors(i/conditions))
        axs[1].set_title('Ready-Set Cue')
        axs[1].set_ylabel('Magnitude')

            # Target
        axs[2].plot(Target[:,i], color=colors(i/conditions))
        axs[2].set_title('Target')
        axs[2].set_ylabel('Target Magnitude')
    plt.show()

# Run Env
if run:
    print("Run Environment")

    # Define Env
    kwargs = {'params': (True, InputNoise, TargetThreshold, ThresholdDelay, TargetRamp)}
    envRun = gym.make(task, **kwargs)
    envRun = monitor.Monitor(envRun, folder='RSGTask/RL/Plots/Run/'+ModelDir, 
                            sv_per=numSteps, verbose=False, sv_fig=True, num_stps_sv_fig=numSteps)
    
    # Run
    for i_episode in range((conditions)*Run_cycle):
        observation = envRun.reset()
        for t in range(numSteps):
            action = envRun.action_space.sample()
            observation, reward, done, info = envRun.step(action)
            if done:
                print("Episode finished after {} timesteps".format((t+1)-info['SetStart']-ThresholdDelay))
                print(info['Interval'])
                break

    envRun.close()
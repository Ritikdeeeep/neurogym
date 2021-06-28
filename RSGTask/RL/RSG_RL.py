# Libraries
import warnings
import gym
import neurogym as ngym
import tensorflow as tf
import numpy as np

from neurogym.utils import plotting
from neurogym.wrappers import monitor

import matplotlib.pyplot as plt

from stable_baselines.common.policies import LstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.schedules import LinearSchedule
from stable_baselines import A2C

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.filterwarnings('ignore')

# Define Flags & Respective Parameters
    # Plots All Conditions
plotTrain = False 

    # Plots Actual Trials
plotTrial = False 
Plot_Trials = 10 # Number of Trials to Plot

    # Export Inputs/Target for MatLab
exportMatLab = True
Export_Cycles = 50 # (Training: 20/10, Testing: 10/5)

    # Run Environment
run = False
Run_Cycles = 1 # Number of Cycles for Run

    # Train Environment
train = False
Train_Cycles = 250 # Number of Cycles for Training # TotalSteps= 3000 * 10 * Train_cycle
LR = 1*(10**-3) # Learning Rate
N_Parallel = 10 # Number of processes to use in parallel


# Define Kwargs
Task = 'MotorTiming-v0'
N_Steps = 3000 # Number of Steps (One: 2000, Two: 3000)
N_Conditions = 10 # Number of Conditions (One: 5, Two: 10)
Input_Noise = 0.01 # Input Noise Percentage
Target_Threshold = 0.05 # Allowed Target Deviation Percentage (Close to WeberFraction of Paper (0.07))
Threshold_Delay = 50 # Delay after Threshold is reached
Wait_Time = 50 # Wait after start for context cue
Target_Ramp = True # Ramp or NaN Target

Model_Dir='Set_1/{}Cycles_{}LR'.format(Train_Cycles, LR)
# Also see Training Log

# Plot Training Env
if plotTrain:
    print("Plot Environment")

    ContextCue = np.empty((N_Steps, N_Conditions))
    ContextCue[:] = np.NaN
    ReadySetCue = np.empty((N_Steps, N_Conditions))
    ReadySetCue[:] = np.NaN
    Target = np.empty((N_Steps, N_Conditions))
    Target[:] = np.NaN

    # Generate Data
    for i in range(N_Conditions):
        kwargs = {'params': (True, Input_Noise, Target_Threshold, Threshold_Delay, Target_Ramp)}
        envPlot = gym.make(Task, **kwargs)
        trial, ob, gt = envPlot._new_trial(Scenario=i, WaitTime=Wait_Time)
        CC = ob[:,0]
        RSC = ob[:,1]
        T = gt[:,0]
        ContextCue[0:0+CC.shape[0], i] = CC
        ReadySetCue[0:0+RSC.shape[0], i] = RSC
        Target[0:0+T.shape[0], i] = T
        envPlot.close()

    # Plot
    fig, axs = plt.subplots(3, sharex=True, sharey=True)
    # plt.xlim(0,numSteps)
    fig.suptitle('RSG Input')
    plt.xlabel("Time")
    colors = ['navy', 'mediumblue', 'royalblue', 'cornflowerblue', 'lightsteelblue', 'darkred', 'firebrick', 'brown', 'indianred', 'lightcoral']    

    for i in range(N_Conditions):
        # Context Cue
        axs[0].plot(ContextCue[:,i], color=colors[i])
        axs[0].set_title('Context Cue')
        axs[0].set_ylabel('Context Magnitude')

            # Set Cue
        axs[1].plot(ReadySetCue[:,i], color=colors[i])
        axs[1].set_title('Ready-Set Cue')
        axs[1].set_ylabel('Magnitude')

            # Target
        axs[2].plot(Target[:,i], color=colors[i])
        axs[2].set_title('Target')
        axs[2].set_ylabel('Target Magnitude')
    plt.show()

# Plot Env
if plotTrial:
    print("Plot Environment")

    # Define Env
    kwargs = {'params': (False, Input_Noise, Target_Threshold, Threshold_Delay, Target_Ramp)}
    envPlot = gym.make(Task, **kwargs)
    env_data = plotting.run_env(envPlot, num_steps=N_Steps, num_trials=Plot_Trials, def_act=0)

    # Plot
    fig, axs = plt.subplots(3, sharex=True, sharey=True)
    fig.suptitle('CSG Input')
    plt.xlabel("Time")

        # Context Cue
    axs[0].plot(env_data['ob'][:,0])
    axs[0].set_title('Context Cue')
    axs[0].set_ylabel('Context Magnitude')

        # Set Cue
    axs[1].plot(env_data['ob'][:,1])
    axs[1].set_title('Set Cue')
    axs[1].set_ylabel('Set Magnitude')

        # Target
    axs[2].plot(env_data['gt'])
    axs[2].set_title('Target')
    axs[2].set_ylabel('Target Magnitude')
    plt.show()

    envPlot.close()

# Run Env
if run:
    print("Run Environment")

    # Define Env
    kwargs = {'params': (True, Input_Noise, Target_Threshold, Threshold_Delay, Target_Ramp)}
    envRun = gym.make(Task, **kwargs)
    envRun = monitor.Monitor(envRun, folder='RSGTask/RL/Plots/Run/'+Model_Dir, 
                            sv_per=N_Steps, verbose=False, sv_fig=True, num_stps_sv_fig=N_Steps)
    
    # Run
    for i_episode in range((N_Conditions)*Run_Cycles):
        observation = envRun.reset()
        for t in range(N_Steps):
            action = envRun.action_space.sample()
            observation, reward, done, info = envRun.step(action)
            if done:
                print("Episode finished after {} timesteps".format((t+1)-info['SetStart']-Threshold_Delay))
                print(info['Interval'])
                break
        print('End of Trial')

    envRun.close()

# Export Arrays for MatLab
if exportMatLab:
    print("Export for MatLab")
    Export_Trials = Export_Cycles*N_Conditions

    Input = np.empty((2, N_Steps, Export_Trials))
    Input[:] = np.NaN
    Target = np.empty((1, N_Steps, Export_Trials))
    Target[:] = np.NaN

    # Generate Data
    for i in range(1,Export_Trials+1):
        kwargs = {'params': (True, Input_Noise, Target_Threshold, Threshold_Delay, Target_Ramp)} # Training
        # kwargs = {'params': (False, Input_Noise, Target_Threshold, Threshold_Delay, Target_Ramp)} # Testing
        envPlot = gym.make(Task, **kwargs)
        trial, ob, gt = envPlot._new_trial(Scenario=((i % N_Conditions)-1), WaitTime=Wait_Time) # Training
        # trial, ob, gt = envPlot._new_trial() # Testing
        CC = ob[:,0]
        RSC = ob[:,1]
        T = gt[:,0]
        Input[0, 0:0+CC.shape[0], i-1] = CC
        Input[1, 0:0+RSC.shape[0], i-1] = RSC
        Target[0, 0:0+T.shape[0], i-1] = T
        envPlot.close()
    np.save(file='RSGTask/MatLab/RSG_2/TrainExample/Input', arr=Input)
    np.save(file='RSGTask/MatLab/RSG_2/TrainExample/Target', arr=Target)

# Train Env
if train:

    print("Train Environment")

    # Define Env
    kwargs = {'params': (True, Input_Noise, Target_Threshold, Threshold_Delay, Target_Ramp)}
    envTrain = gym.make(Task, **kwargs)
    envTrain.reset()
    envTrain = monitor.Monitor(envTrain, folder='RSGTask/RL/Plots/Train/', 
                                sv_per=N_Steps, verbose=False, sv_fig=True, num_stps_sv_fig=N_Steps)
    envTrain = DummyVecEnv([lambda: envTrain])

    # Define Model
    model = A2C(LstmPolicy, envTrain, verbose=1, 
                gamma=1, alpha=1, #max_grad_norm=0.25,
                learning_rate=LR, #lr_schedule='linear', 
                tensorboard_log="RSGTask/RL/Models/RSG_RL_Tensorboard/",
                policy_kwargs={'feature_extraction':"mlp", 'act_fun':tf.nn.tanh ,'n_lstm':200, 'net_arch':[2, 'lstm', 200, 1]})

    # Train
    print("Start Training")
    model.learn(total_timesteps=N_Steps*N_Conditions*Train_Cycles, log_interval=N_Steps*N_Conditions, 
                tb_log_name=Model_Dir, reset_num_timesteps=True)
    model.save('RSGTask/RL/Models/RSG_RL_Models/'+Model_Dir+'.zip')
    print("Done")

    envTrain.close()


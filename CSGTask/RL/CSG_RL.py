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
from stable_baselines.common import set_global_seeds, make_vec_env
from stable_baselines import A2C, PPO2

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.filterwarnings('ignore')

# Define Flags & Respective Parameters
    # Plots All Conditions
plotTrain = False 
Wait_Time = 100 # Fixed Wait Time 

    # Plots Actual Trials
plotTrial = False 
Plot_Trials = 10 # Number of Trials to Plot

    # Export Inputs/Target for MatLab
exportMatLab = True
Export_Cycles = 50

    # Run Environment
run = False
Run_Cycles = 1 # Number of Cycles for Run

    # Train Environment
train = False
Train_Cycles = 250 # Number of Cycles for Training # TotalSteps= 3000 * 10 * Train_cycle
LR = 1*(10**-3) # Learning Rate
N_Parallel = 10 # Number of processes to use in parallel

    # Evaulate Trained Environment
evaluate = False

# Define Kwargs
Task = 'MotorTiming-v0'
N_Steps = 2200 # Number of Steps
N_Conditions = 10 # Number of Conditions
Input_Noise = 0 # Input Noise Percentage
Target_Threshold = 0.05 # Allowed Target Deviation Percentage (Close to WeberFraction of Paper (0.07))
Threshold_Delay = 50 # Delay after Threshold is reached
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
    fig.suptitle('CSG Task')
    plt.xlabel("Time")
    colors = ['navy', 'mediumblue', 'royalblue', 'cornflowerblue', 'lightsteelblue', 'darkred', 'firebrick', 'brown', 'indianred', 'lightcoral']

    for i in range(N_Conditions):
        # Context Cue
        axs[0].plot(ContextCue[:,i], color=colors[i])
        axs[0].set_title('Context Cue')
        axs[0].set_ylabel('Context Magnitude')

            # Set Cue
        axs[1].plot(ReadySetCue[:,i], color=colors[i])
        axs[1].set_title('Set Cue')
        axs[1].set_ylabel('Set Magnitude')

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
    envRun = monitor.Monitor(envRun, folder='CSGTask/RL/Plots/Run/'+Model_Dir, 
                            sv_per=N_Steps, verbose=False, sv_fig=True, num_stps_sv_fig=N_Steps)
    
    # Run
    for i_episode in range((N_Conditions)*Run_Cycles):
        observation = envRun.reset()
        for t in range(N_Steps):
            action = envRun.action_space.sample()
            observation, reward, done, info = envRun.step(action)
            if done:
                print("Episode finished after {} timesteps".format((t+1)-info['SetStart']-info['ThresholdDelay']))
                print(info['Interval'])
                break

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
        kwargs = {'params': (False, Input_Noise, Target_Threshold, Threshold_Delay, Target_Ramp)}
        envPlot = gym.make(Task, **kwargs)
        # trial, ob, gt = envPlot._new_trial(Scenario=((i % N_Conditions)-1), WaitTime=None) # Training
        trial, ob, gt = envPlot._new_trial() # Testing
        CC = ob[:,0]
        RSC = ob[:,1]
        T = gt[:,0]
        Input[0, 0:0+CC.shape[0], i-1] = CC
        Input[1, 0:0+RSC.shape[0], i-1] = RSC
        Target[0, 0:0+T.shape[0], i-1] = T
        envPlot.close()
    np.save(file='CSGTask/MatLab/Testing50/Input', arr=Input)
    np.save(file='CSGTask/MatLab/Testing50/Target', arr=Target)
    
    # fig, axs = plt.subplots(3, sharex=True, sharey=True)
    # axs[0].plot(Input[0,:,:])
    # axs[1].plot(Input[1,:,:])
    # axs[2].plot(Target[0,:,:])
    # plt.show()


# Train Env
if train:
    print("Train Environment")

    # Define Env
    kwargs = {'params': (True, Input_Noise, Target_Threshold, Threshold_Delay, Target_Ramp)}
    envTrain = make_vec_env(Task, monitor_dir='CSGTask/RL/Monitor/'+Model_Dir, task_kwargs=kwargs, n_envs=N_Parallel, seed=0)
    envTrain.reset()
        # Get Parallel Environments to run
        # BatchSize is N_Steps * NumEnvs
        # Changed function to include task_kwargs as well

    # Define Model
    model = A2C(LstmPolicy, envTrain, verbose=1, n_steps=N_Steps,
                gamma=1, alpha=1, max_grad_norm=0.25,
                learning_rate=LR, lr_schedule='exp', 
                tensorboard_log="CSGTask/RL/Models/CSG_RL_Tensorboard/MatLab_Trial/",
                policy_kwargs={'feature_extraction':"mlp", 'act_fun':tf.nn.tanh ,'n_lstm':200, 'net_arch':[2, 200, 'lstm', 1]})
        # Added own exponential decay schedule for LR

    # model = PPO2(LstmPolicy, envTrain, verbose=1, 
    #             nminibatches= 2,
    #             gamma=1, max_grad_norm=0.25,
    #             learning_rate=LR,
    #             tensorboard_log="CSGTask/RL/Models/CSG_RL_Tensorboard/Redefined_Trial/",
    #             policy_kwargs={'feature_extraction':"mlp", 'act_fun':tf.nn.tanh ,'n_lstm':200, 'net_arch':[2, 200, 'lstm', 1]})

    # Train
    print("Start Training")
    model.learn(total_timesteps=N_Steps*N_Conditions*Train_Cycles, log_interval=N_Steps*N_Conditions, 
                tb_log_name=Model_Dir)
    model.save('CSGTask/RL/Models/CSG_RL_Models/'+Model_Dir+'.zip')
    print("Done")

    envTrain.close()

# Evaluate Trained Model
if evaluate:
    print("Run Trained Environment")

    # Define Env
    kwargs = {'params': (False, Input_Noise, Target_Threshold, Threshold_Delay)}
    envRunTrained = gym.make(Task, **kwargs)
    envRunTrained = monitor.Monitor(envRunTrained, folder='CSGTask/RL/Plots/RunTrained/', sv_per=N_Steps, verbose=False, sv_fig=True, num_stps_sv_fig=3500)
    envRunTrained = DummyVecEnv([lambda: envRunTrained])

    # Define Model
    modelTrained = A2C.load(load_path = '/Users/ritikmehta/School/Bsc_NB3/BEP/neurogym/CSGTask/RL/Models/CSG_RL_Models/IN0.5%_TT1.0%/30Cycles_1e-05LR.zip', 
                            env = envRunTrained)

    # Evaluate Model
    for i_episode in range(N_Conditions*Run_Cycles):
        observation = envRunTrained.reset()
        for t in range(N_Steps):
            action = modelTrained.predict(observation)
            observation, reward, done, info = envRunTrained.step(action)
            if done:
                print("Episode finished after {} timesteps".format((t+1)-info['SetStart']-info['ThresholdDelay']))
                print(info['Interval'])
                break
        print("Reach End")

    envRunTrained.close()
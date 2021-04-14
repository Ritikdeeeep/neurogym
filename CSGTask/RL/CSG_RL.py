# Libraries
import warnings
import gym
import neurogym as ngym
import tensorflow as tf
import numpy as np
from neurogym.utils import plotting
import matplotlib.pyplot as plt
from neurogym.wrappers import monitor
from stable_baselines.common.policies import LstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.schedules import LinearSchedule
from stable_baselines import A2C
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
warnings.filterwarnings('ignore')

# Define Parameters
plot = False
run = True
train = False
runTrained = False

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
InputNoise = 0.01
TargetThreshold = 0.01
ThresholdDelay = 100

ModelDir='IN{}%_TT{}%/{}Cycles_{}LR'.format(InputNoise*100, TargetThreshold*100, Train_cycle, LR)

# Plot Env
if plot:
    print("Plot Environment")

    # Define Env
    kwargs = {'params': (False, InputNoise, TargetThreshold, ThresholdDelay)}
    envPlot = gym.make(task, **kwargs)
    env_data = plotting.run_env(envPlot, num_steps=numSteps, num_trials=numTrials, def_act=0)

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
    kwargs = {'params': (True, InputNoise, TargetThreshold, ThresholdDelay)}
    envRun = gym.make(task, **kwargs)
    envRun = monitor.Monitor(envRun, folder='CSGTask/RL/Plots/Run/'+ModelDir, 
                            sv_per=numSteps, verbose=False, sv_fig=True, num_stps_sv_fig=numSteps)
    
    # Run
    for i_episode in range((conditions)*Run_cycle):
        observation = envRun.reset()
        for t in range(numSteps):
            action = envRun.action_space.sample()
            observation, reward, done, info = envRun.step(action)
            if done:
                print("Episode finished after {} timesteps".format((t+1)-info['SetStart']-info['ThresholdDelay']))
                print(info['Interval'])
                break

    envRun.close()

# Train Env
if train:
    print("Train Environment")

    # Define Env
    kwargs = {'params': (True, InputNoise, TargetThreshold, ThresholdDelay)}
    envTrain = gym.make(task, **kwargs)
    envTrain.reset()
    envTrain = monitor.Monitor(envTrain, folder='CSGTask/RL/Plots/Train/', 
                                sv_per=numSteps, verbose=False, sv_fig=True, num_stps_sv_fig=numSteps)
    envTrain = DummyVecEnv([lambda: envTrain])

    # Define Model
    model = A2C(LstmPolicy, envTrain, verbose=1, 
                gamma=1, alpha=1, #max_grad_norm=0.25,
                learning_rate=LR, #lr_schedule='linear', 
                tensorboard_log="CSGTask/RL/Models/CSG_RL_Tensorboard/",
                policy_kwargs={'feature_extraction':"mlp", 'act_fun':tf.nn.tanh ,'n_lstm':200, 'net_arch':[2, 'lstm', 200, 1]})

    # Train
    print("Start Training")
    model.learn(total_timesteps=numSteps*conditions*Train_cycle, log_interval=numSteps*conditions, 
                tb_log_name=ModelDir, reset_num_timesteps=True)
    model.save('CSGTask/RL/Models/CSG_RL_Models/'+ModelDir+'.zip')
    print("Done")

    envTrain.close()

# Run Trained Env
if runTrained:
    print("Run Trained Environment")

    # Define Env
    kwargs = {'params': (False, InputNoise, TargetThreshold, ThresholdDelay)}
    envRunTrained = gym.make(task, **kwargs)
    envRunTrained = monitor.Monitor(envRunTrained, folder='CSGTask/RL/Plots/RunTrained/', sv_per=numSteps, verbose=False, sv_fig=True, num_stps_sv_fig=3500)
    envRunTrained = DummyVecEnv([lambda: envRunTrained])

    # Define Model
    modelTrained = A2C.load(load_path = '/Users/ritikmehta/School/Bsc_NB3/BEP/neurogym/CSGTask/RL/Models/CSG_RL_Models/IN0.5%_TT1.0%/30Cycles_1e-05LR.zip', 
                            env = envRunTrained)

    # Evaluate Model
    for i_episode in range(conditions*Run_cycle):
        observation = envRunTrained.reset()
        for t in range(numSteps):
            action = modelTrained.predict(observation)
            observation, reward, done, info = envRunTrained.step(action)
            if done:
                print("Episode finished after {} timesteps".format((t+1)-info['SetStart']-info['ThresholdDelay']))
                print(info['Interval'])
                break
        print("Reach End")

    envRunTrained.close()
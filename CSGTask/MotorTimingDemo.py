# Libraries
import warnings
import gym
import neurogym as ngym
import tensorflow as tf
from neurogym.utils import plotting
import matplotlib.pyplot as plt
import numpy as np
from neurogym.wrappers import monitor, noise
from stable_baselines.common.policies import LstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
warnings.filterwarnings('ignore')

# Define Parameters
plot = False
run = False
train = True

numSteps = 3500 # Number of Steps
conditions = 10 # Number of Conditions

numTrials = 10 # Number of Trials for Plot
Run_cycle = 5 # Number of Cycles for Run
Train_cycle = 1 # Number of Cycles for Training
    # TotalSteps= 3500 * 10 * Train_cycle

# Define Env
print("Define Environment")
task = 'MotorTiming-v0'
kwargs = {'training': True}
env = gym.make(task, **kwargs)
env.close()

# Plot Env
if plot:
    print("Plot Environment")
    kwargs = {'training': False}
    env_plot = gym.make(task, **kwargs)
    env_data = plotting.run_env(env_plot, num_steps=numSteps, num_trials=numTrials, def_act=0)

    fig, axs = plt.subplots(3, sharex=True, sharey=True)
    fig.suptitle('CSG Input')
    plt.xlabel("Time")

    axs[0].plot(env_data['ob'][:,1])
    axs[0].set_title('Context Cue')
    axs[0].set_ylabel('Context Magnitude')

    axs[1].plot(env_data['ob'][:,3])
    axs[1].set_title('Set Cue')
    axs[1].set_ylabel('Set Magnitude')

    axs[2].plot(env_data['gt'])
    axs[2].set_title('Target')
    axs[2].set_ylabel('Target Magnitude')
    plt.show()

    env_plot.close()

# Run Env
if run:
    print("Run Environment")

    kwargs = {'training': True}
    env_run = gym.make(task, **kwargs)
    for i_episode in range(conditions*Run_cycle):
        observation = env_run.reset()
        for t in range(numSteps):
            action = env_run.action_space.sample()
            observation, reward, done, info = env_run.step(action)
            if done:
                print("Episode finished after {} timesteps".format((t+1)-info['waitTime']))
                break

    env_run.close()

# Train Env
if train:
    print("Train Environment")

    kwargs = {'training': True}
    env_train = gym.make(task, **kwargs)
    env_train.reset()

    env_train = monitor.Monitor(env_train, folder='CSGTask/Plots/', sv_per=50000, verbose=1, sv_fig=True, num_stps_sv_fig=1000)

    env_train = DummyVecEnv([lambda: env_train])
    model = A2C(LstmPolicy, env_train, gamma=1, alpha=1, verbose=1, 
            policy_kwargs={'feature_extraction':"mlp", 'act_fun':tf.nn.tanh ,'n_lstm':200, 'net_arch':[2, 'lstm', 200, 1]})

    print("Start Learning")
    model.learn(total_timesteps=numSteps*conditions*Train_cycle, log_interval=numSteps*conditions)
    model.save('CSGTask/Models/CSGModel')
    print("Done")

    env_train.close()
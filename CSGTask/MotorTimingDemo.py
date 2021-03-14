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
from stable_baselines import A2C
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
warnings.filterwarnings('ignore')

# Define Parameters
plot = False
run = False
train = False
runTrained = True

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

    # Define Env
    kwargs = {'training': False}
    envPlot = gym.make(task, **kwargs)
    env_data = plotting.run_env(envPlot, num_steps=numSteps, num_trials=numTrials, def_act=0)

    # Plot
    fig, axs = plt.subplots(3, sharex=True, sharey=True)
    fig.suptitle('CSG Input')
    plt.xlabel("Time")

        # Context Cue
    axs[0].plot(env_data['ob'][:,1])
    axs[0].set_title('Context Cue')
    axs[0].set_ylabel('Context Magnitude')

        # Set Cue
    axs[1].plot(env_data['ob'][:,3])
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
    kwargs = {'training': True}
    envRun = gym.make(task, **kwargs)

    # Run
    for i_episode in range(conditions*Run_cycle):
        observation = envRun.reset()
        for t in range(numSteps):
            action = envRun.action_space.sample()
            observation, reward, done, info = envRun.step(action)
            if done:
                print("Episode finished after {} timesteps".format((t+1)-info['waitTime']))
                break

    envRun.close()

# Train Env
if train:
    print("Train Environment")

    # Define Env
    kwargs = {'training': True}
    envTrain = gym.make(task, **kwargs)
    envTrain.reset()
    envTrain = monitor.Monitor(envTrain, folder='CSGTask/Plots/', sv_per=50000, verbose=1, sv_fig=True, num_stps_sv_fig=1000)
    envTrain = DummyVecEnv([lambda: envTrain])

    # Define Model
    model = A2C(LstmPolicy, envTrain, gamma=1, alpha=1, verbose=1, 
            policy_kwargs={'feature_extraction':"mlp", 'act_fun':tf.nn.tanh ,'n_lstm':200, 'net_arch':[2, 'lstm', 200, 1]})

    # Train
    print("Start Training")
    model.learn(total_timesteps=numSteps*conditions*Train_cycle, log_interval=numSteps*conditions)
    model.save('CSGTask/Models/CSGModel')
    print("Done")

    envTrain.close()

# Run Trained Env
if runTrained:
    print("Run Trained Environment")

    # Define Env
    kwargs = {'training': False}
    envRunTrained = gym.make(task, **kwargs)
    envRunTrained = DummyVecEnv([lambda: envRunTrained])

    # Define Model
    modelTrained = A2C.load(load_path = '/Users/ritikmehta/School/Bsc_NB3/BEP/neurogym/CSGTask/Models/CSGModel.zip', 
                            env = envRunTrained)

    # Evaluate Model
    # mean_reward, std_reward = evaluate_policy(modelTrained, envRunTrained, n_eval_episodes=10)

    for i_episode in range(conditions):
        observation = envRunTrained.reset()
        for t in range(numSteps):
            action = modelTrained.predict(observation)
            observation, reward, done, info = envRunTrained.step(action)
            if done:
                print("Episode finished after {} timesteps".format((t+1)))
                break
        print("Reach End")

    envRunTrained.close()
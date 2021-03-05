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

# Define Env
print("Define Environment")
task = 'MotorTiming-v0'
kwargs = {'training': True, 'batchSize': 5}
env = gym.make(task, **kwargs)
env.close()

# Plot Env
print("Plot Environment")
kwargs = {'training': False}
env_plot = gym.make(task, **kwargs)
env_data = plotting.run_env(env_plot, num_steps=3500, num_trials=8, def_act=0)

fig, axs = plt.subplots(2, sharex=True, sharey=True)
fig.suptitle('CSG Input')
plt.xlabel("Time")

axs[0].plot(env_data['ob'][:,1])
axs[0].set_title('Context Cue')
axs[0].set_ylabel('Context Magnitude')

axs[1].plot(env_data['ob'][:,3])
axs[1].set_title('Set Cue')
axs[1].set_ylabel('Set Magnitude')

plt.show()
env_plot.close()

# Run Env
print("Run Environment")
repeat = 5
conditions = 8
cycle = 1

kwargs = {'training': True, 'batchSize': repeat}
env_run = gym.make(task, **kwargs)
for i_episode in range((repeat*conditions*cycle)):
    observation = env_run.reset()
    for t in range(3500):
        action = env_run.action_space.sample()
        observation, reward, done, info = env_run.step(action)
        if done:
            print("Episode finished after {} timesteps".format((t+1)-info['wait_time']))
            break
env_run.close()

# Train Env
print("Train Environment")
numSteps = 3500 # Number of steps 
repeat = 5 # Repeats per condition
conditions = 8 # Number of conditions
cycle = 1 # Number of cycles

kwargs = {'training': True, 'batchSize': repeat}
env_train = gym.make(task, **kwargs)
env_train.reset()

env_train = monitor.Monitor(env_train, folder='CSGTask/Plots/', sv_per=50000, verbose=1, sv_fig=True, num_stps_sv_fig=1000)

env_train = DummyVecEnv([lambda: env_train])
model = A2C(LstmPolicy, env_train, gamma=1, alpha=1, verbose=1, 
        policy_kwargs={'feature_extraction':"mlp", 'act_fun':tf.nn.tanh ,'n_lstm':200, 'net_arch':[2, 'lstm', 200, 1]})

print("Start Learning")
# Under repeats
# model.learn(total_timesteps=numSteps*conditions*repeat*cycle, log_interval=numSteps*conditions)
# Under cycles
# model.learn(total_timesteps=numSteps*conditions*repeat, log_interval=3500*8)
#model.save('CSGTask/Models/CSGModel')
print("Done")

env_train.close()
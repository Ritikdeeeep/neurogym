# Libraries
from __future__ import print_function, division, absolute_import
import warnings
import gym
import neurogym as ngym
import tensorflow as tf
from neurogym.utils import plotting
from neurogym.wrappers import monitor
import matplotlib.pyplot as plt
import csv

import SL_Sussillo.integrator as integrator
import SL_Sussillo.rnn as rnn
import SL_Sussillo.utils as utils

import datetime
import h5py
import jax.numpy as np
from jax import random
from jax.experimental import optimizers
import numpy as onp
import os
import sys
import time
from importlib import reload

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
warnings.filterwarnings('ignore')

###################################################################################

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

###################################################################################

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
        onp.savetxt('CSGTask/Models/CSG_SL/Inputs/'+'inputs_{}.csv'.format(i_trial), observationList, delimiter=",")
        
    envInput.close()

# Read Input:
# IN = onp.loadtxt('CSGTask/Models/CSG_SL/Inputs/inputs_0.csv', delimiter=',')

###################################################################################

# Integration parameters
T = 3.5          # Arbitrary amount time, roughly physiological.
ntimesteps = 3500# Divide T into this many bins
bval = 0.01      # bias value limit
sval = 0.025     # standard deviation (before dividing by sqrt(dt))
input_params = (bval, sval, T, ntimesteps)

# Integrator RNN hyperparameters
u = 2         # Number of inputs to the RNN
n = 200       # Number of units in the RNN
o = 1         # Number of outputs in the RNN

# The scaling of the recurrent parameters in an RNN really matters. 
# The correct scaling is 1/sqrt(number of recurrent inputs), which 
# yields an order 1 signal output to a neuron if the input is order 1.
# Given that VRNN uses a tanh nonlinearity, with min and max output 
# values of -1 and 1, this works out.  The scaling just below 1 
# (0.95) is because we know we are making a line attractor so, we 
# might as well start it off basically right 1.0 is also basically 
# right, but perhaps will lead to crazier dynamics.
param_scale = 0.85 # Scaling of the recurrent weight matrix

# Optimization hyperparameters
num_batchs = 10000         # Total number of batches to train on.
batch_size = 128          # How many examples in each batch
eval_batch_size = 1024    # How large a batch for evaluating the RNN
step_size = 0.025          # initial learning rate
decay_factor = 0.99975     # decay the learning rate this much
# Gradient clipping is HUGELY important for training RNNs
max_grad_norm = 10.0      # max gradient norm before clipping, clip to this value.
l2reg = 0.0002           # amount of L2 regularization on the weights
adam_b1 = 0.9             # Adam parameters
adam_b2 = 0.999
adam_eps = 1e-1
print_every = 100          # Print training informatino every so often

###################################################################################

# JAX handles randomness differently than numpy or matlab. 
# one threads the randomness through to each function. 
# It's a bit tedious, but very easy to understand and with reliable effect.
seed = onp.random.randint(0, 1000000) # get randomness from CPU level numpy
print("Seed: %d" % seed)
key = random.PRNGKey(seed) # create a random key for jax for use on device.

# Plot a few input/target examples to make sure things look sane.
ntoplot = 10    # how many examples to plot
# With this split command, we are always getting a new key from the old key,
# and I use first key as as source of randomness for new keys.
#     key, subkey = random.split(key, 2)
#     ## do something random with subkey
#     key, subkey = random.split(key, 2)
#     ## do something random with subkey
# In this way, the same top level randomness source stays random.

# The number of examples to plot is given by the number of 
# random keys in this function.
key, skey = random.split(key, 2)
skeys = random.split(skey, ntoplot) # get ntoplot random keys
inputs, targets = integrator.build_inputs_and_targets_jit(input_params, skeys)

# Plot the input to the RNN and the target for the RNN.
integrator.plot_batch(ntimesteps, inputs, targets)

###################################################################################

# Init some parameters for training.
key, subkey = random.split(key, 2)
init_params = rnn.random_vrnn_params(subkey, u, n, o, g=param_scale)
rnn.plot_params(init_params)

###################################################################################

# Create a decay function for the learning rate
decay_fun = optimizers.exponential_decay(step_size, decay_steps=1, decay_rate=decay_factor)

batch_idxs = onp.linspace(1, num_batchs)
plt.plot(batch_idxs, [decay_fun(b) for b in batch_idxs])
plt.axis('tight')
plt.xlabel('Batch number')
plt.ylabel('Learning rate')

###################################################################################

reload(rnn)
# Initialize the optimizer.  Please see jax/experimental/optimizers.py
opt_init, opt_update, get_params = optimizers.adam(decay_fun, adam_b1, adam_b2, adam_eps)
opt_state = opt_init(init_params)

# Run the optimization loop, first jit'd call will take a minute.
start_time = time.time()
all_train_losses = []
for batch in range(num_batchs):
    key, subkey = random.split(key, 2)
    skeys = random.split(subkey, batch_size)
    inputs, targets = integrator.build_inputs_and_targets_jit(input_params, skeys)
    opt_state = rnn.update_w_gc_jit(batch, opt_state, opt_update, get_params, inputs,
                                  targets, max_grad_norm, l2reg)
    if batch % print_every == 0:
        params = get_params(opt_state)
        all_train_losses.append(rnn.loss_jit(params, inputs, targets, l2reg))
        train_loss = all_train_losses[-1]['total']
        batch_time = time.time() - start_time
        step_size = decay_fun(batch)
        s = "Batch {} in {:0.2f} sec, step size: {:0.5f}, training loss {:0.4f}"
        print(s.format(batch, batch_time, step_size, train_loss))
        start_time = time.time()
        
# List of dicts to dict of lists
all_train_losses = {k: [dic[k] for dic in all_train_losses] for k in all_train_losses[0]}

###################################################################################

# Show the loss through training.
xlims = [2, 50]
plt.figure(figsize=(16,4))
plt.subplot(141)
plt.plot(all_train_losses['total'][xlims[0]:xlims[1]], 'k')
plt.title('Total')

plt.subplot(142)
plt.plot(all_train_losses['lms'][xlims[0]:xlims[1]], 'r')
plt.title('Least mean square')

plt.subplot(143)
plt.plot(all_train_losses['l2'][xlims[0]:xlims[1]], 'g')
plt.title('L2')

plt.subplot(144)
plt.plot(all_train_losses['total'][xlims[0]:xlims[1]], 'k')
plt.plot(all_train_losses['lms'][xlims[0]:xlims[1]], 'r')
plt.plot(all_train_losses['l2'][xlims[0]:xlims[1]], 'g')
plt.title('All losses')

###################################################################################

# Take a batch for an evalulation loss, notice the L2 penalty is 0 for the evaluation.
params = get_params(opt_state)

key, subkey = random.split(key, 2)
skeys = random.split(subkey, batch_size)
inputs, targets = integrator.build_inputs_and_targets_jit(input_params, skeys)
eval_loss = rnn.loss_jit(params, inputs, targets, l2reg=0.0)['total']
eval_loss_str = "{:.5f}".format(eval_loss)
print("Loss on a new large batch: %s" % (eval_loss_str))

###################################################################################

reload(rnn)

# Visualize how good this trained integrator is
def inputs_targets_no_h0s(keys):
    inputs_b, targets_b = \
        integrator.build_inputs_and_targets_jit(input_params, keys)
    h0s_b = None # Use trained h0
    return inputs_b, targets_b, h0s_b

rnn_run = lambda inputs: rnn.batched_rnn_run(params, inputs)

give_trained_h0 = lambda batch_size : np.array([params['h0']] * batch_size)

rnn_internals = rnn.run_trials(rnn_run, inputs_targets_no_h0s, 1, 16)

integrator.plot_batch(ntimesteps, rnn_internals['inputs'], 
                      rnn_internals['targets'], rnn_internals['outputs'], 
                      onp.abs(rnn_internals['targets'] - rnn_internals['outputs']))

###################################################################################

# Visualize the hidden state, as an example.
rnn.plot_examples(ntimesteps, rnn_internals, nexamples=4)

###################################################################################

# Take a look at the trained parameters.
rnn.plot_params(params)

###################################################################################

# Define directories, etc.
task_type = 'pure_int'
rnn_type = 'vrnn'
fname_uniquifier = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
data_dir = os.path.join(os.path.join('/tmp', rnn_type), task_type)

print(data_dir)
print(fname_uniquifier)

###################################################################################

# Save parameters

params_fname = ('trained_params_' + rnn_type + '_' + task_type + '_' + \
                eval_loss_str + '_' + fname_uniquifier + '.h5')
params_fname = os.path.join(data_dir, params_fname)

print("Saving params in %s" % (params_fname))
utils.write_file(params_fname, params)

###################################################################################
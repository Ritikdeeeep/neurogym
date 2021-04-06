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

import CSGTask.SL.SussilloCode.CSG_Input as genInput
import CSGTask.SL.SussilloCode.CSG_rnn as rnn
import CSGTask.SL.SussilloCode.DS_rnn as DSrnn
import CSGTask.SL.SussilloCode.utils as utils

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

# CSG parameters
inputNoise = 0 # Context Cue Noise 
numSteps = 3500
numConditions = 10
training = False
input_params_NP = (inputNoise, numSteps, training)
input_params_JAX = (inputNoise, numSteps, numConditions, training)

# RNN hyperparameters
u = 2         # Number of inputs to the RNN
n = 200       # Number of units in the RNN
o = 1         # Number of outputs in the RNN

param_scale = 0.85 # Scaling of the recurrent weight matrix

# Optimization hyperparameters
num_batchs = 100           # Total number of batches to train on.
batch_size = 10            # How many examples in each batch
eval_batch_size = 20       # How large a batch for evaluating the RNN
step_size = 0.025          # initial learning rate
decay_factor = 0.99975     # decay the learning rate this much
# Gradient clipping is HUGELY important for training RNNs
max_grad_norm = 10.0      # max gradient norm before clipping, clip to this value.
l2reg = 0.0002            # amount of L2 regularization on the weights
adam_b1 = 0.9             # Adam parameters
adam_b2 = 0.999
adam_eps = 1e-1
print_every = 10          # Print training information every so often

###################################################################################

seed = onp.random.randint(0, 1000000) # get randomness from CPU level numpy
print("Seed: %d" % seed)
key = random.PRNGKey(seed) # create a random key for jax for use on device.

# Plot a few input/target examples to make sure things look sane.
ntoplot = 10    # how many examples to plot

key, skey = random.split(key, 2)
skeys = random.split(skey, ntoplot) # get ntoplot random keys

# rng = onp.random.default_rng(seed)
# BatchIters = rng.uniform(low=0, high=numConditions, size=batch_size)
ContextCue, SetCue, Inputs, Targets = genInput.build_input_and_target_NP(input_params_NP, skeys)
genInput.plot_NeuroGym(ContextCue, SetCue, Targets)

################################################################################### Change DSrnn to rnn = Numpy instead of JAX

# Init some parameters for training.
key, subkey = random.split(key, 2)
init_params = DSrnn.random_vrnn_params(subkey, u, n, o, g=param_scale)
DSrnn.plot_params(init_params)

###################################################################################

# Create a decay function for the learning rate
decay_fun = optimizers.exponential_decay(step_size, decay_steps=1, decay_rate=decay_factor)

batch_idxs = onp.linspace(1, num_batchs)
plt.plot(batch_idxs, [decay_fun(b) for b in batch_idxs])
plt.axis('tight')
plt.xlabel('Batch number')
plt.ylabel('Learning rate')

###################################################################################

reload(DSrnn)
# Initialize the optimizer.  Please see jax/experimental/optimizers.py
opt_init, opt_update, get_params = optimizers.adam(decay_fun, adam_b1, adam_b2, adam_eps)
opt_state = opt_init(init_params)

# Run the optimization loop, first jit'd call will take a minute.
start_time = time.time()
all_train_losses = []
for batch in range(num_batchs):
    rng = onp.random.default_rng(batch)
    BatchIters = rng.uniform(low=0, high=numConditions, size=batch_size)
    ContextCue, SetCue, inputs, targets  = genInput.build_input_and_target_NP(input_params_NP, BatchIters)
    inputs = onp.stack((ContextCue, SetCue), axis=2) # Shape (10, 3500, 2), with pairs of ContextCue and SetCue
    opt_state = DSrnn.update_w_gc_jit(batch, opt_state, opt_update, get_params, inputs, targets, max_grad_norm, l2reg)
    if batch % print_every == 0:
        params = get_params(opt_state)
        all_train_losses.append(DSrnn.loss_jit(params, inputs, targets, l2reg))
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

BatchIters = onp.random.uniform(low=0, high=10, size=batch_size)
ContextCue, SetCue, inputs, targets = genInput.build_input_and_target_NP(input_params_NP, BatchIters)
eval_loss = DSrnn.loss_jit(params, inputs, targets, l2reg=0.0)['total']
eval_loss_str = "{:.5f}".format(eval_loss)
print("Loss on a new large batch: %s" % (eval_loss_str))

###################################################################################

# Take a look at the trained parameters.
DSrnn.plot_params(params)

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
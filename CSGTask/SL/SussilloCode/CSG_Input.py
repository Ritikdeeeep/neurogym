# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Routines for creating white noise and integrated white noise."""

from __future__ import print_function, division, absolute_import
import jax.numpy as np
from jax import jit, vmap, partial, lax, random
from jax import random
import matplotlib.pyplot as plt
import CSGTask.SL.SussilloCode.utils as utils
import neurogym
import gym
import numpy as onp
import pandas as pd

# Numpy Version with NeuroGym
def build_input_and_target_NP(input_params, batch):
  inputNoise, numSteps, training = input_params
  task = 'MotorTiming-v0'
  BatchValues = batch

  CC_DF = pd.DataFrame(columns=range(numSteps), index=range(len(BatchValues)))
  SC_DF = pd.DataFrame(columns=range(numSteps), index=range(len(BatchValues)))
  IN_DF = pd.DataFrame(columns=range(numSteps), index=range(2*len(BatchValues)))
  T_DF = pd.DataFrame(columns=range(numSteps), index=range(len(BatchValues)))

  for i in range(len(BatchValues)):
    if training:
      kwargs = {'training': False, 'InputNoise': inputNoise, 'Scenario': i}
    else:
      kwargs = {'training': False, 'InputNoise': inputNoise, 'Scenario': int(batch[i])}
    env = gym.make(task, **kwargs)
    CC, SC, T = env._new_trial(**kwargs)
    CC_DF.loc[i] = CC
    SC_DF.loc[i] = SC
    IN_DF.loc[(2*i)] = CC
    IN_DF.loc[(2*i)+1] = SC
    T_DF.loc[i] = T
    env.close()

  CCList=CC_DF.to_numpy()
  SCList=SC_DF.to_numpy()
  INList=IN_DF.to_numpy()
  TList=T_DF.to_numpy()

  onp.savetxt("CSGTask/SL/Inputs/CC.csv", CCList, delimiter=',')
  onp.savetxt("CSGTask/SL/Inputs/SC.csv", SCList, delimiter=',')
  onp.savetxt("CSGTask/SL/Inputs/IN.csv", INList, delimiter=',')
  onp.savetxt("CSGTask/SL/Inputs/T.csv", TList, delimiter=',')

  return onp.array(CCList), onp.array(SCList), onp.array(INList), onp.array(TList)

# JAXed Version without NeuroGym
def build_input_and_target_JAX(input_params, key):
  """Build CSG input and targets."""
  inputNoise, numSteps, numConditions, training = input_params
  key, skeys = utils.keygen(key, 2)

  def arguments(arg_params, key):
    numConditions, training = arg_params
    
    production_ind = np.array(range(numConditions))
    intervals = np.array([720., 760., 800., 840., 880., 1420., 1460., 1500., 1540., 1580.])

    # Choose index
    if training == False:
      trial = {
        'production_ind': random.choice(next(skeys), production_ind, shape=(), replace=True) # Choose random prod ind using key
        # 'production_ind': onp.random.choice(production_ind, replace=True)
        }
    else:
      trial = {
        'production_ind': next(iter(range(numConditions))) #production_ind[(trial_nr % 10)-1] # Choose next prod ind for training
        }

    # Select corresponding interval
    trial['production'] = intervals[trial['production_ind']]

    return trial['production']

  # ProdInt is considered JAX value and not integer, does not allow Target Shape definition
  arg_params = (numConditions, training)
  ProdInt = arguments(arg_params, key)

  context_mag = np.array(lax.add(lax.mul((0.3/950), ProdInt), (0.2-(0.3/950)*700)))
  
  trialDuration = numSteps
  burnTime = 50
  setTime = 20

  # waitTime is considered JAX value and not integer, does not allow Set Shape definition
  waitTime = random.choice(next(skeys), np.asarray(list(range(100,200))), shape=(), replace=True) #JAX
  # waitTime = onp.random.choice(list(range(100,200))) # Numpy

  setPre = burnTime + waitTime
  setPost = trialDuration - burnTime - waitTime - setTime 
  targetPre = burnTime + waitTime + setTime
  targetPost = trialDuration - ProdInt

  # Select corresponding context cue (Signal + 0.5% Noise)
  contextCue1 = np.zeros(shape=(burnTime, 1))
  contextSignal = context_mag
  noiseSigmaContext = np.array(contextSignal * inputNoise)
  contextNoise = np.array(random.normal(key = next(skeys), shape=(trialDuration-burnTime, 1))*noiseSigmaContext)
  contextCue2 = lax.add(contextSignal, contextNoise)
  contextCue = np.concatenate((np.asarray(contextCue1), np.asarray(contextCue2)), axis=0)

  setCue1 = np.zeros(shape=(setPre, 1))
  setCue2 = np.full(shape=(setTime, 1), fill_value = 0.4)
  setCue3 = np.zeros(shape=(setPost, 1))
  setCue = np.concatenate((setCue1, np.concatenate((setCue2, setCue3), axis=0)), axis=0)

  inputs = np.stack((contextCue, setCue), axis=2)

  # trial['production'] is a JAX value, but is not accepted for shape
  # Has to be with JAX, otherwise not updated in batches
  target0 = np.zeros(shape=(targetPre, 1))
  target1 = np.full(shape=(ProdInt, 1), fill_value=np.nan)
  target2 = np.ones(shape=(targetPost, 1))
  target = np.concatenate((target0, np.concatenate((target1, target2), axis=0)), axis=0)

  return contextCue, setCue, inputs, target

# Now batch it and jit.
build_input_and_target = build_input_and_target_JAX
build_inputs_and_targets = vmap(build_input_and_target, in_axes=(None, 0))
build_inputs_and_targets_jit = jit(build_inputs_and_targets, static_argnums=(0,))

def plot_NeuroGym(Cue, Set, T):
  fig, axs = plt.subplots(3, sharex=True, sharey=True)
  fig.suptitle('CSG Input')
  plt.xlabel("Time")

  # Context Cue
  axs[0].plot(Cue.T)
  axs[0].set_title('Context Cue')
  axs[0].set_ylabel('Context Magnitude')

  # Set Cue
  axs[1].plot(Set.T)
  axs[1].set_title('Set Cue')
  axs[1].set_ylabel('Set Magnitude')

  # Target
  axs[2].plot(T.T)
  axs[2].set_title('Target')
  axs[2].set_ylabel('Target Magnitude')
  # plt.show()
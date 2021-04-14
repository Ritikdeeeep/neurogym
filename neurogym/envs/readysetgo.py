#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ready-set-go task."""

import numpy as np
import jax
from jax import lax, random
import jax.numpy as jnp

import neurogym as ngym
from neurogym import spaces

# Name the task you're using to MotorTiming

class ReadySetGo(ngym.TrialEnv):
    r"""Agents have to measure and produce different time intervals.

    A stimulus is briefly shown during a ready period, then again during a
    set period. The ready and set periods are separated by a measure period,
    the duration of which is randomly sampled on each trial. The agent is
    required to produce a response after the set cue such that the interval
    between the response and the set cue is as close as possible to the
    duration of the measure period.

    Args:
        gain: Controls the measure that the agent has to produce. (def: 1, int)
        prod_margin: controls the interval around the ground truth production
            time within which the agent receives proportional reward
    """
    metadata = {
        'paper_link': 'https://www.sciencedirect.com/science/article/pii/' +
        'S0896627318304185',
        'paper_name': '''Flexible Sensorimotor Computations through Rapid
        Reconfiguration of Cortical Dynamics''',
        'tags': ['timing', 'go-no-go', 'supervised']
    }

    def __init__(self, dt=80, rewards=None, timing=None, gain=1,
                 prod_margin=0.2):
        super().__init__(dt=dt)
        self.prod_margin = prod_margin

        self.gain = gain

        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            'fixation': 100,
            'ready': 83,
            'measure': lambda: self.rng.uniform(800, 1500),
            'set': 83}
        if timing:
            self.timing.update(timing)

        self.abort = False
        # set action and observation space
        name = {'fixation': 0, 'ready': 1, 'set': 2}
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3,),
                                            dtype=np.float32, name=name)

        name = {'fixation': 0, 'go': 1}
        self.action_space = spaces.Discrete(2, name=name)  # (fixate, go)

    def _new_trial(self, **kwargs):
        measure = self.sample_time('measure')
        trial = {
            'measure': measure,
            'gain': self.gain
        }
        trial.update(kwargs)

        trial['production'] = measure * trial['gain']

        self.add_period(['fixation', 'ready'])
        self.add_period('measure', duration=measure, after='fixation')
        self.add_period('set', after='measure')
        self.add_period('production', duration=2*trial['production'],
                        after='set')

        self.add_ob(1, where='fixation')
        self.set_ob(0, 'production', where='fixation')
        self.add_ob(1, 'ready', where='ready')
        self.add_ob(1, 'set', where='set')

        # set ground truth
        gt = np.zeros((int(2*trial['production']/self.dt),))
        gt[int(trial['production']/self.dt)] = 1
        self.set_groundtruth(gt, 'production')

        return trial

    def _step(self, action):
        trial = self.trial
        reward = 0
        ob = self.ob_now
        gt = self.gt_now
        new_trial = False
        if self.in_period('fixation'):
            if action != 0:
                new_trial = self.abort
                reward = self.rewards['abort']
        if self.in_period('production'):
            if action == 1:
                new_trial = True  # terminate
                # time from end of measure:
                t_prod = self.t - self.end_t['measure']
                eps = abs(t_prod - trial['production'])
                # actual production time
                eps_threshold = self.prod_margin*trial['production']+25
                if eps > eps_threshold:
                    reward = self.rewards['fail']
                else:
                    reward = (1. - eps/eps_threshold)**1.5
                    reward = max(reward, 0.1)
                    reward *= self.rewards['correct']
                    self.performance = 1

        return ob, reward, False, {'new_trial': new_trial, 'gt': gt}

class MotorTiming_CSG_SL_NP(ngym.TrialEnv):
    # CSG with Superised Learning with Numpy:
        # To export the observation space and ground truth
    """Agents have to produce different time intervals
    using different effectors (actions).

    Args:
        prod_margin: controls the interval around the ground truth production
                    time within which the agent receives proportional reward
    """

    metadata = {
        'paper_link': 'https://www.nature.com/articles/s41593-017-0028-6',
        'paper_name': '''Flexible timing by temporal scaling of
         cortical responses''',
        'tags': ['timing', 'go-no-go', 'supervised']
    }

    def __init__(self, dt=1, params=None):
        super().__init__(dt=dt)
        # Unpack Parameters
        Training, InputNoise, TargetThreshold, ThresholdDelay, Scenario = params

        self.production_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.intervals = [720, 760, 800, 840, 880, 1420, 1460, 1500, 1540, 1580] 
        self.context_mag= np.add(np.multiply((0.3/950), self.intervals), (0.2-(0.3/950)*700))
        self.scenario = Scenario
        # WeberFraction as the production margin (acceptable deviation)
        # Leave out for now, reimplement later otherwise unfair
        # self.weberFraction = float((100-50)/(1500-800))
        # self.prod_margin = self.weberFraction

        self.training = Training 
        self.trial_nr = 1
        self.InputNoise = InputNoise
        self.TargetThreshold = TargetThreshold
        self.ThresholdDelay = ThresholdDelay

        # Binary Rewards for incorrect and correct
        self.rewards = {'incorrect': 0., 'correct': +1.}    

        # Set Action and Observation Space
        # Allow Ramping between 0-1
        self.action_space = spaces.Box(0, 1, shape=(1,), dtype=np.float32)   
        # Context Cue: Burn Time followed by Cue & Set Cue: Wait followed by Set Spike
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32)

    
    def _new_trial(self, **kwargs):
        # Define Times
        self.trialDuration = 3500
        self.waitTime = int(self.rng.uniform(100, 200))
        self.burn = 50
        self.set = 20

        # Choose index (0-9) at Random
        if self.training == False:
            if self.scenario is not None:
                trial = {
                'production_ind': self.scenario
                }
            else:
                trial = {
                    'production_ind': self.rng.choice(self.production_ind)
                }

        # Choose index by Cycling through all conditions for Training
        if self.training == True: 
            trial = {
                'production_ind': self.production_ind[(self.trial_nr % 10)-1]
            }

        trial.update(kwargs)

        # Select corresponding interval
        trial['production'] = self.intervals[trial['production_ind']]

        # Select corresponding context cue (Signal + 0.5% Noise)
        contextSignal = self.context_mag[trial['production_ind']]
        noiseSigmaContext = contextSignal * self.InputNoise
        contextNoise = np.random.normal(0, noiseSigmaContext, (self.trialDuration-self.burn))
        contextCue = contextSignal + contextNoise

        # Define periods
        self.add_period('burn', duration= self.burn)
        self.add_period('cue', duration= self.trialDuration-self.burn, after='burn')
        self.add_period('wait', duration= self.waitTime, after='burn')
        self.add_period('set', duration= self.set, after='wait')
        self.add_period('production', duration=self.trialDuration-(self.set+self.waitTime+self.burn), after='set')

        # Set Burn to [0,0,0,0]
        ob = self.view_ob('burn')
        ob[:, 0] = 0
        ob[:, 1] = 0

        # Set Cue to contextCue
        ob = self.view_ob('cue')
        ob[:, 0] = contextCue

        # Set Set to 0.4
        ob = self.view_ob('set')
        ob[:, 1] = 0.4
        
        # Set Ground Truth as 0 at set and 1 at trial production with NaN inbetween   
        gt = np.empty([int((self.trialDuration/self.dt)),])
        gt[:] = np.nan
        gt[0:self.burn+self.waitTime+self.set] = 0
        gt[int(trial['production']):-1] = 1
        # gt = np.reshape(gt, [int(self.trialDuration/self.dt)] + list(self.action_space.shape))
        # self.set_groundtruth(gt)
        
        ob = self.view_ob(None)
        return ob[:,0], ob[:,1], gt

    def _step(self, action):
        trial = self.trial
        reward = 0
        ob = self.ob_now
        gt = self.gt_now
        new_trial = False

        if self.in_period('burn'):
            self.SetReward = False
            self.ThresholdReward = False
            self.TimeAfterThreshold = 0

        if self.in_period('set'):
            if action <= 0.05: # Should start close to 0
                reward = self.rewards['correct']
                self.SetReward = True

            if self.SetReward:
                reward = self.rewards['correct']
                self.performance = 1

        if self.in_period('production'): 
            if  action >= 0.90: # Action is over Threshold
                t_prod = self.t - self.end_t['set']  # Measure Time from Set
                eps = abs(t_prod - trial['production']) # Difference between Produced_Interval and Interval
                eps_threshold = int(trial['production']*self.TargetThreshold) # Allowed margin to produced interval

                if eps <= eps_threshold: # If Difference is below Margin, Finish Trial
                    reward = self.rewards['correct']
                    self.ThresholdReward = True
                else:
                    reward = self.rewards['incorrect']

            if self.ThresholdReward == True:
                reward = self.rewards['correct']
                self.performance = 1
                self.TimeAfterThreshold += 1

                if self.TimeAfterThreshold >= self.ThresholdDelay: # Give reward 100 steps after Success
                    # new_trial = True
                    self.ThresholdReward = False

            if self.t >= 3499:
                new_trial = True
                
        if new_trial == True:
            self.trial_nr += 1

        return ob, reward, new_trial, {
            'new_trial': new_trial, 
            'gt': gt, 
            'SetStart': self.waitTime+self.burn, 
            'Interval': trial['production'], 
            'ThresholdDelay': self.ThresholdDelay}

class MotorTiming_CSG_SL_JAX(ngym.TrialEnv):
    # CSG with Superised Learning with JAX:
        # To export the observation space and ground truth
    """Agents have to produce different time intervals
    using different effectors (actions).

    Args:
        prod_margin: controls the interval around the ground truth production
                    time within which the agent receives proportional reward
    """

    metadata = {
        'paper_link': 'https://www.nature.com/articles/s41593-017-0028-6',
        'paper_name': '''Flexible timing by temporal scaling of
         cortical responses''',
        'tags': ['timing', 'go-no-go', 'supervised']
    }

    def __init__(self, dt=1, params = None):
        super().__init__(dt=dt)
        # Unpack Parameters
        Training, InputNoise, TargetThreshold, ThresholdDelay, Scenario = params

        self.production_ind = jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.scenario = Scenario
        self.intervals = jnp.array([720., 760., 800., 840., 880., 1420., 1460., 1500., 1540., 1580.])
        self.context_mag= jnp.array(lax.add(lax.mul((0.3/950), self.intervals), (0.2-(0.3/950)*700)))
        self.seed = np.random.randint(0, 1000000)
        # WeberFraction as the production margin (acceptable deviation)
        # Leave out for now, reimplement later otherwise unfair
        # self.weberFraction = float((100-50)/(1500-800))
        # self.prod_margin = self.weberFraction

        self.training = Training 
        self.trial_nr = 1
        self.InputNoise = InputNoise
        self.TargetThreshold = TargetThreshold
        self.ThresholdDelay = ThresholdDelay

        # Binary Rewards for incorrect and correct
        self.rewards = {'incorrect': 0., 'correct': +1.}   

        # Set Action and Observation Space
        # Allow Ramping between 0-1
        self.action_space = spaces.Box(0, 1, shape=(1,), dtype=np.float32)   
        # Context Cue: Burn Time followed by Cue & Set Cue: Wait followed by Set Spike
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32)
    
    def _new_trial(self, **kwargs):
        # Define Times
        self.trialDuration = 3500
        self.waitTime = int(self.rng.uniform(100, 200))
        self.burn = 50
        self.set = 20

        # Choose index (0-9) at Random
        if self.training == False:
            if self.scenario is not None:
                trial = {
                'production_ind': self.scenario
                }
            else:
                trial = {
                    'production_ind': self.rng.choice(self.production_ind)
                }

        # Choose index by Cycling through all conditions for Training
        if self.training == True: 
            trial = {
                'production_ind': self.production_ind[(self.trial_nr % 10)-1]
            }

        trial.update(kwargs)

        # Select corresponding interval
        trial['production'] = self.intervals[trial['production_ind']]

        # Select corresponding context cue (Signal + 0.5% Noise)
        contextSignal = self.context_mag[trial['production_ind']]
        noiseSigmaContext = jnp.array(contextSignal * self.InputNoise)
        contextNoise = jnp.array(jax.random.normal(key = random.PRNGKey(self.seed), shape=(self.trialDuration-self.burn, 1))*noiseSigmaContext)
        contextCue = jnp.array(contextSignal + contextNoise)

        # Define periods
        self.add_period('burn', duration= self.burn)
        self.add_period('cue', duration= self.trialDuration-self.burn, after='burn')
        self.add_period('wait', duration= self.waitTime, after='burn')
        self.add_period('set', duration= self.set, after='wait')
        self.add_period('production', duration=self.trialDuration-(self.set+self.waitTime+self.burn), after='set')

        # Set Burn to [0,0,0,0]
        ob = self.view_ob('burn')
        ob[:,0] = 0
        ob[:,1] = 0
        #jax.ops.index_update(x=ob, idx=jax.ops.index[:,0:1], y=0)


        # Set Cue to contextCue
        ob = self.view_ob('cue')
        ob = jax.ops.index_update(x=ob, idx=jax.ops.index[:,0], y=jnp.reshape(contextCue, (len(contextCue))))

        # Set Set to 0.4
        ob = self.view_ob('set')
        ob = jax.ops.index_update(x=ob, idx=jax.ops.index[:,1], y=0.4)

        
        # Set Ground Truth as 0 at set and 1 at trial production with NaN inbetween   
        gt0 = jnp.zeros(shape=(self.burn+self.waitTime+self.set,1))
        gt1 = np.zeros(shape=(trial['production']))
        gt1[:] = np.nan
        gt2 = jnp.ones(shape=(self.trialDuration-trial['production'],1))
        gt = jnp.concatenate(gt0, jnp.concatenate(jnp.asarray(gt1), gt2))
        # gt = jnp.ones([int((self.trialDuration/self.dt)),])
        # gt1 = jax.ops.index_update(x=gt, idx=jax.ops.index[:], y=jnp.nan)
        # gt2 = jax.ops.index_update(x=gt1, idx=jax.ops.index[0:self.burn+self.waitTime+self.set], y=0)

        # self.set_groundtruth(gt)

        ob = jnp.asarray(self.view_ob(None))

        return ob[:,0], ob[:,1], gt[:]

    def _step(self, action):
        trial = self.trial
        reward = 0
        ob = self.ob_now
        gt = self.gt_now
        new_trial = False

        if self.in_period('burn'):
            self.SetReward = False
            self.ThresholdReward = False
            self.TimeAfterThreshold = 0

        if self.in_period('set'):
            if action <= 0.05: # Should start close to 0
                reward = self.rewards['correct']
                self.SetReward = True

            if self.SetReward:
                reward = self.rewards['correct']
                self.performance = 1

        if self.in_period('production'): 
            if  action >= 0.90: # Action is over Threshold
                t_prod = self.t - self.end_t['set']  # Measure Time from Set
                eps = abs(t_prod - trial['production']) # Difference between Produced_Interval and Interval
                eps_threshold = int(trial['production']*self.TargetThreshold) # Allowed margin to produced interval

                if eps <= eps_threshold: # If Difference is below Margin, Finish Trial
                    reward = self.rewards['correct']
                    self.ThresholdReward = True
                else:
                    reward = self.rewards['incorrect']

            if self.ThresholdReward == True:
                reward = self.rewards['correct']
                self.performance = 1
                self.TimeAfterThreshold += 1

                if self.TimeAfterThreshold >= self.ThresholdDelay: # Give reward 100 steps after Success
                    # new_trial = True
                    self.ThresholdReward = False

            if self.t >= 3499:
                new_trial = True
                
        if new_trial == True:
            self.trial_nr += 1

        return ob, reward, new_trial, {
            'new_trial': new_trial, 
            'gt': gt, 
            'SetStart': self.waitTime+self.burn, 
            'Interval': trial['production'], 
            'ThresholdDelay': self.ThresholdDelay}

class MotorTiming_CSG_RL(ngym.TrialEnv):
    # CSG with Reinforcement Learning:
        # To use with stable baselines
        # Different versions:
            # Long Trial: New Trial at 3500ms
            # Redefined Rewards: Also give negative rewards
            # Vanilla: New Trial after 100ms of Threshold Crossing and Binary Rewards
    """Agents have to produce different time intervals
    using different effectors (actions).

    Args:
        prod_margin: controls the interval around the ground truth production
                    time within which the agent receives proportional reward
    """

    metadata = {
        'paper_link': 'https://www.nature.com/articles/s41593-017-0028-6',
        'paper_name': '''Flexible timing by temporal scaling of
         cortical responses''',
        'tags': ['timing', 'go-no-go', 'supervised']
    }

    def __init__(self, dt=1, params=None):
        super().__init__(dt=dt)
        # Unpack Parameters
        Training, InputNoise, TargetThreshold, ThresholdDelay = params

        # Several different intervals: their length and corresponding magnitude
        self.production_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] 
        self.intervals = [720, 760, 800, 840, 880, 1420, 1460, 1500, 1540, 1580] 
        self.context_mag= np.add(np.multiply((0.3/950), self.intervals), (0.2-(0.3/950)*700))
         
        # WeberFraction as the production margin (acceptable deviation)
        # Leave out for now, reimplement later otherwise unfair
        # self.weberFraction = float((100-50)/(1500-800))
        # self.prod_margin = self.weberFraction
        
        self.training = Training # Training Label
        self.trial_nr = 1 # Trial Counter
        
        self.InputNoise = InputNoise # Input Noise Percentage
        self.TargetThreshold = TargetThreshold # Target Threshold Percentage
        self.ThresholdDelay = ThresholdDelay # Reward Delay after Threshold Crossing

        # Binary Rewards for incorrect and correct
        self.rewards = {'incorrect': 0., 'correct': +1.} #, 'wrong': -0.2}     

        # Set Action and Observation Space
        # Allow Ramping between 0-1
        self.action_space = spaces.Box(0, 1, shape=(1,), dtype=np.float32)   
        # Context Cue: Burn Time followed by Cue & Set Cue: Wait followed by Set Spike
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32)

    def _new_trial(self, **kwargs):
        # Define Times
        self.trialDuration = 3500 # Total Trial Duration
        self.waitTime = int(self.rng.uniform(100, 200)) # Random wait time between burn and set
        self.burn = 50 # Duration of Burn period before context cue
        self.set = 20 # Duration of Set Period

        # Choose interval index (0-9) at Random
        if self.training == False:
            trial = {
                'production_ind': self.rng.choice(self.production_ind)
            }

        # Choose index by Cycling through all conditions for Training
        if self.training == True: 
            trial = {
                'production_ind': self.production_ind[(self.trial_nr % len(self.production_ind))-1]
            }

        trial.update(kwargs)

        # Select corresponding interval length
        trial['production'] = self.intervals[trial['production_ind']]

        # Calculate corresponding context cue magnitude (Signal + 0.5% Noise)
        contextSignal = self.context_mag[trial['production_ind']]
        noiseSigmaContext = contextSignal * self.InputNoise
        contextNoise = np.random.normal(0, noiseSigmaContext, (self.trialDuration-self.burn))
        contextCue = contextSignal + contextNoise

        # Define periods
        self.add_period('burn', duration= self.burn)
        self.add_period('cue', duration= self.trialDuration-self.burn, after='burn')
        self.add_period('wait', duration= self.waitTime, after='burn')
        self.add_period('set', duration= self.set, after='wait')
        self.add_period('production', duration=self.trialDuration-(self.set+self.waitTime+self.burn), after='set')

        # Set Burn to [0,0,0,0]
        ob = self.view_ob('burn')
        ob[:, 0] = 0
        ob[:, 1] = 0

        # Set Cue to contextCue
        ob = self.view_ob('cue')
        ob[:, 0] = contextCue

        # Set Set to 0.4
        ob = self.view_ob('set')
        ob[:, 1] = 0.4
        
        # Set Ground Truth to Form Ramp & Reshape to Match Action Space
        # t_ramp = range(0, int(trial['production']))
        # gt_ramp = np.multiply(1/trial['production'], t_ramp)
        # gt_step = np.ones((int((self.trialDuration-(trial['production']+self.set+self.waitTime+self.burn))/self.dt),), dtype=np.float)
        # gt = np.concatenate((gt_ramp, gt_step)).astype(np.float)
        # gt = np.reshape(gt, [int(self.trialDuration-(self.set+self.waitTime+self.burn)/self.dt)] + list(self.action_space.shape))
        # self.set_groundtruth(gt, period='production')

        # Set Ground Truth as 0 at set and 1 at trial production with NaN inbetween       
        gt = np.empty([int((self.trialDuration/self.dt)),])
        gt[:] = np.nan
        gt[0:self.burn+self.waitTime+self.set] = 0
        gt[int(trial['production']):-1] = 1
        gt = np.reshape(gt, [int(self.trialDuration/self.dt)] + list(self.action_space.shape))
        self.set_groundtruth(gt)

        return trial

    def _step(self, action):
        trial = self.trial
        reward = 0
        ob = self.ob_now
        gt = self.gt_now
        new_trial = False

        if self.in_period('burn'):
            self.SetReward = False
            self.ThresholdReward = False
            self.TimeAfterThreshold = 0

        if self.in_period('set'):
            if action <= 0.05: # Should start close to 0
                reward = self.rewards['correct']
                self.SetReward = True
            # else:
            #     reward = self.rewards['wrong']

            if self.SetReward:
                reward = self.rewards['correct']
                self.performance = 1

        if self.in_period('production'): 
            if  action >= 0.90: # Action is over Threshold
                t_prod = self.t - self.end_t['set']  # Measure Time from Set
                eps = abs(t_prod - trial['production']) # Difference between Produced_Interval and Interval
                eps_threshold = int(trial['production']*self.TargetThreshold) # Allowed margin to produced interval

                if eps <= eps_threshold: # If Difference is below Margin, Finish Trial
                    reward = self.rewards['correct']
                    new_trial = True
                    self.ThresholdReward = True
                # else:
                #     reward = self.rewards['wrong']

            if self.ThresholdReward == True:
                reward = self.rewards['correct']
                self.performance = 1
                self.TimeAfterThreshold += 1

                if self.TimeAfterThreshold >= self.ThresholdDelay: # Give reward 100 steps after Success
                    new_trial = True
                    self.ThresholdReward = False
        
        # if self.t >= 3500:
        #     new_trial = True

        if new_trial == True:
            self.trial_nr += 1

        return ob, reward, new_trial, {
            'new_trial': new_trial, 
            'gt': gt, 
            'SetStart': self.waitTime+self.burn, 
            'Interval': trial['production'], 
            'ThresholdDelay': self.ThresholdDelay}

class MotorTiming(ngym.TrialEnv):
    # RSG with Reinforcement Learning:
    """Agents have to produce different time intervals
    using different effectors (actions).

    Args:
        prod_margin: controls the interval around the ground truth production
                    time within which the agent receives proportional reward
    """

    metadata = {
        'paper_link': 'https://www.nature.com/articles/s41593-017-0028-6',
        'paper_name': '''Flexible timing by temporal scaling of
         cortical responses''',
        'tags': ['timing', 'go-no-go', 'supervised']
    }

    def __init__(self, dt=1, params= None):
        super().__init__(dt=dt)
        # Unpack Parameters
        Training, InputNoise, TargetThreshold, ThresholdDelay = params

        # Several different intervals with their corresponding their length 
        self.production_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] 
        self.intervals = [480, 560, 640, 720, 800, 800, 900, 1000, 1100, 1200] 
        # Possible Context Cues:
        # (HandShortLeft, HandShortRight, EyeShortLeft, EyeShortRight, HandLongLeft, HandLongRight, EyeLongLeft, EyeLongRight)
        # Short: 0.1 - 0.4 & Long: 0.5 - 0.8
        self.context_mag = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
            # To what extent do we incorporate Hand/Eye and Left/Right
        
        # WeberFraction as the production margin (acceptable deviation)
        # Leave out for now, reimplement later otherwise unfair
        # self.weberFraction = float((100-50)/(1500-800))
        # self.prod_margin = self.weberFraction

        self.training = Training # Training Label
        self.trial_nr = 1 # Trial Counter
        self.InputNoise = InputNoise # Input Noise Percentage
        self.TargetThreshold = TargetThreshold # Target Threshold Percentage
        self.ThresholdDelay = ThresholdDelay

        # Binary Rewards for incorrect and correct
        self.rewards = {'incorrect': 0., 'correct': +1.}    

        # Set Action and Observation Space
        # Allow Ramping between 0-1
        self.action_space = spaces.Box(0, 1, shape=(1,), dtype=np.float32)   
        # Context Cue: Burn Time followed by Cue & Ready-Set Cue: Wait followed by Ready-Set Spike
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32)

    def _new_trial(self, **kwargs):
        # Define Times
        self.trialDuration = 3500
        self.waitTime = int(self.rng.uniform(100, 200))
        self.burn = 50
        self.spike = 10

        # Choose index (0-9) at Random
        if self.training == False:
            trial = {
                'production_ind': self.rng.choice(self.production_ind)
            }

        # Choose index by Cycling through all conditions for Training
        if self.training == True: 
            trial = {
                'production_ind': self.production_ind[(self.trial_nr % len(self.production_ind))-1]
            }

        trial.update(kwargs)

        # Select corresponding interval
        trial['production'] = self.intervals[trial['production_ind']]

        # Select corresponding context cue (Signal + 0.5% Noise)
        if trial['production_ind'] <= 4:
            contextSignal = self.rng.choice(self.context_mag[0:3])
        else:
            contextSignal = self.rng.choice(self.context_mag[4:7])
        noiseSigmaContext = contextSignal * self.InputNoise
        contextNoise = np.random.normal(0, noiseSigmaContext, (self.trialDuration-self.burn))
        contextCue = contextSignal + contextNoise

        # Define periods
        self.add_period('burn', duration= self.burn)
        self.add_period('cue', duration= self.trialDuration-self.burn, after='burn')
        self.add_period('wait', duration= self.waitTime, after='burn')
        self.add_period('ready', duration= self.spike, after='wait')
        self.add_period('estimation', duration= trial['production'], after='ready')
        self.add_period('set', duration= self.spike, after='estimation')
        self.add_period('production', duration=self.trialDuration-(self.spike+trial['production']+self.spike+self.waitTime+self.burn), after='set')

        # Set Burn to 0
        ob = self.view_ob('burn')
        ob[:, 0] = 0
        ob[:, 1] = 0

        # Set Cue to contextCue
        ob = self.view_ob('cue')
        ob[:, 0] = contextCue

        # Set Wait to contextCue
        ob = self.view_ob('wait')
        ob[:, 1] = 0

        # Set Ready to 1
        ob = self.view_ob('ready')
        ob[:, 1] = 1

        # Set Estimation to 0
        ob = self.view_ob('estimation')
        ob[:, 1] = 0

        # Set Set to 1
        ob = self.view_ob('set')
        ob[:, 1] = 1

        # Set Production to 0
        ob = self.view_ob('production')
        ob[:, 1] = 0

        # Set Ground Truth as 0 at set and 1 at trial production with NaN inbetween       
        gt = np.empty([int((self.trialDuration/self.dt)),])
        gt[:] = np.nan
        gt[0:self.burn+self.waitTime+self.spike+trial['production']+self.spike] = 0
        gt[self.burn+self.waitTime+self.spike+trial['production']+self.spike+trial['production']:-1] = 1
        gt = np.reshape(gt, [int(self.trialDuration/self.dt)] + list(self.action_space.shape))
        self.set_groundtruth(gt)

        return trial

    def _step(self, action):
        trial = self.trial
        reward = 0
        ob = self.ob_now
        gt = self.gt_now
        new_trial = False

        if self.in_period('burn'):
            self.SetReward = False
            self.ReadyReward = False
            self.EstReward = False
            self.ThresholdReward = False
            self.TimeAfterThreshold = 0

        if self.in_period('ready'):
            if action <= 0.05: # Should start close to 0
                reward = self.rewards['correct']
                self.ReadyReward = True

            if self.ReadyReward:
                reward = self.rewards['correct']
                self.performance = 1

        if self.in_period('estimation'):
            if action <= 0.05: # Should start close to 0
                reward = self.rewards['correct']
                self.EstReward = True

            if self.EstReward:
                reward = self.rewards['correct']
                self.performance = 1

        if self.in_period('set'):
            if action <= 0.05: # Should start close to 0
                reward = self.rewards['correct']
                self.SetReward = True

            if self.SetReward:
                reward = self.rewards['correct']
                self.performance = 1

        if self.in_period('production'):
            if  action >= 0.90: # Action is over Threshold
                t_prod = self.t - self.end_t['set']  # Measure Time from Set
                eps = abs(t_prod - trial['production']) # Difference between Produced_Interval and Interval
                eps_threshold = int(trial['production']*self.TargetThreshold) # Allowed margin to produced interval

                if eps <= eps_threshold: # If Difference is below Margin, Finish Trial
                    reward = self.rewards['correct']
                    new_trial = True
                    self.ThresholdReward = True

            if self.ThresholdReward == True:
                reward = self.rewards['correct']
                self.performance = 1
                self.TimeAfterThreshold += 1

                if self.TimeAfterThreshold >= self.ThresholdDelay: # Give reward 100 steps after Success
                    new_trial = True
                    self.ThresholdReward = False

        if new_trial == True:
            self.trial_nr += 1

        return ob, reward, new_trial, {
            'new_trial': new_trial, 
            'gt': gt, 
            'SetStart': self.spike+trial['production']+self.spike+self.waitTime+self.burn, 
            'Interval': trial['production'], 
            'ThresholdDelay': self.ThresholdDelay}

class OneTwoThreeGo(ngym.TrialEnv):
    r"""Agents reproduce time intervals based on two samples.

    Args:
        prod_margin: controls the interval around the ground truth production
                    time within which the agent receives proportional reward
    """
    metadata = {
        'paper_link': 'https://www.nature.com/articles/s41593-019-0500-6',
        'paper_name': "Internal models of sensorimotor integration "
                      "regulate cortical dynamics",
        'tags': ['timing', 'go-no-go', 'supervised']
    }

    def __init__(self, dt=80, rewards=None, timing=None, prod_margin=0.2):
        super().__init__(dt=dt)

        self.prod_margin = prod_margin

        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            'fixation': ngym.random.TruncExp(400, 100, 800, rng=self.rng),
            'target': ngym.random.TruncExp(1000, 500, 1500, rng=self.rng),
            's1': 100,
            'interval1': (600, 700, 800, 900, 1000),
            's2': 100,
            'interval2': 0,
            's3': 100,
            'interval3': 0,
            'response': 1000}
        if timing:
            self.timing.update(timing)

        self.abort = False
        # set action and observation space
        name = {'fixation': 0, 'stimulus': 1, 'target': 2}
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3,),
                                            dtype=np.float32, name=name)
        name = {'fixation': 0, 'go': 1}
        self.action_space = spaces.Discrete(2, name=name)

    def _new_trial(self, **kwargs):
        interval = self.sample_time('interval1')
        trial = {
            'interval': interval,
        }
        trial.update(kwargs)

        self.add_period(['fixation', 'target', 's1'])
        self.add_period('interval1', duration=interval, after='s1')
        self.add_period('s2', after='interval1')
        self.add_period('interval2', duration=interval, after='s2')
        self.add_period('s3', after='interval2')
        self.add_period('interval3', duration=interval, after='s3')
        self.add_period('response', after='interval3')

        self.add_ob(1, where='fixation')
        self.add_ob(1, ['s1', 's2', 's3'], where='stimulus')
        self.add_ob(1, where='target')
        self.set_ob(0, 'fixation', where='target')

        # set ground truth
        self.set_groundtruth(1, period='response')

        return trial

    def _step(self, action):
        # ---------------------------------------------------------------------
        # Reward and inputs
        # ---------------------------------------------------------------------
        trial = self.trial
        reward = 0
        ob = self.ob_now
        gt = self.gt_now
        new_trial = False
        if self.in_period('interval3') or self.in_period('response'):
            if action == 1:
                new_trial = True  # terminate
                # time from end of measure:
                t_prod = self.t - self.end_t['s3']
                eps = abs(t_prod - trial['interval'])  # actual production time
                eps_threshold = self.prod_margin*trial['interval']+25
                if eps > eps_threshold:
                    reward = self.rewards['fail']
                else:
                    reward = (1. - eps/eps_threshold)**1.5
                    reward = max(reward, 0.1)
                    reward *= self.rewards['correct']
                    self.performance = 1
        else:
            if action != 0:
                new_trial = self.abort
                reward = self.rewards['abort']

        return ob, reward, False, {'new_trial': new_trial, 'gt': gt}


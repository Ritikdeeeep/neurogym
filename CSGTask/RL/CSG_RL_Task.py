#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cue-set-go task."""

import numpy as np

import neurogym as ngym
from neurogym import spaces

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

        # Calculate Trial Duration
        self.trialDuration = self.burn + self.waitTime + self.set + trial['production'] + self.ThresholdDelay

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

        if new_trial == True:
            self.trial_nr += 1

        return ob, reward, new_trial, {
            'new_trial': new_trial, 
            'gt': gt, 
            'SetStart': self.waitTime+self.burn, 
            'Interval': trial['production'], 
            'ThresholdDelay': self.ThresholdDelay}
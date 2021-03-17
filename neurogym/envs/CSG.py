#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cue-set-go task."""

import numpy as np

import neurogym as ngym
from neurogym import spaces


class CSG(ngym.TrialEnv):
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

    def __init__(self, dt=1, rewards=None, timing=None, training=False, InputNoise=None, TargetThreshold=None):
        super().__init__(dt=dt)
        self.production_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] 
        self.intervals = [720, 760, 800, 840, 880, 1420, 1460, 1500, 1540, 1580] 
        self.context_mag= np.add(np.multiply((0.3/950), self.intervals), (0.2-(0.3/950)*700))
         
        # WeberFraction as the production margin (acceptable deviation)
        # Leave out for now, reimplement later otherwise unfair
        # self.weberFraction = float((100-50)/(1500-800))
        # self.prod_margin = self.weberFraction

        self.training = training 
        self.trial_nr = 1
        self.InputNoise = InputNoise
        self.TargetThreshold = TargetThreshold

        # Binary Rewards for incorrect and correct
        self.rewards = {'incorrect': 0., 'correct': +1.}
        if rewards:
            self.rewards.update(rewards)     

        self.timing = { 
            'cue': 50,
            'set': 100}
            
        if timing:
            self.timing.update(timing)

        self.abort = False
        # Set Action and Observation Space
        # Allow Ramping between 0-1
        self.action_space = spaces.Box(0, 1, shape=(1,), dtype=np.float32)   
        # Context Cue: Burn Time followed by Cue & Set Cue: Wait followed by Set Spike
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32)

    def _new_trial(self, **kwargs):
        # Define Times
        self.trialDuration = 3500
        self.waitTime = int(self.rng.uniform(100, 200))
        self.burn = 50
        self.set = 20

        # Choose index (0-9) at Random
        if self.training == False:
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
        ob[:, 2] = 0
        ob[:, 3] = 0

        # Set Cue to contextCue
        ob = self.view_ob('cue')
        ob[:, 1] = contextCue

        # Set Set to 0.4
        ob = self.view_ob('set')
        ob[:, 3] = 0.4
        
        # Set Ground Truth to Form Ramp & Reshape to Match Action Space
        t_ramp = range(0, int(trial['production']))
        gt_ramp = np.multiply(1/trial['production'], t_ramp)
        gt_step = np.ones((int((self.trialDuration-(trial['production']+self.set+self.waitTime+self.burn))/self.dt),), dtype=np.float)
        gt = np.concatenate((gt_ramp, gt_step)).astype(np.float)
        gt = np.reshape(gt, [int(self.trialDuration-(self.set+self.waitTime+self.burn)/self.dt)] + list(self.action_space.shape))
        self.set_groundtruth(gt, period='production')

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

        if self.in_period('set'):
            if action == 0: # Should start at 0
                reward = self.rewards['correct']
                self.SetReward = True

            if self.SetReward:
                reward = self.rewards['correct']
                self.performance = 1

        if self.in_period('production'): 
            if  action >= 0.90: # Measure Produced_Interval when Action is over Threshold
                t_prod = self.t - self.end_t['set']  # Time from Set till Action <= 0.9 
                eps = abs(t_prod - trial['production']) # Difference between Produced_Interval and Interval
                eps_threshold = int(trial['production']*self.TargetThreshold) # Allowed margin to produced interval

                if eps <= eps_threshold: # If Difference is below Margin, Finish Trial
                    new_trial = True
                    reward = self.rewards['correct']
                    self.ThresholdReward = True
                else:
                    reward = self.rewards['incorrect']

            if self.ThresholdReward == True:
                reward = self.rewards['correct']
                self.performance = 1

        if new_trial == True:
            self.trial_nr += 1

        return ob, reward, new_trial, {'new_trial': new_trial, 'gt': gt, 'Burn_WaitTime': self.waitTime+self.burn, 'Interval': trial['production'], }

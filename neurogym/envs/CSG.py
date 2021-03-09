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

    def __init__(self, dt=1, rewards=None, timing=None, training=True):
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
# Correct action space to allow for ramping
        self.action_space = spaces.Box(0, 1, shape=(1,), dtype=np.float32)  

        # Context Cue: Burn Time followed by Cue & Set Cue: Wait followed by Set Spike
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32)

    def _new_trial(self, **kwargs):
        # Choose index (0-7) at Random
        trial = {
            'production_ind': self.rng.choice(self.production_ind)
        }

        # Choose index by Cycling through all conditions for Training
        if self.training == True: 
            trial['production_ind'] = self.production_ind[(self.trial_nr % 8)-1]

        trial.update(kwargs)

        # Select corresponding interval
        trial['production'] = self.intervals[trial['production_ind']]

        # Select corresponding context cue (Signal + 0.5% Noise)
        contextSignal = self.context_mag[trial['production_ind']]
        noiseSigmaContext = contextSignal * 0.005
        contextNoise = np.random.normal(0, noiseSigmaContext, 3450)
        contextCue = contextSignal + contextNoise

        # Define Times
        self.trialDuration = 3500
        self.waitTime = int(self.rng.uniform(100, 200))
        self.burn = 50
        self.set = 20

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
        
        # Set Ground Truth to Form Ramp
        t_ramp = range(0, int(trial['production']))
        gt_ramp = np.multiply(1/trial['production'], t_ramp)
        gt_step = np.ones((int((self.trialDuration-(trial['production']+self.set+self.waitTime+self.burn))/self.dt),), dtype=np.float)
        gt = np.concatenate((gt_ramp, gt_step)).astype(np.float)
        gt = np.reshape(gt, [self.trialDuration-(self.set+self.waitTime+self.burn) / self.dt] + list(self.action_space.shape))
        self.set_groundtruth(gt, period='production')

        return trial

    def _step(self, action):
        # ---------------------------------------------------------------------
        # Reward and inputs
        # ---------------------------------------------------------------------
        trial = self.trial
        reward = 0
        ob = self.ob_now
        gt = float(self.gt_now)
        new_trial = False
        wait_Time = self.waitTime

        if self.in_period('burn') or self.in_period('wait'):
            if action != 0:
                new_trial = self.abort
                reward = self.rewards['incorrect']

        if self.in_period('production'): 
            if action == 1:
                t_prod = self.t - self.end_t['set']  # Time from set till end of measure
                eps = abs(t_prod - trial['production']) # Difference between Produced and Interval
# Redefine threshold to only consider the mean value
                # eps_threshold = int(self.prod_margin*trial['production']) # Threshold on Production Time
                eps_threshold = 3

                if eps > eps_threshold:
                    reward = self.rewards['incorrect']
                else:
                    new_trial = True  # New trail, because now its in the correct period and over threshold
# Redefine reward for correct as 1
                    #reward = (1. - eps/eps_threshold)**1.5
                    #reward = max(reward, 0.25)
                    #reward *= self.rewards['correct']
                    reward = self.rewards['correct']
                    self.performance = 1

        if new_trial==True:
            self.trial_nr += 1

        return ob, reward, new_trial, {'new_trial': new_trial, 'gt': gt, 'wait_time': wait_Time}

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
        self.production_ind = [0, 1, 2, 3, 4, 5, 6, 7] 
        self.intervals = [800, 850, 900, 950, 1500, 1550, 1600, 1650] 
        self.context_ind=[0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55]

# If we consider the weberFraction as the production margin (acceptable deviation)
        self.weberFraction = float((100-50)/(1500-800))
        self.prod_margin = self.weberFraction

        self.training = training

        self.trial_nr = 1

        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': -0.2}
        if rewards:
            self.rewards.update(rewards)

        self.timing = { 
            'cue': 50,
            'set': 100}
            
        if timing:
            self.timing.update(timing)

        self.abort = False
        # Set Action and Observation Space
        # Fixate & Go
        self.action_space = spaces.Discrete(2)  

        # Context Cue: Burn Time followed by Cue
        # Set Cue: Wait followed by Set Spike
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32)

    def _new_trial(self, **kwargs):
        # Choose index (0-7)
        trial = {
            'production_ind': self.rng.choice(self.production_ind)
        }

        # Cycle through all conditions
        if self.training == True: 
            trial['production_ind'] = self.production_ind[(self.trial_nr % 8)-1]

        trial.update(kwargs)

        # Select corresponding interval (Signal + WeberFraction Noise)
        #productionInterval = self.intervals[trial['production_ind']]
# If we consider the weberFraction as Trial Definition
        #noiseSigmaProduction = productionInterval * self.weberFraction
        #productionNoise = float(np.random.normal(0, noiseSigmaProduction, 1))
        #trial['production'] = int(productionInterval + productionNoise)
# Else
        trial['production'] = self.intervals[trial['production_ind']]

        # Select corresponding context cue (Signal + 1% Noise)
        contextSignal = self.context_ind[trial['production_ind']]
        noiseSigmaContext = contextSignal * 0.01
        contextNoise = np.random.normal(0, noiseSigmaContext, 3450)
        contextCue = contextSignal + contextNoise

        # Define Wait Time
        self.waitTime = int(self.rng.uniform(100, 200))
        
        # Add periods
        self.add_period('burn', duration= 50)
        self.add_period('cue', duration= 3450, after='burn')
        self.add_period('wait', duration= self.waitTime, after='burn')
        self.add_period('set', duration= 20, after='wait')
        self.add_period('production', duration=2*trial['production'], after='set')

        # Set Burn to [0,0,0,0]
        ob = self.view_ob('burn')
        ob[:, 0] = 0
        ob[:, 1] = 0
        ob[:, 2] = 0
        ob[:, 3] = 0

        # Set Cue to contextCue
        ob = self.view_ob('cue')
        ob[:, 1] = contextCue

        # Set Set value to 1
        ob = self.view_ob('set')
        ob[:, 3] = 1
        
        # Set Ground Truth
        gt = np.zeros((int(2*trial['production']/self.dt),))
        gt[int(trial['production']/self.dt)] = 1
        self.set_groundtruth(gt, 'production')

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
        wait_Time = self.waitTime

        if self.in_period('burn') or self.in_period('wait'):
            if action != 0:
                new_trial = self.abort
                reward = self.rewards['abort']

        if self.in_period('production'):
            if action == 1:
                t_prod = self.t - self.end_t['set']  # Time from end of measure
                eps = abs(t_prod - trial['production']) # Difference between Produced and Interval
                eps_threshold = int(self.prod_margin*trial['production']) # Threshold on Production Time

                if eps > eps_threshold:
                    reward = self.rewards['fail']
                else:
                    new_trial = True  # New trail, because now its in the correct period and over threshold
# Allowed to put new_trial condition here of keep when in production?
                    reward = (1. - eps/eps_threshold)**1.5
                    reward = max(reward, 0.25)
                    reward *= self.rewards['correct']
                    self.performance = 1

        if new_trial==True:
            self.trial_nr += 1

        return ob, reward, new_trial, {'new_trial': new_trial, 'gt': gt, 'wait_time': wait_Time}

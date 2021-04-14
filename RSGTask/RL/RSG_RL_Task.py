#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ready-set-go task."""

import numpy as np
import neurogym as ngym
from neurogym import spaces

class RSG_RL(ngym.TrialEnv):
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

    def __init__(self, dt=1, rewards=None, timing=None, training=False, InputNoise=None, TargetThreshold=None):
        super().__init__(dt=dt)
        # Several different intervals with their corresponding their length 
        self.production_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] 
        self.intervals = [480, 560, 640, 720, 800, 800, 900, 1000, 1100, 1200] 
        # Possible Context Cues:
        # (HandShortLeft, HandShortRight, HandLongLeft, HandLongRight, EyeShortLeft, EyeShortRight, EyeLongLeft, EyeLongRight)
        # Short: 0.1 - 0.4
        # Long: 0.5 - 0.8
        self.context_mag = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
# To what extent do we incorporate Hand/Eye and Left/Right
        
        # WeberFraction as the production margin (acceptable deviation)
        # Leave out for now, reimplement later otherwise unfair
        # self.weberFraction = float((100-50)/(1500-800))
        # self.prod_margin = self.weberFraction

        self.training = training # Training Label
        self.trial_nr = 1 # Trial Counter
        self.InputNoise = InputNoise # Input Noise Percentage
        self.TargetThreshold = TargetThreshold # Target Threshold Percentage

        # Binary Rewards for incorrect and correct
        self.rewards = {'incorrect': 0., 'correct': +1.}
        if rewards:
            self.rewards.update(rewards)     

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
        self.set = 10
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
                    new_trial = True
                    self.ThresholdReward = True

            if self.ThresholdReward == True:
                reward = self.rewards['correct']
                self.performance = 1
                self.TimeAfterThreshold += 1

                if self.TimeAfterThreshold >= 100: # Give reward 100 steps after Success
                    new_trial = True
                    self.ThresholdReward = False

        if new_trial == True:
            self.trial_nr += 1

        return ob, reward, new_trial, {
            'new_trial': new_trial, 
            'gt': gt, 
            'Burn_WaitTime': self.waitTime+self.burn, 
            'Interval': trial['production'], 
            'ThresholdDelay': 100}
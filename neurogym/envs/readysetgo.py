#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ready-set-go task."""

import numpy as np

import neurogym as ngym
from neurogym import spaces


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

class MotorTiming(ngym.TrialEnv):
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
        self.rewards = {'incorrect': 0., 'correct': +1., 'completed': +2.}
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
        self.action_space = spaces.Box(0, 1.1, shape=(1,), dtype=np.float32)  

        # Context Cue: Burn Time followed by Cue & Set Cue: Wait followed by Set Spike
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32)

    def _new_trial(self, **kwargs):
        # Choose index (0-9) at Random
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
        waitTime = self.waitTime

        if self.in_period('burn') or self.in_period('wait') or self.in_period('set'):
            if action == 0: # Shouldn't act during Burn, Wait or Set
                reward = self.rewards['correct']

        if self.in_period('production'): 
            if action == gt: # Act during Production should be Ground Truth
                reward = self.rewards['correct']
            else: 
                reward = self.rewards['incorrect']

            if  0.95 < action <= 1: # Measure Produced_Interval when Action reaches 1
# Interval, since it does not seem to reach 1 without training 
                t_prod = self.t - self.end_t['set']  # Time from Set till Action == 1
                eps = abs(t_prod - trial['production']) # Difference between Produced_Interval and Interval
                eps_threshold = 10
# Redefine Threshold here as well
                # int(self.prod_margin*trial['production'])

                if eps < eps_threshold: # If Difference is below Threshold, Finish Trial
                    new_trial = True
                    reward = self.rewards['completed']
                    self.performance = 1
                else:
                    reward = self.rewards['incorrect']

        if new_trial==True:
            self.trial_nr += 1

        return ob, reward, new_trial, {'new_trial': new_trial, 'gt': gt, 'waitTime': waitTime}

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


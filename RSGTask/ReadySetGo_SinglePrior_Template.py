import numpy as np
import neurogym as ngym
from neurogym import spaces

class ReadySetGo_SinglePrior(ngym.TrialEnv):
    # RSG with Reinforcement Learning:
    """Agents have to estimate and produce different time intervals
    using different effectors (actions).
    """

    def __init__(self, dt=1, params= None):
        super().__init__(dt=dt)
        # Unpack Parameters
        Training, InputNoise, TargetThreshold, ThresholdDelay, TargetRamp = params

        # Several different intervals with their corresponding their length 
        self.production_ind = [0, 1, 2, 3, 4] 
        self.intervals = [480, 560, 640, 720, 800]
        # Short Context Cue:
        self.context_mag = 0.3
        
        # WeberFraction as the production margin (acceptable deviation)
        # Leave out for now, reimplement later otherwise unfair
        # self.weberFraction = float((100-50)/(1500-800))
        # self.prod_margin = self.weberFraction

        self.training = Training # Training Label
        self.trial_nr = 1 # Trial Counter
        self.input_noise = InputNoise # Input Noise Percentage
        self.target_threshold = TargetThreshold # Target Threshold Percentage, Later becomes WeberFraction
        self.threshold_delay = ThresholdDelay
        self.target_ramp = TargetRamp
        # Binary Rewards for incorrect and correct
        self.rewards = {'incorrect': 0., 'correct': +1.}    

        # Set Action and Observation Space
        # Allow Ramping between 0-1
        self.action_space = spaces.Box(0, 1, shape=(1,), dtype=np.float32)   
        # Context Cue: Burn Time followed by Cue
        # Ready-Set Cue: Burn+Wait followed by Ready-Set Spike
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32)

    def _new_trial(self, Scenario=None, WaitTime=None, **kwargs):
        # Define Times
        if WaitTime is not None:
            self.wait_time = WaitTime
        else:
            self.wait_time = int(self.rng.uniform(100, 200))
        self.burn = 50
        self.spike = 20

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
        
        # Choose given Scenario
        if Scenario is not None:
            trial = {
                'production_ind': Scenario
            }

        # Select corresponding interval
        trial['production'] = self.intervals[trial['production_ind']]

        # Calculate Trial Duration
        self.trial_duration = 2000
        
        # Select corresponding context cue (Signal + InputNoise)
        contextSignal = self.context_mag
        noiseSigmaContext = contextSignal * self.input_noise
        contextNoise = np.random.normal(0, noiseSigmaContext, (self.trial_duration-self.burn))
        contextCue = contextSignal + contextNoise

        # Define periods
        self.add_period('burn', duration= self.burn)
        self.add_period('cue', duration= self.trial_duration-self.burn, after='burn')
        self.add_period('wait', duration= self.wait_time, after='burn')
        self.add_period('ready', duration= self.spike, after='wait')
        self.add_period('estimation', duration= trial['production'], after='ready')
        self.add_period('set', duration= self.spike, after='estimation')
        self.add_period('production', duration=self.trial_duration-(self.spike+trial['production']+self.spike+self.wait_time+self.burn), after='set')

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

        # Set Ready to 0.4
        ob = self.view_ob('ready')
        ob[:, 1] = 0.4

        # Set Estimation to 0
        ob = self.view_ob('estimation')
        ob[:, 1] = 0

        # Set Set to 0.4
        ob = self.view_ob('set')
        ob[:, 1] = 0.4

        # Set Production to 0
        ob = self.view_ob('production')
        ob[:, 1] = 0

        ob = self.view_ob()

        # Set Ground Truth as 0 at Set and 1 at Trial Production with NaN or Ramp inbetween
        gt = np.empty([int(((self.trial_duration)/self.dt)),])
        gt[:] = np.nan
        gt[self.burn+self.wait_time:self.burn+self.wait_time+self.spike] = 0
        gt[self.burn+self.wait_time+self.spike+trial['production']:self.burn+self.wait_time+self.spike+trial['production']+self.spike] = 0
        if self.target_ramp == True:
            t_ramp = range(0, int(trial['production']))
            gt_ramp = np.multiply(1/trial['production'], t_ramp)
            gt[self.burn+self.wait_time+self.spike+trial['production']+self.spike:self.burn+self.wait_time+self.spike+trial['production']+self.spike+trial['production']] = gt_ramp

        gt[self.burn+self.wait_time+self.spike+trial['production']+self.spike+trial['production']:self.burn+self.wait_time+self.spike+trial['production']+self.spike+trial['production']+self.threshold_delay] = 1
        gt = np.reshape(gt, [int((self.trial_duration)/self.dt)] + list(self.action_space.shape))
        self.set_groundtruth(gt)

        return trial, ob, gt

    def _step(self, action):
        trial = self.trial
        reward = 0
        ob = self.ob_now
        gt = self.gt_now
        NewTrial = False

        if self.in_period('burn'):
            self.set_reward = False
            self.threshold_reward = False
            self.t_threshold = 0

        if self.in_period('set'):
            if action <= 0.05: # Should start close to 0
                reward = self.rewards['correct']
                self.set_reward = True

            if self.set_reward:
                reward = self.rewards['correct']
                self.performance = 1

        if self.in_period('production'):
            if  action >= 0.95: # Action is over Threshold
                t_prod = self.t - self.end_t['set']  # Measure Time from Set
                eps = abs(t_prod - trial[0]['production']) # Difference between Produced_Interval and Interval
                eps_threshold = int(trial[0]['production']*self.target_threshold) # Allowed margin to produced interval

                if eps <= eps_threshold: # If Difference is below Margin, Finish Trial
                    reward = self.rewards['correct']
                    self.threshold_reward = True

            if self.threshold_reward == True:
                reward = self.rewards['correct']
                self.performance = 1
                self.t_threshold += 1

                if self.t_threshold >= self.threshold_delay: # Give reward ThresholDelay steps after Success
                    NewTrial = True
                    self.threshold_reward = False

            if self.t > self.trial_duration:
                NewTrial = True

        if NewTrial == True:
            self.trial_nr += 1

        return ob, reward, NewTrial, {
            'new_trial': NewTrial, 
            'TrialDuration': self.trial_duration,
            'gt': gt,
            'Interval': trial[0]['production'],
            'SetStart': self.spike+trial[0]['production']+self.spike+self.wait_time+self.burn,
            }
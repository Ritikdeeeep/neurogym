# Libraries
import gym
import neurogym as ngym
import numpy as np

from neurogym import spaces
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Define Task
class CSG(ngym.TrialEnv):
    # CSG with Reinforcement Learning

    """Agents have to produce different time intervals based of context cue

    Args:
        Training: Turn on for training to produce cyclin batches of trials (boolean)
        InputNoise: Fraction of noise added to the input (int)
        TargetRamp: Ramping or NaN target (boolean)
    """

    def __init__(self, dt=1, params=None):
        super().__init__(dt=dt)
        # Unpack Parameters
        Training, InputNoise, TargetRamp = params # Unpack arguments

        # Several different intervals: their index, length and magnitude
        self.production_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] 
        self.intervals = [720, 760, 800, 840, 880, 1420, 1460, 1500, 1540, 1580] 
        self.context_mag = np.add(np.multiply((0.3/950), self.intervals), (0.2-(0.3/950)*700))
        
        self.training = Training # Training Label
        self.trial_nr = 1 # Trial Counter
        self.input_noise = InputNoise # Input Noise Percentage
        self.target_threshold = 0.05 # Target Threshold Percentage
        self.threshold_delay = 50 # Reward Delay after Threshold Crossing
        self.target_ramp = TargetRamp # Ramping or NaN Target

        # Binary Rewards for incorrect and correct
        self.rewards = {'incorrect': 0., 'correct': +1.}

        # Set Action and Observation Space
        # Allow Ramping between 0-1
        self.action_space = spaces.Box(0, 1, shape=(1,), dtype=np.float32)   
        # Context Cue and Set Cue
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32)

    def _new_trial(self, Scenario=None, WaitTime=None, **kwargs):
        # Define Times
        if WaitTime is not None:
            self.wait_time = WaitTime
        else:
            self.wait_time = int(self.rng.uniform(100, 200))

        self.burn = 50 # Duration of Burn period before context cue
        self.set = 20 # Duration of Set Period
        self.trial_duration = 2200

        # Choose interval index (0-9) at Random
        if self.training == False:
            trial = {
                'production_ind': self.rng.choice(self.production_ind)
            }

        # Choose index by Cycling through all conditions when Training
        if self.training == True: 
            trial = {
                'production_ind': self.production_ind[(self.trial_nr % len(self.production_ind))-1]
            }
        
        # Choose given Scenario
        if Scenario is not None:
            trial = {
                'production_ind': Scenario
            }

        trial.update(kwargs)

        # Select corresponding interval length
        trial['production'] = self.intervals[trial['production_ind']]

        # Calculate corresponding context cue magnitude (Signal + InputNoise)
        contextSignal = self.context_mag[trial['production_ind']]
        noiseSigmaContext = contextSignal * self.input_noise
        contextNoise = np.random.normal(0, noiseSigmaContext, (self.trial_duration-self.burn))
        contextCue = contextSignal + contextNoise

        # Define periods
        self.add_period('burn', duration= self.burn)
        self.add_period('cue', duration= self.trial_duration-self.burn, after='burn')
        self.add_period('wait', duration= self.wait_time, after='burn')
        self.add_period('set', duration= self.set, after='wait')
        self.add_period('production', duration=self.trial_duration-(self.set+self.wait_time+self.burn), after='set')

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

        ob = self.view_ob()
        
        # Set Ground Truth as Ramp or NaN 
        if self.target_ramp == False:      
            gt = np.empty([int(((self.trial_duration)/self.dt)),])
            gt[:] = np.nan
            gt[0:self.burn+self.wait_time+self.set] = 0
            gt[self.burn+self.wait_time+self.set+int(trial['production']):self.burn+self.wait_time+self.set+int(trial['production'])+self.threshold_delay] = 1
            gt = np.reshape(gt, [int((self.trial_duration)/self.dt)] + list(self.action_space.shape))

        if self.target_ramp == True:
            gt = np.empty([int(((self.trial_duration)/self.dt)),])
            gt[:] = np.nan
            gt[self.burn:self.burn+self.wait_time+self.set] = 0
            gt[self.burn+self.wait_time+self.set+int(trial['production']):self.burn+self.wait_time+self.set+int(trial['production'])+self.threshold_delay] = 1
            gt[self.burn+self.wait_time+self.set:self.burn+self.wait_time+self.set+int(trial['production'])] = np.multiply(1/trial['production'], range(0, int(trial['production'])))
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

            if self.set_reward: # Keep giving reward during set
                reward = self.rewards['correct']
                self.performance = 1

        if self.in_period('production'): 
            if  action >= 0.95: # Action is over Threshold
                t_prod = self.t - self.end_t['set']  # Measure Time from Set
                eps = abs(t_prod - trial[0]['production']) # Difference between Produced_Interval and Interval
                eps_threshold = int(trial[0]['production']*self.target_threshold) # Allowed margin to produced interval

                if eps <= eps_threshold: # If Difference is below Margin, End Trial after Threshold Delay
                    reward = self.rewards['correct']
                    self.ThresholdReward = True

            if self.threshold_reward == True:
                reward = self.rewards['correct']
                self.performance = 1
                self.t_threshold += 1

                if self.t_threshold >= self.threshold_delay: # Give reward ThresholdDelay steps after Success
                    NewTrial = True
                    self.threshold_reward = False

            if self.t > self.trial_duration:
                NewTrial = True 

        if NewTrial == True:
            self.trial_nr += 1

        return ob, reward, NewTrial, {
            'new_trial': NewTrial, 
            'gt': gt, 
            'SetStart': self.wait_time+self.burn, 
            'Interval': trial[0]['production'], 
            'ThresholdDelay': self.threshold_delay}

# Define Parameters
N_Steps = 2200
N_Conditions = 10 

# Define Kwargs
Input_Noise = 0.01 
Target_Ramp = False # True or False depending on Ramp or NaN respectively

# Plot of the possible trial conditions
ContextCue = np.empty((N_Steps, N_Conditions))
ContextCue[:] = np.NaN
ReadySetCue = np.empty((N_Steps, N_Conditions))
ReadySetCue[:] = np.NaN
Target = np.empty((N_Steps, N_Conditions))
Target[:] = np.NaN

# Generate Data
for i in range(N_Conditions):
    kwargs = {'params': (True, Input_Noise, Target_Ramp)}
    envPlot = CSG(**kwargs)
    Wait_Time = 50 
    trial, ob, gt = envPlot._new_trial(Scenario=i, WaitTime=Wait_Time)
    CC = ob[:,0]
    RSC = ob[:,1]
    T = gt[:,0]
    ContextCue[0:0+CC.shape[0], i] = CC
    ReadySetCue[0:0+RSC.shape[0], i] = RSC
    Target[0:0+T.shape[0], i] = T
    envPlot.close()

# Plot
fig, axs = plt.subplots(3, sharex=True, sharey=True)
fig.suptitle('RSG Input')
plt.xlabel("Time")
colors = ['navy', 'mediumblue', 'royalblue', 'cornflowerblue', 'lightsteelblue', 'darkred', 'firebrick', 'brown', 'indianred', 'lightcoral']

for i in range(N_Conditions):
    # Context Cue
    axs[0].plot(ContextCue[:,i], color=colors[i])
    axs[0].set_title('Context Cue')
    axs[0].set_ylabel('Context Magnitude')

        # Set Cue
    axs[1].plot(ReadySetCue[:,i], color=colors[i])
    axs[1].set_title('Ready-Set Cue')
    axs[1].set_ylabel('Magnitude')

        # Target
    axs[2].plot(Target[:,i], color=colors[i])
    axs[2].set_title('Target')
    axs[2].set_ylabel('Target Magnitude')
plt.show()
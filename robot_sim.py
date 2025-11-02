# robot_sim.py

import random
from config import *
from hmm_logic import HMM_Localizer # Need HMM_Localizer to access its transition/sensor models

class Robot:
    def __init__(self, start_r, start_c):
        self.r = start_r
        self.c = start_c
        self.state_index = coords_to_state(start_r, start_c)
        self.hmm = HMM_Localizer() # Use the HMM to get sensor parameters

    def move(self, action):
        """Simulates the robot's movement based on the probabilistic transition model."""
        dr, dc = self._get_action_delta(action)

        # Determine the probability of ending up in an adjacent state
        possible_next_states = []
        probabilities = []

        T_action = self.hmm.T[:, self.state_index, action]
        for s_t in range(NUM_STATES):
            if T_action[s_t] > 0.001: # Only consider states with non-zero probability
                possible_next_states.append(s_t)
                probabilities.append(T_action[s_t])

        # Sample the next state based on the probabilities
        if probabilities:
            next_state_index = random.choices(possible_next_states, weights=probabilities, k=1)[0]
            self.state_index = next_state_index
            self.r, self.c = state_to_coords(self.state_index)
            return True
        return False

    def sense(self):
        """Simulates the noisy sensor reading based on the Sensor Model."""
        true_obs_index = self.hmm._get_true_sensor_state(self.state_index)

        # The sensor model (emission matrix) column for the current true state
        E_column = self.hmm.E[:, self.state_index]

        # Sample the observation index based on the probabilistic sensor model
        possible_observations = list(range(16))

        # Sample from E[obs, s_t]
        observation = random.choices(possible_observations, weights=E_column, k=1)[0]

        return observation

    def _get_action_delta(self, action):
        action_deltas = [(-1, 0), (1, 0), (0, 1), (0, -1)] # N, S, E, W
        return action_deltas[action]
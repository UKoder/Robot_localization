# hmm_logic.py

import numpy as np
from config import *

class HMM_Localizer:
    def __init__(self):
        # Initial belief is uniform over all *open* states
        self.belief = self._initialize_prior()
        # Transition and Sensor models are pre-calculated since they are static
        self.T = self._calculate_transition_model()
        self.E = self._calculate_sensor_model()

    def _initialize_prior(self):
        """Sets a uniform prior probability over all non-obstacle tiles."""
        prior = np.zeros(NUM_STATES)
        open_states = 0
        for r in range(MAP_HEIGHT):
            for c in range(MAP_WIDTH):
                if MAP_LAYOUT[r][c] == 0:
                    state = coords_to_state(r, c)
                    prior[state] = 1.0
                    open_states += 1

        # Normalize the prior belief
        if open_states > 0:
            prior /= open_states
        return prior

    def _calculate_transition_model(self):
        """Calculates the Transition Matrix T[s_t | s_{t-1}]."""
        T = np.zeros((NUM_STATES, NUM_STATES, 4)) # T[s_t, s_{t-1}, action]

        # Actions: 0=North, 1=South, 2=East, 3=West (N, S, E, W)
        action_deltas = [(-1, 0), (1, 0), (0, 1), (0, -1)]

        for prev_s in range(NUM_STATES):
            pr, pc = state_to_coords(prev_s)

            # Skip states that are obstacles
            if MAP_LAYOUT[pr][pc] == 1:
                continue

            for action in range(4):
                dr, dc = action_deltas[action]

                # Target coordinates if successful
                target_r, target_c = pr + dr, pc + dc

                # --- Success/Bumping Logic ---
                if 0 <= target_r < MAP_HEIGHT and 0 <= target_c < MAP_WIDTH and MAP_LAYOUT[target_r][target_c] == 0:
                    # Successful move
                    success_s = coords_to_state(target_r, target_c)
                    T[success_s, prev_s, action] += P_SUCCESS
                    T[prev_s, prev_s, action] += P_STAY # Stays put with some probability
                else:
                    # Bumps into wall, stays put
                    T[prev_s, prev_s, action] += P_SUCCESS + P_STAY

                # --- Slipping Logic (No-op for simplicity, but critical in real HMMs) ---
                # A full HMM would calculate probabilities for slipping to other adjacent free cells.

                # Ensure each column (prev_s, action) sums to 1 (normalization)
                current_sum = np.sum(T[:, prev_s, action])
                if current_sum > 0:
                    T[:, prev_s, action] /= current_sum
                else: # Fallback for edge cases
                    T[prev_s, prev_s, action] = 1.0

        return T

    def _get_true_sensor_state(self, state):
        """Determines the true sensor readings (4-bit binary) for a given map state."""
        r, c = state_to_coords(state)
        # Check N, S, E, W for obstacles (1=obstacle, 0=free)
        N = MAP_LAYOUT[r-1][c] if r > 0 else 1
        S = MAP_LAYOUT[r+1][c] if r < MAP_HEIGHT - 1 else 1
        E = MAP_LAYOUT[r][c+1] if c < MAP_WIDTH - 1 else 1
        W = MAP_LAYOUT[r][c-1] if c > 0 else 1

        # Convert 4 boolean values to a single observation index (0 to 15)
        # N(8) S(4) E(2) W(1)
        obs_index = N * 8 + S * 4 + E * 2 + W * 1
        return obs_index

    def _calculate_sensor_model(self):
        """Calculates the Sensor (Emission) Matrix E[obs | s_t]."""
        E = np.zeros((16, NUM_STATES)) # E[observation_index, s_t]

        for state in range(NUM_STATES):
            if MAP_LAYOUT[state_to_coords(state)[0]][state_to_coords(state)[1]] == 1:
                continue

            true_obs = self._get_true_sensor_state(state)

            # The sensor is a 4-bit reading. The true reading has 4 bits correct.
            for obs in range(16):
                mismatches = bin(obs ^ true_obs).count('1') # Hamming distance

                # Simplified model: probability decreases with more mismatches
                E[obs, state] = (P_SENSOR_ACCURATE ** (4 - mismatches)) * (P_SENSOR_INACCURATE ** mismatches)

            # Normalize E[obs, state] column to sum to 1 (optional, depends on model formulation)
            E[:, state] /= np.sum(E[:, state])

        return E

    def filter(self, action, observation):
        """
        Performs the HMM Forward Algorithm step (Prediction followed by Update).

        1. Prediction (State Transition): belief_bar = T * belief_prev
        2. Update (Sensor Observation): belief_new = E * belief_bar
        """
        # --- 1. Prediction (Transition) ---
        # Matrix multiplication: Sum over previous state s_prev
        # belief_bar[s_t] = SUM_s_prev ( T[s_t, s_prev, action] * belief_prev[s_prev] )
        T_action = self.T[:, :, action] # Get the transition matrix for the specific action

        # belief_bar is the prior belief P(S_t | O_{1:t-1}, A_{1:t})
        belief_bar = T_action @ self.belief

        # --- 2. Update (Sensor) ---
        # Hadamard product (element-wise multiplication)
        # belief_new[s_t] = E[observation, s_t] * belief_bar[s_t]
        sensor_vector = self.E[observation, :]
        belief_new = sensor_vector * belief_bar

        # --- 3. Normalization ---
        # Sum of probabilities must equal 1
        belief_new /= np.sum(belief_new)

        self.belief = belief_new
        return self.belief

#
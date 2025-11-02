# config.py

# --- Pygame Setup ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800
FPS = 30
TILE_SIZE = 80 # Size of each grid cell in pixels
MAP_WIDTH = 10 # Number of tiles wide (10x10 grid)
MAP_HEIGHT = 10
NUM_STATES = MAP_WIDTH * MAP_HEIGHT
FONT_SIZE = 14

# --- HMM Parameters ---

# Map definition (10x10 grid)
# 0 = Open (White), 1 = Obstacle (Black)
# The HMM and robot only move in open spaces.
MAP_LAYOUT = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 1, 0, 1, 1, 1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0, 1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0, 1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0, 1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]

# State representation: Convert (row, col) to a single index (0 to 99)
def coords_to_state(row, col):
    return row * MAP_WIDTH + col

def state_to_coords(state):
    row = state // MAP_WIDTH
    col = state % MAP_WIDTH
    return row, col

# --- Probabilities ---

# P(S_t | S_{t-1}, Action) - Transition Model
# When attempting to move 'Forward' (N, S, E, W),
# this is the probability of succeeding (0.8) vs. slipping (0.05) or staying put (0.15)
P_SUCCESS = 0.8
P_SLIP_SIDE = 0.05
P_STAY = 0.1
P_BUMP = 1.0 # If a move hits a wall, the robot definitely stays in place.

# P(Observation | S_t) - Sensor Model (Noisy)
# Observations are based on detecting an obstacle (1) or free space (0) in the 4 cardinal directions.
# A sensor reading '1' means an obstacle was detected in that direction.
# Example: P(Sensor detects obstacle | True state is obstacle free) = 0.2 (False Negative/Noise)
P_SENSOR_ACCURATE = 0.9  # P(Sensor=X | True=X)
P_SENSOR_INACCURATE = 0.1 # P(Sensor=X | True!=X)
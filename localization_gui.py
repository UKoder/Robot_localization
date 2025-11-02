import random
import pygame
import sys
import numpy as np
import time # For more precise animation timing

from config import *
from hmm_logic import HMM_Localizer
from robot_sim import Robot

# --- Pygame Initialization ---
pygame.init()
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("HMM Robot Localization (Animated)")
CLOCK = pygame.time.Clock()
FONT = pygame.font.Font(None, FONT_SIZE)
BIG_FONT = pygame.font.Font(None, 24)

# --- Asset Loading ---
try:
    ROBOT_IMG = pygame.image.load('assets/robot_sprite.png').convert_alpha()
    ROBOT_IMG = pygame.transform.scale(ROBOT_IMG, (TILE_SIZE, TILE_SIZE))
except pygame.error as e:
    print(f"Error loading robot_sprite.png: {e}. Using fallback circle.")
    ROBOT_IMG = None

# --- Colors ---
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_OBSTACLE = (80, 80, 80) # Darker gray for obstacles
COLOR_ROBOT_TRUE = (0, 150, 0) # Darker green for true robot
COLOR_GRID = (100, 100, 100) # Lighter grid lines
COLOR_BELIEF_LOW = (0, 0, 100) # Dark blue for low probability
COLOR_BELIEF_HIGH = (255, 255, 0) # Yellow for high probability
COLOR_TEXT = (255, 255, 255) # White text for belief numbers on dark background
COLOR_INFO_TEXT = (0, 0, 0) # Black text for info panel

# --- Animation Constants ---
ANIMATION_DURATION_MOVE = 0.5 # seconds for robot movement
ANIMATION_DURATION_HMM_STEP = 0.8 # seconds for total HMM filter step (spread + observe)
ANIMATION_STAGE_SPLIT = 0.4 # % of HMM_STEP for prediction spread, rest for observation update

# --- Visualization Functions ---

def get_color_from_belief(prob, max_prob):
    """Generates a color for the heatmap based on probability."""
    if max_prob == 0: return COLOR_BELIEF_LOW

    # Interpolate between a low and high belief color
    norm_prob = prob / max_prob
    r = int(COLOR_BELIEF_LOW[0] + (COLOR_BELIEF_HIGH[0] - COLOR_BELIEF_LOW[0]) * norm_prob)
    g = int(COLOR_BELIEF_LOW[1] + (COLOR_BELIEF_HIGH[1] - COLOR_BELIEF_LOW[1]) * norm_prob)
    b = int(COLOR_BELIEF_LOW[2] + (COLOR_BELIEF_HIGH[2] - COLOR_BELIEF_LOW[2]) * norm_prob)
    return (r, g, b)

def draw_map_and_belief(screen, current_belief, max_prob):
    """Draws the grid, obstacles, and the belief distribution heatmap."""

    for r in range(MAP_HEIGHT):
        for c in range(MAP_WIDTH):
            x = c * TILE_SIZE
            y = r * TILE_SIZE
            rect = pygame.Rect(x, y, TILE_SIZE, TILE_SIZE)

            state = coords_to_state(r, c)

            if MAP_LAYOUT[r][c] == 0:
                prob = current_belief[state]
                color = get_color_from_belief(prob, max_prob)
                pygame.draw.rect(screen, color, rect)

                # Draw probability text
                if prob > 0.001: # Only draw for relevant probabilities
                    text_surface = FONT.render(f"{prob:.2f}", True, COLOR_TEXT)
                    text_rect = text_surface.get_rect(center=rect.center)
                    screen.blit(text_surface, text_rect)
            else:
                # Draw obstacle
                pygame.draw.rect(screen, COLOR_OBSTACLE, rect)

            # Draw grid lines
            pygame.draw.rect(screen, COLOR_GRID, rect, 1)

def draw_robot_animated(screen, robot_prev_pos, robot_curr_pos, animation_progress):
    """Draws the true, hidden position of the robot with interpolation."""
    prev_r, prev_c = robot_prev_pos
    curr_r, curr_c = robot_curr_pos

    # Interpolate position
    interp_c = prev_c + (curr_c - prev_c) * animation_progress
    interp_r = prev_r + (curr_r - prev_r) * animation_progress

    x = int(interp_c * TILE_SIZE)
    y = int(interp_r * TILE_SIZE)

    if ROBOT_IMG:
        screen.blit(ROBOT_IMG, (x, y))
    else:
        # Simple circle fallback
        center_x = x + TILE_SIZE // 2
        center_y = y + TILE_SIZE // 2
        pygame.draw.circle(screen, COLOR_ROBOT_TRUE, (center_x, center_y), TILE_SIZE // 3)

def draw_info(screen, action_names, current_action, current_obs, current_stage):
    """Displays information on the right side of the screen."""
    info_x = SCREEN_WIDTH - 200 # Assuming enough space
    info_y = 50

    text_surfaces = []
    text_surfaces.append(BIG_FONT.render("HMM Localization", True, COLOR_INFO_TEXT))
    text_surfaces.append(BIG_FONT.render("------------------", True, COLOR_INFO_TEXT))
    text_surfaces.append(BIG_FONT.render(f"Last Action: {action_names.get(current_action, 'None')}", True, COLOR_INFO_TEXT))
    text_surfaces.append(BIG_FONT.render(f"Last Obs (Bin): {bin(current_obs) if current_obs is not None else 'None'}", True, COLOR_INFO_TEXT))
    text_surfaces.append(BIG_FONT.render(f"Stage: {current_stage}", True, COLOR_INFO_TEXT))

    for i, text_surf in enumerate(text_surfaces):
        screen.blit(text_surf, (info_x, info_y + i * 30))

# --- Main Game Loop ---
if __name__ == "__main__":
    # Initialize components
    start_r, start_c = 8, 8
    while MAP_LAYOUT[start_r][start_c] == 1:
        start_r, start_c = random.randint(0, MAP_HEIGHT - 1), random.randint(0, MAP_WIDTH - 1)

    robot = Robot(start_r, start_c)
    hmm = robot.hmm

    action_names = {0: 'N (Up)', 1: 'S (Down)', 2: 'E (Right)', 3: 'W (Left)'}
    current_action = None
    current_obs = None

    # Animation states
    ANIM_STATE_IDLE = 0
    ANIM_STATE_ROBOT_MOVE = 1 # Robot physically moving
    ANIM_STATE_HMM_PREDICT = 2 # Belief spreading (Prediction)
    ANIM_STATE_HMM_UPDATE = 3 # Belief contracting (Update based on observation)

    animation_state = ANIM_STATE_IDLE
    animation_timer = 0.0

    # Store previous and predicted beliefs for animation
    belief_at_start_of_step = np.copy(hmm.belief)
    belief_after_prediction = np.zeros(NUM_STATES) # belief_bar

    robot_prev_coords = (robot.r, robot.c) # Store previous coordinates for smooth animation

    running = True
    while running:
        dt = CLOCK.tick(FPS) / 1000.0 # Time since last frame in seconds

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # --- User Input: Action Selection (only if idle) ---
            if event.type == pygame.KEYDOWN and animation_state == ANIM_STATE_IDLE:
                action = None
                if event.key == pygame.K_UP: action = 0  # North
                elif event.key == pygame.K_DOWN: action = 1  # South
                elif event.key == pygame.K_RIGHT: action = 2  # East
                elif event.key == pygame.K_LEFT: action = 3  # West

                if action is not None:
                    current_action = action
                    robot_prev_coords = (robot.r, robot.c) # Store before robot moves

                    # 1. Trigger Robot Move Animation
                    animation_state = ANIM_STATE_ROBOT_MOVE
                    animation_timer = 0.0

        # --- Animation Update Logic ---
        if animation_state != ANIM_STATE_IDLE:
            animation_timer += dt

        if animation_state == ANIM_STATE_ROBOT_MOVE:
            if animation_timer >= ANIMATION_DURATION_MOVE:
                # Robot movement finished, now trigger HMM steps
                # Perform the *actual* robot move and sense once animation is done
                robot.move(current_action)
                current_obs = robot.sense()

                # Store belief before prediction for animation
                belief_at_start_of_step = np.copy(hmm.belief)

                # Perform HMM prediction step (but don't update hmm.belief yet)
                # This calculates belief_bar = T * belief_prev
                T_action = hmm.T[:, :, current_action]
                belief_after_prediction = T_action @ belief_at_start_of_step

                animation_state = ANIM_STATE_HMM_PREDICT
                animation_timer = 0.0 # Reset timer for HMM step

        elif animation_state == ANIM_STATE_HMM_PREDICT:
            # Belief spread animation
            if animation_timer >= ANIMATION_DURATION_HMM_STEP * ANIMATION_STAGE_SPLIT:
                animation_state = ANIM_STATE_HMM_UPDATE
                # No timer reset here, continue with the same HMM_STEP timer

        elif animation_state == ANIM_STATE_HMM_UPDATE:
            # Belief contraction animation
            if animation_timer >= ANIMATION_DURATION_HMM_STEP:
                # Animation finished, perform the final HMM update and return to idle
                hmm.filter(current_action, current_obs) # This updates hmm.belief
                animation_state = ANIM_STATE_IDLE
                animation_timer = 0.0

        # --- Drawing ---
        SCREEN.fill(COLOR_WHITE)

        # Determine which belief to draw for animation
        current_belief_to_draw = np.copy(hmm.belief) # Default to final belief

        current_stage_text = "Idle"
        if animation_state == ANIM_STATE_ROBOT_MOVE:
            current_stage_text = "Robot Moving..."
            # Draw belief at start of step
            max_prob = np.max(belief_at_start_of_step) if np.max(belief_at_start_of_step) > 0 else 1.0
            draw_map_and_belief(SCREEN, belief_at_start_of_step, max_prob)

            # Draw animated robot
            progress = animation_timer / ANIMATION_DURATION_MOVE
            draw_robot_animated(SCREEN, robot_prev_coords, (robot.r, robot.c), progress)

        elif animation_state == ANIM_STATE_HMM_PREDICT:
            current_stage_text = "HMM: Prediction (Spread)"
            # Interpolate belief from previous state to predicted state
            progress = animation_timer / (ANIMATION_DURATION_HMM_STEP * ANIMATION_STAGE_SPLIT)
            interp_belief = belief_at_start_of_step * (1 - progress) + belief_after_prediction * progress
            max_prob = np.max(interp_belief) if np.max(interp_belief) > 0 else 1.0
            draw_map_and_belief(SCREEN, interp_belief, max_prob)
            draw_robot_animated(SCREEN, (robot.r, robot.c), (robot.r, robot.c), 1.0) # Robot is at target pos

        elif animation_state == ANIM_STATE_HMM_UPDATE:
            current_stage_text = "HMM: Update (Observe)"
            # Interpolate belief from predicted state to final updated state
            final_belief_after_update = np.copy(hmm.belief) # Get the final updated belief
            hmm_temp_belief = np.copy(belief_after_prediction) # Start from predicted

            # Apply sensor model for visualization purposes, without normalizing yet
            sensor_vector = hmm.E[current_obs, :]
            hmm_temp_belief = sensor_vector * hmm_temp_belief

            progress = (animation_timer - (ANIMATION_DURATION_HMM_STEP * ANIMATION_STAGE_SPLIT)) / \
                       (ANIMATION_DURATION_HMM_STEP * (1 - ANIMATION_STAGE_SPLIT))

            interp_belief = hmm_temp_belief * progress + belief_after_prediction * (1 - progress)
            # Normalize for drawing purposes only
            interp_belief /= np.sum(interp_belief) if np.sum(interp_belief) > 0 else 1.0


            max_prob = np.max(interp_belief) if np.max(interp_belief) > 0 else 1.0
            draw_map_and_belief(SCREEN, interp_belief, max_prob)
            draw_robot_animated(SCREEN, (robot.r, robot.c), (robot.r, robot.c), 1.0) # Robot is at target pos

        else: # ANIM_STATE_IDLE
            max_prob = np.max(hmm.belief) if np.max(hmm.belief) > 0 else 1.0
            draw_map_and_belief(SCREEN, hmm.belief, max_prob)
            draw_robot_animated(SCREEN, (robot.r, robot.c), (robot.r, robot.c), 1.0) # Robot stays at target pos

        draw_info(SCREEN, action_names, current_action, current_obs, current_stage_text)

        pygame.display.flip()

    pygame.quit()
    sys.exit()
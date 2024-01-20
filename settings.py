"""
Monza Game - Settings and Global Variables
Description: Defines global settings, constants, and initial conditions.
Author(s): Daniel Sotelo, Jiajun Xu, Vladyslav Korenyak
Date: 09/10/2023
Version: 2.0
"""

# Import libraries
import pygame
import math

# Initialize the pygame
pygame.init()

# FPS settings
FPS = 60
fpsClock = pygame.time.Clock()

# Window settings
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 700

# Color settings
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (139, 0, 12)
BLUE = (0, 0, 255)
PURPLE = (205, 1, 189)
BROWN = (43, 23, 0)
CARAMEL = (97, 54, 19)
MID_BROWN = (124, 71, 0)
BEIGE = (239, 228, 176)

# Coin parameters
r_coin = 15 # Radius
w_coin = 2 # Width
m_coin = 0.007 # Mass

# Initial conditions of the coin in local axis {x1, y1}
x1_0 = 0 
y1_0 = 200
dx1_0 = 0

# Labyrinth circle parameters
r_circle = 251.43 # Radius
r_area = 205.72 # Radius of utile area
w_circle = 3 # Width
x_offset = SCREEN_WIDTH // 2  # Position of origin of global coordinates {x, y}
y_offset = SCREEN_HEIGHT // 2 # Position of origin of global coordinates {x, y}
beta = 2 * math.pi/180 # Angle of inclination with respect to the vertical

# Obstacle parameters
w_obstacle = 3  # Width

# Physical update parameters
dt = 0.01 # Integration time step 
g = 9810 # Gravity
mu_v = 0.01 # Viscosity friction coefficient
mu_ps = 0.001 # Static friction coefficient with the wall
mu_pd = 0.0005 # Dynamic friction coefficient with the wall
mu_s = 0.001 # Static friction coefficient with the rail
mu_f = 0.0005 # Dynamic friction coefficient with the rail
k = 0.8 # Inelastic impact coefficient
v_min = 400 # Minimum velocity for bouncing

# Define the parameters for the parabollic obstacles for the four difficulties
global diff_eqs
diff_eqs = []

# Beginner difficulty
diff_eqs.append([{'coeff': (-0.00054, 160.35), 'x_right': 138, 'x_left': -62.11},
                 {'coeff': (-0.00054, 114.29), 'x_right': 0, 'x_left': -180},
                 {'coeff': (-0.00054, 68.57), 'x_right': 198, 'x_left': 0},
                 {'coeff': (-0.00054, 22.86), 'x_right': 0, 'x_left': -203},
                 {'coeff': (-0.00054, -22.86), 'x_right': 198, 'x_left': 0},
                 {'coeff': (-0.00054, -68.57), 'x_right': 0, 'x_left': -183},
                 {'coeff': (-0.00054, -114.29), 'x_right': 159, 'x_left': 0},
                 {'coeff': (-0.00054, -160.35), 'x_right': -25.54, 'x_left': -116}])

# Intermediate difficulty
diff_eqs.append([{'coeff': (-0.00054, 160.35), 'x_right': 138, 'x_left': -62.11},
                 {'coeff': (-0.00054, 114.29), 'x_right': 62.11, 'x_left': -180},
                 {'coeff': (-0.00054, 68.57), 'x_right': 198, 'x_left': -47.58},
                 {'coeff': (-0.00054, 22.86), 'x_right': 49.69, 'x_left': -203},
                 {'coeff': (-0.00054, -22.86), 'x_right': 198, 'x_left': -48.47},
                 {'coeff': (-0.00054, -68.57), 'x_right': 44.06, 'x_left': -183},
                 {'coeff': (-0.00054, -114.29), 'x_right': 159, 'x_left': -54.09},
                 {'coeff': (-0.00054, -160.35), 'x_right': -25.54, 'x_left': -116}])

# Expert difficulty
diff_eqs.append([{'coeff': (-0.00054, 160.35), 'x_right': 138, 'x_left': -62.11},
                 {'coeff': (-0.00054, 114.29), 'x_right': 123.94, 'x_left': -180},
                 {'coeff': (-0.00054, 68.57), 'x_right': 198, 'x_left': -95.03},
                 {'coeff': (-0.00054, 22.86), 'x_right': 99.24, 'x_left': -203},
                 {'coeff': (-0.00054, -22.86), 'x_right': 198, 'x_left': -96.8},
                 {'coeff': (-0.00054, -68.57), 'x_right': 88.03, 'x_left': -183},
                 {'coeff': (-0.00054, -114.29), 'x_right': 159, 'x_left': -108},
                 {'coeff': (-0.00054, -160.35), 'x_right': -25.54, 'x_left': -116}])

# Legendary difficulty
diff_eqs.append([{'coeff': (-0.00054, 160.35), 'x_right': 138, 'x_left': -62.11},
                 {'coeff': (-0.00054, 114.29), 'x_right': 123.94, 'x_left': -180},
                 {'coeff': (-0.00054, 68.57), 'x_right': 198, 'x_left': -142.24},
                 {'coeff': (-0.00054, 22.86), 'x_right': 148.51, 'x_left': -203},
                 {'coeff': (-0.00054, -22.86), 'x_right': 198, 'x_left': -144.88},
                 {'coeff': (-0.00054, -68.57), 'x_right': 131.8, 'x_left': -183},
                 {'coeff': (-0.00054, -114.29), 'x_right': 159, 'x_left': -108},
                 {'coeff': (-0.00054, -160.35), 'x_right': -25.54, 'x_left': -116}])
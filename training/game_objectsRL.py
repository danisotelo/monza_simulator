# Import libraries

import torch
import torch.nn as nn
import torch.optim as optim
import random
import pygame
import math
import numpy as np
from pygame.locals import *
import matplotlib.pyplot as plt
import os

# import multiprocessing
# import pyqtgraph as pg
# from pyqtgraph.Qt import QtWidgets
# import numpy as np

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
global diff_eqs_training
diff_eqs = []
diff_eqs_training = []

SECONDS = int((1/dt) * 5) #seconds for reinforcement learning





# This function will run in a separate process
# def plot_process(reward_queue):
#     app = QtWidgets.QApplication([])
#     plot_window = pg.GraphicsLayoutWidget()
#     plot_window.show()
#     plot_widget = plot_window.addPlot(title="Rewards Over Time")

#     while True:
#         if not reward_queue.empty():
#             reward = reward_queue.get()
#             if reward is None:
#                 break  # Exit signal received
#             plot_widget.plot(np.arange(len(reward)), reward, clear=True)

#     pg.exec()  # Start the Qt event loop

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

diff_eqs_training.append([{'coeff': (-0.00054, 22.86), 'x_right': 50, 'x_left': -203},
                          {'coeff': (-0.00054, -22.86), 'x_right': 198, 'x_left': -144.88}])

# Convert local coordinates to global coordinates
def local_to_global(x1, y1, theta):
    x = x1*math.cos(theta) + y1*math.sin(theta)
    y = y1*math.cos(theta) - x1*math.sin(theta)
    return x, y

# Convert global coordinates to pygame coordinates
def world_to_screen(x, y):
    return int(x + x_offset), int(y_offset - y)

# Class to define the Monza coin    
class Coin:
    def __init__(self, diff, x1_0, y1_0, dx1_0):

        # Local position in {x1, y1}, circle labyrinth reference
        self.x1 = x1_0
        self.y1 = y1_0
        
        # Global position
        self.x = x1_0
        self.y = y1_0

        # Velocities
        self.dx = dx1_0
        self.dy = 0

        # Accelerations
        self.ddx = 0
        self.ddy = 0

        # Coin parameters
        self.mass = m_coin
        self.radius = r_coin
        self.width = w_coin
        self.color = BROWN
        
        self.n = 0 # The floor in which the ball is (0 if it falls from the top)
        self.t = 0 # Timer for the fall
        
        # Take the equation for the stage in which the coin is
        self.diff = diff
        self.eq_data = diff_eqs[self.diff][self.n]
        
        self.flag_fall = True # Flag to indicate the coin when to fall
        self.flag_out = False # Flag to indicate the coin exited the utile area
        self.flag_fly = False # Flag to indicate the coin has lost contact with the rail (negative normal force)
        self.flag_end = False # Flag to indicate the coin has lost contact with the rail (negative normal force)

        # List that will contain the coordinates of the game
        self.positions = []

    # Coin representation on the screen
    def draw(self, screen):
        pygame.draw.circle(screen, self.color, world_to_screen(self.x, self.y), self.radius, 0) # Width = 0 for the ball to be solid
    
    # Coin fall motion model   
    def fall(self, theta):
        if self.n < len(diff_eqs[self.diff]):
            # Check if it has reached the obstacle
            if self.y1 + self.dy*dt/10 <= self.eq_data['coeff'][0] * ((self.x1 + self.dx *dt/10) ** 2) + self.eq_data['coeff'][1] + self.radius:
                
                # Calculate the velocity modulus
                v_mod = math.sqrt(self.dx**2 + self.dy**2)
                # If the modulus is greater than the minimum velocity the coin bounces
                if v_mod >= v_min:
                    if self.dx != 0: #The general case of bouncing
                        gamma = math.atan(self.dy/self.dx)
                        phi = math.atan(0.00108*self.x1)
                        self.dx = np.sign(self.dx) * abs(math.cos(gamma - 2*phi) * math.e**(-k) * v_mod)
                        self.dy = abs(math.sin(gamma - 2*phi) * math.e**(-k) * v_mod)
                        self.x1 += self.dx*dt
                        self.y1 += self.dy*dt
                        self.x, self.y = local_to_global(self.x1, self.y1, theta) # Local to global coordinates
                        
                        if self.n == 1 and self.x1 < -120.6  + self.radius:
                            self.dx = -self.dx * math.e**(-k)
                        
                    else: # In case of the initial vertical fall
                        self.dy = math.e**(-k) * v_mod
                        self.y1 += self.dy*dt
                        self.x, self.y = local_to_global(self.x1, self.y1, theta) # Local to global coordinates
                        
                else: # If the velocity is lower than the minimum, then it continues to the next path
                    self.y1 = self.eq_data['coeff'][0] * (self.x1 ** 2) + self.eq_data['coeff'][1] + self.radius
                    self.x, self.y = local_to_global(self.x1, self.y1, theta) # Local to global coordinates
                    self.flag_fall = False
                    pass
                
            else:  # If it has not reached the obstacle it continues with its trajectory
                
                if self.dx != 0: # The general case
                    # New velocities and positions for instant dt
                    gamma = math.atan(self.dy/self.dx)
                    self.ddx = g*math.cos(beta)*math.sin(theta) - self.dx*mu_v/self.mass - mu_pd*g*math.sin(beta)*math.cos(gamma)
                    self.x1 += self.dx*dt + 0.5*self.ddx*dt**2
                    self.dx += self.ddx*dt
                    self.ddy = self.dy*mu_v/self.mass + mu_pd*g*math.sin(beta)*math.sin(gamma) - g*math.cos(beta)*math.cos(theta)
                    self.y1 += self.dy*dt + 0.5*self.ddy*dt**2
                    self.dy += self.ddy*dt
                    self.x, self.y = local_to_global(self.x1, self.y1, theta) # Local to global coordinates
                    
                    # if self.n == 1 and self.x1 < -120.6  + self.radius: #check bounce against vertical wall
                    #     self.dx = -self.dx * math.e**(-k)
                    #     self.x1 = -120.6 + self.radius + 0.01
                    
                else: # In case of the initial vertical fall
                    self.y1 += self.dy*dt - 0.5*g*dt**2
                    self.dy -= g*dt
                    
        else:
            # New velocities and positions for instant dt
            gamma = math.atan(self.dy/self.dx)
            self.ddx = g*math.cos(beta)*math.sin(theta) - self.dx*mu_v/self.mass - mu_pd*g*math.sin(beta)*math.cos(gamma)
            self.x1 += self.dx*dt + 0.5*self.ddx*dt**2
            self.dx += self.ddx*dt
            self.ddy = self.dy*mu_v/self.mass + mu_pd*g*math.sin(beta)*math.sin(gamma) - g*math.cos(beta)*math.cos(theta)
            self.y1 += self.dy*dt + 0.5*self.ddy*dt**2
            self.dy += self.ddy*dt
            self.x, self.y = local_to_global(self.x1, self.y1, theta) # Local to global coordinates
            # Arrive to the end of the game
            if self.y1 <= -200:
                self.flag_end = True
    
    # Coin update dynamics
    def update(self, theta):
        # Register position of ball
        self.t += dt
        self.x, self.y = local_to_global(self.x1, self.y1, theta) 
        self.positions.append((self.x1, self.y1, theta, self.t))

        if (self.flag_fall == True):
            self.fall(theta)
        else:
            # Angles
            phi = math.atan(0.00108*self.x1)
            alpha = theta + phi
            
            # Check if the angle is enough for the coin to start rolling
            if abs(math.sin(alpha)) > abs(mu_ps * math.tan(beta)):
                # Calculation of the normal force
                N = self.mass * (g * math.cos(alpha) * math.cos(beta) - 0.00108 * self.dx**2 / (1 + (0.00108*self.x1)**2) ** (1/2))
                if N > 0:
                    if mu_s >= (self.mass * g * math.sin(alpha) * math.cos(beta) - self.dx * mu_v - mu_pd * self.mass * g * math.sin(beta))/(3 * N):
                        # Acceleration for rolling condition
                        self.ddx = 2/3 * math.cos(phi) * (g*math.sin(alpha) * math.cos(beta) - self.dx * math.sqrt(1 + (0.00108 * self.x1)**2) * (mu_v/self.mass) - mu_pd * g * math.sin(beta))
                    else:
                        # Acceleration for sliding condition
                        self.ddx = math.cos(phi) * (g * math.sin(alpha) * math.cos(beta) - self.dx * math.sqrt(1 + (0.00108 * self.x1)**2) * (mu_v / self.mass) - mu_pd * g * math.sin(beta) - N * mu_f / self.mass)
                        
                    # Calculation of positions and velocities
                    self.x1 += self.dx * dt + 0.5 * self.ddx * (dt**2) #UARM
                    self.y1 = self.eq_data['coeff'][0] * (self.x1 ** 2) + self.eq_data['coeff'][1] + self.radius
                    self.dx += self.ddx * dt
                    self.dy = 0
                    self.x, self.y = local_to_global(self.x1, self.y1, theta) # Local to global coordinates  
                    
                    # If it arrives to one of the borders of the rail then it falls
                    if (self.x1 <= self.eq_data['x_left'] or self.x1 >= self.eq_data['x_right']):
                        self.n += 1
                        if self.n == len(diff_eqs[self.diff]):
                            self.flag_fall = True
                        else:
                            self.eq_data = diff_eqs[self.diff][self.n]
                            self.flag_fall = True
                    
                    # If it arrives to one of the walls then the velocity is inverted (inelastic contact)
                    # if self.n == 0 and self.x1 > 81.9 - self.radius:
                    #     self.dx = -self.dx * math.e**(-k)
                    #     self.x1 = 81.9 - self.radius - 0.01 #-epsilon
                    # elif self.n == 1 and self.x1 < -120.6 + self.radius:
                    #     self.dx = -self.dx * math.e**(-k)
                    #     self.x1 = -120.6 + self.radius + 0.01 #+epsilon
                        
                else:
                    self.flag_fly = True
                    
            else:
                self.x, self.y = local_to_global(self.x1, self.y1, theta) # Local to global coordinates
            
    def check_exit(self):
        if math.sqrt(self.x**2 + self.y**2) >= r_area + self.radius/4:
            self.flag_out = True
        else:
            pass       

class Coin_Training:
    def __init__(self, diff, x1_0, y1_0, dx1_0):

        # Local position in {x1, y1}, circle labyrinth reference
        self.x1 = x1_0
        self.y1 = y1_0
        
        # Global position
        self.x = x1_0
        self.y = y1_0

        # Velocities
        self.dx = dx1_0
        self.dy = 0

        # Accelerations
        self.ddx = 0
        self.ddy = 0

        # Coin parameters
        self.mass = m_coin
        self.radius = r_coin
        self.width = w_coin
        self.color = BROWN
        
        self.n = 0 # The floor in which the ball is (0 if it falls from the top)
        self.t = 0 # Timer for the fall
        
        # Take the equation for the stage in which the coin is
        self.diff = diff
        self.eq_data = diff_eqs_training[self.diff][self.n]
        
        self.flag_fall = True # Flag to indicate the coin when to fall
        self.flag_out = False # Flag to indicate the coin exited the utile area
        self.flag_fly = False # Flag to indicate the coin has lost contact with the rail (negative normal force)
        self.flag_end = False # Flag to indicate the coin has lost contact with the rail (negative normal force)

        # List that will contain the coordinates of the game
        self.positions = []

    # Coin representation on the screen
    def draw(self, screen):
        pygame.draw.circle(screen, self.color, world_to_screen(self.x, self.y), self.radius, 0) # Width = 0 for the ball to be solid
    
    # Coin fall motion model   
    def fall(self, theta):
        if self.n < len(diff_eqs_training[self.diff]):
            # Check if it has reached the obstacle
            if self.y1 + self.dy*dt/10 <= self.eq_data['coeff'][0] * ((self.x1 + self.dx *dt/10) ** 2) + self.eq_data['coeff'][1] + self.radius:
                
                # Calculate the velocity modulus
                v_mod = math.sqrt(self.dx**2 + self.dy**2)
                # If the modulus is greater than the minimum velocity the coin bounces
                if v_mod >= v_min:
                    if self.dx != 0: #The general case of bouncing
                        gamma = math.atan(self.dy/self.dx)
                        phi = math.atan(0.00108*self.x1)
                        self.dx = np.sign(self.dx) * abs(math.cos(gamma - 2*phi) * math.e**(-k) * v_mod)
                        self.dy = abs(math.sin(gamma - 2*phi) * math.e**(-k) * v_mod)
                        self.x1 += self.dx*dt
                        self.y1 += self.dy*dt
                        self.x, self.y = local_to_global(self.x1, self.y1, theta) # Local to global coordinates
                        
                        if self.n == 1 and self.x1 < -120.6  + self.radius:
                            self.dx = -self.dx * math.e**(-k)
                        
                    else: # In case of the initial vertical fall
                        self.dy = math.e**(-k) * v_mod
                        self.y1 += self.dy*dt
                        self.x, self.y = local_to_global(self.x1, self.y1, theta) # Local to global coordinates
                        
                else: # If the velocity is lower than the minimum, then it continues to the next path
                    self.y1 = self.eq_data['coeff'][0] * (self.x1 ** 2) + self.eq_data['coeff'][1] + self.radius
                    self.x, self.y = local_to_global(self.x1, self.y1, theta) # Local to global coordinates
                    self.flag_fall = False
                    pass
                
            else:  # If it has not reached the obstacle it continues with its trajectory
                
                if self.dx != 0: # The general case
                    # New velocities and positions for instant dt
                    gamma = math.atan(self.dy/self.dx)
                    self.ddx = g*math.cos(beta)*math.sin(theta) - self.dx*mu_v/self.mass - mu_pd*g*math.sin(beta)*math.cos(gamma)
                    self.x1 += self.dx*dt + 0.5*self.ddx*dt**2
                    self.dx += self.ddx*dt
                    self.ddy = self.dy*mu_v/self.mass + mu_pd*g*math.sin(beta)*math.sin(gamma) - g*math.cos(beta)*math.cos(theta)
                    self.y1 += self.dy*dt + 0.5*self.ddy*dt**2
                    self.dy += self.ddy*dt
                    self.x, self.y = local_to_global(self.x1, self.y1, theta) # Local to global coordinates
                    
                    # if self.n == 1 and self.x1 < -120.6  + self.radius: #check bounce against vertical wall
                    #     self.dx = -self.dx * math.e**(-k)
                    #     self.x1 = -120.6 + self.radius + 0.01
                    
                else: # In case of the initial vertical fall
                    self.y1 += self.dy*dt - 0.5*g*dt**2
                    self.dy -= g*dt
                    
        else:
            # New velocities and positions for instant dt
            gamma = math.atan(self.dy/self.dx)
            self.ddx = g*math.cos(beta)*math.sin(theta) - self.dx*mu_v/self.mass - mu_pd*g*math.sin(beta)*math.cos(gamma)
            self.x1 += self.dx*dt + 0.5*self.ddx*dt**2
            self.dx += self.ddx*dt
            self.ddy = self.dy*mu_v/self.mass + mu_pd*g*math.sin(beta)*math.sin(gamma) - g*math.cos(beta)*math.cos(theta)
            self.y1 += self.dy*dt + 0.5*self.ddy*dt**2
            self.dy += self.ddy*dt
            self.x, self.y = local_to_global(self.x1, self.y1, theta) # Local to global coordinates
            # Arrive to the end of the game
            if self.y1 <= -200:
                self.flag_end = True
    
    # Coin update dynamics
    def update(self, theta):
        # Register position of ball
        self.t += dt
        self.x, self.y = local_to_global(self.x1, self.y1, theta) 
        self.positions.append((self.x1, self.y1, theta, self.t))

        if (self.flag_fall == True):
            self.fall(theta)
        else:
            # Angles
            phi = math.atan(0.00108*self.x1)
            alpha = theta + phi
            
            # Check if the angle is enough for the coin to start rolling
            if abs(math.sin(alpha)) > abs(mu_ps * math.tan(beta)):
                # Calculation of the normal force
                N = self.mass * (g * math.cos(alpha) * math.cos(beta) - 0.00108 * self.dx**2 / (1 + (0.00108*self.x1)**2) ** (1/2))
                if N > 0:
                    if mu_s >= (self.mass * g * math.sin(alpha) * math.cos(beta) - self.dx * mu_v - mu_pd * self.mass * g * math.sin(beta))/(3 * N):
                        # Acceleration for rolling condition
                        self.ddx = 2/3 * math.cos(phi) * (g*math.sin(alpha) * math.cos(beta) - self.dx * math.sqrt(1 + (0.00108 * self.x1)**2) * (mu_v/self.mass) - mu_pd * g * math.sin(beta))
                    else:
                        # Acceleration for sliding condition
                        self.ddx = math.cos(phi) * (g * math.sin(alpha) * math.cos(beta) - self.dx * math.sqrt(1 + (0.00108 * self.x1)**2) * (mu_v / self.mass) - mu_pd * g * math.sin(beta) - N * mu_f / self.mass)
                        
                    # Calculation of positions and velocities
                    self.x1 += self.dx * dt + 0.5 * self.ddx * (dt**2) #UARM
                    self.y1 = self.eq_data['coeff'][0] * (self.x1 ** 2) + self.eq_data['coeff'][1] + self.radius
                    self.dx += self.ddx * dt
                    self.dy = 0
                    self.x, self.y = local_to_global(self.x1, self.y1, theta) # Local to global coordinates  
                    
                    # If it arrives to one of the borders of the rail then it falls
                    if (self.x1 <= self.eq_data['x_left'] or self.x1 >= self.eq_data['x_right']):
                        self.n += 1
                        if self.n == len(diff_eqs_training[self.diff]):
                            self.flag_fall = True
                        else:
                            self.eq_data = diff_eqs_training[self.diff][self.n]
                            self.flag_fall = True
                    
                    # If it arrives to one of the walls then the velocity is inverted (inelastic contact)
                    # if self.n == 0 and self.x1 > 81.9 - self.radius:
                    #     self.dx = -self.dx * math.e**(-k)
                    #     self.x1 = 81.9 - self.radius - 0.01 #-epsilon
                    # elif self.n == 1 and self.x1 < -120.6 + self.radius:
                    #     self.dx = -self.dx * math.e**(-k)
                    #     self.x1 = -120.6 + self.radius + 0.01 #+epsilon
                        
                else:
                    self.flag_fly = True
                    
            else:
                self.x, self.y = local_to_global(self.x1, self.y1, theta) # Local to global coordinates
            
    def check_exit(self):
        if math.sqrt(self.x**2 + self.y**2) >= r_area + self.radius/4:
            self.flag_out = True
        else:
            pass       


# Class to define the circular background
class Circle:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.radius = r_circle
        self.r_area = r_area
        self.angle = 0

        # Load and scale the image
        self.image = pygame.image.load("assets/game/background.png")

    def draw(self, screen):
        # Rotate the image
        rotated_image = pygame.transform.rotate(self.image, - self.angle * 180/math.pi)
        
        # Draw the image
        position = world_to_screen(self.x, self.y)
        rect = rotated_image.get_rect()
        rect.center = position
        screen.blit(rotated_image, rect.topleft)

        # Valid area limits
        pygame.draw.circle(screen, MID_BROWN, world_to_screen(self.x, self.y), self.r_area, w_circle)

    def rotate(self, delta_angle):
        self.angle -= delta_angle # Clockwise sense


# Class to define the obstacles of the labyrinth
class Obstacle:
    def __init__(self, circle, diff):
        self.circle = circle
        self.equations_data = diff_eqs[diff]
        self.points_list = []
        # self.points_lines_list = [(-120.6, 106.5), (-120.6, 164), (81.9, 156.5), (81.9, 186),
        #                           (-25.54, -160.4), (-4, -204), (0, -114.3), (43, -199)]
        for eq_data in self.equations_data:
            x_values = np.linspace(eq_data['x_left'], eq_data['x_right'], num = 100).tolist()
            points = [(x, int(eq_data['coeff'][0] * (x ** 2) + eq_data['coeff'][1])) for x in x_values]
            self.points_list.append(points)
        self.color = CARAMEL

    def draw(self, screen):
        for points in self.points_list:
            rotated_points = []
            for point in points:
                distance = math.sqrt((point[0]**2) + (point[1]**2))
                angle = math.atan2(point[1], point[0]) - self.circle.angle
                x = self.circle.x + distance * math.cos(angle)
                y = self.circle.y + distance * math.sin(angle)
                rotated_points.append((x, y))
            
            for i in range(1, len(rotated_points)):
                start_point = world_to_screen(rotated_points[i-1][0], rotated_points[i-1][1])
                end_point = world_to_screen(rotated_points[i][0], rotated_points[i][1])
                pygame.draw.line(screen, self.color, start_point, end_point, w_obstacle)
        
        # Draw the straight lines
        # for line in range(1, len(self.points_lines_list), 2):
        #         distance_1 = math.sqrt((self.points_lines_list[line-1][0]**2) + (self.points_lines_list[line-1][1]**2))
        #         angle_1 = math.atan2(self.points_lines_list[line-1][1], self.points_lines_list[line-1][0]) - self.circle.angle
        #         x_1 = self.circle.x + distance_1 * math.cos(angle_1)
        #         y_1 = self.circle.y + distance_1 * math.sin(angle_1)
                
        #         distance_2 = math.sqrt((self.points_lines_list[line][0]**2) + (self.points_lines_list[line][1]**2))
        #         angle_2 = math.atan2(self.points_lines_list[line][1], self.points_lines_list[line][0]) - self.circle.angle
        #         x_2 = self.circle.x + distance_2 * math.cos(angle_2)
        #         y_2 = self.circle.y + distance_2 * math.sin(angle_2)
                
        #         pygame.draw.line(screen, self.color, world_to_screen(x_1, y_1), world_to_screen(x_2, y_2), w_obstacle)


class Obstacle_Training:
    def __init__(self, circle, diff):
        self.circle = circle
        self.equations_data = diff_eqs_training[diff]
        self.points_list = []
        # self.points_lines_list = [(-120.6, 106.5), (-120.6, 164), (81.9, 156.5), (81.9, 186),
        #                           (-25.54, -160.4), (-4, -204), (0, -114.3), (43, -199)]
        for eq_data in self.equations_data:
            x_values = np.linspace(eq_data['x_left'], eq_data['x_right'], num = 100).tolist()
            points = [(x, int(eq_data['coeff'][0] * (x ** 2) + eq_data['coeff'][1])) for x in x_values]
            self.points_list.append(points)
        self.color = CARAMEL #ñam ñam

    def draw(self, screen):
        for points in self.points_list:
            rotated_points = []
            for point in points:
                distance = math.sqrt((point[0]**2) + (point[1]**2))
                angle = math.atan2(point[1], point[0]) - self.circle.angle
                x = self.circle.x + distance * math.cos(angle)
                y = self.circle.y + distance * math.sin(angle)
                rotated_points.append((x, y))
            
            for i in range(1, len(rotated_points)):
                start_point = world_to_screen(rotated_points[i-1][0], rotated_points[i-1][1])
                end_point = world_to_screen(rotated_points[i][0], rotated_points[i][1])
                pygame.draw.line(screen, self.color, start_point, end_point, w_obstacle)


# Class to define the Monza game
class Monza:
    def __init__(self, screen):
        self.screen = screen
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 26)

        self.reset()
        
    
    def reset(self):
        # Training
        # random hole position for RL training
        global diff_eqs_training
        self.aux_x_hole = random.randint(-150, 150)
        if self.aux_x_hole > 0:
            self.aux_x1_0 = random.randint(-150, -20)
            self.direction = 1
            diff_eqs_training[0] = [{'coeff': (-0.00054, 22.86), 'x_right': self.aux_x_hole, 'x_left': -203},
                          {'coeff': (-0.00054, -22.86), 'x_right': 198, 'x_left': -203}]
        else:
            self.aux_x1_0 = random.randint(20, 150)
            self.direction = -1
            diff_eqs_training[0] = [{'coeff': (-0.00054, 22.86), 'x_right': 203, 'x_left': self.aux_x_hole},
                          {'coeff': (-0.00054, -22.86), 'x_right': 198, 'x_left': -198}]
            

        # Create the game objects   
        self.diff = 0  #random level = random.randint(0, 3)
        self.circle = Circle(0, 0) # Labyrinth circle
        self.coin = Coin_Training(self.diff, self.aux_x1_0, 100, dx1_0) # Coin with coordinates in {x1, y1}
        self.obstacle = Obstacle_Training(self.circle, self.diff) # Obstacles
        self.frame_iteration = 0
        self.level = 0
        #self.x_values = [(entry['x_right'], entry['x_left']) for entry in diff_eqs_training[0]]
        #self.x_values.append((0, 0))
        self.x_reference = 0
        self.game_over = False
        self.aux_actions = [0,0,0]
        self.distance_to_hole = abs(self.aux_x_hole - self.coin.x)
        self.flag = 0
        self.timer = 0

        

        

    def play_step(self, action):
        #print(self.direction)
        self.frame_iteration += 1
        self.aux_actions = action
        #print(self.distance_to_hole)
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                quit()
        
        self.screen.fill(BROWN)
            
        # Represent the objects on the screen
        self.circle.draw(self.screen)
        self.obstacle.draw(self.screen)
        self.coin.draw(self.screen)



        # 3. check if game over
        reward = 0
        
        if self.direction == 1: # hole to the right

            if self.coin.dx <= 0: # should go to the right
                if np.array_equal(action, [0, 1, 0]):
                    reward = 1
                else: 
                    reward = -1
            
            if self.coin.dx > 0:
                if self.coin.dx < 100:
                    if self.coin.ddx > 0: # should do nothing
                        if np.array_equal(action, [1, 0, 0]):
                            reward = 1
                        else: 
                            reward = -1
                    else: # should go to the right
                        if np.array_equal(action, [0, 1, 0]):
                            reward = 1
                        else: 
                            reward = -1

                else: # above 50
                    if self.coin.ddx > 0: #should try to break
                        if np.array_equal(action, [0, 0, 1]):
                            reward = 1
                        else: 
                            reward = -1
                    else: # should do nothing
                        if np.array_equal(action, [1, 0, 0]):
                            reward = 1
                        else: 
                            reward = -1

        else: # hole to the left
            if self.coin.dx >= 0: # should go to the left
                if np.array_equal(action, [0, 0, 1]):
                    reward = 1
                else: 
                    reward = -1
            
            if self.coin.dx < 0:
                if self.coin.dx > -100:
                    if self.coin.ddx < 0: # should do nothing
                        if np.array_equal(action, [1, 0, 0]):
                            reward = 1
                        else: 
                            reward = -1
                    else: # should go to the left
                        if np.array_equal(action, [0, 0, 1]):
                            reward = 1
                        else: 
                            reward = -1

                else: # above 50
                    if self.coin.ddx < 0: #should try to break
                        if np.array_equal(action, [0, 1, 0]):
                            reward = 1
                        else: 
                            reward = -1
                    else: # should do nothing
                        if np.array_equal(action, [1, 0, 0]):
                            reward = 1
                        else: 
                            reward = -1


        if self.coin.flag_out or self.coin.flag_fly or abs(self.coin.dx) > 600 or abs(self.circle.angle) > 0.7 :
            if self.game_over == False:
                if abs(self.coin.dx) > 600:
                    print("\n speed too high")
                if abs(self.circle.angle) > 0.7:
                    print("\n angle too high")
            self.game_over = True
            return reward, self.game_over
        
        if self.frame_iteration > SECONDS:
            self.game_over = True
            return reward, self.game_over
        

        
        # 2. move
        # Control the rotation of the circle labyrinth (theta)
        if self.direction == -1:
            if np.array_equal(action, [0, 1, 0]):
                self.circle.rotate(-0.005)
                #print("angle: ", self.circle.angle)
            if np.array_equal(action, [0, 0, 1]):
                self.circle.rotate(0.005)
                #print("angle: ", self.circle.angle)
        else:
            if np.array_equal(action, [0, 0, 1]):
                self.circle.rotate(-0.005)
                #print("angle: ", self.circle.angle)
            if np.array_equal(action, [0, 1, 0]):
                self.circle.rotate(0.005)
                #print("angle: ", self.circle.angle)

        # Update the theta angle for the coin
        theta = self.circle.angle

        # Coin update
        self.coin.update(theta)


        # Check if the coin has exited the utile area
        self.coin.check_exit()

        pygame.display.flip()
        fpsClock.tick(FPS)

        return reward, self.game_over #score
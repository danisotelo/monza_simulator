"""
Monza Game - Controllers
Description: Contains the controllers to control the Monza game automatically.
Author(s): Daniel Sotelo, Jiajun Xu, Vladyslav Korenyak
Date: 09/10/2023
Version: 2.0
"""

# Import libraries
import numpy as np
from settings import *

'''
Sliding Mode Control (SMC) defines a "sliding surface" (error function)
The control aims to drive the system state to this surface and then slide
along it until reaching the desired state. Parameters can be furtherly
tuned to improve performance:
 - Beginner: lambda = 100, k = 0.002
 - Intermediate: lambda = 100, k = 0.004
 - Expert: lambda = 110, k = 0.0095
 - Legendary: lambda = 120, k = 0.0095
'''

class SlidingModeController:
    def __init__(self, diff):
        self.prev_error = 0
        self.diff = diff
        self.lambda_ = [100, 100, 110, 120]
        self.k = [0.002, 0.004, 0.0095, 0.0095] # Control gain
    
    def get_target_position(self, stage):
        if stage >= len(diff_eqs[self.diff]):
            target_position = "Exit"
        else:
            eq_data = diff_eqs[self.diff][stage]
            if stage % 2 == 0:
                target_position = eq_data["x_left"]
            else:
                target_position = eq_data["x_right"]
        
        return target_position
        
    def control(self, coin_position, target_position):
        # Define the error as the difference between the target position and coin's position
        e = target_position - coin_position
        # Calculate the derivative of the error (rate of change)
        e_prime = e - self.prev_error
        # Define the sliding surface
        s = e + self.lambda_[self.diff] * e_prime
        # Implement the control law
        control_action = -self.k[self.diff] * np.sign(s)
        
        self.prev_error = e # Store the error for the next iteration

        return control_action
    
# Controlador de David


# Controlador de Vladys
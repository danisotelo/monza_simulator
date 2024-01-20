"""
Monza Game - Controllers
Description: Contains the controllers to control the Monza game automatically.
Author(s): Daniel Sotelo, Jiajun Xu, Vladyslav Korenyak
Date: 03/12/2023
Version: 3.0
"""

# Import libraries
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from settings import *

import torch
import numpy as np
from model import Linear_QNet

'''
Sliding Mode Control (SMC) defines a "sliding surface" (error function)
The control aims to drive the system state to this surface and then slide
along it until reaching the desired state. Parameters can be furtherly
tuned to improve performance:
 - Beginner: lambda = 100, k = 0.002
 - Intermediate: lambda = 100, k = 0.004
 - Expert: lambda = 110, k = 0.0095
 - Legendary: lambda = 140, k = 0.0095
'''

class SlidingModeController:
    def __init__(self, diff):
        self.prev_error = 0
        self.diff = diff
        self.lambda_ = [100, 100, 110, 140]
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


'''
Fuzzy Logic Control (FLC) uses degrees of truth, rather than strict binary 
(True or False), to analyze, make decisions, and control situations where
input data might be imprecise or ambiguos. The following controller is
adapted to beat Monza for all the different levels. 
'''


class FuzzyLogicController:
    def __init__(self, diff):
        self.diff = diff
        self.prev_dist = 0
        self.angle = 0
        self.coin = 0

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
    
    def control(self, coin_position, coin_n, target_position):

        # Obtain the current floor of the coin
        eq_data = diff_eqs[self.diff][coin_n]
        r_max = (eq_data["x_right"]-eq_data["x_left"])  # maximum range of x per floor

        # Input and output variables
        dist = target_position - coin_position
        vel = dist - self.prev_dist

        # Define input and output variables
        distance = ctrl.Antecedent(np.arange(-r_max, r_max, 5), 'distance')  # distance = target_position - coin_position
        velocity = ctrl.Antecedent(np.arange(-100, 100, 0.5), 'velocity')
        angle = ctrl.Consequent(np.arange(-0.04, 0.04, 0.001), 'angle')

        # Define the fuzzy sets for the input and output variables
        distance['left'] = fuzz.trimf(distance.universe, [-r_max, -r_max/4, 0])
        distance['center'] = fuzz.trimf(distance.universe, [-30.8, 0, 30.8])
        distance['right'] = fuzz.trimf(distance.universe, [0, r_max/4, r_max])

        velocity['negative+'] = fuzz.trimf(velocity.universe, [-100, -100, -1.15])
        velocity['negative'] = fuzz.trimf(velocity.universe, [-1.5, -1, 0])
        velocity['stop'] = fuzz.trimf(velocity.universe, [-1, 0, 1])
        velocity['positive'] = fuzz.trimf(velocity.universe, [0, 1, 1.5])
        velocity['positive+'] = fuzz.trimf(velocity.universe, [1.15, 100, 100])


        angle['left'] = fuzz.trimf(angle.universe, [-0.02, -0.005, 0])
        angle['mantain'] = fuzz.trimf(angle.universe, [-0.005, 0, 0.005])
        angle['right'] = fuzz.trimf(angle.universe, [0, 0.005, 0.02])

        # Define fuzzy rules
        rule1 = ctrl.Rule(distance['right'] & velocity['negative'], angle['right']) # going in the correct direction (clockwise)
        rule2 = ctrl.Rule(distance['right'] & velocity['negative+'], angle['left']) # going in the correct direction and accelerated (counterclockwise)
        rule3 = ctrl.Rule(distance['right'] & velocity['positive'], angle['right']) # going in the opposite direction (clockwise)
        rule4 = ctrl.Rule(distance['right'] & velocity['stop'], angle['right']) # heading in the right direction (clockwise)
        rule5 = ctrl.Rule(distance['center'] & velocity['negative'], angle['left']) # turning to not let the ball fall out and make the inertia fall slowly to the right to the next floor

        rule6 = ctrl.Rule(distance['left'] & velocity['positive'], angle['left']) # going in the correct direction (counterclockwise)
        rule7 = ctrl.Rule(distance['left'] & velocity['positive+'], angle['right']) # going in the correct direction and accelerated (clockwise)
        rule8 = ctrl.Rule(distance['left'] & velocity['negative'], angle['left']) # going in the opposite direction (counterclockwise)
        rule9 = ctrl.Rule(distance['left'] & velocity['stop'], angle['left']) # heading in the right direction (counterclockwise)
        rule10 = ctrl.Rule(distance['center'] & velocity['positive'], angle['right']) # turning to not let the ball fall out and make the inertia fall slowly to the left to the next floor

        # Creating and simulating a fuzzy control system
        control_system = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10])
        simulator = ctrl.ControlSystemSimulation(control_system)
        
        # Inputs for Fuzzify + Defuzzify according to centeroid
        simulator.input['distance'] = dist 
        simulator.input['velocity'] = vel 
        # Calculate the output
        try:
            simulator.compute()
            angle_output = simulator.output['angle']
            self.angle = angle_output
        except (AssertionError, ValueError):
            angle_output = 0

        self.prev_dist = dist
        return -angle_output


'''
This Reinforcement Learning Controller, equipped with an Artificial Neural Network,
is trained to maintain a consistent speed while navigating a parabolic path. 
It dynamically adjusts its control inputs based on real-time environmental and
system states to achieve optimal performance."
'''

#'''
class ReinforcementLearningController:

    def __init__(self, diff):
        self.model = Linear_QNet(3, 20, 3)
        self.diff = diff

    def get_target_position(self, stage):
        if stage >= len(diff_eqs[self.diff]):
            target_position = "Exit"
        else:
            if stage % 2 == 0:
                target_position = -1
            else:
                target_position = 1
    
        return target_position

    def load_model(self, model_file):
        # Load the saved state dictionary
        state_dict = torch.load(model_file, map_location=torch.device('cpu'))  # 'cpu' or 'cuda' if you are using GPU
        # Load the state dictionary into the model
        self.model.load_state_dict(state_dict)
        
    def get_state(self, game, target_position):

        # Normalized and standardized state
        if target_position == -1:
            if game.coin.n == 6:
                current_state = [
                    game.circle.angle,  # Already in [-1, 1]
                    game.coin.dx / 800,  # Scaled to [-1, 1]
                    target_position  # Assuming this is already normalized
                ] # If you read this you owe me a beer :)
            else:
                current_state = [
                    game.circle.angle,  # Already in [-1, 1]
                    game.coin.dx / 1200,  # Scaled to [-1, 1]
                    target_position  # Assuming this is already normalized
                ]
        else:
            current_state = [
                -game.circle.angle,  # Already in [-1, 1]
                -game.coin.dx / 1500,  # Scaled to [-1, 1]
                -target_position  # Assuming this is already normalized
            ]    

        return np.array(current_state, dtype=np.float32)

    def get_action(self, state):

        final_move = [0,0,0]
        state0 = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state0)
        move = torch.argmax(prediction).item()
        final_move[move] = 1
        return final_move
#'''

'''
# If you want to have fun seing some bugs use this controller (pretty difficult to do it manually though!)
# Are you able to reach the end of the game taking advantage of the bugs?
 
class ReinforcementLearningController:

    def __init__(self, diff):
        self.model = Linear_QNet(3, 20, 3)
        self.diff = diff

    def get_target_position(self, stage):
        if stage >= len(diff_eqs[self.diff]):
            target_position = "Exit"
        else:
            if stage % 2 == 0:
                target_position = -1
            else:
                target_position = 1
    
        return target_position

    def load_model(self, model_file):
        # Load the saved state dictionary
        state_dict = torch.load(model_file, map_location=torch.device('cpu'))  # 'cpu' or 'cuda' if you are using GPU
        # Load the state dictionary into the model
        self.model.load_state_dict(state_dict)
        
    def get_state(self, game, target_position):

        # Normalized and standardized state
        if target_position == -1:
            if game.coin.n == 7:
                current_state = [
                    game.circle.angle,  # Already in [-1, 1]
                    game.coin.dx / 900,  # Scaled to [-1, 1]
                    target_position  # Assuming this is already normalized
                ]
            elif game.coin.n == 4:
                current_state = [
                    game.circle.angle,  # Already in [-1, 1]
                    game.coin.dx / 900,  # Scaled to [-1, 1]
                    target_position  # Assuming this is already normalized
                ]
            elif game.coin.n == 6:
                current_state = [
                    game.circle.angle,  # Already in [-1, 1]
                    game.coin.dx / 1500,  # Scaled to [-1, 1]
                    target_position  # Assuming this is already normalized
                ]                
            else:
                current_state = [
                    game.circle.angle,  # Already in [-1, 1]
                    game.coin.dx / 1500,  # Scaled to [-1, 1]
                    target_position  # Assuming this is already normalized
                ]                
        else:
            current_state = [
                -game.circle.angle,  # Already in [-1, 1]
                -game.coin.dx / 1500,  # Scaled to [-1, 1]
                -target_position  # Assuming this is already normalized
            ]    

        return np.array(current_state, dtype=np.float32)

    def get_action(self, state):

        final_move = [0,0,0]
        state0 = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state0)
        move = torch.argmax(prediction).item()
        final_move[move] = 1
        return final_move
        
'''  

"""
Monza Game - Controllers
Description: Contains the controllers to control the Monza game automatically.
Author(s): Daniel Sotelo, Jiajun Xu, Vladyslav Korenyak
Date: 09/10/2023
Version: 2.0
"""

# Import libraries
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from settings import *

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


# Controlador de David
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

        # Obtener el piso de la moneda actual
        eq_data = diff_eqs[self.diff][coin_n]
        r_max = (eq_data["x_right"]-eq_data["x_left"])  # rango máximo de x por piso

        # Variables de entrada y salida
        dist = target_position - coin_position
        vel = dist - self.prev_dist

        # Definir las variables de entrada y salida
        distancia = ctrl.Antecedent(np.arange(-r_max, r_max, 5), 'distancia')  # distancia = target_position - coin_position
        velocidad = ctrl.Antecedent(np.arange(-100, 100, 0.5), 'velocidad')
        angulo = ctrl.Consequent(np.arange(-0.04, 0.04, 0.001), 'angulo')

        
        # Definir los conjuntos difusos para las variables de entrada y salida
        distancia['izquierda'] = fuzz.trimf(distancia.universe, [-r_max, -r_max/4, 0])
        distancia['centro'] = fuzz.trimf(distancia.universe, [-30.8, 0, 30.8])
        distancia['derecha'] = fuzz.trimf(distancia.universe, [0, r_max/4, r_max])

        velocidad['negativa+'] = fuzz.trimf(velocidad.universe, [-100, -100, -1.15])
        velocidad['negativa'] = fuzz.trimf(velocidad.universe, [-1.5, -1, 0])
        velocidad['parada'] = fuzz.trimf(velocidad.universe, [-1, 0, 1])
        velocidad['positiva'] = fuzz.trimf(velocidad.universe, [0, 1, 1.5])
        velocidad['positiva+'] = fuzz.trimf(velocidad.universe, [1.15, 100, 100])


        angulo['izquierda'] = fuzz.trimf(angulo.universe, [-0.02, -0.005, 0])
        angulo['mantener'] = fuzz.trimf(angulo.universe, [-0.005, 0, 0.005])
        angulo['derecha'] = fuzz.trimf(angulo.universe, [0, 0.005, 0.02])

        """
        distancia.view()
        velocidad.view()
        angulo.view()
        plt.show()
        """
        # Definir las reglas difusas
        regla1 = ctrl.Rule(distancia['derecha'] & velocidad['negativa'], angulo['derecha']) # yendo sentido correcto (horario)
        regla2 = ctrl.Rule(distancia['derecha'] & velocidad['negativa+'], angulo['izquierda']) # yendo sentido correcto y acelerado (antihorario)
        regla3 = ctrl.Rule(distancia['derecha'] & velocidad['positiva'], angulo['derecha']) # yendo en sentido contrario (horario)
        regla4 = ctrl.Rule(distancia['derecha'] & velocidad['parada'], angulo['derecha']) # me dirijo a sentido correcto (horario)
        regla5 = ctrl.Rule(distancia['centro'] & velocidad['negativa'], angulo['izquierda']) # giro para que no se vaya y que la inercia haga caer la bola por la derecha

        regla6 = ctrl.Rule(distancia['izquierda'] & velocidad['positiva'], angulo['izquierda']) # yendo sentido correcto (antihorario)
        regla7 = ctrl.Rule(distancia['izquierda'] & velocidad['positiva+'], angulo['derecha']) # yendo sentido correcto y acelerado (horario)
        regla8 = ctrl.Rule(distancia['izquierda'] & velocidad['negativa'], angulo['izquierda']) # yendo en sentido contrario (antihorario)
        regla9 = ctrl.Rule(distancia['izquierda'] & velocidad['parada'], angulo['izquierda']) # me dirijo a sendito correcto (antihorario)
        regla10 = ctrl.Rule(distancia['centro'] & velocidad['positiva'], angulo['derecha']) # giro para que no se vaya y dejo que la inercia haga caer la bola por la izquierda

        # Crear y simular un sistema de control difuso
        sistema_control = ctrl.ControlSystem([regla1, regla2, regla3, regla4, regla5, regla6, regla7, regla8, regla9, regla10])
        simulador = ctrl.ControlSystemSimulation(sistema_control)
        
        # Inputs para Fuzzificar + Defuzzificar según centroid
        simulador.input['distancia'] = dist 
        simulador.input['velocidad'] = vel 
        # Calcular la salida
        try:
            simulador.compute()
            angulo_output = simulador.output['angulo']
            self.angle = angulo_output
        except (AssertionError, ValueError):
            angulo_output = 0

        self.prev_dist = dist
        return -angulo_output

# Controlador de Vladys
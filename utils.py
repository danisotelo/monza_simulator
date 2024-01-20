"""
Monza Game - Utility Functions
Description: Contains helper and utility functions used across the game.
Author(s): Daniel Sotelo, Jiajun Xu, Vladyslav Korenyak
Date: 09/10/2023
Version: 2.0
"""

# Import libraries
import pygame
import math
import matplotlib.pyplot as plt
from settings import *
import numpy as np

def draw_mit_license_and_authors(screen, color):
    # Select font for the text
    font = pygame.font.Font(None, 15) 
        
    # Draw the MIT License text
    license_text = "This software is released under the MIT License"
    authors_text = "Created by Daniel Sotelo, Jiajun Xu, and Vladyslav Korenyak"
    
    license_surf = font.render(license_text, True, color)
    authors_surf = font.render(authors_text, True, color)
    
    license_rect = license_surf.get_rect(centerx = SCREEN_WIDTH // 2, bottom = SCREEN_HEIGHT - 10)
    authors_rect = authors_surf.get_rect(centerx = SCREEN_WIDTH // 2, bottom = SCREEN_HEIGHT - 20)
    
    screen.blit(license_surf, license_rect)
    screen.blit(authors_surf, authors_rect)
    
# Convert local coordinates to global coordinates
def local_to_global(x1, y1, theta):
    x = x1*math.cos(theta) + y1*math.sin(theta)
    y = y1*math.cos(theta) - x1*math.sin(theta)
    return x, y

# Convert global coordinates to pygame coordinates
def world_to_screen(x, y):
    return int(x + x_offset), int(y_offset - y)

# Plot the path of the ball in a game
def plot_graphs(positions):
    x = [pos[0] for pos in positions]
    y = [pos[1] for pos in positions]
    thetas = [-angle[2] * 180 / math.pi for angle in positions]
    t = [time[3] for time in positions]

    # Compute angular velocity (omega)
    thetas_rad = [-angle[2] for angle in positions]  # θ in radians
    omega = np.diff(thetas_rad) / np.diff(t)  # ω = Δθ / Δt

    # Plot for position
    plt.plot(x, y, color = (0.486, 0.278, 0), linestyle="-", linewidth=2)
    plt.title("Path of the coin", fontsize=14)
    plt.xlabel(r"$x$ $(mm)$", fontsize=12)
    plt.ylabel(r"$y$ $(mm)$", fontsize=12)
    plt.grid(True)
    plt.show()

    # Plot for angle
    plt.plot(t, thetas, color = (0.486, 0.278, 0), linestyle="-", linewidth=2)
    plt.title("Evolution of the angle of inclination", fontsize=14)
    plt.xlabel(r"$t$ $(s)$", fontsize=12)
    plt.ylabel(r"$\theta$ ($^\circ$)", fontsize=12)
    plt.grid(True)
    plt.show()

    # Plot for angular velocity
    t_mid = t[:-1] + np.diff(t) / 2  # Compute midpoints of time intervals for plotting
    plt.plot(t_mid, omega, color = (0.486, 0.278, 0), linestyle="-", linewidth=2)
    plt.title("Angular Velocity Over Time", fontsize=14)
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Angular Velocity (rad/s)", fontsize=12)
    plt.grid(True)
    plt.show()
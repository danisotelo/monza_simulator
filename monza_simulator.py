'''
=================================================================================================================
                                                MONZA SIMULATOR
=================================================================================================================
- Authors: Daniel Sotelo, Jiajun Xu and Vladyslav Korenyak
- Date: 03/12/2023
- Version: 3.0
- Description: The following program allows the  user to play the game  Monza either manually or using one of the
               the three provided  controllers. There are four levels of difficulty and the game is over when the
               ball falls off the screen. Once the game ends, the user can choose to play again or exit the game.
               It is also possible to plot the graphs with the results of the game, and a record of the number of
               wins, number of losses and best time is saved.

- Requirements: Install pygame.py - https://www.pygame.org/download.shtml (or run "pip install pygame")
                Install skfuzzy.py https://pypi.org/project/scikit-fuzzy/ - (or run "pip install scikit-fuzzy")
                For running the Reinforcement Learning Controller, you also must install PyTorch.
                Instructions are in the following web: https://pytorch.org/get-started/locally/.
=================================================================================================================
'''

# Import libraries
import pygame
from settings import *
from game_objects import Monza
from menus import main_menu, diff_menu, controllers_menu, post_game_menu

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def main():
    
    # Load and set game icon
    icon = pygame.image.load("img/icon.ico")
    pygame.display.set_icon(icon)
    
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Monza Simulator")
    
    # Initialize parameters
    counter_wins = 0
    counter_losses = 0
    best_time = float("inf")
    diff = 0
    
    # Intialize the main menu
    next_screen = "main_menu"
    while True:
        if next_screen == "main_menu":
            next_screen = main_menu(screen)
        elif next_screen == "diff_menu_manual": # Difficulty selection menu for manual mode
            diff = diff_menu(screen)
            if diff == "main_menu":
                next_screen ="main_menu"
            else:
                next_screen = "start_game_manual"
        elif next_screen == "diff_menu_controller": # Difficulty selection menu for controller mode
            diff = diff_menu(screen)
            if diff == "main_menu":
                next_screen ="main_menu"
            else:
                selected_controller = controllers_menu(screen)
                next_screen = "start_game_controller"
            
        if next_screen == "start_game_manual": # Manual mode
            game = Monza(screen, diff, counter_wins, counter_losses, best_time, "manual")
            result = game.run()
            # Update parameters
            if result == "win":
                counter_wins += 1
                if game.coin.t < best_time:
                    best_time = game.coin.t
            elif result == "lose":
                counter_losses += 1
            elif result == "lost_contact":
                counter_losses += 1
            next_screen = post_game_menu(screen, result, game.coin.positions, "manual") # Postgame menu
            
        elif next_screen == "start_game_controller": # Controller mode
            if selected_controller == "diff_menu_controller":
               next_screen = "diff_menu_controller"
            elif selected_controller in [1, 2, 3]: 
                game = Monza(screen, diff, counter_wins, counter_losses, best_time, "controller", selected_controller)
                result = game.run()
                # Update parameters
                if result == "win":
                    #print(game.coin.t)
                    counter_wins += 1
                    if game.coin.t < best_time:
                        best_time = game.coin.t
                elif result == "lose":
                    counter_losses += 1
                elif result == "lost_contact":
                    counter_losses += 1
                next_screen = post_game_menu(screen, result, game.coin.positions, "controller") # Postgame menu
            else:
                print("Controller ", selected_controller, " has not been implemented yet.")
                next_screen = "main_menu"

if __name__ == "__main__":
    
    # Display ASCII art
    with open('assets/ascii_monza_art.txt', 'r') as file:
        print(file.read())
        
    # Run the program
    main() 

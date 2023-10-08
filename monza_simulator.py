'''
=================================================================================================================
                                                MONZA SIMULATOR
=================================================================================================================
- Authors: Daniel Sotelo, Jiajun Xu and Vladyslav Korenyak
- Date: 09/10/2023
- Version: 2.0
- Description: The following program allows the  user to play the game  Monza either manually or using one of the
               the three provided  controllers. There are four levels of difficulty and the game is over when the
               ball falls off the screen. Once the game ends, the user can choose to play again or exit the game.
               It is also possible to plot the graphs with the results of the game, and a record of the number of
               wins, number of losses and best time is saved.

- Requirements: Install pygame.py - https://www.pygame.org/download.shtml (or run "pip install pygame")
=================================================================================================================
'''

# Import libraries
import pygame
from settings import *
from game_objects import Monza
from menus import main_menu, diff_menu, controllers_menu, post_game_menu

def main():
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
            next_screen = "start_game_manual"
        elif next_screen == "diff_menu_controller": # Difficulty selection menu for controller mode
            diff = diff_menu(screen)
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
            if selected_controller == 1:
                game = Monza(screen, diff, counter_wins, counter_losses, best_time, "controller", selected_controller)
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
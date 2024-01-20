"""
Monza Game - Menus
Description: Contains functions and logic related to the game's main menu and post-game menu.
Author(s): Daniel Sotelo, Jiajun Xu, Vladyslav Korenyak
Date: 09/10/2023
Version: 2.0
"""

# Import libraries
import pygame
import sys
from settings import *
from game_objects import Button
from utils import draw_mit_license_and_authors, plot_graphs

# Create main menu
def main_menu(screen):
    font = pygame.font.Font(None, 36)        
    manual_image = pygame.image.load("assets/main_menu/manual.png")
    controller_image = pygame.image.load("assets/main_menu/controller.png")
    buttons = [
        Button(50, 550, 300, 80, "Manual Gameplay", manual_image),
        Button(SCREEN_WIDTH - 390, 550, 340, 80, "Controller Gameplay", controller_image)
    ]
    
    # Create title
    title_image = pygame.image.load("assets/main_menu/title_menu.png")
    title_image = pygame.transform.scale(title_image, (554, 156))
    
    # Create main menu image
    img = pygame.image.load("assets/main_menu/menu.png")
    img_rect = img.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2))
        
    while True:
        screen.fill(BEIGE)    
        draw_mit_license_and_authors(screen, BROWN)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                if buttons[0].is_clicked(pos, event):
                    return "diff_menu_manual"
                if buttons[1].is_clicked(pos, event):
                    return "diff_menu_controller"

        # Draw the title and image
        title_rect = title_image.get_rect(center=(SCREEN_WIDTH // 2, 100))
        screen.blit(title_image, title_rect)
        screen.blit(img, img_rect)
        
        # Draw the buttons
        for button in buttons:
            button.draw(screen, font, BROWN)    
        
        pygame.display.flip()

# Define the difficulty selection menu
def diff_menu(screen):
    font = pygame.font.Font(None, 36)
    buttons = [
        Button(20, 375, 175, 250, "Beginner", None, 1),
        Button(215, 375, 175, 250, "Intermediate", None, 1),
        Button(410, 375, 175, 250, "Expert", None, 1),
        Button(605, 375, 175, 250, "Legendary", None, 1),
        Button(10, 660, 80, 30, "Back", None, 1, 25)
    ]
    
    # Difficulty images
    beginner_image = pygame.image.load("assets/diff_menu/beginner.png")
    intermediate_image = pygame.image.load("assets/diff_menu/intermediate.png")
    expert_image = pygame.image.load("assets/diff_menu/expert.png")
    legendary_image = pygame.image.load("assets/diff_menu/legendary.png")
    
    beginner_image = pygame.transform.scale(beginner_image, (120, 107))
    intermediate_image = pygame.transform.scale(intermediate_image, (120, 107))
    expert_image = pygame.transform.scale(expert_image, (120, 107))
    legendary_image = pygame.transform.scale(legendary_image, (120, 107))
    
    # Create title
    title_image = pygame.image.load("assets/main_menu/title_menu.png")
    title_image = pygame.transform.scale(title_image, (554, 156))
    
    while True:
        screen.fill(BEIGE)    
        draw_mit_license_and_authors(screen, BROWN)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                if buttons[0].is_clicked(pos, event):
                    return 0
                if buttons[1].is_clicked(pos, event):
                    return 1
                if buttons[2].is_clicked(pos, event):
                    return 2
                if buttons[3].is_clicked(pos, event):
                    return 3
                if buttons[4].is_clicked(pos, event):
                    return "main_menu"

        # Draw the title and images
        title_rect = title_image.get_rect(center = (SCREEN_WIDTH // 2, 125))
        screen.blit(title_image, title_rect)

        beginner_rect = beginner_image.get_rect(center = (107.5, 290))
        screen.blit(beginner_image, beginner_rect)
        intermediate_rect = intermediate_image.get_rect(center = (302.5, 290))
        screen.blit(intermediate_image, intermediate_rect)
        expert_rect = expert_image.get_rect(center = (497.5, 290))
        screen.blit(expert_image, expert_rect)
        legendary_rect = legendary_image.get_rect(center = (692.5, 290))
        screen.blit(legendary_image, legendary_rect)
        
        
        # Draw the buttons
        for button in buttons:
            button.draw(screen, font, BROWN)    
        
        pygame.display.flip()
        
# Define the controllers menu
def controllers_menu(screen):
    font = pygame.font.Font(None, 36)
    buttons = [
        Button(65, 375, 180, 250, ["Controller 1", " ", "Sliding Mode", "Controller", "(SMC)"], None, 1),
        Button(310, 375, 180, 250, ["Controller 2", " ", "Fuzzy Logic", " ", " "], None, 1),
        Button(555, 375, 180, 250, ["Controller 3", " ", "Reinforcement", "Learning", " "], None, 1),
        Button(10, 660, 80, 30, "Back", None, 1, 25)
    ]
    
    # Controller images
    controller_image = pygame.image.load("assets/controllers_menu/SMC.png") # Cambiar imagenes cuando tengamos las ideas para cada uno de los controladores
    controller_image = pygame.transform.scale(controller_image, (80, 80))
    
    # Create title
    title_image = pygame.image.load("assets/main_menu/title_menu.png")
    title_image = pygame.transform.scale(title_image, (554, 156))
    
    while True:
        screen.fill(BEIGE)    
        draw_mit_license_and_authors(screen, BROWN)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                if buttons[0].is_clicked(pos, event):
                    return 1
                if buttons[1].is_clicked(pos, event):
                    return 2
                if buttons[2].is_clicked(pos, event):
                    return 3
                if buttons[3].is_clicked(pos, event):
                    return "diff_menu_controller"

        # Draw the title and images
        title_rect = title_image.get_rect(center = (SCREEN_WIDTH // 2, 125))
        screen.blit(title_image, title_rect)

        controller_rect = controller_image.get_rect(center = (155, 290))
        screen.blit(controller_image, controller_rect)
        controller_rect = controller_image.get_rect(center = (400, 290))
        screen.blit(controller_image, controller_rect)
        controller_rect = controller_image.get_rect(center = (645, 290))
        screen.blit(controller_image, controller_rect)
        
        # Draw the buttons
        for button in buttons:
            button.draw(screen, font, BROWN)    
        
        pygame.display.flip()

# Define post-game menu
def post_game_menu(screen, result, positions, mode):    
    
    font = pygame.font.Font(None, 36)
    
    # Load and scale images
    win_image = pygame.image.load("assets/postgame_menu/win_game.png")
    win_image = pygame.transform.scale(win_image, (275, 241))
    try_image = pygame.image.load("assets/postgame_menu/exit_utile_area.png")
    try_image = pygame.transform.scale(try_image, (300, 241))
    
    try_again_button = pygame.image.load("assets/postgame_menu/try_again.png")
    main_menu_button = pygame.image.load("assets/postgame_menu/main_menu.png")
    plot_graphs_button = pygame.image.load("assets/postgame_menu/plot_graphs.png")
    exit_button = pygame.image.load("assets/postgame_menu/exit.png")

    # Define buttons
    buttons = [
        Button(100, 310, 275, 150, "Try Again", try_again_button),
        Button(100, 490, 275, 150, "Main Menu", main_menu_button),
        Button(425, 310, 275, 150, "Plot Graphs", plot_graphs_button),
        Button(425, 490, 275, 150, "Exit", exit_button)
    ]

    while True:
        screen.fill(BEIGE)
        draw_mit_license_and_authors(screen, BROWN)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                if buttons[0].is_clicked(pos, event):
                    if mode == "manual":
                        return "start_game_manual"
                    else:
                        return "start_game_controller"
                if buttons[1].is_clicked(pos, event):
                    return "main_menu"
                if buttons[2].is_clicked(pos, event):
                    plot_graphs(positions)
                    pygame.quit()
                    sys.exit()
                if buttons[3].is_clicked(pos, event):
                    pygame.quit()
                    sys.exit()
        
        # Display a message related to the result of the game (Win/Lose)
        if result == "win":
            message_1 = "Congratulations! You have reached the end! :D"
            screen.blit(win_image, ((SCREEN_WIDTH - win_image.get_width()) // 2, 60))
        elif result == "lose":
            message_1 = "The coin has exited the utile area! :("
            screen.blit(try_image, ((SCREEN_WIDTH - try_image.get_width()) // 2, 60))
        elif result == "lost_contact":
            message_1 = "The coin has lost contact with the rail! :("
            screen.blit(try_image, ((SCREEN_WIDTH - try_image.get_width()) // 2, 60))

        text_1 = font.render(message_1, True, BROWN)
        rect_1 = text_1.get_rect(center = (SCREEN_WIDTH // 2, 50))
        screen.blit(text_1, rect_1)
        
        # Draw the buttons
        for button in buttons:
            button.draw(screen, font, BROWN)
            
        pygame.display.flip()
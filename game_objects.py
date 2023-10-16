"""
Monza Game - Game Objects
Description: Defines game objects like the coin, circle, obstacles, and their behaviors.
Author(s): Daniel Sotelo, Jiajun Xu, Vladyslav Korenyak
Date: 09/10/2023
Version: 2.0
"""

# Import libraries
import pygame
from pygame.locals import *
import math
import numpy as np
from settings import *
from controllers import *
from utils import local_to_global, world_to_screen, draw_mit_license_and_authors

# Class to define a button
class Button:
    def __init__(self, x, y, w, h, texts, image=None, centering=0, text_size=35):
        self.rect = pygame.Rect(x, y, w, h)
        self.texts = texts if isinstance(texts, list) else [texts]
        self.image = image
        self.centering = centering
        if image is not None:
            aspect_ratio = image.get_width() / image.get_height()
            self.image = pygame.transform.scale(image, (int(w/6), int((w/6) / aspect_ratio)))
        self.text_size = text_size
    def draw(self, screen, font, color, border_radius=10):
        pygame.draw.rect(screen, color, self.rect, border_radius=border_radius)
        
        x_offset = 0
        if self.image is not None:
            image_rect = self.image.get_rect()
            image_rect.y = self.rect.y + (self.rect.height - image_rect.height) // 2
            image_rect.x = self.rect.x + 10
            screen.blit(self.image, image_rect.topleft)
            x_offset = image_rect.width + 10

        total_text_height = sum([font.size(txt)[1] for txt in self.texts])
        current_y_offset = (self.rect.height - total_text_height) // 2

        for text in self.texts:
            font = pygame.font.Font(None, self.text_size)
            text_surf = font.render(text, True, BEIGE)
            text_rect = text_surf.get_rect()
            text_rect.y = self.rect.y + current_y_offset
            if self.centering == 1:
                text_rect.x = self.rect.x + (self.rect.width - text_rect.width) // 2
            else:
                text_rect.x = self.rect.x + x_offset + 10
            
            screen.blit(text_surf, text_rect)
            current_y_offset += font.size(text)[1]

    def is_clicked(self, pos, clicked):
        if self.rect.collidepoint(pos) and clicked:
            return True
        return False


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
                    ddx = g*math.cos(beta)*math.sin(theta) - self.dx*mu_v/self.mass - mu_pd*g*math.sin(beta)*math.cos(gamma)
                    self.x1 += self.dx*dt + 0.5*ddx*dt**2
                    self.dx += ddx*dt
                    ddy = self.dy*mu_v/self.mass + mu_pd*g*math.sin(beta)*math.sin(gamma) - g*math.cos(beta)*math.cos(theta)
                    self.y1 += self.dy*dt + 0.5*ddy*dt**2
                    self.dy += ddy*dt
                    self.x, self.y = local_to_global(self.x1, self.y1, theta) # Local to global coordinates
                    
                    if self.n == 1 and self.x1 < -120.6  + self.radius:
                        self.dx = -self.dx * math.e**(-k)
                        ddx = -ddx * math.e**(-k)
                    
                else: # In case of the initial vertical fall
                    self.y1 += self.dy*dt - 0.5*g*dt**2
                    self.dy -= g*dt
                    
        else:
            # New velocities and positions for instant dt
            gamma = math.atan(self.dy/self.dx)
            ddx = g*math.cos(beta)*math.sin(theta) - self.dx*mu_v/self.mass - mu_pd*g*math.sin(beta)*math.cos(gamma)
            self.x1 += self.dx*dt + 0.5*ddx*dt**2
            self.dx += ddx*dt
            ddy = self.dy*mu_v/self.mass + mu_pd*g*math.sin(beta)*math.sin(gamma) - g*math.cos(beta)*math.cos(theta)
            self.y1 += self.dy*dt + 0.5*ddy*dt**2
            self.dy += ddy*dt
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
                        ddx = 2/3 * math.cos(phi) * (g*math.sin(alpha) * math.cos(beta) - self.dx * math.sqrt(1 + (0.00108 * self.x1)**2) * (mu_v/self.mass) - mu_pd * g * math.sin(beta))
                    else:
                        # Acceleration for sliding condition
                        ddx = math.cos(phi) * (g * math.sin(alpha) * math.cos(beta) - self.dx * math.sqrt(1 + (0.00108 * self.x1)**2) * (mu_v / self.mass) - mu_pd * g * math.sin(beta) - N * mu_f / self.mass)
                        
                    # Calculation of positions and velocities
                    self.x1 += self.dx * dt + 0.5 * ddx * (dt**2) #UARM
                    self.y1 = self.eq_data['coeff'][0] * (self.x1 ** 2) + self.eq_data['coeff'][1] + self.radius
                    self.dx += ddx * dt
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
                    if self.n == 0 and self.x1 > 81.9 - self.radius:
                        self.dx = -self.dx * math.e**(-k)
                        ddx = -ddx * math.e**(-k)
                    elif self.n == 1 and self.x1 < -120.6  + self.radius:
                        self.dx = -self.dx * math.e**(-k)
                        ddx = -ddx * math.e**(-k)
                        
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
        self.points_lines_list = [(-120.6, 106.5), (-120.6, 164), (81.9, 156.5), (81.9, 186),
                                  (-25.54, -160.4), (-4, -204), (0, -114.3), (43, -199)]
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
        for line in range(1, len(self.points_lines_list), 2):
                distance_1 = math.sqrt((self.points_lines_list[line-1][0]**2) + (self.points_lines_list[line-1][1]**2))
                angle_1 = math.atan2(self.points_lines_list[line-1][1], self.points_lines_list[line-1][0]) - self.circle.angle
                x_1 = self.circle.x + distance_1 * math.cos(angle_1)
                y_1 = self.circle.y + distance_1 * math.sin(angle_1)
                
                distance_2 = math.sqrt((self.points_lines_list[line][0]**2) + (self.points_lines_list[line][1]**2))
                angle_2 = math.atan2(self.points_lines_list[line][1], self.points_lines_list[line][0]) - self.circle.angle
                x_2 = self.circle.x + distance_2 * math.cos(angle_2)
                y_2 = self.circle.y + distance_2 * math.sin(angle_2)
                
                pygame.draw.line(screen, self.color, world_to_screen(x_1, y_1), world_to_screen(x_2, y_2), w_obstacle)
                
# Class to define the Monza game
class Monza:
    def __init__(self, screen, diff, counter_wins, counter_losses, best_time, mode, selected_controller = 0):
        self.screen = screen
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 26)
        self.mode = mode

        # Create the game objects   
        self.circle = Circle(0, 0) # Labyrinth circle
        self.coin = Coin(diff, x1_0, y1_0, dx1_0) # Coin with coordinates in {x1, y1}
        self.obstacle = Obstacle(self.circle, diff) # Obstacles
        
        # Create title
        self.title_image = pygame.image.load("assets/game/title_game.png")
        self.title_image = pygame.transform.scale(self.title_image, (300, 92))
        
        # Create wins and losses register
        self.register_image = pygame.image.load("assets/game/register.png")
        self.register_image = pygame.transform.scale(self.register_image, (81, 91))
        self.counter_wins = counter_wins
        self.counter_losses = counter_losses
        self.best_time = best_time
        
        # Initialize selected controllers
        if self.mode == "controller":
            self.controller_n = selected_controller
            if self.controller_n == 1:
                self.controller = SlidingModeController(diff)
            elif self.controller_n == 2:
                self.controller = FuzzyLogicController(diff)
    def render_number(self, number, offset = 0):
        # Render the text with the chosen font
        font = pygame.font.Font(None, 20)
        text = font.render(str(number), True, BEIGE)
        
        # Get the rect (position) of the rendered text
        text_rect = text.get_rect()
        
        # You can set the position of the text on the screen
        text_rect.right = SCREEN_WIDTH - 40
        text_rect.y = SCREEN_HEIGHT - 125 + offset
        
        # Blit (draw) the text onto the screen
        self.screen.blit(text, text_rect)
        
    def run(self):
        running = True
        while running and not self.coin.flag_out and not self.coin.flag_fly and not self.coin.flag_end:
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False

            self.screen.fill(BROWN)
            draw_mit_license_and_authors(self.screen, BEIGE)
            
            # Represent the objects on the screen
            self.circle.draw(self.screen)
            self.obstacle.draw(self.screen)
            self.coin.draw(self.screen)

            # Draw the title image
            title_rect = self.title_image.get_rect(center=(SCREEN_WIDTH // 2, 60))
            self.screen.blit(self.title_image, title_rect)
            
            # Draw the register image
            register_position = (SCREEN_WIDTH - 220, SCREEN_HEIGHT - 132)
            self.screen.blit(self.register_image, register_position)
            self.render_number(self.counter_wins)
            self.render_number(self.counter_losses, 61)
            if self.best_time != float("inf"):
                self.render_number(str(round(self.best_time, 2)) + " s", 31)
            else:
                self.render_number("No records", 31)

            if self.mode == "manual":
                # Control the rotation of the circle labyrinth (theta)
                keys = pygame.key.get_pressed()
                if keys[pygame.K_LEFT]:
                    self.circle.rotate(0.005)
                    print(self.circle.angle)
                if keys[pygame.K_RIGHT]:
                    self.circle.rotate(-0.005)
            
            # Si es SlidingModeController
            elif self.controller_n == 1:
                target_position = self.controller.get_target_position(self.coin.n)
                if target_position !=  "Exit":
                    control_action = self.controller.control(self.coin.x1, target_position)
                    self.circle.rotate(control_action)
                else:
                    self.flag_out = True

            # Si es FuzzyLogicController
            elif self.controller_n == 2:
                target_position = self.controller.get_target_position(self.coin.n)
                if target_position !=  "Exit":
                    control_action = self.controller.control(self.coin.x1, self.coin.n, target_position)
                    self.circle.rotate(control_action)
                else:
                    self.flag_out = True
                
            # Update the theta angle for the coin
            theta = self.circle.angle

            # Coin update
            self.coin.update(theta)
            
            # Check if the coin has exited the utile area
            self.coin.check_exit()
            pygame.display.flip()
            fpsClock.tick(FPS)
        
        # Check different flags to display different postgame menus
        if self.coin.flag_out:
            return "lose"
        elif self.coin.flag_end:
            return "win"
        elif self.coin.flag_fly:
            return "lost_contact"
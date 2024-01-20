import torch
import random
import numpy as np
from collections import deque
from game_objectsRL import *
from model2 import Linear_QNet, QTrainer
from helper import plot
import pickle

MAX_MEMORY = 10_000
BATCH_SIZE = 1000
LR = 0.0000005

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.99 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(3, 20, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.state_history = []

    def load_model(self, model_file):
        # Load the saved state dictionary
        state_dict = torch.load(model_file, map_location=torch.device('cpu'))  # 'cpu' or 'cuda' if you are using GPU
        # Load the state dictionary into the model
        self.model.load_state_dict(state_dict)
        # If you are loading the model for inference, set the model to evaluation mode
        #self.model.eval()
        
    def get_state(self, game):

        # Normalized and standardized state
        if game.direction == -1:
            
            current_state = [
                game.circle.angle,  # Already in [-1, 1]
                #game.coin.x / 250,  # Scaled to [-1, 1]
                game.coin.dx / 600,  # Scaled to [-1, 1]
                #(game.distance_to_hole - 125) / 125,  # Shifted and scaled to [-1, 1]
                game.direction  # Assuming this is already normalized
            ]

        else:
            current_state = [
                -game.circle.angle,  # Already in [-1, 1]
                #game.coin.x / 250,  # Scaled to [-1, 1]
                -game.coin.dx / 600,  # Scaled to [-1, 1]
                #(game.distance_to_hole - 125) / 125,  # Shifted and scaled to [-1, 1]
                -game.direction  # Assuming this is already normalized
                #previous_action      
            ]    

        return np.array(current_state, dtype=np.float32)
    

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached
        #print(self.memory)

    def save_training_data(self, filename='training_data.pkl'):
        if not isinstance(filename, str):
            raise TypeError("filename must be a string")

        with open(filename, 'wb') as f:
            pickle.dump(self.memory, f)

    def train_long_memory(self):
        #print("\nlen self memory: ", len(self.memory))
        if len(self.memory) > BATCH_SIZE:
            #print("\nlen self memory: ", len(self.memory))
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)


    def train_long_memory_from_file(self, filename='training_data.pkl', epochs=1000):
        try:
            with open(filename, 'rb') as f:
                sample = pickle.load(f)
        except (FileNotFoundError, IOError):
            print(f"Training file {filename} not found. Training aborted.")
            return

        # Training loop
        for epoch in range(epochs):
            states, actions, rewards, next_states, dones = zip(*sample)
            self.trainer.train_step(states, actions, rewards, next_states, dones)
            print(f"Epoch {epoch + 1}/{epochs} completed.")
            


    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)


    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 0
            
        final_move = [0,0,0]
        # FOR AUTOMATIC CONTROL

        if random.randint(0, 100) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        # FOR MANUAL CONTROL

        # keys = pygame.key.get_pressed()
        # if keys[pygame.K_LEFT]:
        #     final_move = [0,1,0]
        # if keys[pygame.K_RIGHT]:
        #     final_move = [0,0,1]

        return final_move

def trainJustFromMemory():
    agent = Agent() #model_file='model/LastModelManual1000epochs.pth'
    agent.train_long_memory_from_file()     
    agent.model.save() #model_file='model/LastModelManual1000epochs.pth'

def train(filename):
    
    plot_scores = []
    plot_mean_scores = []
    plot_rewards = []
    total_score = 0
    record = 0
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    game = Monza(screen)

    agent = Agent() #model_file="model/AccLastModelManual1000epochsLR0001.pth"
    agent.load_model(filename)
    
    accum_reward = 0

    while True:
        # get old state
        state = agent.get_state(game)

        # get move
        move = agent.get_action(state)
        
        batched_reward = 0
        # perform move and get new state. 10 steps per action --> 10 actions per second
        reward, done = game.play_step(move)
        #for i in range(1):
        batched_reward = batched_reward + reward
            # if not done:
            #     reward, done = game.play_step(move)
        state_new = agent.get_state(game)

        if batched_reward < -10:
            batched_reward = -10

        print("\nreward: ", batched_reward)
        plot_rewards.append(batched_reward)
        accum_reward = accum_reward + batched_reward

        # train short memory
        agent.train_short_memory(state, move, batched_reward, state_new, done)

        # remember
        agent.remember(state, move, batched_reward, state_new, done)

        if done:
            # train long memory
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if accum_reward > record:
                record = accum_reward
                agent.model.save('record_model.pth')

            agent.model.save("LastModel.pth")

            print('Game', agent.n_games, 'Score', accum_reward, 'Record:', record)

            plot_scores.append(accum_reward)
            total_score += accum_reward
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            accum_reward = 0


            plot(plot_scores, plot_mean_scores, plot_rewards)
            plot_rewards = []
            

            # for manual learning

            # if agent.n_games == 20:
            #     #print(type(agent.memory))
            #     agent.save_training_data('training_data.pkl')
            #     print("\n games saved to pkl file, starting training")   
            #     agent.train_long_memory_from_file()     
            #     agent.model.save("AccLastModelManual1000epochsLR0001.pth")
            #     return
                
def evaluate_networks():
    num_networks = 1000
    network_scores = []

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    game = Monza(screen)

    for i in range(num_networks):
        # Initialize a new network (agent)
        agent = Agent()

        # Play the game with the initialized network
        score = play_game(agent, game)
        game.reset()

        # Save the network and its score
        agent.model.save(f"model_montecarlo{i}.pth")
        network_scores.append((f"model_montecarlo{i}.pth", score, game.aux_x_hole, game.aux_x1_0))

    return network_scores

def play_game(agent, game):
    score = 0

    while True:
        # get state
        state = agent.get_state(game)

        # get move
        move = agent.get_action(state)
        
        # perform move and get reward
        reward, done = game.play_step(move)

        score = score + reward

        if done:
            game.reset()
            return score

def evaluate_strategy(): #perfect
    num_tries = 10
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    game = Monza(screen)

    for i in range(num_tries):

        while True:

            if game.direction == 1: # hole to the right
                
                if game.coin.dx <= 0: # should go to the right
                    action = [0, 1, 0]
            
                if game.coin.dx > 0:
                    if game.coin.dx < 100:
                        if game.coin.ddx > 0: # should do nothing
                            action = [1, 0, 0]
                        else: # should go to the right
                            action = [0, 1, 0]

                    else: # above 50
                        if game.coin.ddx > 0: #should try to break
                            action = [0, 0, 1]
                        else: # should do nothing
                            action = [1, 0, 0]

            else: # hole to the left
                if game.coin.dx >= 0: # should go to the left
                    action = [0, 0, 1]
                
                if game.coin.dx < 0:
                    if game.coin.dx > -100:
                        if game.coin.ddx < 0: # should do nothing
                            action = [1, 0, 0]
                        else: # should go to the left
                            action = [0, 0, 1]

                    else: # above 50
                        if game.coin.ddx < 0: #should try to break
                            action = [0, 1, 0]
                        else: # should do nothing
                            action = [1, 0, 0]
        
            reward, done = game.play_step(action)
                
            print("\n playstep: ", game.frame_iteration, "\n reward: ",reward ) 

            if done:
                game.reset()
                break
    


    """
    Loads an agent from a saved model file.
    Args:
    filename (str): The path to the model file.
    """
def evaluate_agent(filename):
    num_evals = 10
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    game = Monza(screen)
    agent = Agent()
    agent.load_model(filename)
    accum_score = 0

    for i in range(num_evals):
        score = play_game(agent, game)
        accum_score = accum_score + score
    
    mean_score = accum_score/num_evals
    return mean_score

if __name__ == '__main__':
    # evaluate_strategy() # Sirve para probar si los rewards son adecuados y si siguiendo la estrategia de máximo reward se gana el juego


    # montecarlo_scores = evaluate_networks() # Se inicializan y se prueban 1000 modelos aleatorios
    # # Now, save the scores to a text file
    # with open('network_scores.txt', 'w') as file:
    #     for model, score, aux_x_hole, aux_x1_0  in montecarlo_scores:
    #         file.write(f'{model}: {score} hole {aux_x_hole} x1 {aux_x1_0} \n')

    # mean_score = evaluate_agent('model\zecord_model.pth') # Se evalúa un modelo en particular
    # print(mean_score)

    train('model\LastModel.pth') # Se entrena un modelo en particular
# monza_simulator

The following is a Python program that simulates the Monza game. This game consists of a coin that needs to pass through a vertical wall labyrinth in such a way that it doesn't fall by the edges. The control is made by rotating the circular labyrinth. The program was written as part of the subject Non Linear Systems at Technical University of Madrid.

The program can be used either as a game to play manually or as a **non-linear controller testing platform**. It counts with four different levels of difficulty (beginner, intermediate, expert and legendary) depending on the length of the parabollic rails. With respect to the controllers, three different ones have been programmed following different non-linear control techniques. The program saves the number of times you won and lost and also the best achieved time. After the game you can access to graphic information about the path the coin followed and the actuation over the system with respect to time.

<p align="center">
  <img src="https://github.com/danisotelo/labyrinth_solver/blob/main/img/labyrinth.png" alt="Image Description">
</p>

## Getting Started
### Cloning the Repository
To clone the repository and start using the **monza_simulator**, follow these steps:

1. Open a terminal or command prompt.
2. Navigate to the directory where you want to clone the repository.
3. Run the following command

```
git clone https://github.com/danisotelo/monza_simulator.git
```
### Installing Dependencies
Before running the program, you need to install the required dependencies. The main dependency for this project is the **pygame** library. To install **pygame**, run:
```
pip install pygame
```

## Running the Program
After cloning the repository and installing the required dependencies, navigate to the directory containin the **monza_simulator** code and run:
```
python monza_simulator.py
```
This will start the Monza game simulator. Enjoy playing!


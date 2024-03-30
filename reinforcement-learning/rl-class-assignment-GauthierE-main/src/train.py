from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import torch
import torch.nn as nn
import gymnasium as gym
import random
import numpy as np
# from tqdm import tqdm # comment this line before pushing
from evaluate import evaluate_HIV, evaluate_HIV_population
from copy import deepcopy

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
) 

state_dim = env.observation_space.shape[0]
n_action = env.action_space.n 


class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        self.device = device
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)
    
config = {'nb_actions': env.action_space.n,
        'learning_rate': 0.001,
        'gamma': 0.99,
        'buffer_size': 1000000,
        'epsilon_min': 0.01,
        'epsilon_max': 1.,
        'epsilon_decay_period': 10000,
        'epsilon_delay_decay': 100,
        'batch_size': 800,
        'gradient_steps': 3,
        'criterion': torch.nn.MSELoss(),
        'update_target': 600,
        'nb_neurons': 256}

class DQNNetwork(nn.Module):
    def __init__(self, state_dim, nb_neurons, n_action):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, nb_neurons)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(nb_neurons, nb_neurons)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(nb_neurons, nb_neurons*2)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(nb_neurons*2, nb_neurons)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(nb_neurons, nb_neurons)
        self.relu5 = nn.ReLU()
        self.fc6 = nn.Linear(nb_neurons, n_action)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.relu4(self.fc4(x))
        x = self.relu5(self.fc5(x))
        x = self.fc6(x)
        return x

class ProjectAgent:

    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gamma = config['gamma']
        self.batch_size = config['batch_size']
        self.nb_actions = config['nb_actions']
        self.memory = ReplayBuffer(config['buffer_size'], device)
        self.epsilon_max = config['epsilon_max']
        self.epsilon_min = config['epsilon_min']
        self.epsilon_stop = config['epsilon_decay_period']
        self.epsilon_delay = config['epsilon_delay_decay']
        self.epsilon_step = (self.epsilon_max - self.epsilon_min) / self.epsilon_stop
        self.model = DQNNetwork(state_dim, config['nb_neurons'], self.nb_actions).to(device)
        self.target_model = DQNNetwork(state_dim, config['nb_neurons'], self.nb_actions).to(device).eval()
        self.criterion = config['criterion']
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.optimizer2 = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.nb_gradient_steps = config['gradient_steps']
        self.update_target = config['update_target']
    
    def act(self, observation, use_random=False):
        device = "cuda" if next(self.model.parameters()).is_cuda else "cpu"
        with torch.no_grad():
            Q = self.model(torch.Tensor(observation).unsqueeze(0).to(device))
            return torch.argmax(Q).item()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self):
        device = torch.device('cpu')
        self.model = DQNNetwork(state_dim, config['nb_neurons'], self.nb_actions).to(device)
        self.model.load_state_dict(torch.load("src/model_save_dqn4.pt", map_location=device))
        self.model.eval()
    
    def greedy_action(self, network, state):
        device = "cuda" if next(network.parameters()).is_cuda else "cpu"
        with torch.no_grad():
            Q = network(torch.Tensor(state).unsqueeze(0).to(device))
            return torch.argmax(Q).item()

    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.target_model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 
    
    def train(self):

        device = "cuda" if torch.cuda.is_available() else "cpu"
        score = 0
        max_episode = 200
        test_episode = 100
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0

        while episode < max_episode:
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)
            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = self.greedy_action(self.model, state)
            # step
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward
            # train
            for _ in range(self.nb_gradient_steps): 
                self.gradient_step()
            # replace target    
            if step % self.update_target == 0: 
                self.target_model.load_state_dict(self.model.state_dict())

            step += 1
            if done or trunc:
                episode += 1
                if episode > test_episode:
                    test_score = evaluate_HIV(agent=self, nb_episode=1)
                else :
                    test_score = 0
                print("Episode ", '{:3d}'.format(episode), 
                      ", epsilon ", '{:6.2f}'.format(epsilon), 
                      ", batch size ", '{:5d}'.format(len(self.memory)), 
                      ", episode return ", '{:.2e}'.format(episode_cum_reward),
                      ", test score ", '{:.2e}'.format(test_score),
                      sep='')
                state, _ = env.reset()
                # save the best model on the test set
                if test_score > score:
                    score = test_score
                    self.best_model = deepcopy(self.model).to(device)
                    self.save("model_save_dqn4.pt")
                episode_return.append(episode_cum_reward)
                episode_cum_reward = 0
            else:
                state = next_state

        self.model.load_state_dict(self.best_model.state_dict())
        self.save("model_save_dqn4.pt")
        return episode_return


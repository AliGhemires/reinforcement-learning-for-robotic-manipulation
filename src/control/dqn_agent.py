import numpy as np
from collections import deque
from typing import Any, Dict, Tuple, List, Optional
import random

class DQNAgent:
    """
    A Deep Q-Learning Agent for reinforcement learning tasks.

    This agent interacts with the environment, learns from experiences, and optimizes its policy
    using the Q-learning algorithm by updating action-value functions.
    """

    def __init__(self,
                 state_size: int,
                 action_size: int,
                 lr: float = 0.001,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 batch_size: int = 64,
                 memory_size: int = 2000) -> None:
        """
        Initialize the DQN agent with required parameters.

        Args:
            state_size (int): The size of the state space.
            action_size (int): The number of possible actions.
            lr (float): The learning rate for the Q-learning updates.
            gamma (float): The discount factor for future rewards.
            epsilon (float): The exploration factor.
            epsilon_decay (float): The decay rate for exploration.
            epsilon_min (float): The minimum value for epsilon.
            batch_size (int): The batch size for training.
            memory_size (int): The maximum size of the experience replay memory.
        """
        if not (0 < lr <= 1):
            raise ValueError("Learning rate must be between 0 and 1.")
        if not (0 <= gamma <= 1):
            raise ValueError("Discount factor must be between 0 and 1.")
        if not (0 < epsilon <= 1):
            raise ValueError("Initial epsilon must be between 0 and 1.")

        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size

        self.memory = deque(maxlen=memory_size)
        self.q_table = np.zeros((state_size, action_size), dtype=np.float32)

    def remember(self, state: Tuple[float, ...], action: int, reward: float, next_state: Tuple[float, ...], done: bool) -> None:
        """
        Store experience in memory.

        Args:
            state (Tuple[float, ...]): The current state before the action.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (Tuple[float, ...]): The next state after action is taken.
            done (bool): Whether this is the end of the episode.
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state: Tuple[float, ...]) -> int:
        """
        Choose the next action based on the current state.

        Args:
            state (Tuple[float, ...]): The current state.

        Returns:
            int: The selected action to be taken.
        """
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)  # Explore
        q_values = self.q_table[state]
        return np.argmax(q_values)  # Exploit

    def replay(self) -> None:
        """
        Train the agent based on the stored experiences in memory.
        Sample a batch from memory and update the Q-values.
        """
        if len(self.memory) < self.batch_size:
            return  # Wait for enough experiences

        mini_batch_indices = np.random.choice(len(self.memory), self.batch_size)
        for index in mini_batch_indices:
            state, action, reward, next_state, done = self.memory[index]
            target = reward
            if not done:
                target += self.gamma * np.max(self.q_table[next_state])
            q_value_update = self.lr * (target - self.q_table[state][action])
            self.q_table[state][action] += q_value_update

        # Decay epsilon after each episode
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)

    def save(self, name: str) -> None:
        """
        Save the model to disk.

        Args:
            name (str): The filename to save the model.
        """
        # Serialization logic can be added here
        pass

    def load(self, name: str) -> None:
        """
        Load the model from disk.

        Args:
            name (str): The filename from which to load the model.
        """
        # Deserialization logic can be added here
        pass

    # Additional utility methods and considerations could be added here,
    # such as target networks, Dueling DQN architectures, Prioritized Experience Replay, etc.

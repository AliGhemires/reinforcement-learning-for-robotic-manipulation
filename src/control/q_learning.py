import numpy as np
import random
from collections import deque
from typing import Tuple, List, Any, Dict

class QLearning:
    """
    Implementation of Q-Learning algorithm for reinforcement learning tasks.
    This class manages the Q-table, performs action selection through
    epsilon-greedy strategy, and updates the Q-values.
    """

    def __init__(self,
                 state_size: int,
                 action_size: int,
                 learning_rate: float = 0.001,
                 discount_factor: float = 0.99,
                 exploration_rate: float = 1.0,
                 exploration_decay: float = 0.995,
                 min_exploration_rate: float = 0.1,
                 memory_size: int = 2000):
        """
        Initializes the QLearning agent with the specified parameters.
        :param state_size: The dimensionality of the state space, must be positive.
        :param action_size: The number of possible actions, must be positive.
        :param learning_rate: The rate at which the model learns.
        :param discount_factor: The discount factor for future rewards.
        :param exploration_rate: The initial exploration rate for epsilon-greedy.
        :param exploration_decay: The decay rate for exploration over time.
        :param min_exploration_rate: The minimum exploration rate.
        :param memory_size: The size of the replay memory.
        """
        assert state_size > 0, "State size must be a positive integer."
        assert action_size > 0, "Action size must be a positive integer."
        
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate

        # Initialize Q-table with zeros and validate shape
        self.q_table = np.zeros((state_size, action_size), dtype=np.float32)
        self.memory = deque(maxlen=memory_size)  # Experience replay memory

    def select_action(self, state: int) -> int:
        """
        Select an action based on the current state using epsilon-greedy strategy.
        :param state: The current state of the environment.
        :return: The selected action.
        """
        assert 0 <= state < self.state_size, "State index out of range."
        if np.random.rand() < self.exploration_rate:
            return random.randint(0, self.action_size - 1)  # Explore a random action
        else:
            return int(np.argmax(self.q_table[state]))  # Exploit the best known action

    def update_q_table(self, state: int, action: int,
                       reward: float, next_state: int, done: bool) -> None:
        """
        Update the Q-value for a given state-action pair using the Bellman Equation.
        :param state: The current state.
        :param action: The action taken.
        :param reward: The reward received from the action.
        :param next_state: The state reached after taking the action.
        :param done: Status indicating whether the episode has ended.
        """
        assert 0 <= state < self.state_size, "State index out of range."
        assert 0 <= action < self.action_size, "Action index out of range."
        assert 0 <= next_state < self.state_size, "Next state index out of range."

        # Calculate the target Q-value using max over possible next actions
        target = reward
        if not done:
            target += self.discount_factor * np.max(self.q_table[next_state])

        # Update the Q-value using a simple gradient descent step
        old_value = self.q_table[state, action]
        self.q_table[state, action] += self.learning_rate * (target - old_value)

    def store_experience(self, experience: Tuple[int, int, float, int, bool]) -> None:
        """
        Store experience in memory for later replay.
        :param experience: A tuple containing (state, action, reward, next_state, done).
        """
        self.memory.append(experience)

    def replay(self, batch_size: int) -> None:
        """
        Sample a batch of experiences from memory and update Q-values for them.
        :param batch_size: Number of experiences to sample for replay.
        """
        if len(self.memory) < batch_size:
            return  # Not enough experiences to sample

        # Sample a random batch from memory
        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in batch:
            self.update_q_table(state, action, reward, next_state, done)

    def decay_exploration(self) -> None:
        """
        Decrease the exploration rate with each episode until it reaches the minimum.
        """
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)

    def get_q_values(self, state: int) -> np.ndarray:
        """
        Retrieve the Q-values for all actions in a given state.
        :param state: The current state.
        :return: Q-values for the current state.
        """
        assert 0 <= state < self.state_size, "State index out of range."
        return self.q_table[state].copy()  # Return a copy to prevent external modifications

from typing import List, Tuple, Any
import numpy as np
import random
import tensorflow as tf
from collections import deque

class DQNAgent:
    """
    Deep Q-Network (DQN) agent for reinforcement learning.
    This agent utilizes a deep learning model to estimate Q-values
    and selects actions based on an epsilon-greedy policy.
    """

    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.001,
                 gamma: float = 0.99, epsilon: float = 1.0, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01, batch_size: int = 64, memory_size: int = 2000):
        """
        Initialize the DQNAgent.

        Parameters:
        - state_size: The dimension of the state space.
        - action_size: The number of possible actions.
        - learning_rate: Learning rate for the optimizer.
        - gamma: Discount factor for future rewards.
        - epsilon: Initial exploration rate.
        - epsilon_decay: Rate at which exploration decreases.
        - epsilon_min: Minimum exploration rate.
        - batch_size: Number of experiences to sample from memory.
        - memory_size: Size of the replay memory.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)  # Replay memory
        self.model = self._build_model()  # Neural network for Q-value approximation
        
    def _build_model(self) -> tf.keras.Model:
        """
        Build a neural network model for estimating Q-values.
        The model takes states as input and outputs Q-values for each action.
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'),
            tf.keras.layers.Dense(24, activation='relu', kernel_initializer='he_uniform'),
            tf.keras.layers.Dense(self.action_size, activation='linear', kernel_initializer='he_uniform')  # Q-values
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        """
        Store experiences in replay memory.
        
        Parameters:
        - state: Current state.
        - action: Action taken.
        - reward: Reward received.
        - next_state: Next state after action.
        - done: Boolean indicating if the episode was finished after this transition.
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state: np.ndarray) -> int:
        """
        Choose an action using an epsilon-greedy policy.

        Parameters:
        - state: Current state of the environment.

        Returns:
        - Selected action based on policy.
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Explore
        state = np.expand_dims(state, axis=0)
        q_values = self.model.predict(state, verbose=0)  # Exploit
        return np.argmax(q_values[0])  # Return action with max Q-value

    def replay(self) -> None:
        """
        Train the model using a batch of experiences from memory.
        """
        if len(self.memory) < self.batch_size:
            return  # Not enough samples to replay
        minibatch = random.sample(self.memory, self.batch_size)
        states, targets_f = [], []

        for state, action, reward, next_state, done in minibatch:
            target = reward
            next_state = np.expand_dims(next_state, axis=0)
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(np.expand_dims(state, axis=0), verbose=0)
            target_f[0][action] = target
            states.append(state)
            targets_f.append(target_f[0])

        self.model.fit(np.array(states), np.array(targets_f), epochs=1, verbose=0, batch_size=self.batch_size)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay  # Decay epsilon cautiously to allow for exploration

    def load(self, name: str) -> None:
        """
        Load model weights from a file.

        Parameters:
        - name: Path to the file containing the weights.
        """
        self.model.load_weights(name)

    def save(self, name: str) -> None:
        """
        Save model weights to a file.

        Parameters:
        - name: Path to the file where weights will be saved.
        """
        self.model.save_weights(name)

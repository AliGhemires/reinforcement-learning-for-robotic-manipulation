import random
from collections import deque
from typing import Any, Tuple, List, Deque

class ExperienceReplay:
    """
    A class to implement Experience Replay for reinforcement learning.

    This class stores experiences (state, action, reward, next_state) in a 
    buffer and allows for sampling minibatches of experiences. Using 
    experience replay helps to break the temporal correlation between 
    consecutive experiences, improving learning stability and sample efficiency.
    """
    
    def __init__(self, capacity: int) -> None:
        """
        Initializes the Experience Replay buffer.

        Args:
            capacity (int): The maximum number of experiences to store.

        Raises:
            ValueError: If `capacity` is not a positive integer.
        """
        if not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("Capacity must be a positive integer.")
        self.capacity: int = capacity
        self.memory: Deque[Tuple[Any, Any, float, Any]] = deque(maxlen=capacity)

    def push(self, experience: Tuple[Any, Any, float, Any]) -> None:
        """
        Stores an experience in the replay buffer.

        Args:
            experience (Tuple[Any, Any, float, Any]): A tuple containing
            (state, action, reward, next_state).

        Raises:
            ValueError: If `experience` is not a tuple of length 4.
        """
        if not isinstance(experience, tuple) or len(experience) != 4:
            raise ValueError("Experience must be a tuple of format: (state, action, reward, next_state).")
        self.memory.append(experience)

    def sample(self, batch_size: int) -> List[Tuple[Any, Any, float, Any]]:
        """
        Samples a minibatch of experiences from the replay buffer.

        Args:
            batch_size (int): The number of experiences to sample.

        Returns:
            List[Tuple[Any, Any, float, Any]]: A list of sampled experiences.

        Raises:
            ValueError: If `batch_size` is not a positive integer or exceeds the number
            of stored experiences.
        """
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("Batch size must be a positive integer.")
        if batch_size > len(self.memory):
            raise ValueError("Batch size cannot exceed the number of stored experiences ({}).".format(len(self.memory)))
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        """
        Returns the current number of experiences stored in the buffer.

        Returns:
            int: The number of experiences stored.
        """
        return len(self.memory)

# Example of instantiation and usage:
# experience_replay = ExperienceReplay(capacity=10000)
# experience_replay.push((state, action, reward, next_state))
# batch = experience_replay.sample(batch_size=32)  

# Note: The experience should be tuples of length 4 (state, action, reward, next_state).  
# Proper handling of each component type and appropriate measures to avoid running out of memory by respecting the defined capacity are implemented.

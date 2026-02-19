import random
from collections import deque
from typing import Any, Tuple, List, Optional

class ExperienceReplay:
    """
    A class that implements experience replay using a circular buffer.
    Experience replay is a crucial component in reinforcement learning,
    where it helps to optimize learning by reusing past experiences.
    """

    def __init__(self, capacity: int) -> None:
        """
        Initializes the ExperienceReplay with a given capacity.
        The capacity determines how many experiences are retained.

        Args:
            capacity (int): Maximum number of experiences to store.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be greater than zero.")
        self.capacity: int = capacity
        self.buffer: deque = deque(maxlen=capacity)  # Circular buffer

    def push(self, experience: Tuple[Any, Any, float, Any, bool]) -> None:
        """
        Push a new experience to the buffer. If the buffer is full,
        the oldest experience will be overwritten.

        Args:
            experience (Tuple[Any, Any, float, Any, bool]): A tuple containing the experience
            in the form (state, action, reward, next_state, done).
        """
        if not isinstance(experience, tuple) or len(experience) != 5:
            raise ValueError("Experience must be a tuple of form (state, action, reward, next_state, done) and of length 5.")
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> List[Tuple[Any, Any, float, Any, bool]]:
        """
        Sample a batch of experiences from the buffer.
        If the buffer has fewer experiences than requested batch size,
        the full buffer will be returned.

        Args:
            batch_size (int): The size of the batch to sample.

        Returns:
            List[Tuple[Any, Any, float, Any, bool]]: A list of sampled experiences.
        """
        if batch_size <= 0:
            raise ValueError("Batch size must be greater than zero.")
        return random.sample(self.buffer, min(len(self.buffer), batch_size))

    def __len__(self) -> int:
        """
        Returns the current size of the experience replay buffer.
        """
        return len(self.buffer)

    def clear(self) -> None:
        """
        Clears all the experiences stored in the buffer.
        """
        self.buffer.clear()

    def is_full(self) -> bool:
        """
        Checks if the experience replay buffer has reached its full capacity.

        Returns:
            bool: True if buffer is full, otherwise False.
        """
        return len(self.buffer) == self.capacity

# Example usage:
# replay_buffer = ExperienceReplay(capacity=10000)
# replay_buffer.push((state, action, reward, next_state, done))
# batch = replay_buffer.sample(batch_size=32)  # Retrieve a batch of experiences

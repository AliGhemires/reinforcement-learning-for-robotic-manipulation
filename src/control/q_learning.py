class QLearning:
    def __init__(self, state_size, action_size, learning_rate=0.01, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.q_table = {}  # Initialize Q-table

    def choose_action(self, state):
        # your logic for choosing action
        pass
    
    def update_q_value(self, state, action, reward, next_state):
        # your logic for updating Q value
        pass

    # Additional methods as required by your implementation

    # Ensure method signatures match those expected in tests

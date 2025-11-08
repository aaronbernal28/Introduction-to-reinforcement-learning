import numpy as np
from typing import Tuple, List, Optional

# Actions
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class GridWorld:
    """
    A grid world environment with stochastic transitions.
    
    Grid layout (3x4):
    [0]  [1]  [2]  [3]
    [4]  [X]  [5]  [6]
    [7]  [8]  [9]  [10]
    
    Where:
    - [X] is a wall (state 5 doesn't exist in the numbering)
    - State 3 is terminal with reward +1
    - State 6 is terminal with reward -1
    - All other states have reward -0.04
    """
    
    def __init__(self, gamma: float = 0.9):
        """
        Initialize the GridWorld environment.
        
        Args:
            gamma: Discount factor
        """
        self.gamma = gamma
        self.n_states = 11  # Total states (excluding the wall)
        self.n_actions = 4
        
        # Grid dimensions
        self.rows = 3
        self.cols = 4
        
        # Terminal states
        self.terminal_states = [3, 6]
        
        # State rewards
        self.rewards = np.full(self.n_states, -0.04)
        self.rewards[3] = 1.0   # Positive terminal
        self.rewards[6] = -1.0  # Negative terminal
        
        # Transition probabilities (intended, slip)
        self.intended_prob = 0.94
        self.slip_prob = 0.02
        
        # Map grid positions to state indices
        self.grid_to_state = {
            (0, 0): 0, (0, 1): 1, (0, 2): 2, (0, 3): 3,
            (1, 0): 4,            (1, 2): 5, (1, 3): 6,
            (2, 0): 7, (2, 1): 8, (2, 2): 9, (2, 3): 10
        }
        
        self.state_to_grid = {v: k for k, v in self.grid_to_state.items()}
        
        # Current state is initialized randomly
        self.non_terminal = [s for s in range(self.n_states) if s not in self.terminal_states]
        self.current_state = np.random.choice(self.non_terminal)

    def is_terminal(self, state: int) -> bool:
        """Check if a state is terminal."""
        return state in self.terminal_states
    
    def get_next_state(self, state: int, action: int) -> int:
        """
        Get the next state given current state and action (deterministic part).
        
        Args:
            state: Current state
            action: Action to take
            
        Returns:
            Next state index
        """
        if self.is_terminal(state):
            return state
        
        row, col = self.state_to_grid[state]
        
        # Calculate intended next position
        if action == UP:
            next_row, next_col = row - 1, col
        elif action == RIGHT:
            next_row, next_col = row, col + 1
        elif action == DOWN:
            next_row, next_col = row + 1, col
        elif action == LEFT:
            next_row, next_col = row, col - 1
        else:
            raise ValueError("Invalid action")
        
        # Return to current state. If next position is invalid (wall or out of bounds), stay in the same state
        return self.grid_to_state.get((next_row, next_col), state)
    
    def get_transition_prob(self, state: int, action: int, next_state: int) -> float:
        """
        Get transition probability P(s'|s,a).
        
        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            
        Returns:
            Probability of transitioning to next_state
        """
        if self.is_terminal(state):
            # If in terminal state, stay there
            return 1.0 if next_state == state else 0.0
        
        prob = 0.0
        
        # Check each possible outcome
        for a in range(self.n_actions):
            resulting_state = self.get_next_state(state, a)
            
            if resulting_state == next_state:
                if a == action:
                    prob += self.intended_prob
                else:
                    # Slip actions 0.02 each
                    prob += self.slip_prob
        
        return prob
    
    def build_transition_matrix(self, policy: np.ndarray) -> np.ndarray:
        """
        Build the transition probability matrix P for a given policy.
        
        Args:
            policy: Policy matrix (n_states, n_actions) with action probabilities
            
        Returns:
            Transition matrix P (n_states, n_states)
        """
        P = np.zeros((self.n_states, self.n_states))
        
        for s in range(self.n_states):
            for a in range(self.n_actions):
                for s_next in range(self.n_states):
                    P[s, s_next] += policy[s, a] * self.get_transition_prob(s, a, s_next)
        
        return P
    
    def step(self, action: int) -> Tuple[int, float, bool]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (next_state, reward, done)
        """
        # Determine actual action (with slipping)
        rand = np.random.random()
        if rand < self.intended_prob:
            actual_action = action
        else:
            # Slip to one of the other three directions
            other_actions = [a for a in range(self.n_actions) if a != action]
            actual_action = np.random.choice(other_actions)
        
        # Get next state
        next_state = self.get_next_state(self.current_state, actual_action)
        reward = self.rewards[self.current_state]
        done = self.is_terminal(next_state)
        
        self.current_state = next_state

        #if done:
        #    # Restart the environment if terminal state is reached
        #    self.reset()
        
        return next_state, reward, done
    
    def reset(self, state: Optional[int] = None) -> int:
        """
        Reset the environment.
        
        Args:
            state: Optional initial state (random if None)
            
        Returns:
            Initial state
        """
        if state is None:
            # Choose random non-terminal state
            self.current_state = np.random.choice(self.non_terminal)
        else:
            self.current_state = state
        
        return self.current_state
    
    def visualize_policy(self, policy: np.ndarray) -> str:
        """
        Visualize a policy as a string with arrows.
        
        Args:
            policy: Policy array (n_states,) with action indices
            
        Returns:
            String representation of the policy
        """
        arrows = ['↑', '→', '↓', '←']
        grid = [[' ' for _ in range(self.cols)] for _ in range(self.rows)]
        
        for state, (row, col) in self.state_to_grid.items():
            if self.is_terminal(state):
                grid[row][col] = 'T' if self.rewards[state] > 0 else 'X'
            else:
                grid[row][col] = arrows[int(policy[state])]
        
        # Mark wall
        grid[1][1] = '█'
        
        result = []
        for row in grid:
            result.append(' | '.join(row))
        
        return '\n'.join(result)
    
    def visualize_values(self, values: np.ndarray) -> str:
        """
        Visualize state values as a grid.
        
        Args:
            values: Value array (n_states,)
            
        Returns:
            String representation of values
        """
        grid = [['      ' for _ in range(self.cols)] for _ in range(self.rows)]
        
        for state, (row, col) in self.state_to_grid.items():
            grid[row][col] = f'{values[state]:6.3f}'
        
        # Mark wall
        grid[1][1] = '  ██  '
        
        result = []
        for row in grid:
            result.append(' | '.join(row))
        
        return '\n'.join(result)

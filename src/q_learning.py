import numpy as np
from typing import Tuple, List, Optional


class QLearning:
    """Q-Learning algorithm for finding optimal policy."""
    
    def __init__(self, env):
        """
        Initialize Q-Learning agent.
        
        Args:
            env: GridWorld environment
        """
        self.env = env
        self.n_states = env.n_states
        self.n_actions = env.n_actions
        self.gamma = env.gamma
    
    def epsilon_greedy_policy(self, Q: np.ndarray, state: int, epsilon: float) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            Q: Action-value function
            state: Current state
            epsilon: Exploration rate
            
        Returns:
            Selected action
        """
        if np.random.random() < epsilon:
            # Explore: random action
            return np.random.randint(self.n_actions)
        else:
            # Exploit: greedy action
            return np.argmax(Q[state])
    
    def uniform_random_policy(self, state: int) -> int:
        """
        Select action uniformly at random.
        
        Args:
            state: Current state
            
        Returns:
            Random action
        """
        return np.random.randint(self.n_actions)
    
    def train(self,
              n_episodes: int = 1000,
              max_steps: int = 100,
              epsilon: float = 0.1,
              epsilon_decay: float = 1.0,
              epsilon_min: float = 0.01,
              alpha: Optional[float] = None,
              behavior_policy: str = 'epsilon_greedy',
              Q_init: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """
        Train using Q-Learning algorithm.
        
        Args:
            n_episodes: Number of episodes
            max_steps: Maximum steps per episode
            epsilon: Initial exploration rate (for epsilon-greedy)
            epsilon_decay: Epsilon decay rate per episode
            epsilon_min: Minimum epsilon value
            alpha: Learning rate (1/n_episodes if None)
            behavior_policy: 'epsilon_greedy' or 'uniform' for exploration
            Q_init: Initial Q-values
            
        Returns:
            Tuple of (Q-values, policy, rewards per episode)
        """
        # Initialize Q-values
        if Q_init is None:
            Q = np.zeros((self.n_states, self.n_actions))
        else:
            Q = Q_init.copy()
        
        episode_rewards = []
        current_epsilon = epsilon
        
        for episode in range(n_episodes):
            # Set learning rate
            if alpha is None:
                learning_rate = 1.0 / (episode + 1)
            else:
                learning_rate = alpha
            
            # Reset environment
            state = self.env.reset()
            
            episode_reward = 0
            
            for step in range(max_steps):
                # Choose action according to behavior policy
                if behavior_policy == 'uniform':
                    action = self.uniform_random_policy(state)
                else:  # epsilon_greedy
                    action = self.epsilon_greedy_policy(Q, state, current_epsilon)
                
                # Take action
                next_state, reward, done = self.env.step(action)
                episode_reward += reward
                
                # Q-Learning update: Q(S,A) ← Q(S,A) + α[R + γ max_a Q(S',a) - Q(S,A)]
                if done:
                    td_target = reward
                else:
                    td_target = reward + self.gamma * np.max(Q[next_state])
                
                td_error = td_target - Q[state, action]
                Q[state, action] = Q[state, action] + learning_rate * td_error
                
                if done:
                    break
                
                state = next_state
            
            episode_rewards.append(episode_reward)
            
            # Decay epsilon (only for epsilon-greedy)
            if behavior_policy == 'epsilon_greedy':
                current_epsilon = max(epsilon_min, current_epsilon * epsilon_decay)
        
        # Extract greedy policy
        policy = np.argmax(Q, axis=1)
        
        return Q, policy, episode_rewards
    
    def get_value_function(self, Q: np.ndarray) -> np.ndarray:
        """
        Extract value function from Q-values.
        
        Args:
            Q: Action-value function
            
        Returns:
            State-value function
        """
        return np.max(Q, axis=1)

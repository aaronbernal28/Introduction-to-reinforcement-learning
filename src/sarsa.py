import numpy as np
from typing import Tuple, List, Optional


class SARSA:
    """SARSA algorithm for finding optimal policy."""
    
    def __init__(self, env):
        """
        Initialize SARSA learner.
        
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
    
    def train(self,
              n_episodes: int = 1000,
              max_steps: int = 100,
              epsilon: float = 0.1,
              epsilon_decay: float = 1.0,
              epsilon_min: float = 0.01,
              alpha: Optional[float] = None,
              Q_init: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """
        Train using SARSA algorithm.
        
        Args:
            n_episodes: Number of episodes
            max_steps: Maximum steps per episode
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay rate per episode
            epsilon_min: Minimum epsilon value
            alpha: Learning rate (1/n_episodes if None)
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
            
            # Choose initial action
            action = self.epsilon_greedy_policy(Q, state, current_epsilon)
            
            episode_reward = 0
            
            for step in range(max_steps):
                # Take action
                next_state, reward, done = self.env.step(action)
                episode_reward += reward
                
                # Choose next action
                next_action = self.epsilon_greedy_policy(Q, next_state, current_epsilon)
                
                # SARSA update: Q(S,A) ← Q(S,A) + α[R + γQ(S',A') - Q(S,A)]
                if done:
                    td_target = reward
                else:
                    td_target = reward + self.gamma * Q[next_state, next_action]
                
                td_error = td_target - Q[state, action]
                Q[state, action] = Q[state, action] + learning_rate * td_error
                
                if done:
                    break
                
                state = next_state
                action = next_action
            
            episode_rewards.append(episode_reward)
            
            # Decay epsilon
            current_epsilon = max(epsilon_min, current_epsilon * epsilon_decay)
        
        # Extract policy
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

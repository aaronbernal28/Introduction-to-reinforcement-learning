import numpy as np
from typing import Tuple, List, Optional


class TDLearning:
    """Temporal Difference Learning algorithms."""
    
    def __init__(self, env):
        """
        Initialize TD learner.
        
        Args:
            env: GridWorld environment
        """
        self.env = env
        self.n_states = env.n_states
        self.n_actions = env.n_actions
        self.gamma = env.gamma
    
    def td0_policy_evaluation(self,
                             policy: np.ndarray,
                             n_episodes: int = 1000,
                             max_steps: int = 100,
                             alpha: Optional[float] = None,
                             V_init: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[float]]:
        """
        Evaluate a policy using TD(0).
        
        Args:
            policy: Policy matrix (n_states, n_actions) or policy array (n_states,)
            n_episodes: Number of episodes
            max_steps: Maximum steps per episode
            alpha: Learning rate (1/n_episodes if None)
            V_init: Initial value function
            
        Returns:
            Tuple of (value function, list of errors per episode)
        """
        # Initialize value function
        if V_init is None:
            V = np.zeros(self.n_states)
        else:
            V = V_init.copy()
        
        # Check if policy is deterministic or stochastic
        if policy.ndim == 1:
            # Deterministic policy
            policy_type = 'deterministic'
        else:
            # Stochastic policy
            policy_type = 'stochastic'
        
        errors = []
        
        for episode in range(n_episodes):
            # Set learning rate
            if alpha is None:
                learning_rate = 1.0 / (episode + 1)
            else:
                learning_rate = alpha
            
            # Reset environment
            state = self.env.reset()
            episode_error = 0
            
            for step in range(max_steps):
                # Select action according to policy
                if policy_type == 'deterministic':
                    action = policy[state]
                else:
                    action = np.random.choice(self.n_actions, p=policy[state])
                
                # Take action
                next_state, reward, done = self.env.step(action)
                
                # TD(0) update: V(S) ← V(S) + α[R + γV(S') - V(S)]
                td_target = reward + self.gamma * V[next_state]
                td_error = td_target - V[state]
                V[state] = V[state] + learning_rate * td_error
                
                episode_error += abs(td_error)
                
                if done:
                    break
                
                state = next_state
            
            errors.append(episode_error)
        
        return V, errors

import numpy as np
from typing import Tuple, Optional


class MDPSolver:
    """Solver for Markov Decision Processes using Dynamic Programming."""
    
    def __init__(self, env):
        """
        Initialize MDP solver.
        
        Args:
            env: GridWorld environment
        """
        self.env = env
        self.n_states = env.n_states
        self.n_actions = env.n_actions
        self.gamma = env.gamma
    
    def direct_policy_evaluation(self, policy: np.ndarray) -> np.ndarray:
        """
        Compute value function by solving V = (I - γP)^(-1)R directly.
        
        Args:
            policy: Policy matrix (n_states, n_actions) with action probabilities
            
        Returns:
            Value function V(s)
        """
        # Build transition matrix P under policy
        P = self.env.build_transition_matrix(policy)
        
        # Get reward vector
        R = self.env.rewards.copy()
        
        # For terminal states, set their row in P to zeros (absorbing)
        for terminal in self.env.terminal_states:
            P[terminal, :] = 0.0
            P[terminal, terminal] = 1.0
        
        # Solve V
        I = np.eye(self.n_states)
        V = np.linalg.solve(I - self.gamma * P, R)
        
        return V
    
    def iterative_policy_evaluation(self, 
                                   policy: np.ndarray,
                                   theta: float = 1e-4,
                                   max_iterations: int = 1000,
                                   V_init: Optional[np.ndarray] = None) -> Tuple[np.ndarray, int]:
        """
        Evaluate a policy using iterative policy evaluation.
        
        Implements: V_{k+1}(s) = ∑_a π(a|s) [R(s,a) + γ ∑_{s'} P(s'|s,a) V_k(s')]
        
        Args:
            policy: Policy matrix (n_states, n_actions) with action probabilities
            theta: Convergence threshold
            max_iterations: Maximum number of iterations
            V_init: Initial value function (random if None)
            
        Returns:
            Tuple of (value function, number of iterations)
        """
        # Initialize value function
        if V_init is None:
            V = np.zeros(self.n_states)
        else:
            V = V_init.copy()
        
        iteration = 0
        
        for iteration in range(max_iterations):
            delta = 0
            V_new = np.zeros(self.n_states)
            
            for s in range(self.n_states):
                v_new = 0
                for a in range(self.n_actions):
                    # R(s,a) - reward for state-action pair
                    R_s_a = self.env.rewards[s]
                    
                    expected_value = 0
                    for s_prime in range(self.n_states):
                        if self.env.is_terminal(s):
                            # Terminal state: absorbing (stays in same state)
                            P_s_prime = 1.0 if s_prime == s else 0.0
                        else:
                            P_s_prime = self.env.get_transition_prob(s, a, s_prime)
                        expected_value += P_s_prime * V[s_prime]
                    
                    action_value = R_s_a + self.gamma * expected_value
                    
                    v_new += policy[s, a] * action_value
                
                V_new[s] = v_new
                delta = max(delta, abs(V[s] - V_new[s]))
            
            V = V_new
            
            if delta < theta:
                break
        
        return V, iteration + 1
    
    def value_iteration(self, 
                       theta: float = 1e-6,
                       max_iterations: int = 1000) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Find optimal value function using value iteration.
        
        Implements: V_{k+1}(s) = max_a [R(s,a) + γ ∑_{s'} P(s'|s,a) V_k(s')]
        
        Args:
            theta: Convergence threshold
            max_iterations: Maximum number of iterations
            
        Returns:
            Tuple of (optimal value function, optimal policy, number of iterations)
        """
        V = np.zeros(self.n_states)
        
        iteration = 0
        
        for iteration in range(max_iterations):
            delta = 0
            V_new = np.zeros(self.n_states)
            
            for s in range(self.n_states):
                action_values = np.zeros(self.n_actions)
                
                for a in range(self.n_actions):
                    R_s_a = self.env.rewards[s]
                    
                    expected_value = 0
                    for s_prime in range(self.n_states):
                        if self.env.is_terminal(s):
                            # Terminal state: absorbing (stays in same state)
                            P_s_prime = 1.0 if s_prime == s else 0.0
                        else:
                            P_s_prime = self.env.get_transition_prob(s, a, s_prime)
                        expected_value += P_s_prime * V[s_prime]
                    
                    action_values[a] = R_s_a + self.gamma * expected_value
                
                V_new[s] = np.max(action_values)
                delta = max(delta, abs(V[s] - V_new[s]))
            
            V = V_new
            
            if delta < theta:
                break
        
        # Extract optimal policy
        policy = self.extract_policy(V)
        
        return V, policy, iteration + 1
    
    def extract_policy(self, V: np.ndarray) -> np.ndarray:
        """
        Extract greedy policy from value function.
        
        Implements: π*(s) = argmax_a [R(s,a) + γ ∑_{s'} P(s'|s,a) V(s')]
        
        Args:
            V: Value function
            
        Returns:
            Policy array (n_states,) with action indices
        """
        policy = np.zeros(self.n_states, dtype=int)
        
        for s in range(self.n_states):
            if self.env.is_terminal(s):
                policy[s] = 0  # Arbitrary action for terminal states
                continue
            
            action_values = np.zeros(self.n_actions)
            
            for a in range(self.n_actions):
                R_s_a = self.env.rewards[s]
                
                expected_value = 0
                for s_prime in range(self.n_states):
                    P_s_prime = self.env.get_transition_prob(s, a, s_prime)
                    expected_value += P_s_prime * V[s_prime]
                
                action_values[a] = R_s_a + self.gamma * expected_value
            
            policy[s] = np.argmax(action_values)
        
        return policy
    
    def policy_improvement(self, V: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Improve policy based on value function.
        
        Implements: π'(s) = argmax_a [R(s,a) + γ ∑_{s'} P(s'|s,a) V(s')]
        
        Args:
            V: Current value function
            
        Returns:
            Tuple of (improved policy matrix, policy_stable flag)
        """
        policy = np.zeros((self.n_states, self.n_actions))
        policy_stable = True
        
        for s in range(self.n_states):
            if self.env.is_terminal(s):
                policy[s, 0] = 1.0  # Arbitrary action for terminal states
                continue
            
            action_values = np.zeros(self.n_actions)
            
            for a in range(self.n_actions):
                R_s_a = self.env.rewards[s]
                
                expected_value = 0
                for s_prime in range(self.n_states):
                    P_s_prime = self.env.get_transition_prob(s, a, s_prime)
                    expected_value += P_s_prime * V[s_prime]
                
                action_values[a] = R_s_a + self.gamma * expected_value
            
            best_action = np.argmax(action_values)
            policy[s, best_action] = 1.0
        
        return policy, policy_stable
    
    def policy_iteration(self, 
                        initial_policy: Optional[np.ndarray] = None,
                        theta: float = 1e-6,
                        max_iterations: int = 100) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Find optimal policy using policy iteration.
        
        Args:
            initial_policy: Initial policy (random if None)
            theta: Convergence threshold for evaluation
            max_iterations: Maximum number of iterations
            
        Returns:
            Tuple of (optimal value function, optimal policy matrix, number of iterations)
        """
        # Initialize policy
        if initial_policy is None:
            policy = np.ones((self.n_states, self.n_actions)) / self.n_actions
        else:
            policy = initial_policy.copy()
        
        iteration = 0
        
        for iteration in range(max_iterations):
            # Policy Evaluation
            V, _ = self.iterative_policy_evaluation(policy, theta=theta)
            
            # Policy Improvement
            policy_new, policy_stable = self.policy_improvement(V)
            
            # Check if policy has converged
            if np.allclose(policy, policy_new):
                policy = policy_new
                break
            
            policy = policy_new
        
        return V, policy, iteration + 1
    
    def verify_bellman_optimality(self, V: np.ndarray, tolerance: float = 1e-4) -> bool:
        """
        Verify that a value function satisfies the Bellman optimality equation.
        
        Args:
            V: Value function to verify
            tolerance: Numerical tolerance
            
        Returns:
            True if Bellman equation is satisfied
        """
        for s in range(self.n_states):
            if self.env.is_terminal(s):
                continue
            
            # Compute max over actions
            max_value = float('-inf')
            
            for a in range(self.n_actions):
                R_s_a = self.env.rewards[s]
                
                expected_value = 0
                for s_prime in range(self.n_states):
                    P_s_prime = self.env.get_transition_prob(s, a, s_prime)
                    expected_value += P_s_prime * V[s_prime]
                
                action_value = R_s_a + self.gamma * expected_value
                
                max_value = max(max_value, action_value)
            
            if abs(V[s] - max_value) > tolerance:
                return False
        
        return True
    
    def compare_policies(self, V1: np.ndarray, V2: np.ndarray) -> Tuple[bool, float]:
        """
        Compare two policies by their value functions.
        
        Args:
            V1: First value function
            V2: Second value function
            
        Returns:
            Tuple of (is_V1_better, average_difference)
        """
        diff = V1 - V2
        avg_diff = np.mean(diff[~np.isin(np.arange(self.n_states), self.env.terminal_states)])
        is_better = np.all(diff >= -1e-6)  # V1 >= V2 for all states
        
        return is_better, avg_diff

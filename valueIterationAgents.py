import mdp, util
from learningAgents import ValueEstimationAgent
import collections


class ValueIterationAgent(ValueEstimationAgent):
    """
        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        """
          Run the value iteration algorithm for the specified number of iterations.
        """
        for i in range(self.iterations):
            new_values = util.Counter()
            for state in self.mdp.getStates():
                if not self.mdp.isTerminal(state):
                    # For each state, compute the best value by checking all possible actions
                    best_value = float('-inf')
                    for action in self.mdp.getPossibleActions(state):
                        q_value = self.computeQValueFromValues(state, action)
                        best_value = max(best_value, q_value)
                    new_values[state] = best_value
            self.values = new_values  # Update the values after all states are updated

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the value function stored in self.values.
        """
        q_value = 0
        # For each possible next state, calculate the expected value of taking the action
        for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            reward = self.mdp.getReward(state, action, next_state)
            # Update q_value based on reward and the value of the next state
            q_value += prob * (reward + self.discount * self.values[next_state])
        return q_value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state according to the values.
          If there are no legal actions (i.e., terminal state), return None.
        """
        if self.mdp.isTerminal(state):
            return None

        best_action = None
        best_q_value = float('-inf')

        # Find the best action by computing Q-value for each action
        for action in self.mdp.getPossibleActions(state):
            q_value = self.computeQValueFromValues(state, action)
            if q_value > best_q_value:
                best_q_value = q_value
                best_action = action

        return best_action

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def getPolicy(self, state):
        """
          Return the policy (best action) at the state.
        """
        return self.computeActionFromValues(state)

    def getAction(self, state):
        """
          Return the action (policy) at the state.
        """
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        """
          Return the Q-value of (state, action).
        """
        return self.computeQValueFromValues(state, action)

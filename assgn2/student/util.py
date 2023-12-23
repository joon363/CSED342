import collections, random

############################################################

# An algorithm that solves an MDP (i.e., computes the optimal
# policy).
class MDPAlgorithm:
    # Set:
    # - self.pi: optimal policy (mapping from state to action)
    # - self.V: values (mapping from state to best values)
    def solve(self, mdp): raise NotImplementedError("Override me")

############################################################
class ValueIteration(MDPAlgorithm):
    '''
    Solve the MDP using value iteration.  Your solve() method must set
    - self.V to the dictionary mapping states to optimal values
    - self.pi to the dictionary mapping states to an optimal action
    Note: epsilon is the error tolerance: you should stop value iteration when
    all of the values change by less than epsilon.
    The ValueIteration class is a subclass of util.MDPAlgorithm (see util.py).
    '''

    def computeQ(self, mdp, V, state, action):
        # Return Q(state, action) based on V(state).
        return sum(prob * (reward + mdp.discount() * V[newState]) \
                   for newState, prob, reward in mdp.succAndProbReward(state, action))

    def computeOptimalPolicy(self, mdp, V):
        # Return the optimal policy given the values V.
        pi = {}
        for state in mdp.states:
            pi[state] = max((self.computeQ(mdp, V, state, action), action) for action in mdp.actions(state))[1]
        return pi

    def solve(self, mdp, epsilon=0.001):
        V = collections.defaultdict(float)  # state -> value of state
        numIters = 0
        while True:
            newV = {}
            for state in mdp.states:
                newV[state] = max(self.computeQ(mdp, V, state, action) for action in mdp.actions(state))
            numIters += 1
            if max(abs(V[state] - newV[state]) for state in mdp.states) < epsilon:
                V = newV
                break
            V = newV

        # Compute the optimal policy now
        pi = self.computeOptimalPolicy(mdp, V)
        print("ValueIteration: %d iterations" % numIters)
        self.pi = pi
        self.V = V

# An abstract class representing a Markov Decision Process (MDP).
class MDP:
    def __init__(self):
        self._states = None

    # Return the start state.
    def startState(self): raise NotImplementedError("Override me")

    # Return True if the state is an end state
    def isEnd(self, state):
        for action in self.actions(state):
            for succ, prob, reward in self.succAndProbReward(state, action):
                return False
        return True

    # Return set of actions possible from |state|.
    def actions(self, state): raise NotImplementedError("Override me")

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.
    # Mapping to notation from class:
    #   state = s, action = a, newState = s', prob = T(s, a, s'), reward = Reward(s, a, s')
    # If IsEnd(state), return the empty list.
    def succAndProbReward(self, state, action): raise NotImplementedError("Override me")

    def discount(self): raise NotImplementedError("Override me")

    @property
    def states(self):
        if self._states is None:
            self._states = self.computeStates()
        return self._states

    # Compute set of states reachable from startState.  Helper function for
    # MDPAlgorithms to know which states to compute values and policies for.
    # This function returns |states|, which is the set of all states.
    def computeStates(self):
        states = set()
        queue = []
        states.add(self.startState())
        queue.append(self.startState())
        while len(queue) > 0:
            state = queue.pop()
            for action in self.actions(state):
                for newState, prob, reward in self.succAndProbReward(state, action):
                    if newState not in states:
                        states.add(newState)
                        queue.append(newState)
        return states
        # print "%d states" % len(states)
        # print states

############################################################

# A simple example of an MDP where states are integers in [-n, +n].
# and actions involve moving left and right by one position.
# We get rewarded for going to the right.
class NumberLineMDP(MDP):
    def __init__(self, n=5): self.n = n
    def startState(self): return 0
    def actions(self, state): return [-1, +1]
    def succAndProbReward(self, state, action):
        return [(state, 0.4, 0),
                (min(max(state + action, -self.n), +self.n), 0.6, state)]
    def discount(self): return 0.9

############################################################
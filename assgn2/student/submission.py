import util, math, random
from collections import defaultdict
from util import ValueIteration


############################################################
# Problem 1a: BlackjackMDP

class BlackjackMDP(util.MDP):
    def __init__(self, cardValues, multiplicity, threshold, peekCost):
        """
        cardValues: array of card values for each card type
        multiplicity: number of each card type
        threshold: maximum total before going bust
        peekCost: how much it costs to peek at the next card
        """
        super().__init__()

        self.cardValues = cardValues
        self.multiplicity = multiplicity
        self.threshold = threshold
        self.peekCost = peekCost

    # Return the start state.
    # Look at this function to learn about the state representation.
    # The first element of the tuple is the sum of the cards in the player's
    # hand.
    # The second element is the index (not the value) of the next card, if the player peeked in the
    # last action.  If they didn't peek, this will be None.
    # The final element is the current deck.
    def startState(self):
        return (0, None, (self.multiplicity,) * len(self.cardValues))  # total, next card (if any), multiplicity for each card

    # Return set of actions possible from |state|.
    # You do not need to modify this function.
    # All logic for dealing with end states should be done in succAndProbReward
    def actions(self, state):
        return ['Take', 'Peek', 'Quit']

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.  Indicate a terminal state (after quitting or
    # busting) by setting the deck to None. 
    # When the probability is 0 for a particular transition, don't include that 
    # in the list returned by succAndProbReward.
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_ANSWER (our solution is 44 lines of code, but don't worry if you deviate from this)
        totalValue, nextCard, deckCounts = state
        if deckCounts==None:
            return []
        card_type_num=len(deckCounts)
        card_total_num=sum(deckCounts)
        return_list=[]
        reward=0

        if action=='Take':
            for i in range(card_type_num):
                if deckCounts[i]==0:
                    continue
                #manipulating tuple
                new_deckcounts=list(deckCounts)
                new_deckcounts[i]=new_deckcounts[i]-1
                new_deckcounts=tuple(new_deckcounts)
                
                #adding reward
                new_totalValue=totalValue+self.cardValues[i]
                
                #checking whether game ends
                #if deck is empty, reward=0, deck=None
                #if reward > threshold, reward=0, deck=None
                iszero=True
                for num in new_deckcounts:
                    if num!=0:
                        iszero=False
                        break
                if iszero:
                    new_deckcounts=None
                    reward=new_totalValue
                elif self.threshold<new_totalValue:
                    reward=0
                    new_deckcounts=None

                #adding new state
                next_state=((new_totalValue,None,new_deckcounts))
                if card_total_num==1:
                    #to prevent prob=1.0
                    prob = 1
                else: 
                    prob=deckCounts[i]/card_total_num
                return_tuple=(next_state,prob,reward)
                return_list.append(return_tuple)
            return return_list
        
        if action=='Peek':
            #peeking twice: return nothing
            if nextCard!=None:
                return []
            #peeking
            for i in range(card_type_num):
                if deckCounts[i]==0:
                    continue
                next_state=(totalValue,i,deckCounts)
                return_tuple=(next_state,deckCounts[i]/card_total_num,-self.peekCost)
                return_list.append(return_tuple)
            return return_list
        
        if action=='Quit':
            #quit with reward 
            return [( (totalValue, None, None),1,totalValue )]
        
        # END_YOUR_ANSWER

    def discount(self):
        return 1

############################################################
# Problem 1b: ValueIterationDP

class ValueIterationDP(ValueIteration):
    '''
    Solve the MDP using value iteration with dynamic programming.
    '''
    def solve(self, mdp):
        V = {}  # state -> value of state

        # BEGIN_YOUR_ANSWER (our solution is 13 lines of code, but don't worry if you deviate from this)
        #dictionary for dp
        V_DP={}

        #using recursive function  
        def DP(state,V_DP):
            #change computeQ
            values_action=[]
            for action in mdp.actions(state):
                sum=0
                succlist=mdp.succAndProbReward(state,action)
                if succlist==[]: #no next states(game end)
                    values_action+=[sum]
                    continue
                
                #iterate next states
                for newState,prob,reward in succlist:
                    if prob==0:
                        continue
                    #dp
                    if newState not in V_DP:
                        V_DP[newState]=DP(newState,V_DP)
                    #add Q
                    sum+=prob * (reward + mdp.discount() * V_DP[newState])
                values_action+=[sum]
            #choose max Q
            V[state]=max(values_action)
            return V[state]
        
        #start iteration
        DP(mdp.startState(),V_DP)
        
        # END_YOUR_ANSWER

        # Compute the optimal policy now
        pi = self.computeOptimalPolicy(mdp, V)
        self.pi = pi
        self.V = V

############################################################
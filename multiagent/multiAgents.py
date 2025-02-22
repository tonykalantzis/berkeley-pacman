# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        ######################################################################
        #          The evaluation Function in Reflex agent (Prj2-Q1) :       #
        # Here we get a gamestate and a proposed position                    #
        # Using these we evaluate a score of the move                        #
        # INIT:                                                              #
        #   constants to reward penalize the pacman                          #
        #   food-wise we get the new_food as list and all their distances    #
        #   ghost-wise we get the ghost-dists from new pos                   #
        #   and a bool value if we have any ghosts present                   #
        # DECISION:                                                          #
        #   if scared ghost close -> eat it                                  #
        #   else                                                             #
        #     if we have food left -> incetivise to eat the closest          #
        #     if closest ghost close to threshold GO AWAY                    #
        ######################################################################

        # INIT
        score = successorGameState.getScore()

        # food related stuff
        FOOD_WEIGHT = 10.0
        new_food_list = newFood.asList()
        food_distances = [util.manhattanDistance(newPos, food) for food in new_food_list]

        # ghost related stuff
        GHOST_WEIGHT = 11.0
        GHOST_PROZIMITY_THRESHOLD = 2
        new_ghost_distances=[util.manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
        scared_ghosts_present = any(ghost.scaredTimer > 0 for ghost in newGhostStates)

        # DECISION
        # if scared ghost close eat it
        if scared_ghosts_present and min(new_ghost_distances) == 0:
            score += GHOST_WEIGHT
        else:
            #if we have food left eat closest
            if len(new_food_list) > 0:
                score += FOOD_WEIGHT / min(food_distances)

            # if ghost close to no no zone go away
            if min(new_ghost_distances) < GHOST_PROZIMITY_THRESHOLD :
                score -= GHOST_WEIGHT

        return score

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        ######################################################################
        #                       Minimax (Prj2-Q2) :                          #
        #  Here we implement the Minimax algorithm with all the cases        #
        # Our base case is if terminal state or reached full depth           #
        # max_value:                                                         #
        #   choses max from every child node of pacman's child states        #
        #   used olny by pacman, acting as the maximizing level of tree      #
        # min_value:                                                         #
        #   if we have pacman                                                #
        #     it goes to next depth while keeping the min  direction:(down)  #
        #     also it escapes minimizing level of tree                       #
        #   else if it has ghost                                             #
        #     it goes to next agent minimizing and keeping the min           #
        #     it expands the minimizing level direction:(right)              #
        #                                                                    #
        # Big help1: https://www.youtube.com/watch?v=KU9Ch59-4vw             #
        # Big help2: https://www.javatpoint.com/mini-max-algorithm-in-ai     #
        ######################################################################

        pacman_index = gameState.getNumAgents() - 1

        def base_case(state, depth):
            # check if the game state is a terminal state or if the depth limit is reached
            return state.isWin() or state.isLose() or depth == self.depth

        def max_value(state, depth):
            # this function is exclusive for mr pacman
            # Maximizer function
            if base_case(state, depth):
                return self.evaluationFunction(state)

            v =float('-inf')
            for action in state.getLegalActions(0):
                # maximize the choices
                v = max(v, min_value(state.generateSuccessor(0, action), depth, 1))

            return v

        def min_value(state, depth, agent):
            # minimizer function
            if base_case(state, depth):
                return self.evaluationFunction(state)

            v = float('inf')
            for action in state.getLegalActions(agent):
                if agent == pacman_index:
                    #if we have pacman go to next depth trying to maximize
                    v = min(v, max_value(state.generateSuccessor(agent, action), depth + 1))
                else:
                    #if we have ghost go to next agent trying to minimize
                    v = min(v, min_value(state.generateSuccessor(agent, action), depth, agent + 1))

            return v

        # evaluate each action for pacman and choose the one with the maximum value
        actions = gameState.getLegalActions(0)
        best_action = max(actions, key=lambda action: min_value(gameState.generateSuccessor(0, action), 0, 1))

        return best_action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        ######################################################################
        #                       Minimax (Prj2-Q3) :                          #
        #  Implementation of Minimax algorithm with alpha beta pruning       #
        # The point is to throw out not needed stuff so we have less cases   #
        # base case: terminal state of limit depth                           #
        # max_value: find max like before and test beta cut-off              #
        # min_value: find min like before at test alpha cut-off              #
        # init: for every action find the value and if new best update alpha #
        ######################################################################

        pacman_index = gameState.getNumAgents() - 1

        def base_case(state, depth):
            # check if the game state is a terminal state or if the depth limit is reached
            return state.isWin() or state.isLose() or depth == self.depth

        def max_value(state, depth, alpha, beta):
            # this function is exclusive for mr pacman
            # Maximizer function
            if base_case(state, depth):
                return self.evaluationFunction(state)

            v =float('-inf')
            for action in state.getLegalActions(0):
                # maximize the choices
                v = max(v, min_value(state.generateSuccessor(0, action),\
                                     depth, 1, alpha, beta))
                if v > beta:
                    return v # beta cut-off
                alpha = max(alpha,v)

            return v

        def min_value(state, depth, agent, alpha, beta):
            # minimizer function
            if base_case(state, depth):
                return self.evaluationFunction(state)

            v = float('inf')
            for action in state.getLegalActions(agent):
                if agent == pacman_index:
                    #if we have pacman go to next depth trying to maximize
                    v = min(v, \
                        max_value(state.generateSuccessor(agent, action),\
                                  depth + 1, alpha, beta))
                else:
                    #if we have ghost go to next agent trying to minimize
                    v = min(v, \
                        min_value(state.generateSuccessor(agent, action),\
                                  depth, agent + 1, alpha, beta))

                if v < alpha:
                    return v # alpha cut-off

                beta = min(beta,v)

            return v

        # evaluate each action for pacman and choose the one with the maximum value
        alpha = float('-inf')
        beta = float('inf')
        actions = gameState.getLegalActions(0)
        best_action = None
        best_value = float('-inf')

        for action in actions:
            value = min_value(gameState.generateSuccessor(0, action), 0, 1, alpha, beta)
            if value > best_value:
                best_value = value
                best_action = action
            alpha = max(alpha, best_value)

        return best_action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        ######################################################################
        #            The Expendimax Agent funciton (Prj2-Q4) :               #
        # Here we dont assume that the oponent (ghost) plays optimally       #
        # base_case: terminal or no more depth                               #
        # max_value: check if base case for every action choose max exp      #
        # exp_value:                                                         #
        #   check if base case                                               #
        #   for every action get correct value adding them                   #
        #     pacman-> max                                                   #
        #     ghost -> exp                                                   #
        #   their sum / count-of-moves gives an average -> exp value         #
        # Help: https://www.youtube.com/watch?v=jaFRyzp7yWw -- 16:10         #
        ######################################################################

        pacman_index = gameState.getNumAgents() - 1

        def base_case(state, depth):
            # check if the game state is a terminal state or if the depth limit is reached
            return state.isWin() or state.isLose() or depth == self.depth

        def max_value(state, depth):
            actions = state.getLegalActions(0)

            if base_case(state, depth):
                return self.evaluationFunction(state)

            v = float('-inf')
            for action in actions:
                v = max(v, exp_value(state.generateSuccessor(0, action),depth, 1))

            return v

        def exp_value(state, depth, agent):

            actions = state.getLegalActions(agent)

            if base_case(state,depth):
                return self.evaluationFunction(state)

            v = 0
            divisor = len(actions)

            for action in actions:
                if agent == pacman_index:
                    #if we have pacman go to next depth trying to maximize
                    sum_cursor = max_value(state.generateSuccessor(agent,action),depth + 1)
                else:
                    sum_cursor = exp_value(state.generateSuccessor(agent,action), depth, agent + 1)

                v += sum_cursor

            return v / divisor


        actions = gameState.getLegalActions(0)
        best_action = None
        best_value = float('-inf')

        for action in actions:
            value = exp_value(gameState.generateSuccessor(0, action), 0, 1)
            if value > best_value:
                best_value = value
                best_action = action

        return best_action



def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION:
    -As shown before I setted some constants that penalize/reward the score(easily changed)
    -I got the current game features
    -If we win we GOT THIS
    -FOOD STUFF: Food score is more the less food we have (insentive to have more food and not wait)
    -GHOST STUFF:
            if we lose -> -A LOT because we lose
            if we get close to ghost (closer than threshold)-> -Significant because danger
            else the farther away, the better

    By mistake it kinda camps the ghost to execute it and get score
    """
    "*** YOUR CODE HERE ***"
    FOOD_WEIGHTS = 10
    GHOST_WEIGHTS = 11
    GHOST_THRESHOLD = 1
    DANGER_WEIGHTS = 1000
    MAKE_OR_BREAK_WEIGHTS = 10000

    # get current game state features
    state = currentGameState
    pacman_position = state.getPacmanPosition()
    food_positions = state.getFood().asList()
    ghost_states = state.getGhostStates()

    # initialize the score based on the current game score
    score = state.getScore()

    # check if the game is in a terminal state (win or lose)
    if state.isWin():
        return MAKE_OR_BREAK_WEIGHTS

    #FOOD STUFF
    # evaluate based on remaining food
    food_score = -FOOD_WEIGHTS * len(food_positions)

    # GHOST stuff
    ghost_score = score
    # evaluate based on ghost positions and distances
    for ghost_state in ghost_states:
        ghost_position = ghost_state.getPosition()
        ghost_distance = manhattanDistance(pacman_position, ghost_position)
        if ghost_distance == 0:
            # pacman is caught by a ghost, NO NO case
            return -MAKE_OR_BREAK_WEIGHTS
        elif ghost_distance <= GHOST_THRESHOLD:
            # pacman is in danger, consider avoiding
            ghost_score -= DANGER_WEIGHTS
        else:
            # evaluate based on ghost distance
            ghost_score += GHOST_WEIGHTS / ghost_distance

    # combine scores for a final evaluation
    final_score = food_score + ghost_score

    return final_score


# Abbreviation
better = betterEvaluationFunction

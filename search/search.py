# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))

    """
    "*** YOUR CODE HERE ***"
    ######################################################################
    #          The DFS algorithm in a maze(graph) dictionary:            #
    # Priority of moves: North->South->East->West (getSuccesors)         #
    # Fringe: Nodes discovered but not fully explored or processed yet   #
    # Fringe in DFS: Stack (LiFo)                                        #
    # Tuples in Fringe Stack:(current state,[previous moves to state)]   #
    # We dont care about the cost                                        #
    # Explored: A set with all nodes we have visited                     #
    # Start state: The agents start info: (coordinates)                  #
    # Here we get the first succesful path                               #
    # Big Help: https://www.youtube.com/watch?v=sTRK9mQgYuc              #
    ######################################################################
    fringe = util.Stack() #init an empty stack
    explored = set() #we have not visited any yet
    start_state = problem.getStartState()


    #an empty list of previous moves because we are at start
    fringe.push( (start_state,[]) ) #we do not care about the cost yet(3rd one)

    #while we dont have any more nodes to process
    while (fringe.isEmpty() is False ): #while not empty
        state, prev_moves = fringe.pop()

        if state not in explored:
            explored.add(state)
            #if we found the end state return the steps:
            if problem.isGoalState(state):
                return prev_moves #nodes_expanded->146, moves->13

            #now since we mark the newly-found nodes for exploration in fringe
            # for every succesor -> mark him (if not already) explored
            # and add to fringe the new state and new direction to path
            list_of_child_nodes = problem.getSuccessors(state)
            for child_node in list_of_child_nodes:
                cursor_state = child_node[0]
                move = child_node[1]
                if cursor_state not in explored:
                    fringe.push( (cursor_state, prev_moves + [move] ) )

    #util.raiseNotDefined()

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    ######################################################################
    #          The BFS algorithm in a maze(graph) dictionary:            #
    # Priority of moves: North->South->East->West (getSuccesors)         #
    # Fringe: Nodes discovered but not fully explored or processed yet   #
    # Fringe in BFS: Queue (FiFo)                                        #
    # Tuples in Fringe Stack:(current state,[previous moves to state)]   #
    # We still dont care about the cost                                  #
    # Explored: A set with all nodes we have visited                     #
    # Start state: The agents start info: (coordinates)                  #
    # Here we get the shortest path                                      #
    # Big Help: https://www.youtube.com/watch?v=D14YK-0MtcQ              #
    ######################################################################
    "*** YOUR CODE HERE ***"
    fringe = util.Queue() #init an empty queue for FiFo
    explored = set() #we have not visited any yet
    start_state = problem.getStartState()

    #an empty list of previous moves because we are at start
    fringe.push( (start_state,[]) ) #we do not care about the cost yet(3rd one)

    #while we dont have any more nodes to process
    while (fringe.isEmpty() is False ):
        state, prev_moves = fringe.pop()

        if state not in explored:
            explored.add(state)
            #if we found the end state return the steps:
            if problem.isGoalState(state):
                return prev_moves

            #now since we mark the newly-found nodes for exploration in fringe
            # for every succesor -> mark him (if not already) explored
            # and add to fringe the new state and new direction to path
            list_of_child_nodes = problem.getSuccessors(state)
            for child_node in list_of_child_nodes:
                cursor_state = child_node[0]
                move = child_node[1]
                if cursor_state not in explored:
                    fringe.push( (cursor_state, prev_moves + [move] ) )

    util.raiseNotDefined()

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    ######################################################################
    #          The UCS algorithm in a maze(graph) dictionary:            #
    # Priority of moves: North->South->East->West (getSuccesors)         #
    # Fringe: Nodes discovered but not fully explored or processed yet   #
    # Fringe in UCS: Priority Queue (Costs of paths)                     #
    # Tuples in Fringe Stack, we add the cost factor                     #
    # (current state,[previous moves to state), cost of current )        #
    # Explored: A set with all nodes we have visited                     #
    # Start state: The agents start info: (coordinates)                  #
    # Here we get the shortest path                                      #
    ######################################################################
    "*** YOUR CODE HERE ***"
    fringe = util.PriorityQueue() #init a priority queue because we have costs now
    explored = set() #we have not visited any yet
    start_state = problem.getStartState()


    #an empty list of previous moves because we are at start
    # (x,y | [empty previous moves] | 0 cost to get there,) , 0 Priority
    fringe.push( (start_state,[], 0), 0 )

    #while we dont have any more nodes to process
    while (fringe.isEmpty() is False ): #while not empty
        state, prev_moves, curr_cost = fringe.pop()

        if state not in explored:
            #add the current state to the set of fully explored states
            explored.add(state)
            #if we found the end state return the steps:
            if problem.isGoalState(state):
                return prev_moves

            #now since we mark the newly-found nodes for exploration in fringe
            # for every succesor add to fringe:
            # 1.)the new state
            # 2.)new direction to path
            # 3.) the new cost, which using PQ we prioritize the smallest cost
            list_of_child_nodes = problem.getSuccessors(state)
            for child_node in list_of_child_nodes:
                cursor_state = child_node[0]
                move = child_node[1]
                new_cost =  curr_cost + child_node[2]
                if cursor_state not in explored:
                    fringe.push( (cursor_state, prev_moves + [move], new_cost), new_cost )

    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    ##########################################################################
    #      The A* informed algorithm in a maze(graph) dictionary:            #
    # Why informed? Because the algorithm considers where the goal is        #
    # How? by considering both cost and direction with a function f:         #
    # f(n) = g(n) + h(n) where                                               #
    #   g(n)-> cost to get there as previously                               #
    #   h(n)-> a heuristic function that calculates "distance to end goal"   #
    # Initial state: We consider every node's cost as infitity               #
    # Fringe: Nodes discovered but not fully explored or processed yet       #
    # Fringe in UCS: Priority Queue (Costs of paths)                         #
    # Here we get the shortest path with less expanded nodes                 #
    # Help I got: https://www.youtube.com/watch?v=W9zSr9jnoqY&t=93s          #
    ##########################################################################
    "*** YOUR CODE HERE ***"
    fringe = util.PriorityQueue() #init a priority queue because consider costs
    explored = set() #we have not visited any yet
    start_state = problem.getStartState()


    #an empty list of previous moves because we are at start
    # (x,y | [empty previous moves] | 0 cost to get there,) , 0 Priority
    fringe.push( (start_state,[], 0), 0 )

    #while we dont have any more nodes to process
    while (fringe.isEmpty() is False ): #while not empty
        state, prev_moves, curr_cost = fringe.pop()

        if state not in explored:
            #add the current state to the set of fully explored states
            explored.add(state)
            #if we found the end state return the steps:
            if problem.isGoalState(state):
                return prev_moves

            #now since we mark the newly-found nodes for exploration in fringe
            # for every succesor add to fringe:
            # 1.)the new state
            # 2.)new direction it took on path
            # 3.) the new cost, which using PQ we prioritize the smallest cost
            # we use f(n) mentioned above
            list_of_child_nodes = problem.getSuccessors(state)
            for child_node in list_of_child_nodes:
                cursor_state = child_node[0]
                move = child_node[1]
                g_n =  curr_cost + child_node[2] # new cost of path
                h_n = heuristic(cursor_state, problem) #heuristic cost
                f_n = g_n + h_n #combined cost
                if cursor_state not in explored:
                    fringe.push( (cursor_state, prev_moves + [move], g_n), f_n )
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

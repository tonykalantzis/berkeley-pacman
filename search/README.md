# Pacman Project 1

## Απαντήσεις

### Question 1
In this question I implemented the depthFirstSearch function of the search.py file.I used a Stack as a Fringe, utilizing its Last-In-First-Out property.
Also, I used a set of visited nodes, in order to not revisit them again.
Then starting from the start state the search iterates through all the child nodes searching for a goal state. It does so, in a depth first manner, because of the fringe being a Stack.

### Question 2
In this question I Implemented the breadthFirstSearch function of the search.py file. I used a Queue as a Fringe , utilizing its First-In-First-Out property. We have a set of explored nodes here too. The algorithm is the same as the dfs but we have a queue for the fringe.

### Question 3
Here I implemented the uniformCostSearch function of the search.py file. The fringe used is a Priority Queue, because we now consider costs too. The Priority Queue prioritises the nodes with the smallest cost, this way it finds the best path on our maze. The cost is a sum of all the moves.

### Question 4
Here I implemented the aStarSearch function of the search.py file. The fringe used is a Priority Queue, because we now consider costs too. The Priority Queue prioritises the nodes with the smallest cost along with value of a heuristic given, this way it finds the best path on our maze while having a general "direction".

### Question 5
In this question I implemented the CornersProblem class.
That inludes:
* The getStartState function a tuple of format(agent position, (corners visited))
* The isGoalState function. (if visited all corners we have it)
* The getSuccessors function returns all the successor states of a node.

### Question 6
In this question I implemented the cornersHeuristic method, which provides a heuristic value for the A* algorithm. The heuristic Ι chose to use calculates the following:
It returns the maximum value of the manhattan paths to the corners.

### Question 7
Here I implemented the foodHeuristic method. I created the heuristic using the maze distance for every food in a corner. Then I took the max of those distances as my heuristic.

### Question 8
In this question I implemented the findPathToClosestDot.
Using the astar function of the anyfoodsearch problem, the findPathToClosestDot returs the node closest to the agent.

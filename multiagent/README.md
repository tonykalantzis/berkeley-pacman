# Pacman Project 2

## Answers

### Question 1
*  **The Reflex agent**: using some constant I reward/punish the pacman based on specific moves. So it has "Reflexes" on the ghost and its suroundings.

    ```python
    if scared ghost close:
        eat it
    else
        if we have food left:
            incetivise to eat the closest
        if closest ghost close to threshold:
            GO AWAY
    ```

### Question 2
*  **Minimax** the minimax algorithm has 4 parts:

    1. The **base case** which is either a terminal state or max depth
    2. The **max value** which for every child state gets max of the min_values of the successors.
    3. The **min value** which for every child state if pacman->maximizes if ghost->minimizes
    4. Finds the best action for the current player (player 0) by considering the maximum of the minimum values of the successor states after each legal action using the minimax stuff we implemented above.

### Question 3
*  **Minimax with alpha beta pruning** this also has 4 parts:

    1. The **base case** which is either a terminal state or max depth
    2. The **max value** same as before but cuts off the sub-trees with v > beta(beta cut-offs).
    3. The **min value** same as before but cuts off the sub-trees with v < alpha(alpha cut-offs).
    4. Finds the best action for the current player (player 0) by considering the maximum values of the successor states after each legal action using the minimax stuff we implemented above, while cutting off sub-trees.

### Question 4
*  **Expectimax** here rather than min we have an exp statistical value:
    1. The **base case** which is either a terminal state or max depth
    2. The **max value** checks if end and chooses exp max (pacman's method)
    3. The **exp value** checks if end and call exp or max based on the agent. After summing all possible actions it gets the weighted average and returns this, having the expendimax function.
    4. Finds the best action for the current player (player 0) by considering the maximum values of the expected successor states after each legal action using the expectimax stuff we implemented above.

### Question 5
*  **betterEvaluationFunction** here I implemented my own evaluation function:
    * I tried to see the bigger picture(evaluate states)
    * I setted some constants that penalize/reward the score(easily changed)
    * If we win we GET THIS
    * FOOD STUFF: Food score better the less food we have
    * GHOST STUFF:
    * if we lose -> subtract A LOT
    * if we get close to ghost (closer than threshold)-> minus Significant because danger
    * else the farther away, the better




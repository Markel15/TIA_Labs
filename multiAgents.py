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


import random

import util
from game import Agent, Directions
from util import manhattanDistance


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
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
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
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
        distancia_comida = float('inf')
        posiciones_comidas = newFood.asList()
        if len(posiciones_comidas) == 0: #si no queda comida devolver 0 en vez de infinito
            distancia_comida = 0
        for pos_comida in posiciones_comidas: #Calcular distancia a la comida más cercana
            distancia_comida = min(distancia_comida, manhattanDistance(pos_comida, newPos))

        distancia_ghost = 1000
        for estado_ghost in newGhostStates:
            if estado_ghost.scaredTimer == 0: #Solo tomar en cuenta si el fantasma no está asustado
                ghost_x, ghost_y = estado_ghost.getPosition()
                distancia_ghost = min(distancia_ghost, manhattanDistance(newPos,(ghost_x, ghost_y)))

        distancia_comida = distancia_comida * (-1) #Hacer negativo el valor de la distancia, cuanto más lejos esté peor valoración
        cantidad_de_comida = len(posiciones_comidas) * (-1) # Cuanta más comida quede peor

        return successorGameState.getScore()-4/(distancia_ghost+1) + distancia_comida/2 + cantidad_de_comida  #Tendréis que comentar esta linea y devolver el valor que calculeis


def scoreEvaluationFunction(currentGameState):
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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        super().__init__()
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, game_state):
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
        return self.value(game_state, 0, self.depth)[1]  # usar solo la accion [1] y no el valor

    def value(self, game_state, agentIndex, depth):
        if depth == 0 or game_state.isWin() or game_state.isLose(): #Si depth=0, parar, ya no hay que mirar más opciones en el arbol
            return self.evaluationFunction(game_state), None
        if agentIndex == 0:  # pacman
            return self.max_value(game_state, agentIndex, depth)
        else:  # fantasmas
            return self.min_value(game_state, agentIndex, depth)

    def max_value(self, game_state, agentIndex, depth):
        v = float('-inf')
        best_accion = None

        # Actualizar agentIndex
        next_agentIndex = (agentIndex + 1) % game_state.getNumAgents()
        # Actualizar depth si se han analizado todos los agentes (ha vuelto a pacman) depth-1
        if next_agentIndex == 0:
            new_depth = depth - 1
        else:
            new_depth = depth

        acciones = game_state.getLegalActions(agentIndex)

        for accion in acciones:
            sucesor = game_state.generateSuccessor(agentIndex, accion)
            sucesor_v = self.value(sucesor, next_agentIndex, new_depth)[0]
            if sucesor_v > v:
                v = sucesor_v
                best_accion = accion
        return v, best_accion

    def min_value(self, game_state, agentIndex, depth):
        v = float('inf')
        best_accion = None

        next_agentIndex = (agentIndex + 1) % game_state.getNumAgents()
        if next_agentIndex == 0:
            new_depth = depth - 1
        else:
            new_depth = depth

        acciones = game_state.getLegalActions(agentIndex)
        for accion in acciones:
            sucesor = game_state.generateSuccessor(agentIndex, accion)
            sucesor_v = self.value(sucesor, next_agentIndex, new_depth)[0]
            if sucesor_v < v:
                v = sucesor_v
                best_accion = accion

        return v, best_accion

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        return self.alphabeta(game_state, 0, self.depth, float('-inf'), float('inf'))[1]

    def alphabeta(self, game_state, agentIndex, depth, alpha, beta):
        if depth == 0 or game_state.isWin() or game_state.isLose():
            return self.evaluationFunction(game_state), None

        if agentIndex == 0:  # Pacman (maximizing player)
            return self.max_value(game_state, agentIndex, depth, alpha, beta)
        else:  # Ghosts (minimizing players)
            return self.min_value(game_state, agentIndex, depth, alpha, beta)

    def max_value(self, game_state, agentIndex, depth, alpha, beta):
        v = float('-inf')
        best_accion = None

        acciones = game_state.getLegalActions(agentIndex)
        if not acciones:
            return self.evaluationFunction(game_state), None

        next_agentIndex = (agentIndex + 1) % game_state.getNumAgents()
        new_depth = depth if next_agentIndex != 0 else depth - 1

        for accion in acciones:
            sucesor = game_state.generateSuccessor(agentIndex, accion)
            sucesor_v = self.alphabeta(sucesor, next_agentIndex, new_depth, alpha, beta)[0]

            if sucesor_v > v:
                v = sucesor_v
                best_accion = accion

            if v > beta:  # Poda
                return v, best_accion

            alpha = max(alpha, v)

        return v, best_accion

    def min_value(self, game_state, agentIndex, depth, alpha, beta):
        v = float('inf')
        best_accion = None

        acciones = game_state.getLegalActions(agentIndex)
        if not acciones:
            return self.evaluationFunction(game_state), None

        next_agentIndex = (agentIndex + 1) % game_state.getNumAgents()
        new_depth = depth if next_agentIndex != 0 else depth - 1

        for accion in acciones:
            sucesor = game_state.generateSuccessor(agentIndex, accion)
            sucesor_v = self.alphabeta(sucesor, next_agentIndex, new_depth, alpha, beta)[0]

            if sucesor_v < v:
                v = sucesor_v
                best_accion = accion

            if v < alpha:  # Poda
                return v, best_accion

            beta = min(beta, v)

        return v, best_accion



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def getAction(self, game_state):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction
        """
        return self.value(game_state, 0, self.depth)[1]

    def value(self, game_state, agentIndex, depth):
        if depth == 0 or game_state.isWin() or game_state.isLose():
            return self.evaluationFunction(game_state), None

        if agentIndex == 0:  # Pacman
            return self.max_value(game_state, agentIndex, depth)
        else:  # Fantasmas
            return self.exp_value(game_state, agentIndex, depth)

    def max_value(self, game_state, agentIndex, depth):
        v = float('-inf')
        best_accion = None

        acciones = game_state.getLegalActions(agentIndex)
        if not acciones:
            return self.evaluationFunction(game_state), None

        next_agentIndex = (agentIndex + 1) % game_state.getNumAgents()
        new_depth = depth if next_agentIndex != 0 else depth - 1

        for accion in acciones:
            sucesor = game_state.generateSuccessor(agentIndex, accion)
            sucesor_v = self.value(sucesor, next_agentIndex, new_depth)[0]

            if sucesor_v > v:
                v = sucesor_v
                best_accion = accion

        return v, best_accion

    def exp_value(self, game_state, agentIndex, depth):
        v = 0  # Valor esperado

        acciones = game_state.getLegalActions(agentIndex)
        if not acciones:
            return self.evaluationFunction(game_state), None

        next_agentIndex = (agentIndex + 1) % game_state.getNumAgents()
        new_depth = depth if next_agentIndex != 0 else depth - 1

        probabilidad = 1.0 / len(acciones)  # Probabilidad uniforme para cada acción

        for accion in acciones:
            sucesor = game_state.generateSuccessor(agentIndex, accion)
            sucesor_v = self.value(sucesor, next_agentIndex, new_depth)[0]
            v += probabilidad * sucesor_v  # Suma de todos los valores esperados * probabilidad de que ocurra cada uno

        return v, None


def betterEvaluationFunction(currentGameState):
    """
    Your extreme, unstoppable evaluation function (question 5).
    """
    # Información útil del estado actual
    pacmanPos = currentGameState.getPacmanPosition()
    foodGrid = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    # 1. Calcular la distancia a la comida más cercana
    food_positions = foodGrid.asList()
    if len(food_positions) > 0:
        min_food_distance = min(manhattanDistance(pacmanPos, food) for food in food_positions)
    else:
        min_food_distance = 0  # Si no queda comida, distancia mínima es 0

    # Calcular la distancia a los fantasmas y manejar el estado de asustado
    min_ghost_distance = float('inf')
    ghost_penalty = 0
    for i, ghost in enumerate(ghostStates):
        ghostPos = ghost.getPosition()
        ghost_distance = manhattanDistance(pacmanPos, ghostPos)

        if scaredTimes[i] > 0:  # El fantasma está asustado, podemos ir a por él
            ghost_penalty += 450 / (ghost_distance + 1)  # Más cerca = más incentivo
        else:  # El fantasma no está asustado, hay que evitarlo
            min_ghost_distance = min(min_ghost_distance, ghost_distance)

    # Penalización por estar demasiado cerca de un fantasma no asustado
    if min_ghost_distance < 2:
        ghost_penalty -= 1000  # Gran penalización si estamos muy cerca de un fantasma peligroso

    # Considerar la cantidad de comida restante
    remaining_food = currentGameState.getNumFood()

    # Considerar las cápsulas de energía
    capsules = currentGameState.getCapsules()
    num_capsules = len(capsules)
    capsule_bonus = 0
    if num_capsules > 0:
        capsule_bonus = 100 / min(manhattanDistance(pacmanPos, capsule) for capsule in capsules)

    # Calcular la puntuación final
    score = currentGameState.getScore()
    evaluation = (
            score  # Valor del estado actual
            + 1.0 / (min_food_distance + 1)  # Incentivar acercarse a la comida
            + ghost_penalty  # Incentivar evitar fantasmas o cazarlos si están asustados
            - 4 * remaining_food  # Penalizar por la cantidad de comida que queda
            + capsule_bonus  # Bonificación por acercarse a las cápsulas
    )

    return evaluation


# Abbreviation
better = betterEvaluationFunction

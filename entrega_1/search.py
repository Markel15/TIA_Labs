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

from abc import ABC, abstractmethod

import util


class SearchProblem(ABC):
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    @abstractmethod
    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    @abstractmethod
    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    @abstractmethod
    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    @abstractmethod
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
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    frontera = util.Stack()
    frontera.push((problem.getStartState(), []))  # Primero guardar el camino vacio como inicializacion
    lista_visitados = set()

    while not frontera.isEmpty():
        nodo, path = frontera.pop()

        if problem.isGoalState(nodo):
            return path
        else:
            if nodo not in lista_visitados:
                lista_visitados.add(nodo)
                for sucesor in problem.getSuccessors(nodo):
                    pos = sucesor[0]  # Posición
                    movimiento = sucesor[1]
                    if pos not in lista_visitados:
                        # Agregar la próxima posicion a expandir a la pila y actualizar el camino hasta el momento
                        new_path = path + [f"{movimiento}"]
                        frontera.push((pos, new_path))
    return []


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    frontera = util.Queue()
    frontera.push((problem.getStartState(), []))  # Primero guardar el camino vacio como inicializacion
    lista_visitados = set()

    while not frontera.isEmpty():
        nodo, path = frontera.pop()

        if problem.isGoalState(nodo):
            return path
        else:
            if nodo not in lista_visitados:
                lista_visitados.add(nodo)
                for sucesor in problem.getSuccessors(nodo):
                    pos = sucesor[0]  # Posición
                    movimiento = sucesor[1]#eliminado el siguiente if, era innecesario
                    new_path = path + [f"{movimiento}"]
                    frontera.push((pos, new_path))
    return []

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    frontera = util.PriorityQueue()
    frontera.push((problem.getStartState(), [], 0),0)  # Añadir el costo al valor de la cola a parte de la prioridad
    lista_visitados = set()

    while not frontera.isEmpty():
        nodo, path, costo_anterior = frontera.pop()

        if problem.isGoalState(nodo):
            return path
        else:
            if nodo not in lista_visitados:
                lista_visitados.add(nodo)
                for sucesor in problem.getSuccessors(nodo):
                    pos = sucesor[0]  # Posición
                    movimiento = sucesor[1]
                    coste_paso = sucesor[2]
                    new_path = path + [f"{movimiento}"]
                    frontera.push((pos, new_path,costo_anterior+coste_paso),costo_anterior+coste_paso)
    return []


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    frontera = util.PriorityQueue()
    frontera.push((problem.getStartState(), [], 0), 0 + heuristic(problem.getStartState(),problem))
    lista_visitados = set()

    while not frontera.isEmpty():
        nodo, path, costo_anterior = frontera.pop()

        if problem.isGoalState(nodo):
            return path
        else:
            if nodo not in lista_visitados:
                lista_visitados.add(nodo)
                for sucesor in problem.getSuccessors(nodo):
                    pos = sucesor[0]  # Posición
                    movimiento = sucesor[1]
                    coste_paso = sucesor[2]
                    new_path = path + [f"{movimiento}"]
                    coste = coste_paso + costo_anterior
                    frontera.push((pos, new_path, coste), coste + heuristic(pos,problem))
    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

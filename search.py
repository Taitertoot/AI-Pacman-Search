# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
In search.py, you will implement generic search algorithms which are called 
by Pacman agents (in searchAgents.py).
"""

import util

'''
Class Node describes and encodes the behavior of
what each position on our Pacman map layout should contain
It holds information like the current position of this node,
its parent, its current cost from the start to where it is now
and its direction on how it came to be.

This class has one really useful method
'''
class Node:

    def __init__(self, succ, dad = None):

      self.pos = succ[0]
      self.orientation = succ[1]
      self.dad = dad
      
      if dad != None and dad.cost != None:
          self.cost = succ[2] + dad.cost
          
      elif dad == None or dad.cost == None:
          self.cost = succ[2]

    #iterative solution for returning the
    #path of a node, traversing back on itself to the beginning.
    def getPath(self):
      path = list()
      currentNode = self
      while currentNode.orientation != None:
          path.insert(0, currentNode.orientation)
          currentNode = currentNode.dad
      return path
    
    #recursive solution for returning a list
    #of the path this node has to get back to the very
    #start. This is a huge benefit in using a Node structure.
    def treeHierarchy(self, path, node, problem):
        if self is None or node is None:
            return path
        if node.orientation is not None:
            path.insert(0, node.orientation)
        return node.treeHierarchy(path, node.dad, problem) 


class SearchProblem:
  """
  This class outlines the structure of a search problem, but doesn't implement
  any of the methods (in object-oriented terminology: an abstract class).
  
  You do not need to change anything in this class, ever.
  """
  
  def getStartState(self):
     """
     Returns the start state for the search problem 
     """
     util.raiseNotDefined()
    
  def isGoalState(self, state):
     """
       state: Search state
    
     Returns True if and only if the state is a valid goal state
     """
     util.raiseNotDefined()

  def getSuccessors(self, state):
     """
       state: Search state
     
     For a given state, this should return a list of triples, 
     (successor, action, stepCost), where 'successor' is a 
     successor to the current state, 'action' is the action
     required to get there, and 'stepCost' is the incremental 
     cost of expanding to that successor
     """
     util.raiseNotDefined()

  def getCostOfActions(self, actions):
     """
      actions: A list of actions to take
 
     This method returns the total cost of a particular sequence of actions.  The sequence must
     be composed of legal moves
     """
     util.raiseNotDefined()
           

def tinyMazeSearch(problem):
  """
  Returns a sequence of moves that solves tinyMaze.  For any other
  maze, the sequence of moves will be incorrect, so only use this for tinyMaze
  """
  from game import Directions
  s = Directions.SOUTH
  w = Directions.WEST
  return  [s,s,w,s,w,w,s,w]

def gSearch(problem, ds, control, h):

  #Begin pushing and appending our starting position to our frontier and
  #visited nodes list.
  snod = Node((problem.getStartState(), None, None))
  if control == 2:
      ds.push(snod, h(snod.pos, problem))
  elif control == 1:
      ds.push(snod, 0)
  elif control == 0:
      ds.push(snod)
  visited_nodes = []
  visited_nodes.append(snod.pos)

  positions = []
  positions.insert(0, snod.pos)

  if problem.isGoalState(snod.pos):
    return []
  #While we're still searching and expanding nodes...
  while ds:

    #Gets the closest node pacman is at to work with.
    route = ds.pop()
    positions.remove(route.pos)
    
    #otherwise, append this position, and look for the
    #positions after this node.
    visited_nodes.append(route.pos)
    if problem.isGoalState(route.pos):
        return route.getPath()
        #return route.treeHierarchy([], route, problem)
    for succ in problem.getSuccessors(route.pos):
        n = Node(succ, route)
        #if this node is the goal, return the path it took to get here.
        if problem.isGoalState(n.pos):
            return n.getPath()
            #return n.treeHierarchy([], n, problem)
      #if successor is valid and not yet visited, push it onto the stack
        if n.pos not in visited_nodes and n.pos not in positions:

            if control == 0:
                ds.push(n)
           #    positions.insert(0, succ[0])
                positions.append(n.pos)

            elif control == 1:
                ds.push(n, route.cost)
           #    positions.insert(0, succ[0])
                positions.append(n.pos)

            elif control == 2:
                ds.push(n, n.cost + h(n.pos, problem))
            #   positions.insert(0, succ[0])
                positions.append(n.pos)

            else:
                print "Wrong control value."
                return []

  return [] #return empty list if nothing happens. This is not possible, unless
            #python really messes up. I hate python.


def depthFirstSearch(problem):
  """
  Search the deepest nodes in the search tree first [p 85].
  """
  return gSearch(problem, util.Stack(), 0, None) #Same function, uses stack. 
  
def breadthFirstSearch(problem):
  "Search the shallowest nodes in the search tree first. [p 81]"
  return gSearch(problem, util.Queue(), 0, None) #same function, uses queue

def uniformCostSearch(problem):
  "Search the node of least total cost first. "
  return gSearch(problem, util.PriorityQueue(), 1, None) #Same function, uses a PriorityQueue

def nullHeuristic(state, problem=None):
  """
  A heuristic function estimates the cost from the current state to the nearest
  goal in the provided SearchProblem.  This heuristic is trivial.
  """
  return 0

def aStarSearch(problem, heuristic=nullHeuristic):
  "Search the node that has the lowest combined cost and heuristic first."
  return gSearch(problem, util.PriorityQueue(), 2, heuristic)
  #Uses a priority queue and a varying cost.

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

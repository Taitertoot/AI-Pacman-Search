# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from game import Directions
from util import manhattanDistance
import random, util, searchAgents

from game import Agent
'''
This code was developed by Taite Nazifi for
an ISTA450 pacman AI project. All of the code seen here was done
by himself, and he helped two different groups: Sam Hoeger's group,
and Devon Oberdan's group, who also implemented all of their own code.
'''
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
    some Directions.X for some X in the set {North, South, West, East, Stop}
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
    #gets the current score
    score = successorGameState.getScore()
    if newFood.count() > 0:
        score += 100 * (1/newFood.count())
    else:
        return 999
    #IF the ghost isn't scared then we'll end up goin'
    #along our own little way. If he is scared, we won't
    #do anything.
    if (newScaredTimes[0] <= 0) and \
                      (newPos == ((newGhostStates[0].getPosition()[0]-1, newGhostStates[0].getPosition()[1]) or
                      (newGhostStates[0].getPosition()[0]+1, newGhostStates[0].getPosition()[1]) or
                      (newGhostStates[0].getPosition()[0], newGhostStates[0].getPosition()[1]-1) or
                      (newGhostStates[0].getPosition()[0], newGhostStates[0].getPosition()[1]+1))):
            score = -99999
            
    if newFood.count() == currentGameState.getFood().count():
      #This code gets the distance tot he closest food
      #on the grid using packBits to get its neat little
      #data structure. really cool function.
        newGrid = newFood.packBits()
        value = None
        for i in range(newGrid[0]):
            for j in range(newGrid[1]):
                if newFood[i][j] == True:
                    dist = (abs(newPos[0] - i) + abs(newPos[1] - j)), (i,j)
                    if value == None:
                        value = dist
                    if dist[0] < value[0]:
                        value = dist
        if value == None: 
            value = (0, newPos)
        score -= value[0]
    score += newScaredTimes[0] * 10

    return score 

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

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (question 2)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction.
      Here are some method calls that might be useful when implementing minimax.
      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1
      Directions.STOP:
        The stop direction, which is always legal
      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action
      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"
    #Calls the two below functions and recurses down our tree until
    #we've won or lost, or the game has stopped, or we have reached
    #the depth of our current tree. Recursively calls teh other two
    #functions until we have won.
    def miniMax(state, dept = -1, agentIndex = -1):
        agentIndex += 1
        agentIndex = agentIndex % gameState.getNumAgents()
        if agentIndex == 0 :
            dept += 1
        if state.isWin() or state.isLose() or (dept == self.depth):
            return self.evaluationFunction(state)
        if agentIndex == 0 :
            return maxValue(state, dept, agentIndex)
        else :
            return minValue(state, dept, agentIndex)
    #Gets the max value of the current action, maxifies
    #on pacman's location and sees his best options.
    def maxValue(state, dept, agentIndex):
        maxval = float("-inf"), 'Stop'
        actions = state.getLegalActions(agentIndex)
        for action in actions:
            successor = state.generateSuccessor(agentIndex,action)
            val = miniMax(successor, dept, agentIndex)
            if val > maxval[0] :
                maxval = val, action
        if dept != 0:
            return maxval[0]
        return maxval

    #Minimizes on the ghosts actions and sees what would happen
    #if something were to go wrong and pacman were to go in this
    #certain location.
    def minValue(state, dept, agentIndex):
        minval = float("inf")
        actions = state.getLegalActions(agentIndex) 
        for action in actions :
            successor = state.generateSuccessor(agentIndex,action)
            v = miniMax(successor, dept, agentIndex)
            if v < minval :
                minval = v
        return minval

    return miniMax(gameState)[1]

alpha = 0
beta = 0       
class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    global alpha
    global beta
    alpha = float("-inf")
    beta = float("inf")  

    #Calls the two below functions and recurses down our tree until
    #we've won or lost, or the game has stopped, or we have reached
    #the depth of our current tree. Recursively calls teh other two
    #functions until we have won.
    def miniMax(state, dept, agentIndex, alpha, beta):
        agentIndex += 1
        agentIndex = agentIndex % gameState.getNumAgents()
        if agentIndex == 0 :
            dept += 1
        if state.isWin() or state.isLose() or (dept == self.depth) :
            return self.evaluationFunction(state)
        if agentIndex == 0 :
            return maxValue(state, dept, agentIndex, alpha, beta)
        else :
            return minValue(state, dept, agentIndex, alpha, beta)

    #Gets the max value of the current action, maxifies
    #on pacman's location and sees his best options.
    def maxValue(state, dept, agentIndex, alpha, beta):
        maxVal = float("-inf"), 'Stop'
        actions = state.getLegalActions(agentIndex)
        for a in actions :
            successor = state.generateSuccessor(agentIndex,a)
            v = miniMax(successor, dept, agentIndex, alpha, beta)
            if v >= beta :
                return v, a
            if v > maxVal[0] :
                maxVal = v, a
            alpha = max(alpha, v)
        if dept == 0:
          return maxVal
        return maxVal[0]
        
    #Minimizes on the ghosts actions and sees what would happen
    #if something were to go wrong and pacman were to go in this
    #certain location.
    def minValue(state, dept, agentIndex, alpha, beta):
        minVal = float("inf")
        actions = state.getLegalActions(agentIndex) 
        for a in actions :
            successor = state.generateSuccessor(agentIndex,a)
            v = miniMax(successor, dept, agentIndex, alpha, beta)
            if v <= alpha :
                return v
            if v < minVal :
                minVal = v
            beta = min(beta, v)
        return minVal
    
    return miniMax(gameState, -1, -1, alpha, beta)[1]

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction
      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    "*** YOUR CODE HERE ***"
    #Calls the two below functions and recurses down our tree until
    #we've won or lost, or the game has stopped, or we have reached
    #the depth of our current tree. Recursively calls teh other two
    #functions until we have won.
    def miniMax(state, dept = -1, agentIndex = -1):
        agentIndex += 1
        agentIndex = agentIndex % gameState.getNumAgents()
        if agentIndex == 0 :
            dept += 1
        if state.isWin() or state.isLose() or (dept == self.depth) :
            return self.evaluationFunction(state)
        if agentIndex == 0 :
            return maxValue(state, dept, agentIndex)
        else :
            return minValue(state, dept, agentIndex)

    #Gets the max value of the current action, maxifies
    #on pacman's location and sees his best options.
    def maxValue(state, dept, agentIndex):
        maxval = float("-inf")
        actions = state.getLegalActions(agentIndex)
        maxval = (maxval, 'Stop')
        for a in actions :
          successor = state.generateSuccessor(agentIndex,a)
          v = miniMax(successor, dept, agentIndex)
          if v > maxval[0] :
            maxval = v, a
        if dept != 0:
          return maxval[0]
        return maxval

    #Minimizes on the ghosts actions and sees what would happen
    #if something were to go wrong and pacman were to go in this
    #certain location.
    def minValue(state, dept, agentIndex):
        minval = float(0) #Can't use float("inf") here.
        actions = state.getLegalActions(agentIndex) 
        for a in actions:
            successor = state.generateSuccessor(agentIndex,a)
            v = miniMax(successor, dept, agentIndex)
            minval += v/len(actions)
        return minval
    
    return miniMax(gameState)[1]

    

def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).
    DESCRIPTION: <I didn't do this linearly, I used conditionals c: koopa kappa poopa>
  """
  "*** YOUR CODE HERE ***"

  newPos = currentGameState.getPacmanPosition()
  newFood = currentGameState.getFood()
  newGhostStates = currentGameState.getGhostStates()
  newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

  #Adding the game score to our current score tally to help
  #our little monster pacman win.
  score = currentGameState.getScore()
  
  if currentGameState.isWin():
      return 999 + score 
  
  if newFood.count() > 0 :
      score += 200*(1/newFood.count())

  #Chases down scared ghosts, otherwise we tell pacman to
  #RETREAT and GET OUTTA THERE MAN
  for i in range(len(newGhostStates)) :
      ghostPos = newGhostStates[i].getPosition()
      if newScaredTimes[i] < 1 :
          if newPos == (ghostPos[0]+1, ghostPos[1]):
              score = -9999
          elif newPos == (ghostPos[0]-1, ghostPos[1]):
              score = -9999
          elif newPos == (ghostPos[0], ghostPos[1]+1):
              score = -9999
          elif newPos == (ghostPos[0], ghostPos[1]-1):
              score = -9999
      else :
          score += ((1/(util.manhattanDistance(newPos, ghostPos)))*100)
  #This chunk of code gets the distance to
  #the closest food using manhattan distance.
  newGrid = newFood.packBits()
  value = None
  for i in range(newGrid[0]):
      for j in range(newGrid[1]):
          if newFood[i][j] == True:
              dist = (abs(newPos[0] - i) + abs(newPos[1] - j)), (i,j)
              if value == None:
                  value = dist
              if dist[0] < value[0]:
                  value = dist
  if value == None: 
      value = (0, newPos)
  score -= value[0]

  return score

better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.
      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    pass

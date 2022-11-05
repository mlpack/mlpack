/**
 * @file methods/reinforcement_learning/environment/maze.hpp
 * @author Eshaan Agarwal
 *
 * This file is an implementation of Goal based Maze task:
 * In this task agent will given a random generated n*m maze where each cell
 * can have value 1- Goal . 0 - Path, -1 - Wall.
 * Agent is expected to move through 0(path) cells to reach to 1(goal cell).
 * For maze generation we have used a Step based Random DFS Walk and filled 
 * other cells randomly with -1 or 0.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RL_ENVIRONMENT_MAZE_HPP
#define MLPACK_METHODS_RL_ENVIRONMENT_MAZE_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * Implementation of Maze task.
 */
class Maze
{
 public:
  /**
   * Implementation of the state of Maze.
   * Each State is a tuple {(row, column)} in n * n maze matrix.
   */
  class State
  {
   public:
    /**
     * Construct a state instance.
     */
    State() : data(dimension)
    { /* Nothing to do here. */ }

    /**
     * Construct a state instance from given data.
     *
     * @param data Data 
     */
    State(const arma::vec& data) : data(data)
    { /* Nothing to do here */ }

    //! Modify the internal representation of the state.
    arma::vec& Data() { return data; }

    //! Get the internal representation of the state.
    arma::vec Data() const { return data; }

    //! Encode the state to a column vector.
    const arma::vec& Encode() const { return data; }

    //! Get value of row index of the node.
    double Row() const { return data[0]; }
    //! Modify value of row index of the node.
    double& Row() { return data[0]; }

    //! Get value of column index of the node.
    double Column() const { return data[1]; }
    //! Modify value of column index of the node.
    double& Column() { return data[1]; }

    //! Dimension of the encoded state.
    static constexpr size_t dimension = 2;

   private:
    //! Locally-stored {(row, column)}.
    arma::vec data;
  };

  /**
   * Implementation of action of Maze.
   */
  class Action
  {
   public:
   enum actions
    {
      left,
      right,
      down,
      up,
    };
    
    // To store the action ( direction to move )
    Action::actions action;

    // Track the size of the action space.
    static const size_t size = 4;
  };

  /**
   * Construct a maze instance using the given constants.
   * @param rows number of rows in maze.
   * @param columns number of columns in maze.
   * @param maxSteps The number of steps after which the episode terminates.
   * If the value is 0, there is no limit.
   * @param ratioForRandomWalk ratio of Steps for Random DFS Walk in 
   * maze generation.
   */
  Maze(const size_t rows = 4,
       const size_t columns = 4,
       const size_t maxSteps = 200,
       const double ratioForRandomWalk = 3) :
      maxSteps(maxSteps),
      stepsPerformed(0)
  { 
    MazeGeneration(rows, columns, ratioForRandomWalk);
    
    directions = std::unordered_map<Action::actions, arma::vec>({
            { Action::actions::left, arma::vec({0, -1}) },
            { Action::actions::right, arma::vec({0, 1}) },
            { Action::actions::up, arma::vec({-1, 0}) },
            { Action::actions::down, arma::vec({1, 0}) }
        });
  }

  /**
   * Generate Maze using Random Iterative DFS Algorithm with 
   * 0.4 * rows * column steps.
   * @param rows number of rows in maze.
   * @param columns number of columns in maze.
   * @param ratioForRandomWalk ratio of Steps for Random DFS Walk in 
   *        maze generation.
   */
  void MazeGeneration(const size_t rows, const size_t columns,
                      const double ratioForRandomWalk = 3)
  {
    maze = arma::mat(rows,columns);
    arma::umat visited(rows, columns, arma::fill::zeros);

    std::vector<arma::ivec> moves = {arma::ivec({0, -1}), arma::ivec({0, 1}),
                                     arma::ivec({-1, 0}), arma::ivec({1, 0})};

    const size_t maxNumberOfMazeSteps = ratioForRandomWalk * 
                                        std::sqrt(rows * columns);

    std::stack<arma::ivec> cellsInPath;

    arma::ivec startingPoint = {arma::randi(arma::distr_param(0, rows-1)),
                                arma::randi(arma::distr_param(0, columns-1))};

    cellsInPath.push(startingPoint);

    size_t numberOfMazeSteps = 0;
 
    // Loop till stack is empty.
    while (!cellsInPath.empty())
    {
      // Pop a vertex from the stack.
      arma::ivec lastCell = cellsInPath.top();
      cellsInPath.pop();

      ++numberOfMazeSteps;

      // Break for Max number of maze steps.
      if(numberOfMazeSteps > maxNumberOfMazeSteps)
        break;

      // Mark the particular row and column as visited.
      visited(lastCell(0),lastCell(1)) = 1;

      // If visitable nodes are present from that node.
      bool visitableNodes = false;

      // Randomly choose a direction and check if the adjacent node in
      // that direction is visited or not.
      for (size_t counter = 0; counter < 10 ; ++counter)
      {
        const size_t index = arma::randi(arma::distr_param(0, 3));
        arma::ivec nextCell = arma::ivec({lastCell(0) + moves[index](0),
                                          {lastCell(1) + moves[index](1)}});

        // Check if particular row and column index are out of bounds of maze or visited.
        if (!CheckBoundsofMaze(nextCell(0),nextCell(1)) &&
            !visited(nextCell(0),nextCell(1)))
        {
            cellsInPath.push(lastCell);
            cellsInPath.push(nextCell);
            break;
        }
      }
    }

    // Pop last ({row,column}) from the stack
    arma::ivec goalCell = cellsInPath.top();
    cellsInPath.pop();

    // Set the ({row,column}) asa goal cell
    maze(goalCell(0),goalCell(1)) = 1;

    // Set all other cells in stack to 0 
    while (!cellsInPath.empty())
    {
      arma::ivec lastCell = cellsInPath.top();
      cellsInPath.pop();
      
      maze(lastCell(0),lastCell(1)) = 0;
    }

    // Randomly set other cells to -1(wall) or 0(path).
    maze.for_each([] (double& cell) {
    if (cell != 0 && cell != 1)
      {cell = arma::randi(arma::distr_param(-1, 0));}
    });

    // Store goal cell
    goal = arma::vec({double(goalCell(0)),double(goalCell(1))});

    // Store all the path cells as potential starting points
    for (size_t row = 0; row < rows; ++row)
    {
      for (size_t col = 0; col < columns; ++col)
      {
        if(maze(row,col) == 0)
        {
          startingPoints.push_back(arma::vec({double(row),double(col)}));
        }
      }
    }

    maze.print();
  }

  /**
   * Check bounds of current row and column index
   *
   * @param state The current state.
   * @param action The current action.
   * @param nextState The next state.
   * @param transitionGoal The goal for transition.
   * @return reward, it's always 1.0.
   */
  bool CheckBoundsofMaze(double row, double column)
  {
    return row >= maze.n_rows || column>= maze.n_cols || row < 0 || column < 0;
  }

  /**
   * Get reward and next state based on current
   * state and current action.
   *
   * @param state The current state.
   * @param action The current action.
   * @param nextState The next state.
   * @param transitionGoal The goal for transition.
   * @return reward, it's always 1.0.
   */
  double Sample(const State& state,
                const Action& action,
                State& nextState,
                const State& transitionGoal)
  {
    // Update the number of steps performed.
    stepsPerformed++;

    // Make a vector to estimate nextstate.
    arma::vec currentState (state.Data());
    arma::vec direction = directions[action.action];

    arma::vec currentNextState = currentState + direction;
    nextState.Data() = currentState + direction;

    // dont move to that position if invalid
    bool invalid = false;

    // Check if particular row and column index are out
    // of bounds of maze or a wall.
    if(CheckBoundsofMaze(currentNextState(0),currentNextState(1)) || 
        maze(currentNextState(0),currentNextState(1))==-1 ){
        invalid = true;
        nextState.Data() = currentState;
    }   

    // Check if the episode has terminated.
    bool done = IsTerminal(nextState);

    // Do not reward agent if it failed.
    if (done && maxSteps != 0 && stepsPerformed >= maxSteps)
      return 0.0;
    
    if (done && maze(nextState.Row(),nextState.Column()) == 1 && 
        arma::approx_equal(nextState.Data(), transitionGoal.Data(), 
        "absdiff", 1e-5))
      return 1.0;

    return 0.0;
  }

  /**
   * Get reward based on current state and current
   * action.
   *
   * @param state The current state.
   * @param action The current action.
   * @param transitionGoal The transition goal.
   * @return reward, it's always 1.0.
   */
  double Sample(const State& state, const Action& action,
                 const State& transitionGoal)
  {
    State nextState;
    return Sample(state, action, nextState, transitionGoal);
  }

  /**
   * Initial state (row,column) for agent in maze
   *
   * @return Initial state for each episode.
   */
  State InitialSample()
  {
    stepsPerformed = 0;
    size_t index = arma::randi(arma::distr_param(0, startingPoints.size() -1));
    initialState = arma::vec(startingPoints[index]);
    return State(initialState);
  }

  /**
   * Get reward for particular goal
   *
   * @param nextState The next state.
   * @param transitionGoal Transition's goal.
   * @return Initial state for each episode.
   */
  double GetHERReward(const State& nextState,
                    const State& transitionGoal)
  {
    if (arma::approx_equal(nextState.Data(), transitionGoal.Data(), 
        "absdiff", 1e-5) )
    {
      return 1.0;
    }

    return 0.0;
  }

  /**
   * This function checks if the cart has reached the terminal state.
   *
   * @param state The desired state.
   * @return true if state is a terminal state, otherwise false.
   */
  bool IsTerminal(const State& state) const
  { 
    if (maxSteps != 0 && stepsPerformed >= maxSteps)
    {
      Log::Info << "Episode terminated due to the maximum number of steps " << maxSteps <<
          " being taken.";
      return true;
    }
    else if (maze(state.Row(),state.Column()) == 1)
    {
      Log::Info << "Episode terminated as agent has reached desired goal.";
      return true;
    }
    return false;
  }

  /**
   * Initial goal representation for thr environment
   *
   * @return Initial goal for each episode.
   */
  State GoalSample()
  {
    return State(goal);
  }

  //! Get the number of steps performed.
  size_t StepsPerformed() const { return stepsPerformed; }

  //! Get the maximum number of steps allowed.
  size_t MaxSteps() const { return maxSteps; }
  //! Set the maximum number of steps allowed.
  size_t& MaxSteps() { return maxSteps; }

  //! Get the goal for the episode
  arma::vec Goal() const { return goal; }
  //! Set the goal for the episode
  arma::vec& Goal() { return goal; }

  //! Get the maze for the episode
  arma::mat MazeMatrix() const { return maze; }
  //! Set the maze for the episode
  arma::mat& MazeMatrix() { return maze; }

 private:
  //! Random DFS Walk based generated maze matrix which as the environment.
  arma::mat maze;

  //! Starting points from where agent can start
  std::vector<arma::vec> startingPoints;

  //! allowed directions and associated moves
  std::unordered_map<Action::actions, arma::vec> directions;

  //! Locally-stored maximum number of steps.
  size_t maxSteps;

  //! Locally-stored done reward.
  double doneReward;

  //! Locally-stored number of steps performed.
  size_t stepsPerformed;

  //! Locally stored goal for the epsiode
  arma::vec goal;

  //! Locally stored initialState for the epsiode
  arma::vec initialState;
};

} // namespace mlpack

#endif

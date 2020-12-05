/**
 * @file methods/reinforcement_learning/environment/frozen_lake.hpp
 * @author Alex Nguyen
 *
 * This file is an implementation of Frozen Lake task:
 * https://gym.openai.com/envs/FrozenLake-v0/
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RL_ENVIRONMENT_FROZEN_LAKE_HPP
#define MLPACK_METHODS_RL_ENVIRONMENT_FROZEN_LAKE_HPP

#include <mlpack/prereqs.hpp>
#include <stack>

namespace mlpack {
namespace rl {

/**
 * Implementation of Frozen Lake task.
 */
class FrozenLake
{
 public:
  class State
  {
   public:
    /**
     * Default constructor.
     */
    State() {/** Nothing to do here. */}

    /**
     * Construct a state instance from given data. Initialize the
     * current position to the top left of the environment board.
     * 
     * @param height Height of the environment board.
     * @param width Width of the environment board.
     */ 
    State(size_t height, size_t width) : 
      height(height),
      width(width),
      curRow(0),
      curCol(0)
    {
      // Nothing to do here.
    }

    //! Copy constructor.
    State(State const& other) :
      height(other.height),
      width(other.width),
      curRow(other.curRow),
      curCol(other.curCol)
    {
      // Nothing to do here.
    }

    //! Move constructor.
    State(State&& other) :
      height(other.height),
      width(other.width),
      curRow(other.curRow),
      curCol(other.curCol)
    {
      // Nothing to do here.
    }

    //! Operator= copy constructor.
    State& operator=(State const& other)
    {
      if (this != &other)
      {
        height = other.height;
        width = other.width;
        curRow = other.curRow;
        curCol = other.curCol;
      }
      return *this;
    }

    //! Operator= move constructor.
    State& operator=(State&& other)
    {
      if (this != &other)
      {
        height = other.height;
        width = other.width;
        curRow = other.curRow;
        curCol = other.curCol;
      }
      return *this;
    }

    //! Get the height of the environment board.
    size_t NumRows() const {return height;}

    //! Get the width of the environment board.
    size_t NumCols() const {return width;}

    //! Get the current row-position of the player.
    size_t CurRow() const {return curRow;}

    //! Get the current column-position of the player.
    size_t CurCol() const {return curCol;}

    //! Set the current row-position of the player.
    size_t& CurRow() {return curRow;}

    //! Set the current column-position of the player.
    size_t& CurCol() {return curCol;}

    //! Get the state representation.
    size_t GetState() const {return width * curRow + curCol;}

   private:
    //! Locally-stored height of the environment board.
    size_t height;

    //! Locally-stored width of the environment board.
    size_t width;

    //! Locally-stored current row-position of the agent.
    size_t curRow;

    //! Locally-stored current column-position of the agent.
    size_t curCol;
  };

  /**
   * Implementation of action of Frozen Lake.
   */
  class Action
  {
   public:
    enum actions
    {
      Up, Down, Left, Right
    };
    // To store the action.
    Action::actions action;

    // Track the size of the action space.
    static const size_t size = 4;
  };

  /**
   * Construct a Frozen Lake instance using the given constants.
   * 
   * @param maxSteps the maximum number of steps that the agent can
   *    make. Default to 200.
   * @param height Height of the environment board. Default to 4.
   * @param width Width of the environment board. Default to 4.
   * @param platformRate The probability of platform (walkable) tile
   *    distribution. Default to 0.8.
   */ 
  FrozenLake(size_t const maxSteps=200,
             size_t const height=4,
             size_t const width=4,
             double const platformRate=0.8) : 
    maxSteps(maxSteps),
    height(height),
    width(width)
  { 
    // Pre-process platform rate
    this->platformRate = std::min(platformRate, 1.0);
    this->platformRate = std::max(platformRate, 0.0);
  }

  /**
   * Dynamics of Frozen Lake instance. Get reward and next state based on current
   * state and current action.
   * 
   * @param state The current state.
   * @param action The current action.
   * @param nextState The next state.
   * @return reward, 1.0 if the agent step into a goal, -1.0 if the agent step
   *    into a hole, 0.0 otherwise.
   */
  double Sample(const State& state,
                const Action& action,
                State& nextState)
  {
    // Update the number of steps performed.
    stepsPerformed++;

    // Copy the state to the next state.
    nextState = state;
    
    // Calculate new position.
    size_t newRow = state.CurRow();
    size_t newCol = state.CurCol();
    if (action.action == Action::actions::Left)
      newCol = std::max(nextState.CurCol() - 1, (unsigned long) 0);
    else if (action.action == Action::actions::Down)
      newRow = std::min(nextState.CurRow() + 1, height - 1);
    else if (action.action == Action::actions::Right)
      newCol = std::min(nextState.CurCol() + 1, width - 1);
    else if (action.action == Action::actions::Up)
      newRow = std::max(nextState.CurRow() - 1, (unsigned long) 0);
    
    // Update states by setting new current row and column, 
    // and manually set the description of board.
    nextState.CurRow() = newRow;
    nextState.CurCol() = newCol;

    // Check if the episode has terminated.
    bool done = IsTerminal(nextState);

    // Reward agent if it reached the goal.
    if (done && maxSteps != 0 && newRow == height - 1 && 
        newCol == width - 1)
      return 1.0;
    else if (done && boardDescription[newRow][newCol] == 'H')
      return -1.0;

    // Reward 0 otherwise.
    return 0.0;
  }

  /**
   * Dynamics of Frozen Lake. Get reward based on current 
   * state and current action.
   *
   * @param state The current state.
   * @param action The current action.
   * @return reward, 1.0 if the agent step into a goal, -1.0 if the agent step
   *    into a hole, 0.0 otherwise.
   */
  double Sample(const State& state, const Action& action)
  {
    State nextState;
    return Sample(state, action, nextState);
  }

  /**
   * Initial state representation is the current position at (0, 0) 
   * and a new random environment board is created.
   *
   * @return Initial state for each episode.
   */
  State InitialSample()
  {
    stepsPerformed = 0;
    boardDescription = generateRandomBoard();
    return State(height, width);
  }

  /**
   * Initial state representation is the current position at (0, 0) 
   * and a new random bord is created.
   * 
   * @param board Environment board of that the user want to initialize to.
   * @param height Height of the environment board.
   * @param width Width of the environment board.
   * @return A state in which the height and width of the environment board are 
   *    limited to height and width, respectively.
   */
  State InitialSample(std::vector<std::vector<char>> board, size_t height, size_t width)
  {
    stepsPerformed = 0;
    this->height = height;
    this->width = width;
    boardDescription = board;
    return State(height, width);
  }

  /**
   * This function checks if the agent has reached the terminal state.
   *
   * @param state The desired state.
   * @return true if state is a terminal state, otherwise false.
   */
  bool IsTerminal(const State& state) const
  { 
    if (maxSteps != 0 && stepsPerformed >= maxSteps)
    {
      Log::Info << "Episode terminated due to the maximum number of steps"
          "being taken.\n";
      return true;
    }
    else if (boardDescription[state.CurRow()][state.CurCol()] == 'H')
    {
      Log::Info << "Episode terminated due to agent falling in the hole.\n";
      return true;
    }
    else if (boardDescription[state.CurRow()][state.CurCol()] == 'G')
    {
      Log::Info << "Episode terminated, agent reached the goal.\n";
      return true;
    }
    return false;
  }

  //! Get the number of steps performed.
  size_t StepsPerformed() const { return stepsPerformed; }

  //! Get the maximum number of steps allowed.
  size_t MaxSteps() const { return maxSteps; }

  //! Set the maximum number of steps allowed.
  size_t& MaxSteps() { return maxSteps; }

  //! Get the board description.
  std::vector<std::vector<char>> Description() const {return boardDescription;}

  //! Set the board description.
  std::vector<std::vector<char>>& Description() {return boardDescription;}

 private:
  /**
   * Utility function that helps generate random environment board.
   * 
   * @return board, a 2D array of characters that hold the board description.
   *    'S' is the Start Tile, the starting position of the agent.
   *    'G' is the Goal Tile, the agent can walk into a goal tile to win.
   *    'F' is a Frozen Tile, the agent can walk on it.
   *    'H' is a Hole Tile, the agent can fall into it and the game ends.
   */
  std::vector<std::vector<char>> generateRandomBoard()
  {
    bool valid = false;
    std::vector<std::vector<char>> Board;
    //! TODO: Tentative, implement max step to control whether
    // we can never generate a possible board.
    while (!valid)
    {
      // Randomly choose tiles in the board (discrete random distribution).
      Board = genBoardHelper(height, width, platformRate);
      valid = dfsHelper(Board, height, width);
    }
    return Board;
  }

  /**
   * Perform depth-firsth search to see if the environment board has 
   * a solution or not.
   * 
   * @param candidateBoard Environment board description.
   * @param height Height of the environment board.
   * @param width Width of the environment board.
   * @return true if there is a solution, false otherwise. 
   */
  static bool dfsHelper(std::vector<std::vector<char>> candidateBoard, size_t height, size_t width)
  {
    arma::Mat<short> visited(height, width);
    visited.fill(0);
    std::vector<std::array<size_t, 2>> path;
    path.push_back({0, 0});
    while (!path.empty())
    {
      auto node = path.back();
      path.pop_back();

      visited(node[0],node[1]) = 1;
      int directions[4][2] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
      for (auto direction : directions)
      {
        size_t r_new = direction[0] + node[0];
        size_t c_new = direction[1] + node[1];
        if (!visited(r_new, c_new))
        {
          if (r_new >= height || c_new >= width)
              continue;
          if (candidateBoard[r_new][c_new] == 'G')
              return true;
          if (candidateBoard[r_new][c_new] != 'H')
              path.push_back({r_new, c_new});
        }
      }
    }
    return false;
  }

  /**
   * Perform random distribution of tile in the environment board. There are 4 type of tile:
   * 'S' is the Start Tile, the starting position of the agent.
   * 'G' is the Goal Tile, the agent can walk into a goal tile to win.
   * 'F' is a Frozen Tile, the agent can walk on it.
   * 'H' is a Hole Tile, the agent can fall into it and the game ends.
   * 
   * @param m Height of the environment board.
   * @param n Width of the environment board.
   * @return a 2d array that describes the environment board. 
   */
  std::vector<std::vector<char>> genBoardHelper(size_t height, size_t width, double platformRate)
  {
    std::vector<char> board[height];
    for (size_t i = 0; i < height; i++)
    {
      for (size_t j = 0; j < width; j++)
      {
        auto r = arma::randu();
        if (r < 1 - platformRate)
          board[i].push_back('H');
        else
          board[i].push_back('F');
      }
    }
    board[0][0] = 'S';
    board[height - 1][width - 1] = 'G';

    std::vector<std::vector<char>> returnedBoard(board, board + height);
    return returnedBoard;
  }

  /**
   * Utilities function to print board description. 
   * 
   * @param board Environment board's description.
   * @param height Height of the environment board.
   * @param width Width of the environment board.
   */
  static void printBoard(std::vector<std::vector<char>> board, size_t height, size_t width)
  {
    for (size_t i = 0; i < height; i++) 
    {
      for (size_t j = 0; j < width; j++)
      {
        Log::Info << board[i][j] << " ";
      }
      Log::Info << "\n";
    }
    Log::Info << "\n";
  }

  //! Locally-stored maximum number of steps.
  size_t maxSteps;

  //! Locally-stored number of steps performed.
  size_t stepsPerformed;

  //! Locally-stored height of the environment board.
  size_t height;

  //! Locally-stored width of the environment board.
  size_t width;

  //! Locally-stored the probability of how many 
  //  platform (walkable) tile exists.
  double platformRate;

  //! Locally-stored the environment board description.
  std::vector<std::vector<char>> boardDescription;

};
} // namespace rl
} // namespace mlpack

#endif

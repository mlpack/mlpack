/**
 * @file methods/reinforcement_learning/environment/frozen_lake.hpp
 * @author Alex Nguyen
 *
 * This file is an implementation of Frozen Lake task:
 * https://gym.openai.com/envs/FrozenLake-v0/
 *
 * TODO: provide an option to use dynamics directly from OpenAI gym.
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
 * Implementation of Cart Pole task.
 */
class FrozenLake
{
 public:
  class State
  {
   public:
    State()
    { 
      // Nothing to do here.
    }

    // Initialize a new state with current row and column at 0.
    State(size_t nRows, size_t nCols) : 
      nRows(nRows),
      nCols(nCols),
      curRow(0),
      curCol(0)
    {
      // Nothing to do here.
    }

    //! Copy constructor.
    State(State const& other) :
      nRows(other.nRows),
      nCols(other.nCols),
      curRow(other.curRow),
      curCol(other.curCol)
    {
      // Nothing to do here.
    }

    //! Move constructor.
    State(State&& other) :
      nRows(other.nRows),
      nCols(other.nCols),
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
        nRows = other.nRows;
        nCols = other.nCols;
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
        nRows = other.nRows;
        nCols = other.nCols;
        curRow = other.curRow;
        curCol = other.curCol;
      }
      return *this;
    }

    //! Get nRows.
    size_t NumRows() const {return nRows;}

    //! Get nCol.
    size_t NumCols() const {return nCols;}

    //! Get current row.
    size_t CurRow() const {return curRow;}

    //! Get current col.
    size_t CurCol() const {return curCol;}

    //! Set current row.
    size_t& CurRow() {return curRow;}

    //! Set current col.
    size_t& CurCol() {return curCol;}

    //! Get state representation.
    size_t GetState() const {return nCols * curRow + curCol;}

   private:
    size_t nRows;
    size_t nCols;
    size_t curRow;
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

  //! Constructor.
  FrozenLake(size_t const& maxSteps=200,
              size_t const& nRows=4,
              size_t const& nCols=4,
              double const& platformRate=0.8) : 
    maxSteps(maxSteps),
    nRows(nRows),
    nCols(nCols)
  { 
    // Preprocess platform rate
    this->platformRate = std::min(platformRate, 1.0);
    this->platformRate = std::max(platformRate, 0.0);
  }

  //  Step function
  double Sample(const State& state,
                const Action& action,
                State& nextState)
  {
    // Update the number of steps performed.
    stepsPerformed++;

    size_t newRow = state.CurRow();
    size_t newCol = state.CurCol();

    // Copy the state to the next state.
    nextState = state;
    
    // Calculate new row and col.
    if (action.action == Action::actions::Left)
      newCol = std::max(nextState.CurCol() - 1, (unsigned long) 0);
    else if (action.action == Action::actions::Down)
      newRow = std::min(nextState.CurRow() + 1, nRows - 1);
    else if (action.action == Action::actions::Right)
      newCol = std::min(nextState.CurCol() + 1, nCols - 1);
    else if (action.action == Action::actions::Up)
      newRow = std::max(nextState.CurRow() - 1, (unsigned long) 0);
    
    // Update states by setting new current row and column, 
    //    and manually set the description of board.
    nextState.CurRow() = newRow;
    nextState.CurCol() = newCol;

    // Check if the episode has terminated.
    bool done = IsTerminal(nextState);

    // Reward agent if it reached the goal.
    if (done && maxSteps != 0 && newRow == nRows - 1 && 
        newCol == nCols - 1)
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
   * @return reward, it's always 1.0.
   */
  double Sample(const State& state, const Action& action)
  {
    State nextState;
    return Sample(state, action, nextState);
  }

  /**
   * Initial state representation is the current position at (0, 0) 
   * and a new random bord is created.
   *
   * @return Initial state for each episode.
   */
  State InitialSample()
  {
    stepsPerformed = 0;
    boardDescription = generateRandomBoard();
    // Log::Info << boardDescription;
    return State(nRows, nCols);
  }

  /**
   * Initial state representation is the current position at (0, 0) 
   * and a new random bord is created.
   * @param  board: The board of that the user want to initialize to.
   * @param  nRows: Number of rows of the board.
   * @param  nCols: Number of columns of the board.
   * @return the State in which the number of rows and columns are 
   * limited to nRows and nCols, respectively.
   */
  State InitialSample(std::vector<std::vector<char>> board, size_t nRows, size_t nCols)
  {
    stepsPerformed = 0;
    this->nRows = nRows;
    this->nCols = nCols;
    boardDescription = board;
    return State(nRows, nCols);
  }

  /**
   * This function checks if the cart has reached the terminal state.
   *
   * @param state The desired state.
   * @return true if state is a terminal state, otherwise false.
   */
  bool IsTerminal(const State& state) const
  { 
    // printBoard(boardDescription, nRows, nCols);
    // Log::Info << "this turn moved: " << boardDescription[state.CurRow()][state.CurCol()] << "\n";
    if (maxSteps != 0 && stepsPerformed >= maxSteps)
    {
      Log::Info << "Episode terminated due to the maximum number of steps"
          "being taken.\n";
      return true;
    }
    // else if (state.Description()(std::vector<int> t{state.CurRow(), state.CurCol()}) == 'H')
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
   * This is an utility function that help generate random board. 
   * @return board: a 2D array of characters that hold the board description. 
   */
  std::vector<std::vector<char>> generateRandomBoard()
  {
    bool valid = false;
    std::vector<std::vector<char>> Board;
    //! TODO: Tentative, implement max step to control whether
    //    the we can never generate a possible board.
    while (!valid)
    {
      // Randomly choose tiles in the board (discrete random distribution).
      Board = genBoardHelper(nRows, nCols, platformRate);
      valid = dfsHelper(Board, nRows, nCols);
    }
    return Board;
  }

  /**
   * Perform depth-firsth search to see if the board has a solution or not. 
   * @param  candidateBoard: the board description.
   * @param  nRows: number of rows.
   * @param  nCols: number of columns.
   * @return true if there is a solution, false otherwise. 
   */
  static bool dfsHelper(std::vector<std::vector<char>> candidateBoard, size_t nRows, size_t nCols)
  {
    bool visited[nRows][nCols] = {{false}};
    std::vector<std::array<size_t, 2>> path;
    path.push_back({0, 0});
    while (!path.empty())
    {
      auto node = path.back();
      path.pop_back();

      visited[node[0]][node[1]] = true;
      int directions[4][2] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
      for (auto direction : directions)
      {
        size_t r_new = direction[0] + node[0];
        size_t c_new = direction[1] + node[1];
        if (!visited[r_new][c_new])
        {
          if (r_new >= nRows || c_new >= nCols)
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
   * Perform random distribution.
   * @param  m: number of rows.
   * @param  n: number of columns.
   * @return a 2d array that describes the game board. 
   */
  std::vector<std::vector<char>> genBoardHelper (size_t nRows, size_t nCols, double platformRate)
  {
    std::vector<char> board[nRows];
    for (size_t i = 0; i < nRows; i++)
    {
      for (size_t j = 0; j < nCols; j++)
      {
        auto r = arma::randu();
        if (r < 1 - platformRate)
          board[i].push_back('H');
        else
          board[i].push_back('F');
      }
    }
    board[0][0] = 'S';
    board[nRows - 1][nCols - 1] = 'G';

    std::vector<std::vector<char>> returnedBoard(board, board + nRows);
    return returnedBoard;
  }

  /**
   * Utilities function to print board description. 
   * @param  board: the board description.
   * @param  nRows: number of rows in the board.
   * @param  nCols: number of columns in the board.
   */
  static void printBoard(std::vector<std::vector<char>> board, size_t nRows, size_t nCols)
  {
    for (size_t i = 0; i < nRows; i++) 
    {
      for (size_t j = 0; j < nCols; j++)
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

  //! Locally-stored number of rows of the board.
  size_t nRows;

  //! Locally-stored number of columns of the board.
  size_t nCols;

  //! Locally-stored the probability of how many 
  //  platform (walkable) tile exists.
  double platformRate;

  //! Locally-stored the board description.
  std::vector<std::vector<char>> boardDescription;

};
} // namespace rl
} // namespace mlpack

#endif

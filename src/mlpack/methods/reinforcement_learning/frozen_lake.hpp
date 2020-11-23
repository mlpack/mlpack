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

    //! TODO: Do we actually need a default constructor?
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
      // boardDescription(boardDescription)
    {
      // Nothing to do here.
    }

    //! Copy constructor.
    State(State const& other) :
      nRows(other.nRows),
      nCols(other.nCols),
      curRow(other.curRow),
      curCol(other.curCol)
      // boardDescription(other.boardDescription)
    {
      // Nothing to do here.
    }

    //! Move constructor.
    State(State&& other) :
      nRows(std::move(other.nRows)),
      nCols(std::move(other.nCols)),
      curRow(std::move(other.curRow)),
      curCol(std::move(other.curCol))
      // boardDescription(std::move(other.boardDescription))
    {
      other.nRows = 0;
      other.nCols = 0;
      other.curRow = 0;
      other.curCol = 0;
      // other.boardDescription = nullptr;
    }

    //! Copy constructor.
    State& operator=(State const& other)
    {
      nRows = other.nRows;
      nCols = other.nCols;
      curRow = other.curRow;
      curCol = other.curCol;
      // boardDescription = other.boardDescription;
    }

    //! Copy constructor.
    State& operator=(State&& other)
    {
      nRows = std::move(other.nRows);
      nCols = std::move(other.nCols);
      curRow = std::move(other.curRow);
      curCol = std::move(other.curCol);
      // boardDescription = std::move(other.boardDescription);
      other.nRows = 0;
      other.nCols = 0;
      other.curRow = 0;
      other.curCol = 0;
      // other.boardDescription = nullptr;
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

    //! Set newPosition
    // static arma::Mat<char>& newBoardDescription(
    //   arma::Mat<char>& currentBoard, 
    //   size_t const row, 
    //   size_t const col)
    // {
    //   //! TODO: Do we actually need to set the board position?
    //   // or do we just need to keep track of the current wor and column?
    // }


    //! Get the current description of the board.
    // arma::Mat<char> Description() const {return boardDescription;}

   private:
    size_t nRows;
    size_t nCols;
    size_t curRow;
    size_t curCol;
    // arma::Mat<char> boardDescription;
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
    // platformRate(platformRate)
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

    //! TODO: Figure out whether to copy or move pointer.
    // Copy will cost space and time, but move will modify the current data 
    //    in place, so it's a burden that the user want to store history.

    // Move state to the next state pointer
    // nextState = std::move(state);

    // Copy the state to the next state.
    nextState = state;
    
    // Calculate new row and col.
    if (action.action == Action::actions::Left)
    {
      newCol = std::max(nextState.CurCol() - 1, (unsigned long) 0);
    }
    else if (action.action == Action::actions::Down)
    {
      newRow = std::min(nextState.CurRow() + 1, nRows - 1);
    }
    else if (action.action == Action::actions::Right)
    {
      newCol = std::min(nextState.CurCol() + 1, nCols - 1);
    }
    else if (action.action == Action::actions::Up)
    {
      newRow = std::max(nextState.CurRow() - 1, (unsigned long) 0);
    }

    // Update states by setting new current row and column, 
    //    and manually set the description of board
    nextState.CurRow() = newRow;
    nextState.CurCol() = newCol;

    // Check if the episode has terminated.
    bool done = IsTerminal(nextState);

    // Reward agent if it reached the goal.
    if (done && maxSteps != 0 && newRow == nRows - 1 && 
        newCol == nCols - 1)
      return 1;
    else if (done && boardDescription(newRow, newCol) == 'H')
      return -1;
    // Reward 0 otherwise.
    return 0;
  }

  /**
   * Dynamics of Frozen Lake. Get reward based on current state and current.
   * action.
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
   * Initial state representation is the current position at (0, 0) and a new random bord is created.
   *
   * @return Initial state for each episode.
   */
  State InitialSample()
  {
    stepsPerformed = 0;
    boardDescription = generateRandomBoard();
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
    if (maxSteps != 0 && stepsPerformed >= maxSteps)
    {
      Log::Info << "Episode terminated due to the maximum number of steps"
          "being taken.";
      return true;
    }
    // else if (state.Description()(std::vector<int> t{state.CurRow(), state.CurCol()}) == 'H')
    else if (boardDescription(state.CurRow(), state.CurCol()) == 'H')
    {
      Log::Info << "Episode terminated due to agent falling in the hole.";
      return true;
    }
    else if (boardDescription(state.CurRow(), state.CurCol()) == 'G')
    {
      Log::Info << "Episode terminated, agent reached the goal.";
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

  //! Get the board description
  arma::Mat<char> Description() const {return boardDescription;}

 private:

  /**
   * @brief  
   * @note   
   * @param  nRows: 
   * @param  nCols: 
   * @param  platformRate: 
   * @retval 
   */
  arma::Mat<char> generateRandomBoard()
  {
    // Actual code here
    //! TODO: Write the implementation for the generate random board.
    bool valid = false;
    arma::Mat<char> Board;
    while (!valid)
    {
      // Randomly choose tiles in the board (discrete random distribution)
      size_t length = nRows * nCols;
      
      // This code is taken from stack overflow
      std::vector<char> vec(length);
      const std::vector<char> samples{ 'F', 'H' };
      const std::vector<double> probabilities{ platformRate, 1 - platformRate };
      std::default_random_engine generator;
      std::discrete_distribution<int> distribution(probabilities.begin(), probabilities.end());
      std::vector<int> indices(vec.size());
      std::generate(indices.begin(), indices.end(), [&generator, &distribution]() { return distribution(generator); });
      std::transform(indices.begin(), indices.end(), vec.begin(), [&samples](int index) { return samples[index]; });

      arma::Row<char> candidateBoard(vec);
      candidateBoard.reshape(nRows, nCols);
      candidateBoard(0, 0) = 'S';
      candidateBoard(nRows - 1, nCols - 1) = 'G';
      Board = candidateBoard;
      valid = dfsHelper(candidateBoard);
    }
    return Board;
  }

  bool dfsHelper(arma::Mat<char> candidateBoard)
  {
    bool visited[nRows][nCols];
    std::stack<std::array<size_t, 2>> path;
    path.push({0, 0});
    while (!path.empty())
    {
      std::array<size_t, 2> node = path.top();
      path.pop();
      if (!visited[node[0]][node[1]])
      {
        visited[node[0]][node[1]] = true;
        int directions[4][2] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
        for (auto direction : directions)
        {
          size_t r_new = direction[0] + node[0];
          size_t c_new = direction[1] + node[1];
          if (r_new < 0 or r_new >= nRows || c_new < 0 or c_new >= nCols)
              continue;
          if (candidateBoard(r_new, c_new) == 'G')
              return true;
          if (candidateBoard(r_new, c_new) != 'H')
              path.push({r_new, c_new});
        }
      }
    }
    return false;
  }

  //! Locally-stored maximum number of steps.
  size_t maxSteps;

  //! Locally-stored number of steps performed.
  size_t stepsPerformed;

  size_t nRows;

  size_t nCols;

  double platformRate;

  arma::Mat<char> boardDescription;

};

} // namespace rl
} // namespace mlpack

#endif

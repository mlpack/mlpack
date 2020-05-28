/**
 * @file early_stop_at_min_loss.hpp
 * @author Marcus Edel
 * @author Omar Shrit
 *
 * Implementation of the early stop at minimum loss callback function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CALLBACKS_EARLY_STOP_AT_MIN_LOSS_HPP
#define MLPACK_CALLBACKS_EARLY_STOP_AT_MIN_LOSS_HPP

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Early stopping to terminate the optimization process early if the loss stops
 * decreasing.
 */
class EarlyStopAtMinLoss
{
 public:
  /**
   * Set up the early stop at min loss class, which keeps track of the minimum
   * loss and stops the optimization process if the loss stops decreasing.
   *
   * @param patienceIn The number of epochs to wait after the minimum loss has
   *    been reached or no improvement has been made (Default: 10).
   */
  EarlyStopAtMinLoss(const size_t patienceIn = 10, 
                     std::ostream& output = arma::get_cout_stream()) :
      patience(patienceIn),
      bestObjective(std::numeric_limits<double>::max()),
      steps(0),
      output(output)
  { /* Nothing to do here */ }

  /**
   * Set up the early stop at min class, which keeps track
   * of the minimum loss and stops the optimization process if the loss stops
   * decreasing on a specific validation set.
   *
   * @param predictors: data matrix used to predict the responses.
   * @param responses: data matrix used to evaluate the predictions.
   * @param patienceIn The number of epochs to wait after the minimum loss has
   * been reached or no improvement has been made (Default: 10).
   */
  EarlyStopAtMinLoss(const arma::mat& predictors,
                     const arma::mat& responses,
                     const size_t patienceIn = 10,
                     std::ostream& output = arma::get_cout_stream()) :
      patience(patienceIn),
      bestObjective(std::numeric_limits<double>::max()),
      steps(0),
      output(output)
  {
    this->predictors = predictors;
    this->responses = responses;
  }

  /**
   * Callback function called at the end of a pass over the data.
   *
   * @param optimizer The optimizer used to update the function.
   * @param function Function to optimize.
   * @param coordinates Starting point.
   * @param epoch The index of the current epoch.
   * @param objective Objective value of the current point.
   */
  template<typename OptimizerType, typename FunctionType, typename MatType>
  bool EndEpoch(OptimizerType& /* optimizer */,
                FunctionType& function,
                const MatType& /* coordinates */,
                const size_t /* epoch */,
                double objective)
  {
    if ((!predictors.empty()) and (!responses.empty()))
    {
      objective = function.Evaluate(predictors, responses);
      output << "Validation loss: "<< objective << std::endl; 
    }

    if (objective < bestObjective)
    {
      steps = 0;
      bestObjective = objective;
      return false;
    }

    steps++;
    if (steps >= patience)
    {
      Log::Info << "Minimum loss reached; terminate optimization." << std::endl;
      return true;
    }

    return false;
  }

 private:
  //! The number of epochs to wait before terminating the optimization process.
  size_t patience;

  //! Locally-stored best objective.
  double bestObjective;

  //! Locally-stored number of steps since the loss improved.
  size_t steps;

 //! The output stream that all data is to be sent to; example: std::cout.
  std::ostream& output;

  //! The matrix of data points (predictors).
  arma::mat predictors;

  //! The matrix of responses to the input data points.
  arma::mat responses;
};

}
} 

#endif

/** 
 * @file distance_statistic_predictor.hpp
 * @author Yannis Mentekidis
 *
 * This file defines a helper class that uses the function a * k^b * N^c for 
 * some parameters a, b, c that have been fit to either predict the arithmetic 
 * or geometric mean of the squared distance of a point to its k-nearest
 * neighbor, given some dataset size N and its k-nearest neighbor.
 *
 * DistanceStatisticPredictor objects are used by the LSHModel class of mlpack.
 */
#ifndef MLPACK_METHODS_NEIGHBOR_SEARCH_DISTANCE_STATISTIC_PREDICTOR_HPP
#define MLPACK_METHODS_NEIGHBOR_SEARCH_DISTANCE_STATISTIC_PREDICTOR_HPP

// For curve fitting.
#include <mlpack/core/optimizers/lbfgs/lbfgs.hpp>
// Default objective function.
#include "objectivefunction.hpp"

namespace mlpack
{
namespace neighbor
{

template <typename ObjectiveFunction = DefaultObjectiveFunction>
class DistanceStatisticPredictor
{
 public:
  //! Empty constructor.
  DistanceStatisticPredictor() { };

  /** 
   * Function to construct with training set.
   *
   * @param inputSize A vector of input sizes. The first input variable of 
   *     the regression.
   * @param kValues A vector of k values. The second input variable of the
   *     regression.
   * @param statistic A vector of responses - the value of the statistic for
   *     each given inputSize.
   */
  DistanceStatisticPredictor(const arma::Col<size_t>& inputSize, 
                             const arma::Col<size_t>& kValues,
                             const arma::mat& statistic) 
  { Train(inputSize, kValues, statistic); };
  
  //! Default destructor.
  ~DistanceStatisticPredictor() { };

  /**
   * Function that fits the alpha, beta and gamma parameters.
   *
   * @param inputSize A vector of input sizes. The first input variable of 
   *     the regression.
   * @param kValues A vector of k values. The second input variable of the
   *     regression.
   * @param statistic A vector of responses - the value of the statistic for
   *     each given inputSize.
   */
  double Train(const arma::Col<size_t>& inputSize, 
               const arma::Col<size_t>& kValues,
               const arma::mat& statistic);

  /** 
   * Evaluate the statistic for a given dataset size.
   *
   * @param N - a new input size for which to evaluate the expected
   *     statistic.
   */
  double Predict(size_t N, size_t k) 
  { return alpha * std::pow(k, beta) * std::pow(N, gamma); };

  //! Set the alpha parameter.
  void Alpha(double a) { alpha = a; };

  //! Get the alpha parameter.
  double Alpha(void) { return alpha; };
  
  //! Set the beta parameter.
  void Beta(double b) { beta = b; };

  //! Get the beta parameter.
  double Beta(void) { return beta; };

  //! Set the gamma parameter.
  void Gamma(double c) { gamma = c; };

  //! Get the gamma parameter.
  double Gamma(void) { return gamma; };

 private:
  double alpha;
  double beta;
  double gamma;
};

// Fit a curve to the data provided.
template <typename ObjectiveFunction>
double DistanceStatisticPredictor<ObjectiveFunction>::Train(
    const arma::Col<size_t>& inputSize,
    const arma::Col<size_t>& kValues,
    const arma::mat& statistic)
{
  // Objective function for fitting the E(x, k) curve to the statistic.
  ObjectiveFunction f(inputSize, kValues, statistic);

  // Optimizer. Use L_BFGS
  mlpack::optimization::L_BFGS<ObjectiveFunction> opt(f);

  // Get an initial point from the optimizer.
  arma::mat currentPoint = f.GetInitialPoint();
  double result = opt.Optimize(currentPoint);

  // Optimizer is done - set alpha, beta, gamma.
  this->alpha = currentPoint(0, 0);
  this->beta = currentPoint(1, 0);
  this->gamma = currentPoint(2, 0);

  return result;
}

} // namespace neighbor
} // namespace mlpack

#endif

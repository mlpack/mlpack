/**
 * @file objectivefunction.hpp
 * @author Yannis Mentekidis
 *
 * This file implements a class that describes an objective function for
 * minimization. It is used by the LSH model to fit a curve of the form
 * E(k, N) = \alpha \cdot k ^ \beta \cdot N^\gamma
 * to a certain statistic E, which can be either the arithmetic or the geometric
 * mean of distances of a random point and its k-Nearest Neighbors.
 *
 * The objective function to minimize is the mean squared error (MSE):
 *
 * Error =\sum_{i=0}^{M} (y(i) - \alpha \cdot k ^ \beta \cdot N^\gamma)^2
 *
 * The class is designed for use with the L_BFGS optimizer, which is what the
 * lshmodel class uses.
 */

#ifndef MLPACK_METHODS_NEIGHBOR_SEARCH_LSH_DEFAULT_OBJECTIVE_FUNCTION_HPP
#define MLPACK_METHODS_NEIGHBOR_SEARCH_LSH_DEFAULT_OBJECTIVE_FUNCTION_HPP

namespace mlpack {
namespace neighbor {

class DefaultObjectiveFunction
{
  public:
    //! Default constructor - do nothing.
    DefaultObjectiveFunction() { };

    /**
     * Parameterized constructor.
     *
     * @param xData Vector of x - the sizes of the reference set when performing
     *    kNN.
     * @param kData Vector of k - the kth nearest neighbor for which we
     *    calculated the statistic.
     * @param yData Matrix of y, one for each (x, k) value.
     */
    DefaultObjectiveFunction(const arma::Col<size_t>& xData, 
                             const arma::Col<size_t>& kData, 
                             const arma::mat& yData)
      : xData(&xData), kData(&kData), yData(&yData)  { };

    //! Return the number of functions
    size_t NumFunctions(void) const { return yData->n_elem; }

    //! Return a random starting point.
    arma::mat GetInitialPoint() const 
    { return arma::mat(3, 1, arma::fill::randu); }

    /**
     * This function evaluates the objective (MSE) at some coordinates with 
     * some index.
     * Called by the optimizer.
     *
     * @param coordinates Input matrix of coordinates. 
     */
    double Evaluate(const arma::mat& coordinates) const;

    /**
     * This function evaluates the gradient at some coordinates with some index.
     * Called by the optimizer.
     *
     * @param coordinates Input matrix of coordinates.
     * @param gradient Output matrix of gradients for each dimension of the
     *    surface
     */
    void Gradient(const arma::mat& coordinates, 
                  arma::mat& gradient) const;

  private:
    //! Data points for x-axis.
    const arma::Col<size_t>* xData;
    //! Data points for k-axis.
    const arma::Col<size_t>* kData;
    //! Data points for y-axis.
    const arma::mat* yData;
};

/**
 * Returns the value of the objective function for some coordinates (alpha,
 * beta, gamma).
 * This is the mean squared error for the current parameters or coordinates.
 */
double DefaultObjectiveFunction::Evaluate(const arma::mat& coordinates) const
{
  // Use extra variables to make code readable.
  double alpha = coordinates(0, 0);
  double beta = coordinates(1, 0);
  double gamma = coordinates(2, 0);
  double M = (double) NumFunctions();

  // Sum the squared error for each element in yData.
  double sum = 0;
  for (size_t i = 0; i < yData->n_elem; ++i)
  {
    // Map i to (row, col). Columnwise access of yData.
    size_t row = i % yData->n_rows;
    size_t col = (size_t) (i / yData->n_rows); // Integer division (floor).

    // Get the corresponding values.
    size_t x = (*xData)(row);
    size_t k = (*kData)(col);
    double y = (*yData)(row, col);

    // Evaluate (y - a * k ^ b * x ^ c)^2 for the given (x, y) pair.
    sum += pow(y - alpha * std::pow(k, beta) * std::pow(x, gamma), 2); 
  }

  // Return the mean of the squared errors.
  return sum / M;
}

/**
 * Stores the gradient of the objective function in gradient. This is the
 * derivative with respect to (alpha, beta, gamma) evaluated at the current
 * parameters.
 */
void DefaultObjectiveFunction::Gradient(const arma::mat& coordinates,
                                        arma::mat& gradient) const
{
  // Use extra variables to make code readable.
  double alpha = coordinates(0, 0);
  double beta = coordinates(1, 0);
  double gamma = coordinates(2, 0);
  double M = (double) NumFunctions();

  // Allocate 3x1 matrix for gradient. Set all gradients to 0.
  gradient.set_size(3, 1);
  gradient.zeros(3,1);

  // Sum each gradient.
  for (size_t i = 0; i < yData->n_elem; ++i)
  {
    size_t row = i % yData->n_rows;
    size_t col = (size_t) (i / yData->n_rows); // Integer division.
    size_t x = (*xData)(row);
    size_t k = (*kData)(col);
    double y = (*yData)(row, col);

    // The error for these parameters. Precompute for efficiency.
    double error = (y - alpha * std::pow(k, beta) * std::pow(x, gamma));

    // The chain rule factor of the product, for each gradient dimension.
    double alphaChain = 
      - 2.0 * std::pow(k, beta) * std::pow(x, gamma);

    double betaChain = 
      - 2.0 * alpha * std::pow(x, gamma) * std::log(k) * std::pow(k, beta);

    double gammaChain =
      - 2.0 * alpha * std::pow(k, beta) * std::log(x) * std::pow(x, gamma);

    // 3x1 column vector (in matrix form).
    gradient(0, 0) += error * alphaChain;
    gradient(1, 0) += error * betaChain;
    gradient(2, 0) += error * gammaChain;
  }

  // Return the average of each gradient after the summation is complete.
  gradient(0, 0) /= ((double) M);
  gradient(1, 0) /= ((double) M);
  gradient(2, 0) /= ((double) M);
}

} // namespace neighbor
} // namespace mlpack

#endif


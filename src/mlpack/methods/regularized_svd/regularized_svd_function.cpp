/**
 * @file regularized_svd_function.cpp
 * @author Siddharth Agrawal
 *
 * An implementation of the RegularizedSVDFunction class.
 *
 * This file is part of MLPACK 1.0.9.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "regularized_svd_function.hpp"

namespace mlpack {
namespace svd {

RegularizedSVDFunction::RegularizedSVDFunction(const arma::mat& data,
                                               const size_t rank,
                                               const double lambda) :
    data(data),
    rank(rank),
    lambda(lambda)
{
  // Number of users and items in the data.
  numUsers = max(data.row(0)) + 1;
  numItems = max(data.row(1)) + 1;
  
  // Initialize the parameters.
  initialPoint.randu(rank, numUsers + numItems);
}

double RegularizedSVDFunction::Evaluate(const arma::mat& parameters) const
{
  // The cost for the optimization is as follows:
  //          f(u, v) = sum((rating(i, j) - u(i).t() * v(j))^2)
  // The sum is over all the ratings in the rating matrix.
  // 'i' points to the user and 'j' points to the item being considered.
  // The regularization term is added to the above cost, where the vectors u(i)
  // and v(j) are regularized for each rating they contribute to.

  double cost = 0.0;

  for(size_t i = 0; i < data.n_cols; i++)
  {
    // Indices for accessing the the correct parameter columns.
    const size_t user = data(0, i);
    const size_t item = data(1, i) + numUsers;

    // Calculate the squared error in the prediction.
    const double rating = data(2, i);
    double ratingError = rating - arma::dot(parameters.col(user),
                                            parameters.col(item));
    double ratingErrorSquared = ratingError * ratingError;
  
    // Calculate the regularization penalty corresponding to the parameters.
    double userVecNorm = arma::norm(parameters.col(user), 2);
    double itemVecNorm = arma::norm(parameters.col(item), 2);
    double regularizationError = lambda * (userVecNorm * userVecNorm +
                                           itemVecNorm * itemVecNorm);
                                           
    cost += (ratingErrorSquared + regularizationError);
  }
  
  return cost;
}

double RegularizedSVDFunction::Evaluate(const arma::mat& parameters,
                                        const size_t i) const
{
  // Indices for accessing the the correct parameter columns.
  const size_t user = data(0, i);
  const size_t item = data(1, i) + numUsers;
  
  // Calculate the squared error in the prediction.
  const double rating = data(2, i);
  double ratingError = rating - arma::dot(parameters.col(user),
                                          parameters.col(item));
  double ratingErrorSquared = ratingError * ratingError;
  
  // Calculate the regularization penalty corresponding to the parameters.
  double userVecNorm = arma::norm(parameters.col(user), 2);
  double itemVecNorm = arma::norm(parameters.col(item), 2);
  double regularizationError = lambda * (userVecNorm * userVecNorm +
                                         itemVecNorm * itemVecNorm);
                                         
  return (ratingErrorSquared + regularizationError);
}

void RegularizedSVDFunction::Gradient(const arma::mat& parameters,
                                      arma::mat& gradient) const
{
  // For an example with rating corresponding to user 'i' and item 'j', the
  // gradients for the parameters is as follows:
  //           grad(u(i)) = lambda * u(i) - error * v(j)
  //           grad(v(j)) = lambda * v(j) - error * u(i)
  // 'error' is the prediction error for that example, which is:
  //           rating(i, j) - u(i).t() * v(j)
  // The full gradient is calculated by summing the contributions over all the
  // training examples.

  gradient.zeros(rank, numUsers + numItems);

  for(size_t i = 0; i < data.n_cols; i++)
  {
    // Indices for accessing the the correct parameter columns.
    const size_t user = data(0, i);
    const size_t item = data(1, i) + numUsers;

    // Prediction error for the example.
    const double rating = data(2, i);
    double ratingError = rating - arma::dot(parameters.col(user),
                                            parameters.col(item));

    // Gradient is non-zero only for the parameter columns corresponding to the
    // example.
    gradient.col(user) += 2 * (lambda * parameters.col(user) -
                               ratingError * parameters.col(item));
    gradient.col(item) += 2 * (lambda * parameters.col(item) -
                               ratingError * parameters.col(user));
  }
}

}; // namespace svd
}; // namespace mlpack

// Template specialization for the SGD optimizer.
namespace mlpack {
namespace optimization {

template<>
double SGD<mlpack::svd::RegularizedSVDFunction>::Optimize(arma::mat& parameters)
{
  // Find the number of functions to use.
  const size_t numFunctions = function.NumFunctions();

  // To keep track of where we are and how things are going.
  size_t currentFunction = 0;
  double overallObjective = 0;

  // Calculate the first objective function.
  for(size_t i = 0; i < numFunctions; i++)
    overallObjective += function.Evaluate(parameters, i);
    
  const arma::mat data = function.Dataset();

  // Now iterate!
  for(size_t i = 1; i != maxIterations; i++, currentFunction++)
  {
    // Is this iteration the start of a sequence?
    if((currentFunction % numFunctions) == 0)
    {
      // Reset the counter variables.
      overallObjective = 0;
      currentFunction = 0;
    }

    const size_t numUsers = function.NumUsers();

    // Indices for accessing the the correct parameter columns.
    const size_t user = data(0, currentFunction);
    const size_t item = data(1, currentFunction) + numUsers;

    // Prediction error for the example.
    const double rating = data(2, currentFunction);
    double ratingError = rating - arma::dot(parameters.col(user),
                                            parameters.col(item));
                                            
    double lambda = function.Lambda();

    // Gradient is non-zero only for the parameter columns corresponding to the
    // example.
    parameters.col(user) -= stepSize * (lambda * parameters.col(user) -
                                        ratingError * parameters.col(item));
    parameters.col(item) -= stepSize * (lambda * parameters.col(item) -
                                        ratingError * parameters.col(user));

    // Now add that to the overall objective function.
    overallObjective += function.Evaluate(parameters, currentFunction);
  }

  return overallObjective;
}

}; // namespace optimization
}; // namespace mlpack

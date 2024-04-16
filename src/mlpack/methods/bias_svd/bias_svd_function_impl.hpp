/**
 * @file methods/bias_svd/bias_svd_function_impl.hpp
 * @author Siddharth Agrawal
 * @author Wenhao Huang
 *
 * An implementation of the BiasSVDFunction class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_BIAS_SVD_BIAS_SVD_FUNCTION_IMPL_HPP
#define MLPACK_METHODS_BIAS_SVD_BIAS_SVD_FUNCTION_IMPL_HPP

#include "bias_svd_function.hpp"

namespace mlpack {

template <typename MatType>
BiasSVDFunction<MatType>::BiasSVDFunction(const MatType& dataIn,
                                          const size_t rank,
                                          const double lambda) :
    rank(rank),
    lambda(lambda)
{
  MakeAlias(data, dataIn, dataIn.n_rows, dataIn.n_cols, false);

  // Number of users and items in the data.
  numUsers = max(data.row(0)) + 1;
  numItems = max(data.row(1)) + 1;

  // Initialize the parameters.
  // The last row in initialPoint saves use/item bias.
  initialPoint.randu(rank + 1, numUsers + numItems);
}

template<typename MatType>
void BiasSVDFunction<MatType>::Shuffle()
{
  data = data.cols(arma::shuffle(arma::linspace<arma::uvec>(0, data.n_cols - 1,
      data.n_cols)));
}

template <typename MatType>
double BiasSVDFunction<MatType>::Evaluate(const MatType& parameters) const
{
  return Evaluate(parameters, 0, data.n_cols);
}

template <typename MatType>
double BiasSVDFunction<MatType>::Evaluate(const MatType& parameters,
                                          const size_t start,
                                          const size_t batchSize) const
{
  // The cost for the optimization is as follows:
  //    f(u, v, p ,q) = sum((rating(i, j) - p(i) - q(j) - u(i).t() * v(j))^2)
  // The sum is over all the ratings in the rating matrix.
  // 'i' points to the user and 'j' points to the item being considered.
  // The regularization term is added to the above cost, where the vectors u(i)
  // and v(j), bias p(i) and q(j) are regularized for each rating they
  // contribute to.

  // It's possible this loop could be changed so that it's SIMD-vectorized.
  double objective = 0.0;
  for (size_t i = start; i < start + batchSize; ++i)
  {
    // Indices for accessing the the correct parameter columns.
    const size_t user = data(0, i);
    const size_t item = data(1, i) + numUsers;

    // Calculate the squared error in the prediction.
    const double rating = data(2, i);
    const double userBias = parameters(rank, user);
    const double itemBias = parameters(rank, item);
    double ratingError = rating - userBias - itemBias -
        dot(parameters.col(user).subvec(0, rank - 1),
                  parameters.col(item).subvec(0, rank - 1));
    double ratingErrorSquared = ratingError * ratingError;

    // Calculate the regularization penalty corresponding to the parameters.
    double userVecNorm = norm(parameters.col(user), 2);
    double itemVecNorm = norm(parameters.col(item), 2);
    double regularizationError = lambda * (userVecNorm * userVecNorm +
                                           itemVecNorm * itemVecNorm);

    objective += (ratingErrorSquared + regularizationError);
  }

  return objective;
}

template <typename MatType>
void BiasSVDFunction<MatType>::Gradient(const MatType& parameters,
                                        MatType& gradient) const
{
  // For an example with rating corresponding to user 'i' and item 'j', the
  // gradients for the parameters is as follows:
  //           grad(u(i)) = 2 * (lambda * u(i) - error * v(j))
  //           grad(v(j)) = 2 * (lambda * v(j) - error * u(i))
  //           grad(p(i)) = 2 * (lambda * p(i) - error)
  //           grad(q(j)) = 2 * (lambda * q(j) - error)
  // 'error' is the prediction error for that example, which is:
  //           rating(i, j) - p(i) - q(j) - u(i).t() * v(j)
  // The full gradient is calculated by summing the contributions over all the
  // training examples.

  gradient.zeros(rank + 1, numUsers + numItems);

  for (size_t i = 0; i < data.n_cols; ++i)
  {
    // Indices for accessing the the correct parameter columns.
    const size_t user = data(0, i);
    const size_t item = data(1, i) + numUsers;

    // Prediction error for the example.
    const double rating = data(2, i);
    const double userBias = parameters(rank, user);
    const double itemBias = parameters(rank, item);
    double ratingError = rating - userBias - itemBias -
        dot(parameters.col(user).subvec(0, rank - 1),
                  parameters.col(item).subvec(0, rank - 1));

    // Gradient is non-zero only for the parameter columns corresponding to the
    // example.
    gradient.col(user).subvec(0, rank - 1) +=
        2 * (lambda * parameters.col(user).subvec(0, rank - 1) -
        ratingError * parameters.col(item).subvec(0, rank - 1));
    gradient.col(item).subvec(0, rank - 1) +=
        2 * (lambda * parameters.col(item).subvec(0, rank - 1) -
        ratingError * parameters.col(user).subvec(0, rank - 1));
    gradient(rank, user) +=
        2 * (lambda * parameters(rank, user) - ratingError);
    gradient(rank, item) +=
        2 * (lambda * parameters(rank, item) - ratingError);
  }
}

template <typename MatType>
template <typename GradType>
void BiasSVDFunction<MatType>::Gradient(const MatType& parameters,
                                        const size_t start,
                                        GradType& gradient,
                                        const size_t batchSize) const
{
  gradient.zeros(rank + 1, numUsers + numItems);

  // It's possible this could be SIMD-vectorized for additional speedup.
  for (size_t i = start; i < start + batchSize; ++i)
  {
    const size_t user = data(0, i);
    const size_t item = data(1, i) + numUsers;

    // Prediction error for the example.
    const double rating = data(2, i);
    const double userBias = parameters(rank, user);
    const double itemBias = parameters(rank, item);
    double ratingError = rating - userBias - itemBias -
        dot(parameters.col(user).subvec(0, rank - 1),
                  parameters.col(item).subvec(0, rank - 1));

    // Gradient is non-zero only for the parameter columns corresponding to the
    // example.
    for (size_t j = 0; j < rank; ++j)
    {
      gradient(j, user) +=
          2 * (lambda * parameters(j, user) -
          ratingError * parameters(j, item));
      gradient(j, item) +=
          2 * (lambda * parameters(j, item) -
          ratingError * parameters(j, user));
    }
    gradient(rank, user) +=
        2 * (lambda * parameters(rank, user) - ratingError);
    gradient(rank, item) +=
        2 * (lambda * parameters(rank, item) - ratingError);
  }
}

} // namespace mlpack

// Template specialization for the SGD optimizer.
namespace ens {

template <>
template <>
double StandardSGD::Optimize(
    mlpack::BiasSVDFunction<arma::mat>& function,
    arma::mat& parameters)
{
  // Find the number of functions to use.
  const size_t numFunctions = function.NumFunctions();

  // To keep track of where we are and how things are going.
  size_t currentFunction = 0;
  double overallObjective = 0;

  // Calculate the first objective function.
  for (size_t i = 0; i < numFunctions; ++i)
    overallObjective += function.Evaluate(parameters, i);

  const arma::mat data = function.Dataset();

  // Rank of decomposition.
  const size_t rank = function.Rank();

  // Now iterate!
  for (size_t i = 1; i != maxIterations; ++i, currentFunction++)
  {
    // Is this iteration the start of a sequence?
    if ((currentFunction % numFunctions) == 0)
    {
      const size_t epoch = i / numFunctions + 1;
      mlpack::Log::Info << "Epoch " << epoch << "; " << "objective "
          << overallObjective << "." << std::endl;

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
    const double userBias = parameters(rank, user);
    const double itemBias = parameters(rank, item);
    double ratingError = rating - userBias - itemBias -
        dot(parameters.col(user).subvec(0, rank - 1),
                  parameters.col(item).subvec(0, rank - 1));

    double lambda = function.Lambda();

    // Gradient is non-zero only for the parameter columns corresponding to the
    // example.
    parameters.col(user).subvec(0, rank - 1) -= stepSize * 2 *(
        lambda * parameters.col(user).subvec(0, rank - 1) -
        ratingError * parameters.col(item).subvec(0, rank - 1));
    parameters.col(item).subvec(0, rank - 1) -= stepSize * 2 * (
        lambda * parameters.col(item).subvec(0, rank - 1) -
        ratingError * parameters.col(user).subvec(0, rank - 1));
    parameters(rank, user) -= stepSize * 2 * (
        lambda * parameters(rank, user) - ratingError);
    parameters(rank, item) -= stepSize * 2 * (
        lambda * parameters(rank, item) - ratingError);

    // Now add that to the overall objective function.
    overallObjective += function.Evaluate(parameters, currentFunction);
  }

  return overallObjective;
}


template <>
template <>
inline double ParallelSGD<ExponentialBackoff>::Optimize(
    mlpack::BiasSVDFunction<arma::mat>& function,
    arma::mat& iterate)
{
  double overallObjective = DBL_MAX;
  double lastObjective;

  // The order in which the functions will be visited.
  arma::Col<size_t> visitationOrder = arma::linspace<arma::Col<size_t>>(0,
      (function.NumFunctions() - 1), function.NumFunctions());

  const arma::mat data = function.Dataset();
  const size_t numUsers = function.NumUsers();
  const double lambda = function.Lambda();

  // Rank of decomposition.
  const size_t rank = function.Rank();

  // Iterate till the objective is within tolerance or the maximum number of
  // allowed iterations is reached. If maxIterations is 0, this will iterate
  // till convergence.
  for (size_t i = 1; i != maxIterations; ++i)
  {
    // Calculate the overall objective.
    lastObjective = overallObjective;
    overallObjective = 0;

    #pragma omp parallel for reduction(+:overallObjective)
    for (size_t j = 0; j < (size_t) function.NumFunctions(); ++j)
    {
      overallObjective += function.Evaluate(iterate, j);
    }

    // Output current objective function.
    mlpack::Log::Info << "Parallel SGD: iteration " << i << ", objective "
        << overallObjective << "." << std::endl;

    if (std::isnan(overallObjective) || std::isinf(overallObjective))
    {
      mlpack::Log::Warn << "Parallel SGD: converged to " << overallObjective
          << "; terminating with failure. Try a smaller step size?"
          << std::endl;
      return overallObjective;
    }

    if (std::abs(lastObjective - overallObjective) < tolerance)
    {
      mlpack::Log::Info << "SGD: minimized within tolerance " << tolerance
          << "; terminating optimization." << std::endl;
      return overallObjective;
    }

    // Get the stepsize for this iteration
    double stepSize = decayPolicy.StepSize(i);

    if (shuffle) // Determine order of visitation.
      std::shuffle(visitationOrder.begin(), visitationOrder.end(),
          mlpack::RandGen());

    #pragma omp parallel
    {
      // Each processor gets a subset of the instances.
      // Each subset is of size threadShareSize.
      size_t threadId = 0;
      #ifdef MLPACK_USE_OPENMP
        threadId = omp_get_thread_num();
      #endif

      for (size_t j = threadId * threadShareSize;
          j < (threadId + 1) * threadShareSize && j < visitationOrder.n_elem;
          ++j)
      {
        // Indices for accessing the the correct parameter columns.
        const size_t user = data(0, visitationOrder[j]);
        const size_t item = data(1, visitationOrder[j]) + numUsers;

        // Prediction error for the example.
        const double rating = data(2, visitationOrder[j]);
        const double userBias = iterate(rank, user);
        const double itemBias = iterate(rank, item);
        double ratingError = rating - userBias - itemBias -
            dot(iterate.col(user).subvec(0, rank - 1),
                      iterate.col(item).subvec(0, rank - 1));

        arma::mat userVecUpdate = stepSize * 2 * (
            lambda * iterate.col(user).subvec(0, rank - 1) -
            ratingError * iterate.col(item).subvec(0, rank - 1));
        arma::mat itemVecUpdate = stepSize * 2 * (
            lambda * iterate.col(item).subvec(0, rank - 1) -
            ratingError * iterate.col(user).subvec(0, rank - 1));
        double userBiasUpdate = stepSize * 2 * (
            lambda * iterate(rank, user) - ratingError);
        double itemBiasUpdate = stepSize * 2 * (
            lambda * iterate(rank, item) - ratingError);

        // Gradient is non-zero only for the parameter columns corresponding to
        // the example.
        for (size_t i = 0; i < rank; ++i)
        {
          #pragma omp atomic
          iterate(i, user) -= userVecUpdate(i);
          #pragma omp atomic
          iterate(i, item) -= itemVecUpdate(i);
        }
        #pragma omp atomic
        iterate(rank, user) -= userBiasUpdate;
        #pragma omp atomic
        iterate(rank, item) -= itemBiasUpdate;
      }
    }
  }
  mlpack::Log::Info << "\n Parallel SGD terminated with objective : "
    << overallObjective << std::endl;

  return overallObjective;
}

} // namespace ens

#endif

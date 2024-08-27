/**
 * @file methods/svdplusplus/svdplusplus_function_impl.hpp
 * @author Siddharth Agrawal
 * @author Wenhao Huang
 *
 * An implementation of the SVDPlusPlusFunction class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_SVDPLUSPLUS_SVDPLUSPLUS_FUNCTION_IMPL_HPP
#define MLPACK_METHODS_SVDPLUSPLUS_SVDPLUSPLUS_FUNCTION_IMPL_HPP

#include "svdplusplus_function.hpp"
#include <mlpack/core/math/make_alias.hpp>

namespace mlpack {

template <typename MatType>
SVDPlusPlusFunction<MatType>::SVDPlusPlusFunction(
    const MatType& dataIn,
    const arma::sp_mat& implicitData,
    const size_t rank,
    const double lambda) :
    implicitData(implicitData),
    rank(rank),
    lambda(lambda)
{
  MakeAlias(data, dataIn, dataIn.n_rows, dataIn.n_cols, false);

  // Number of users and items in the data.
  numUsers = max(data.row(0)) + 1;
  numItems = max(data.row(1)) + 1;

  // Initialize the parameters.
  // Item matrix: submat(0, numUsers, rank - 1, numUsers + numItems - 1).t()
  // User matrix: submat(0, 0, rank - 1, numUsers - 1)
  // Item bias: row(rank).subvec(numUsers, numUsers + numItems - 1)
  // User bias: row(rank).subvec(0, numUsers - 1)
  // Item implicit matrix: submat(0, numUsers + numItems,
  //     rank - 1, numUsers + 2 * numItems - 1)
  // Unused:
  //     row(rank).subvec(numUsers + numItems, numUsers + 2 * numItems - 1)
  initialPoint.randu(rank + 1, numUsers + 2 * numItems);
}

template<typename MatType>
void SVDPlusPlusFunction<MatType>::Shuffle()
{
  data = data.cols(arma::shuffle(arma::linspace<arma::uvec>(0, data.n_cols - 1,
      data.n_cols)));
}

template <typename MatType>
double SVDPlusPlusFunction<MatType>::Evaluate(const arma::mat& parameters) const
{
  return Evaluate(parameters, 0, data.n_cols);
}

template <typename MatType>
double SVDPlusPlusFunction<MatType>::Evaluate(const arma::mat& parameters,
                                              const size_t start,
                                              const size_t batchSize) const
{
  // The cost for the optimization is as follows:
  //    f(u, v, p, q, y) =
  //        sum((rating(i, j) - p(i) - q(j) - u(i).t() * (v(j) + sum(y(k))))^2)
  // The sum is over all the ratings in the rating matrix.
  // 'i' points to the user and 'j' points to the item being considered.
  // 'k' points to the items which user 'i' interacted with.
  // The regularization term is added to the above cost, where the vectors u(i)
  // and v(j), bias p(i) and q(j), implicit vectors y(k) are regularized for
  // each rating they contribute to.

  // It's possible this loop could be changed so that it's SIMD-vectorized.
  double objective = 0.0;

  // The norm square of implicit item vectors is cached to avoid repeated
  // calculation.
  arma::vec implicitVecsNormSquare(numItems);
  implicitVecsNormSquare.fill(-1);

  for (size_t i = start; i < start + batchSize; ++i)
  {
    // Indices for accessing the the correct parameter columns.
    const size_t user = data(0, i);
    const size_t item = data(1, i) + numUsers;
    const size_t implicitStart = numUsers + numItems;

    // Calculate the squared error in the prediction.
    const double rating = data(2, i);
    const double userBias = parameters(rank, user);
    const double itemBias = parameters(rank, item);

    // Iterate through each item which the user interacted with to calculate
    // user vector.
    arma::vec userVec(rank);
    arma::sp_mat::const_iterator it = implicitData.begin_col(user);
    arma::sp_mat::const_iterator it_end = implicitData.end_col(user);
    size_t implicitCount = 0;
    double regularizationError = 0;
    for (; it != it_end; ++it)
    {
      userVec += parameters.col(implicitStart + it.row()).subvec(0, rank - 1);
      if (implicitVecsNormSquare(it.row()) < 0)
      {
        implicitVecsNormSquare(it.row()) = dot(
            parameters.col(implicitStart + it.row()).subvec(0, rank - 1),
            parameters.col(implicitStart + it.row()).subvec(0, rank - 1));
      }
      regularizationError += lambda * implicitVecsNormSquare(it.row());
      implicitCount += 1;
    }
    if (implicitCount != 0)
    {
      userVec /= std::sqrt(implicitCount);
      regularizationError /= implicitCount;
    }
    userVec += parameters.col(user).subvec(0, rank - 1);

    double ratingError = rating - userBias - itemBias -
        dot(userVec, parameters.col(item).subvec(0, rank - 1));
    double ratingErrorSquared = ratingError * ratingError;

    // Calculate the regularization penalty corresponding to the parameters.
    double userVecNorm = norm(parameters.col(user), 2);
    double itemVecNorm = norm(parameters.col(item), 2);
    regularizationError += lambda * (userVecNorm * userVecNorm +
                                     itemVecNorm * itemVecNorm);

    objective += (ratingErrorSquared + regularizationError);
  }

  return objective;
}

template <typename MatType>
void SVDPlusPlusFunction<MatType>::Gradient(const arma::mat& parameters,
                                            arma::mat& gradient) const
{
  // For an example with rating corresponding to user 'i' and item 'j', the
  // gradients for the parameters is as follows:
  //           uservec = v(j) + sum(y(k))
  //           grad(u(i)) = 2 * (lambda * u(i) - error * v(j))
  //           grad(v(j)) = 2 * (lambda * v(j) - error * uservec)
  //           grad(p(i)) = 2 * (lambda * p(i) - error)
  //           grad(q(j)) = 2 * (lambda * q(j) - error)
  //           grad(y(k)) = 2 * (lambda * y(k) - error / sqrt(N(u)) * v(j))
  // 'error' is the prediction error for that example, which is:
  //           rating(i, j) - p(i) - q(j) - u(i).t() * (v(j) + sum(y(k)))
  // The full gradient is calculated by summing the contributions over all the
  // training examples.

  gradient.zeros(rank + 1, numUsers + 2 * numItems);

  for (size_t i = 0; i < data.n_cols; ++i)
  {
    // Indices for accessing the the correct parameter columns.
    const size_t user = data(0, i);
    const size_t item = data(1, i) + numUsers;
    const size_t implicitStart = numUsers + numItems;

    // Calculate the squared error in the prediction.
    const double rating = data(2, i);
    const double userBias = parameters(rank, user);
    const double itemBias = parameters(rank, item);

    // Iterate through each item which the user interacted with to calculate
    // user vector.
    arma::vec userVec(rank);
    arma::sp_mat::const_iterator it = implicitData.begin_col(user);
    arma::sp_mat::const_iterator it_end = implicitData.end_col(user);
    size_t implicitCount = 0;
    for (; it != it_end; ++it)
    {
      userVec += parameters.col(implicitStart + it.row()).subvec(0, rank - 1);
      implicitCount += 1;
    }
    if (implicitCount != 0)
      userVec /= std::sqrt(implicitCount);
    userVec += parameters.col(user).subvec(0, rank - 1);

    double ratingError = rating - userBias - itemBias -
        dot(userVec, parameters.col(item).subvec(0, rank - 1));

    // Gradient is non-zero only for the parameter columns corresponding to the
    // example.
    gradient.col(user).subvec(0, rank - 1) +=
        2 * (lambda * parameters.col(user).subvec(0, rank - 1) -
        ratingError * parameters.col(item).subvec(0, rank - 1));
    gradient.col(item).subvec(0, rank - 1) +=
        2 * (lambda * parameters.col(item).subvec(0, rank - 1) -
        ratingError * userVec);
    gradient(rank, user) +=
        2 * (lambda * parameters(rank, user) - ratingError);
    gradient(rank, item) +=
        2 * (lambda * parameters(rank, item) - ratingError);
    // Calculate gradients for item implicit vector.
    it = implicitData.begin_col(user);
    it_end = implicitData.end_col(user);
    for (; it != it_end; ++it)
    {
      // Note that implicitCount != 0 if this loop is acutally executed.
      gradient.col(implicitStart + it.row()).subvec(0, rank - 1) +=
          2.0 * (lambda / implicitCount *
          parameters.col(implicitStart + it.row()).subvec(0, rank - 1) -
          ratingError / std::sqrt(implicitCount) *
          parameters.col(item).subvec(0, rank - 1));
    }
  }
}

template <typename MatType>
template <typename GradType>
void SVDPlusPlusFunction<MatType>::Gradient(const arma::mat& parameters,
                                            const size_t start,
                                            GradType& gradient,
                                            const size_t batchSize) const
{
  gradient.zeros(rank + 1, numUsers + 2 * numItems);

  // It's possible this could be SIMD-vectorized for additional speedup.
  for (size_t i = start; i < start + batchSize; ++i)
  {
    // Indices for accessing the the correct parameter columns.
    const size_t user = data(0, i);
    const size_t item = data(1, i) + numUsers;
    const size_t implicitStart = numUsers + numItems;

    // Calculate the squared error in the prediction.
    const double rating = data(2, i);
    const double userBias = parameters(rank, user);
    const double itemBias = parameters(rank, item);

    // Iterate through each item which the user interacted with to calculate
    // user vector.
    arma::vec userVec(rank);
    arma::sp_mat::const_iterator it = implicitData.begin_col(user);
    arma::sp_mat::const_iterator it_end = implicitData.end_col(user);
    size_t implicitCount = 0;
    for (; it != it_end; ++it)
    {
      userVec += parameters.col(implicitStart + it.row()).subvec(0, rank - 1);
      implicitCount += 1;
    }
    if (implicitCount != 0)
      userVec /= std::sqrt(implicitCount);
    userVec += parameters.col(user).subvec(0, rank - 1);

    double ratingError = rating - userBias - itemBias -
        dot(userVec, parameters.col(item).subvec(0, rank - 1));

    // Gradient is non-zero only for the parameter columns corresponding to the
    // example.
    for (size_t j = 0; j < rank; ++j)
    {
      gradient(j, user) +=
          2 * (lambda * parameters(j, user) -
          ratingError * parameters(j, item));
      gradient(j, item) +=
          2 * (lambda * parameters(j, item) -
          ratingError * userVec(j));
    }
    gradient(rank, user) +=
        2 * (lambda * parameters(rank, user) - ratingError);
    gradient(rank, item) +=
        2 * (lambda * parameters(rank, item) - ratingError);
    // Calculate gradients for item implicit vector.
    it = implicitData.begin_col(user);
    it_end = implicitData.end_col(user);
    for (; it != it_end; ++it)
    {
      // Note that implicitCount != 0 if this loop is acutally executed.
      for (size_t j = 0; j < rank; ++j)
      {
        gradient(j, implicitStart + it.row()) +=
            2.0 * (lambda / implicitCount *
            parameters(j, implicitStart + it.row()) -
            ratingError / std::sqrt(implicitCount) *
            parameters(j, item));
      }
    }
  }
}

} // namespace mlpack

// Template specialization for the SGD optimizer.
namespace ens {

template <>
template <>
double StandardSGD::Optimize(
    mlpack::SVDPlusPlusFunction<arma::mat>& function,
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
  const arma::sp_mat implicitData = function.ImplicitDataset();
  const size_t numUsers = function.NumUsers();
  const size_t numItems = function.NumItems();
  const double lambda = function.Lambda();

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

    // Indices for accessing the the correct parameter columns.
    const size_t user = data(0, currentFunction);
    const size_t item = data(1, currentFunction) + numUsers;
    const size_t implicitStart = numUsers + numItems;

    // Calculate the squared error in the prediction.
    const double rating = data(2, currentFunction);
    const double userBias = parameters(rank, user);
    const double itemBias = parameters(rank, item);

    // Iterate through each item which the user interacted with to calculate
    // user vector.
    arma::vec userVec(rank);
    arma::sp_mat::const_iterator it = implicitData.begin_col(user);
    arma::sp_mat::const_iterator it_end = implicitData.end_col(user);
    size_t implicitCount = 0;
    for (; it != it_end; ++it)
    {
      userVec += parameters.col(implicitStart + it.row()).subvec(0, rank - 1);
      implicitCount += 1;
    }
    if (implicitCount != 0)
      userVec /= std::sqrt(implicitCount);
    userVec += parameters.col(user).subvec(0, rank - 1);

    double ratingError = rating - userBias - itemBias -
        dot(userVec, parameters.col(item).subvec(0, rank - 1));

    // Gradient is non-zero only for the parameter columns corresponding to the
    // example.
    parameters.col(user).subvec(0, rank - 1) -= stepSize * 2 * (
        lambda * parameters.col(user).subvec(0, rank - 1) -
        ratingError * parameters.col(item).subvec(0, rank - 1));
    parameters.col(item).subvec(0, rank - 1) -= stepSize * 2 * (
        lambda * parameters.col(item).subvec(0, rank - 1) -
        ratingError * userVec);
    parameters(rank, user) -= stepSize * 2 * (
        lambda * parameters(rank, user) - ratingError);
    parameters(rank, item) -= stepSize * 2 * (
        lambda * parameters(rank, item) - ratingError);
    // Update item implicit vectors.
    it = implicitData.begin_col(user);
    it_end = implicitData.end_col(user);
    for (; it != it_end; ++it)
    {
      // Note that implicitCount != 0 if this loop is acutally executed.
      parameters.col(implicitStart + it.row()).subvec(0, rank - 1) -=
          stepSize * 2.0 * (lambda / implicitCount *
          parameters.col(implicitStart + it.row()).subvec(0, rank - 1) -
          ratingError / std::sqrt(implicitCount) *
          parameters.col(item).subvec(0, rank - 1));
    }

    // Now add that to the overall objective function.
    overallObjective += function.Evaluate(parameters, currentFunction);
  }

  return overallObjective;
}


template <>
template <>
inline double ParallelSGD<ExponentialBackoff>::Optimize(
    mlpack::SVDPlusPlusFunction<arma::mat>& function,
    arma::mat& iterate)
{
  double overallObjective = DBL_MAX;
  double lastObjective;

  // The order in which the functions will be visited.
  arma::Col<size_t> visitationOrder = arma::linspace<arma::Col<size_t>>(0,
      (function.NumFunctions() - 1), function.NumFunctions());

  const arma::mat data = function.Dataset();
  const arma::sp_mat implicitData = function.ImplicitDataset();
  const size_t numUsers = function.NumUsers();
  const size_t numItems = function.NumItems();
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
        const size_t implicitStart = numUsers + numItems;

        // Prediction error for the example.
        const double rating = data(2, visitationOrder[j]);
        const double userBias = iterate(rank, user);
        const double itemBias = iterate(rank, item);
        // Iterate through each item which the user interacted with to calculate
        // user vector.
        arma::vec userVec(rank);
        arma::sp_mat::const_iterator it = implicitData.begin_col(user);
        arma::sp_mat::const_iterator it_end = implicitData.end_col(user);
        size_t implicitCount = 0;
        for (; it != it_end; ++it)
        {
          userVec += iterate.col(implicitStart + it.row()).subvec(0, rank - 1);
          implicitCount += 1;
        }
        if (implicitCount != 0)
          userVec /= std::sqrt(implicitCount);
        userVec += iterate.col(user).subvec(0, rank - 1);

        double ratingError = rating - userBias - itemBias -
        dot(userVec, iterate.col(item).subvec(0, rank - 1));

        arma::mat userVecUpdate = stepSize * 2 * (
            lambda * iterate.col(user).subvec(0, rank - 1) -
            ratingError * iterate.col(item).subvec(0, rank - 1));
        arma::mat itemVecUpdate = stepSize * 2 * (
            lambda * iterate.col(item).subvec(0, rank - 1) -
            ratingError * userVec);
        double userBiasUpdate = stepSize * 2 * (
            lambda * iterate(rank, user) - ratingError);
        double itemBiasUpdate = stepSize * 2 * (
            lambda * iterate(rank, item) - ratingError);

        // Update of item implicit vectors.
        arma::mat itemImplicitUpdate(rank, implicitCount);
        arma::Col<size_t> implicitItems(implicitCount);
        it = implicitData.begin_col(user);
        it_end = implicitData.end_col(user);
        size_t implicitIndex = 0;
        for (; it != it_end; ++it, ++implicitIndex)
        {
          itemImplicitUpdate.col(implicitIndex) =
              stepSize * 2.0 * (lambda / implicitCount *
              iterate.col(implicitStart + it.row()).subvec(0, rank - 1) -
              ratingError / std::sqrt(implicitCount) *
              iterate.col(item).subvec(0, rank - 1));
          implicitItems(implicitIndex) = it.row();
        }

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
        for (size_t k = 0; k < implicitCount; ++k)
        {
          for (size_t i = 0; i < rank; ++i)
          {
            #pragma omp atomic
            iterate(i, implicitStart + implicitItems(k)) -=
                itemImplicitUpdate(i, k);
          }
        }
      }
    }
  }
  mlpack::Log::Info << "\n Parallel SGD terminated with objective : "
      << overallObjective << std::endl;

  return overallObjective;
}

} // namespace ens

#endif

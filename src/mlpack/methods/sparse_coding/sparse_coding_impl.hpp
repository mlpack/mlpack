/**
 * @file sparse_coding_impl.hpp
 * @author Nishant Mehta
 *
 * Implementation of Sparse Coding with Dictionary Learning using l1 (LASSO) or
 * l1+l2 (Elastic Net) regularization.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_SPARSE_CODING_SPARSE_CODING_IMPL_HPP
#define MLPACK_METHODS_SPARSE_CODING_SPARSE_CODING_IMPL_HPP

// In case it hasn't already been included.
#include "sparse_coding.hpp"

namespace mlpack {
namespace sparse_coding {

template<typename DictionaryInitializer>
SparseCoding::SparseCoding(
    const arma::mat& data,
    const size_t atoms,
    const double lambda1,
    const double lambda2,
    const size_t maxIterations,
    const double objTolerance,
    const double newtonTolerance,
    const DictionaryInitializer& initializer) :
    atoms(atoms),
    lambda1(lambda1),
    lambda2(lambda2),
    maxIterations(maxIterations),
    objTolerance(objTolerance),
    newtonTolerance(newtonTolerance)
{
  Train(data, initializer);
}

template<typename DictionaryInitializer>
void SparseCoding::Train(
    const arma::mat& data,
    const DictionaryInitializer& initializer)
{
  // Now, train.
  Timer::Start("sparse_coding");

  // Initialize the dictionary.
  initializer.Initialize(data, atoms, dictionary);

  double lastObjVal = DBL_MAX;

  // Take the initial coding step, which has to happen before entering the main
  // optimization loop.
  Log::Info << "Initial coding step." << std::endl;

  arma::mat codes(atoms, data.n_cols);
  Encode(data, codes);
  arma::uvec adjacencies = find(codes);

  Log::Info << "  Sparsity level: " << 100.0 * ((double) (adjacencies.n_elem))
      / ((double) (atoms * data.n_cols)) << "%." << std::endl;
  Log::Info << "  Objective value: " << Objective(data, codes) << "."
      << std::endl;

  for (size_t t = 1; t != maxIterations; ++t)
  {
    // Print current iteration, and maximum number of iterations (if it isn't
    // 0).
    Log::Info << "Iteration " << t;
    if (maxIterations != 0)
      Log::Info << " of " << maxIterations;
    Log::Info << "." << std::endl;

    // First step: optimize the dictionary.
    Log::Info << "Performing dictionary step... " << std::endl;
    OptimizeDictionary(data, codes, adjacencies);
    Log::Info << "  Objective value: " << Objective(data, codes) << "."
        << std::endl;

    // Second step: perform the coding.
    Log::Info << "Performing coding step..." << std::endl;
    Encode(data, codes);
    // Get the indices of all the nonzero elements in the codes.
    adjacencies = find(codes);
    Log::Info << "  Sparsity level: " << 100.0 * ((double) (adjacencies.n_elem))
        / ((double) (atoms * data.n_cols)) << "%." << std::endl;

    // Find the new objective value and improvement so we can check for
    // convergence.
    double curObjVal = Objective(data, codes);
    double improvement = lastObjVal - curObjVal;
    Log::Info << "  Objective value: " << curObjVal << " (improvement "
        << std::scientific << improvement << ")." << std::endl;

    // Have we converged?
    if (improvement < objTolerance)
    {
      Log::Info << "Converged within tolerance " << objTolerance << ".\n";
      break;
    }

    lastObjVal = curObjVal;
  }

  Timer::Stop("sparse_coding");
}

template<typename Archive>
void SparseCoding::Serialize(Archive& ar, const unsigned int /* version */)
{
  ar & data::CreateNVP(atoms, "atoms");
  ar & data::CreateNVP(dictionary, "dictionary");
  ar & data::CreateNVP(lambda1, "lambda1");
  ar & data::CreateNVP(lambda2, "lambda2");
  ar & data::CreateNVP(maxIterations, "maxIterations");
  ar & data::CreateNVP(objTolerance, "objTolerance");
  ar & data::CreateNVP(newtonTolerance, "newtonTolerance");
}

} // namespace sparse_coding
} // namespace mlpack

#endif

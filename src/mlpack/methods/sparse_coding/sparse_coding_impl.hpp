/**
 * @file methods/sparse_coding/sparse_coding_impl.hpp
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

inline SparseCoding::SparseCoding(
    const size_t atoms,
    const double lambda1,
    const double lambda2,
    const size_t maxIterations,
    const double objTolerance,
    const double newtonTolerance) :
    atoms(atoms),
    lambda1(lambda1),
    lambda2(lambda2),
    maxIterations(maxIterations),
    objTolerance(objTolerance),
    newtonTolerance(newtonTolerance)
{
  // Nothing to do.
}

inline void SparseCoding::Encode(const arma::mat& data,
                                 arma::mat& codes)
{
  // When using the Cholesky version of LARS, this is correct even if
  // lambda2 > 0.
  arma::mat matGram = trans(dictionary) * dictionary;

  codes.set_size(atoms, data.n_cols);
  for (size_t i = 0; i < data.n_cols; ++i)
  {
    // Report progress.
    if ((i % 100) == 0)
      Log::Debug << "Optimization at point " << i << "." << std::endl;

    bool useCholesky = true;
    // Intercept fitting and data normalization is disabled.
    LARS<> lars(useCholesky, lambda1, lambda2, 1e-16 /* default tolerance */,
        false, false);

    // Create an alias of the code (using the same memory), and then LARS will
    // place the result directly into that; then we will not need to have an
    // extra copy.
    arma::vec code = codes.unsafe_col(i);
    arma::rowvec responses = data.unsafe_col(i).t();
    lars.Train(dictionary, responses, false, useCholesky, matGram);
    code = lars.Beta();
  }
}

// Dictionary step for optimization.
inline double SparseCoding::OptimizeDictionary(const arma::mat& data,
                                               const arma::mat& codes,
                                               const arma::uvec& adjacencies)
{
  // Count the number of atomic neighbors for each point x^i.
  arma::uvec neighborCounts = zeros<arma::uvec>(data.n_cols, 1);

  if (adjacencies.n_elem > 0)
  {
    // This gets the column index.  Intentional integer division.
    size_t curPointInd = (size_t) (adjacencies(0) / atoms);

    size_t nextColIndex = (curPointInd + 1) * atoms;
    for (size_t l = 1; l < adjacencies.n_elem; ++l)
    {
      // If l no longer refers to an element in this column, advance the column
      // number accordingly.
      if (adjacencies(l) >= nextColIndex)
      {
        curPointInd = (size_t) (adjacencies(l) / atoms);
        nextColIndex = (curPointInd + 1) * atoms;
      }

      ++neighborCounts(curPointInd);
    }
  }

  // Handle the case of inactive atoms (atoms not used in the given coding).
  std::vector<arma::uword> activeAtoms;
  for (arma::uword j = 0; j < atoms; ++j)
  {
    if (arma::any(codes.row(j) != 0))
      activeAtoms.push_back(j);
  }

  const size_t nActiveAtoms = activeAtoms.size();
  const size_t nInactiveAtoms = atoms - nActiveAtoms;

  // Efficient construction of Z restricted to active atoms.
  arma::mat matActiveZ;
  if (nInactiveAtoms > 0)
  {
    matActiveZ = codes.rows(arma::uvec(activeAtoms));
  }

  if (nInactiveAtoms > 0)
  {
    Log::Warn << "There are " << nInactiveAtoms
        << " inactive atoms. They will be re-initialized randomly.\n";
  }

  Log::Debug << "Solving Dual via Newton's Method.\n";

  // Solve using Newton's method in the dual - note that the final dot
  // multiplication with inv(A) seems to be unavoidable. Although more
  // expensive, the code written this way (we use solve()) should be more
  // numerically stable than just using inv(A) for everything.
  arma::vec dualVars = zeros<arma::vec>(nActiveAtoms);

  // vec dualVars = 1e-14 * ones<vec>(nActiveAtoms);

  // Method used by feature sign code - fails miserably here.  Perhaps the
  // MATLAB optimizer fmincon does something clever?
  // vec dualVars = 10.0 * randu(nActiveAtoms, 1);

  // vec dualVars = diagvec(solve(dictionary, data * trans(codes))
  //    - codes * trans(codes));
  // for (size_t i = 0; i < dualVars.n_elem; ++i)
  //   if (dualVars(i) < 0)
  //     dualVars(i) = 0;

  bool converged = false;

  // If we have any inactive atoms, we must construct these differently.
  arma::mat codesXT;
  arma::mat codesZT;

  if (nInactiveAtoms == 0)
  {
    codesXT = codes * trans(data);
    codesZT = codes * trans(codes);
  }
  else
  {
    codesXT = matActiveZ * trans(data);
    codesZT = matActiveZ * trans(matActiveZ);
  }

  double normGradient = 0;
  double improvement = 0;
  for (size_t t = 1; (t != maxIterations) && !converged; ++t)
  {
    arma::mat A = codesZT + diagmat(dualVars);

    arma::mat matAInvZXT = solve(A, codesXT);

    arma::vec gradient = -sum(square(matAInvZXT), 1);
    gradient += 1;

    arma::mat hessian = -(-2 * (matAInvZXT * trans(matAInvZXT)) % inv(A));

    arma::vec searchDirection = -solve(hessian, gradient);

    // Armijo line search.
    const double c = 1e-4;
    double alpha = 1.0;
    const double rho = 0.9;
    double sufficientDecrease = c * dot(gradient, searchDirection);

    // A maxIterations parameter for the Armijo line search may be a good idea,
    // but it doesn't seem to be causing any problems for now.
    while (true)
    {
      // Calculate objective.
      double sumDualVars = sum(dualVars);
      double fOld = -(-trace(trans(codesXT) * matAInvZXT) - sumDualVars);
      double fNew = -(-trace(trans(codesXT) * solve(codesZT +
          diagmat(dualVars + alpha * searchDirection), codesXT)) -
          (sumDualVars + alpha * sum(searchDirection)));

      if (fNew <= fOld + alpha * sufficientDecrease)
      {
        searchDirection = alpha * searchDirection;
        improvement = fOld - fNew;
        break;
      }

      alpha *= rho;
    }

    // Take step and print useful information.
    dualVars += searchDirection;
    normGradient = norm(gradient, 2);
    Log::Debug << "Newton Method iteration " << t << ":" << std::endl;
    Log::Debug << "  Gradient norm: " << std::scientific << normGradient
        << "." << std::endl;
    Log::Debug << "  Improvement: " << std::scientific << improvement << ".\n";

    if (normGradient < newtonTolerance)
      converged = true;
  }

  if (nInactiveAtoms == 0)
  {
    // Directly update dictionary.
    dictionary = trans(solve(codesZT + diagmat(dualVars), codesXT));
  }
  else
  {
    arma::mat activeDictionary = trans(solve(codesZT +
        diagmat(dualVars), codesXT));

    // Update all atoms.
    size_t currentActiveIndex = 0;
    for (size_t i = 0; i < atoms; ++i)
    {
      if (currentActiveIndex >= activeAtoms.size() ||
          activeAtoms[currentActiveIndex] != i)
      {
        // This atom is inactive.  Reinitialize it randomly.
        dictionary.col(i) = (data.col(RandInt(data.n_cols)) +
                             data.col(RandInt(data.n_cols)) +
                             data.col(RandInt(data.n_cols)));

        dictionary.col(i) /= norm(dictionary.col(i), 2);
      }
      else
      {
        // Update estimate.
        dictionary.col(i) = activeDictionary.col(currentActiveIndex);

        // Increment active index counter.
        ++currentActiveIndex;
      }
    }
  }

  return normGradient;
}

// Project each atom of the dictionary back into the unit ball (if necessary).
inline void SparseCoding::ProjectDictionary()
{
  for (size_t j = 0; j < atoms; ++j)
  {
    double atomNorm = norm(dictionary.col(j), 2);
    if (atomNorm > 1)
    {
      Log::Info << "Norm of atom " << j << " exceeds 1 (" << std::scientific
          << atomNorm << ").  Shrinking...\n";
      dictionary.col(j) /= atomNorm;
    }
  }
}

// Compute the objective function.
inline double SparseCoding::Objective(const arma::mat& data, 
                                      const arma::mat& codes)
    const
{
  double l11NormZ = sum(sum(arma::abs(codes)));
  double froNormResidual = norm(data - (dictionary * codes), "fro");

  if (lambda2 > 0)
  {
    double froNormZ = norm(codes, "fro");
    return 0.5 * (std::pow(froNormResidual, 2.0) + (lambda2 *
        std::pow(froNormZ, 2.0))) + (lambda1 * l11NormZ);
  }
  else // It can be simpler.
  {
    return 0.5 * std::pow(froNormResidual, 2.0) + lambda1 * l11NormZ;
  }
}

template<typename DictionaryInitializer>
double SparseCoding::Train(
    const arma::mat& data,
    const DictionaryInitializer& initializer)
{
  // Now, train.

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

    lastObjVal = curObjVal;

    // Have we converged?
    if (improvement < objTolerance)
    {
      Log::Info << "Converged within tolerance " << objTolerance << ".\n";
      break;
    }
  }

  return lastObjVal;
}

template<typename Archive>
void SparseCoding::serialize(Archive& ar, const uint32_t /* version */)
{
  ar(CEREAL_NVP(atoms));
  ar(CEREAL_NVP(dictionary));
  ar(CEREAL_NVP(lambda1));
  ar(CEREAL_NVP(lambda2));
  ar(CEREAL_NVP(maxIterations));
  ar(CEREAL_NVP(objTolerance));
  ar(CEREAL_NVP(newtonTolerance));
}

} // namespace mlpack

#endif

/**
 * @file methods/local_coordinate_coding/lcc_impl.hpp
 * @author Nishant Mehta
 *
 * Implementation of Local Coordinate Coding
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_LOCAL_COORDINATE_CODING_LCC_IMPL_HPP
#define MLPACK_METHODS_LOCAL_COORDINATE_CODING_LCC_IMPL_HPP

// In case it hasn't been included yet.
#include "lcc.hpp"

namespace mlpack {

template<typename MatType>
template<typename DictionaryInitializer>
LocalCoordinateCoding<MatType>::LocalCoordinateCoding(
    const MatType& data,
    const size_t atoms,
    const double lambda,
    const size_t maxIterations,
    const double tolerance,
    const DictionaryInitializer& initializer) :
    atoms(atoms),
    lambda(lambda),
    maxIterations(maxIterations),
    tolerance(tolerance)
{
  // Train the model.
  Train(data, initializer);
}

template<typename MatType>
inline LocalCoordinateCoding<MatType>::LocalCoordinateCoding(
    const size_t atoms,
    const double lambda,
    const size_t maxIterations,
    const double tolerance) :
    atoms(atoms),
    lambda(lambda),
    maxIterations(maxIterations),
    tolerance(tolerance)
{
  // Nothing to do.
}

template<typename MatType>
template<typename DictionaryInitializer>
double LocalCoordinateCoding<MatType>::Train(
    const MatType& data,
    const DictionaryInitializer& initializer)
{
  // Initialize the dictionary.
  initializer.Initialize(data, atoms, dictionary);

  double lastObjVal = DBL_MAX;

  // Take the initial coding step, which has to happen before entering the main
  // loop.
  Log::Info << "Initial Coding Step." << std::endl;

  MatType codes;
  Encode(data, codes);
  arma::uvec adjacencies = find(codes);

  Log::Info << "  Sparsity level: " << 100.0 * ((double)(adjacencies.n_elem)) /
      ((double)(atoms * data.n_cols)) << "%.\n";
  Log::Info << "  Objective value: " << Objective(data, codes, adjacencies)
      << "." << std::endl;

  for (size_t t = 1; t != maxIterations; t++)
  {
    Log::Info << "Iteration " << t << " of " << maxIterations << "."
        << std::endl;

    // First step: optimize the dictionary.
    Log::Info << "Performing dictionary step..." << std::endl;
    OptimizeDictionary(data, codes, adjacencies);
    double dsObjVal = Objective(data, codes, adjacencies);
    Log::Info << "  Objective value: " << dsObjVal << "." << std::endl;

    // Second step: perform the coding.
    Log::Info << "Performing coding step..." << std::endl;
    Encode(data, codes);
    adjacencies = find(codes);
    Log::Info << "  Sparsity level: " << 100.0 * ((double) (adjacencies.n_elem))
        / ((double)(atoms * data.n_cols)) << "%.\n";

    // Terminate if the objective increased in the coding step.
    double curObjVal = Objective(data, codes, adjacencies);
    if (curObjVal > dsObjVal)
    {
      Log::Warn << "Objective increased in coding step!  Terminating."
          << std::endl;
      break;
    }

    // Find the new objective value and improvement so we can check for
    // convergence.
    double improvement = lastObjVal - curObjVal;
    Log::Info << "Objective value: " << curObjVal << " (improvement "
        << std::scientific << improvement << ")." << std::endl;

    if (improvement < tolerance)
    {
      Log::Info << "Converged within tolerance " << tolerance << ".\n";
      break;
    }

    lastObjVal = curObjVal;
  }

  return lastObjVal;
}

template<typename MatType>
inline void LocalCoordinateCoding<MatType>::Encode(const MatType& data,
                                                   MatType& codes)
{
  MatType invSqDists = 1.0 / (repmat(trans(sum(square(dictionary))), 1,
      data.n_cols) + repmat(sum(square(data)), atoms, 1) - 2 * trans(dictionary)
      * data);

  MatType dictGram = trans(dictionary) * dictionary;
  MatType dictGramTD(dictGram.n_rows, dictGram.n_cols);

  codes.set_size(atoms, data.n_cols);
  for (size_t i = 0; i < data.n_cols; ++i)
  {
    // Report progress.
    if ((i % 100) == 0)
    {
      Log::Debug << "Optimization at point " << i << "." << std::endl;
    }

    ColType invW = invSqDists.unsafe_col(i);
    MatType dictPrime = dictionary * diagmat(invW);

    MatType dictGramTD = diagmat(invW) * dictGram * diagmat(invW);

    bool useCholesky = false;
    // Normalization and fitting and intercept are disabled.
    const double tol = std::is_same_v<typename MatType::elem_type, float> ?
        1e-8 : 1e-16;
    LARS<MatType> lars(useCholesky, 0.5 * lambda, 0, tol, false, false);

    // Run LARS for this point, by making an alias of the point and passing
    // that.
    ColType beta = codes.unsafe_col(i);
    RowType responses = data.unsafe_col(i).t();
    lars.Train(dictPrime, responses, false, useCholesky, dictGramTD);
    beta = lars.Beta();
    beta %= invW; // Remember, beta is an alias of codes.col(i).
  }
}

template<typename MatType>
inline void LocalCoordinateCoding<MatType>::OptimizeDictionary(
    const MatType& data,
    const MatType& codes,
    const arma::uvec& adjacencies)
{
  // Count number of atomic neighbors for each point x^i.
  arma::uvec neighborCounts = zeros<arma::uvec>(data.n_cols, 1);
  if (adjacencies.n_elem > 0)
  {
    // This gets the column index.  Intentional integer division.
    size_t curPointInd = (size_t) (adjacencies(0) / atoms);
    ++neighborCounts(curPointInd);

    size_t nextColIndex = (curPointInd + 1) * atoms;
    for (size_t l = 1; l < adjacencies.n_elem; l++)
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

  // Build dataPrime := [X x^1 ... x^1 ... x^n ... x^n]
  // where each x^i is repeated for the number of neighbors x^i has.
  MatType dataPrime = zeros<MatType>(data.n_rows,
      data.n_cols + adjacencies.n_elem);

  dataPrime(arma::span::all, arma::span(0, data.n_cols - 1)) = data;

  size_t curCol = data.n_cols;
  for (size_t i = 0; i < data.n_cols; ++i)
  {
    if (neighborCounts(i) > 0)
    {
      dataPrime(arma::span::all, arma::span(curCol, curCol + neighborCounts(i)
          - 1)) = repmat(data.col(i), 1, neighborCounts(i));
    }
    curCol += neighborCounts(i);
  }

  // Handle the case of inactive atoms (atoms not used in the given coding).
  std::vector<arma::uword> activeAtoms;
  for (size_t j = 0; j < atoms; ++j)
    if (accu(codes.row(j) != 0) != 0)
      activeAtoms.push_back((arma::uword) j);

  const size_t nActiveAtoms = activeAtoms.size();
  const size_t nInactiveAtoms = atoms - nActiveAtoms;

  // Efficient construction of codes restricted to active atoms.
  MatType codesPrime = zeros<MatType>(nActiveAtoms, data.n_cols +
      adjacencies.n_elem);
  ColType wSquared = ones<ColType>(data.n_cols + adjacencies.n_elem, 1);

  if (nInactiveAtoms > 0)
  {
    Log::Warn << "There are " << nInactiveAtoms
        << " inactive atoms.  They will be re-initialized randomly.\n";

    // Create matrix holding only active codes.
    MatType activeCodes = codes.rows(arma::uvec(activeAtoms));

    // Create reverse atom lookup for active atoms.
    arma::uvec atomReverseLookup(atoms);
    for (size_t i = 0; i < activeAtoms.size(); ++i)
    {
      atomReverseLookup[activeAtoms[i]] = i;
    }

    codesPrime(arma::span::all, arma::span(0, data.n_cols - 1)) = activeCodes;

    // Fill the rest of codesPrime.
    for (size_t l = 0; l < adjacencies.n_elem; ++l)
    {
      // Recover the location in the codes matrix that this adjacency refers to.
      size_t atomInd = adjacencies(l) % atoms;
      size_t pointInd = (size_t) (adjacencies(l) / atoms);

      // Fill matrix.
      codesPrime(atomReverseLookup(atomInd), data.n_cols + l) = 1.0;
      wSquared(data.n_cols + l) = codes(atomInd, pointInd);
    }
  }
  else
  {
    // All atoms are active.
    codesPrime(arma::span::all, arma::span(0, data.n_cols - 1)) = codes;

    for (size_t l = 0; l < adjacencies.n_elem; ++l)
    {
      // Recover the location in the codes matrix that this adjacency refers to.
      size_t atomInd = adjacencies(l) % atoms;
      size_t pointInd = (size_t) (adjacencies(l) / atoms);

      // Fill matrix.
      codesPrime(atomInd, data.n_cols + l) = 1.0;
      wSquared(data.n_cols + l) = codes(atomInd, pointInd);
    }
  }

  if (adjacencies.n_elem > 0)
  {
    wSquared.subvec(data.n_cols, wSquared.n_elem - 1) = lambda *
        abs(wSquared.subvec(data.n_cols, wSquared.n_elem - 1));
  }

  // Solve system.
  if (nInactiveAtoms == 0)
  {
    // No inactive atoms.  We can solve directly.
    MatType A = codesPrime * diagmat(wSquared) * trans(codesPrime);
    MatType B = codesPrime * diagmat(wSquared) * trans(dataPrime);

    dictionary = trans(solve(A, B));
    /*
    dictionary = trans(solve(codesPrime * diagmat(wSquared) * trans(codesPrime),
        codesPrime * diagmat(wSquared) * trans(dataPrime)));
    */
  }
  else
  {
    // Inactive atoms must be reinitialized randomly, so we cannot solve
    // directly for the entire dictionary estimate.
    MatType dictionaryActive =
        trans(solve(codesPrime * diagmat(wSquared) * trans(codesPrime),
                    codesPrime * diagmat(wSquared) * trans(dataPrime)));

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

        // Now normalize the atom.
        dictionary.col(i) /= norm(dictionary.col(i), 2);
      }
      else
      {
        // Update estimate.
        dictionary.col(i) = dictionaryActive.col(currentActiveIndex);

        // Increment active atom counter.
        ++currentActiveIndex;
      }
    }
  }
}

template<typename MatType>
inline double LocalCoordinateCoding<MatType>::Objective(
    const MatType& data,
    const MatType& codes) const
{
  // Compute adjacencies and pass off to other overload.
  return Objective(data, codes, find(codes));
}

template<typename MatType>
inline double LocalCoordinateCoding<MatType>::Objective(
    const MatType& data,
    const MatType& codes,
    const arma::uvec& adjacencies) const
{
  double weightedL1NormZ = 0;

  for (size_t l = 0; l < adjacencies.n_elem; l++)
  {
    // Map adjacency back to its location in the codes matrix.
    const size_t atomInd = adjacencies(l) % atoms;
    const size_t pointInd = (size_t) (adjacencies(l) / atoms);

    weightedL1NormZ += fabs(codes(atomInd, pointInd)) * arma::as_scalar(
        sum(square(dictionary.col(atomInd) - data.col(pointInd))));
  }

  double froNormResidual = norm(data - dictionary * codes, "fro");
  return std::pow(froNormResidual, 2.0) + lambda * weightedL1NormZ;
}

template<typename MatType>
template<typename Archive>
void LocalCoordinateCoding<MatType>::serialize(Archive& ar,
                                               const uint32_t version)
{
  ar(CEREAL_NVP(atoms));

  if (cereal::is_loading<Archive>() && version == 0)
  {
    // Older versions of LocalCoordinateCoding always stored dictionary as an
    // arma::mat.
    arma::mat dictionaryTmp;
    ar(cereal::make_nvp("dictionary", dictionaryTmp));
    dictionary = ConvTo<MatType>::From(dictionaryTmp);
  }
  else
  {
    ar(CEREAL_NVP(dictionary));
  }

  ar(CEREAL_NVP(dictionary));
  ar(CEREAL_NVP(lambda));
  ar(CEREAL_NVP(maxIterations));
  ar(CEREAL_NVP(tolerance));
}

} // namespace mlpack

#endif

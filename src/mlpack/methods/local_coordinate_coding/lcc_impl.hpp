/**
 * @file lcc_impl.hpp
 * @author Nishant Mehta
 *
 * Implementation of Local Coordinate Coding
 */
#ifndef __MLPACK_METHODS_LOCAL_COORDINATE_CODING_LCC_IMPL_HPP
#define __MLPACK_METHODS_LOCAL_COORDINATE_CODING_LCC_IMPL_HPP

// In case it hasn't been included yet.
#include "lcc.hpp"

#define OBJ_TOL 1e-2 // 1E-9

namespace mlpack {
namespace lcc {

template<typename DictionaryInitializer>
LocalCoordinateCoding<DictionaryInitializer>::LocalCoordinateCoding(
    const arma::mat& data,
    const size_t atoms,
    const double lambda) :
    atoms(atoms),
    data(data),
    codes(atoms, data.n_cols),
    lambda(lambda)
{
  // Initialize the dictionary.
  DictionaryInitializer::Initialize(data, atoms, dictionary);
}

template<typename DictionaryInitializer>
void LocalCoordinateCoding<DictionaryInitializer>::Encode(
    const size_t maxIterations)
{
  double lastObjVal = DBL_MAX;

  // Take the initial coding step, which has to happen before entering the main
  // loop.
  Log::Info << "Initial Coding Step." << std::endl;

  OptimizeCode();
  arma::uvec adjacencies = find(codes);

  Log::Info << "  Sparsity level: " << 100.0 * ((double)(adjacencies.n_elem)) /
      ((double)(atoms * data.n_cols)) << "%.\n";
  Log::Info << "  Objective value: " << Objective(adjacencies) << "."
      << std::endl;

  for (size_t t = 1; t <= maxIterations; t++)
  {
    Log::Info << "Iteration " << t << " of " << maxIterations << "."
        << std::endl;

    // First step: optimize the dictionary.
    Log::Info << "Performing dictionary step..." << std::endl;
    OptimizeDictionary(adjacencies);
    double dsObjVal = Objective(adjacencies);
    Log::Info << "  Objective value: " << Objective(adjacencies) << "."
        << std::endl;

    // Second step: perform the coding.
    Log::Info << "Performing coding step..." << std::endl;
    OptimizeCode();
    adjacencies = find(codes);
    Log::Info << "  Sparsity level: " << 100.0 * ((double) (adjacencies.n_elem))
        / ((double)(atoms * data.n_cols)) << "%.\n";

    // Terminate if the objective increased in the coding step.
    double curObjVal = Objective(adjacencies);
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

    if (improvement < OBJ_TOL)
    {
      Log::Info << "Converged within tolerance " << OBJ_TOL << ".\n";
      break;
    }

    lastObjVal = curObjVal;
  }
}

template<typename DictionaryInitializer>
void LocalCoordinateCoding<DictionaryInitializer>::OptimizeCode()
{
  arma::mat invSqDists = 1.0 / (repmat(trans(sum(square(dictionary))), 1,
      data.n_cols) + repmat(sum(square(data)), atoms, 1) - 2 * trans(dictionary)
      * data);

  arma::mat dictGram = trans(dictionary) * dictionary;
  arma::mat dictGramTD(dictGram.n_rows, dictGram.n_cols);

  for (size_t i = 0; i < data.n_cols; i++)
  {
    // report progress
    if ((i % 100) == 0)
    {
      Log::Debug << "Optimization at point " << i << "." << std::endl;
    }

    arma::vec invW = invSqDists.unsafe_col(i);
    arma::mat dictPrime = dictionary * diagmat(invW);

    arma::mat dictGramTD = diagmat(invW) * dictGram * diagmat(invW);

    bool useCholesky = false;
    regression::LARS lars(useCholesky, dictGramTD, 0.5 * lambda);

    // Run LARS for this point, by making an alias of the point and passing
    // that.
    arma::vec beta = codes.unsafe_col(i);
    lars.Regress(dictPrime, data.unsafe_col(i), beta, true);
    beta %= invW; // Remember, beta is an alias of codes.col(i).
  }
}

template<typename DictionaryInitializer>
void LocalCoordinateCoding<DictionaryInitializer>::OptimizeDictionary(
    arma::uvec adjacencies)
{
  // count number of atomic neighbors for each point x^i
  arma::uvec neighborCounts = arma::zeros<arma::uvec>(data.n_cols, 1);
  if (adjacencies.n_elem > 0)
  {
    // this gets the column index
    size_t curPointInd = (size_t) (adjacencies(0) / atoms);
    size_t curCount = 1;
    for (size_t l = 1; l < adjacencies.n_elem; l++)
    {
      if ((size_t) (adjacencies(l) / atoms) == curPointInd)
      {
        curCount++;
      }
      else
      {
        neighborCounts(curPointInd) = curCount;
        curPointInd = (size_t) (adjacencies(l) / atoms);
        curCount = 1;
      }
    }
    neighborCounts(curPointInd) = curCount;
  }

  // Build dataPrime := [X x^1 ... x^1 ... x^n ... x^n]
  // where each x^i is repeated for the number of neighbors x^i has.
  arma::mat dataPrime = arma::zeros(data.n_rows, data.n_cols + adjacencies.n_elem);
  dataPrime(arma::span::all, arma::span(0, data.n_cols - 1)) = data;
  size_t curCol = data.n_cols;
  for (size_t i = 0; i < data.n_cols; i++)
  {
    if (neighborCounts(i) > 0)
    {
      dataPrime(arma::span::all, arma::span(curCol, curCol + neighborCounts(i)
          - 1)) = repmat(data.col(i), 1, neighborCounts(i));
    }
    curCol += neighborCounts(i);
  }

  // Handle the case of inactive atoms (atoms not used in the given coding).
  std::vector<size_t> inactiveAtoms;
  std::vector<size_t> activeAtoms;
  activeAtoms.reserve(atoms);
  for (size_t j = 0; j < atoms; j++)
  {
    if (accu(codes.row(j) != 0) == 0)
      inactiveAtoms.push_back(j);
    else
      activeAtoms.push_back(j);
  }
  size_t nActiveAtoms = activeAtoms.size();
  size_t nInactiveAtoms = inactiveAtoms.size();

  // efficient construction of Z restricted to active atoms
  arma::mat matActiveZ;
  if (inactiveAtoms.empty())
  {
    matActiveZ = codes;
  }
  else
  {
    arma::uvec inactiveAtomsVec = arma::conv_to<arma::uvec>::from(
        inactiveAtoms);
    RemoveRows(codes, inactiveAtomsVec, matActiveZ);
  }

  arma::uvec atomReverseLookup = arma::uvec(atoms);
  for (size_t i = 0; i < nActiveAtoms; i++)
  {
    atomReverseLookup(activeAtoms[i]) = i;
  }

  if (nInactiveAtoms > 0)
  {
    Log::Info << "There are " << nInactiveAtoms << " inactive atoms. They will"
        << " be re-initialized randomly.\n";
  }

  arma::mat codesPrime = arma::zeros(nActiveAtoms,
      data.n_cols + adjacencies.n_elem);
  codesPrime(arma::span::all, arma::span(0, data.n_cols - 1)) = matActiveZ;

  arma::vec wSquared = arma::ones(data.n_cols + adjacencies.n_elem, 1);
  for (size_t l = 0; l < adjacencies.n_elem; l++)
  {
    size_t atomInd = adjacencies(l) % atoms;
    size_t pointInd = (size_t) (adjacencies(l) / atoms);
    codesPrime(atomReverseLookup(atomInd), data.n_cols + l) = 1.0;
    wSquared(data.n_cols + l) = codes(atomInd, pointInd);
  }

  wSquared.subvec(data.n_cols, wSquared.n_elem - 1) = lambda *
      abs(wSquared.subvec(data.n_cols, wSquared.n_elem - 1));

  //Log::Debug << "about to solve\n";
  arma::mat dictionaryEstimate;
  if (inactiveAtoms.empty())
  {
    arma::mat A = codesPrime * diagmat(wSquared) * trans(codesPrime);
    arma::mat B = codesPrime * diagmat(wSquared) * trans(dataPrime);

    dictionaryEstimate = trans(solve(A, B));
    /*
    dictionaryEstimate =
      trans(solve(codesPrime * diagmat(wSquared) * trans(codesPrime),
                  codesPrime * diagmat(wSquared) * trans(dataPrime)));
    */
  }
  else
  {
    dictionaryEstimate = arma::zeros(data.n_rows, atoms);
    arma::mat dictionaryActiveEstimate =
      trans(solve(codesPrime * diagmat(wSquared) * trans(codesPrime),
                  codesPrime * diagmat(wSquared) * trans(dataPrime)));
    for (size_t j = 0; j < nActiveAtoms; j++)
    {
      dictionaryEstimate.col(activeAtoms[j]) = dictionaryActiveEstimate.col(j);
    }

    for (size_t j = 0; j < nInactiveAtoms; j++)
    {
      // Reinitialize randomly.
      // Add three atoms together.
      dictionaryEstimate.col(inactiveAtoms[j]) =
          (data.col(math::RandInt(data.n_cols)) +
           data.col(math::RandInt(data.n_cols)) +
           data.col(math::RandInt(data.n_cols)));

      // Now normalize the atom.
      dictionaryEstimate.col(inactiveAtoms[j]) /=
          norm(dictionaryEstimate.col(inactiveAtoms[j]), 2);
    }
  }

  dictionary = dictionaryEstimate;
}

template<typename DictionaryInitializer>
double LocalCoordinateCoding<DictionaryInitializer>::Objective(
    arma::uvec adjacencies)
{
  double weightedL1NormZ = 0;
  size_t nAdjacencies = adjacencies.n_elem;
  for (size_t l = 0; l < nAdjacencies; l++)
  {
    size_t atomInd = adjacencies(l) % atoms;
    size_t pointInd = (size_t) (adjacencies(l) / atoms);
    weightedL1NormZ += fabs(codes(atomInd, pointInd)) *
        as_scalar(sum(square(dictionary.col(atomInd) - data.col(pointInd))));
  }

  double froNormResidual = norm(data - dictionary * codes, "fro");
  return froNormResidual * froNormResidual + lambda * weightedL1NormZ;
}

void RemoveRows(const arma::mat& X, arma::uvec rows_to_remove, arma::mat& X_mod)
{
  arma::uword n_cols = X.n_cols;
  arma::uword n_rows = X.n_rows;
  arma::uword n_to_remove = rows_to_remove.n_elem;
  arma::uword n_to_keep = n_rows - n_to_remove;

  if (n_to_remove == 0)
  {
    X_mod = X;
  }
  else
  {
    X_mod.set_size(n_to_keep, n_cols);

    arma::uword cur_row = 0;
    arma::uword remove_ind = 0;
    // first, check 0 to first row to remove
    if (rows_to_remove(0) > 0)
    {
      // note that this implies that n_rows > 1
      arma::uword height = rows_to_remove(0);
      X_mod(arma::span(cur_row, cur_row + height - 1), arma::span::all) =
          X(arma::span(0, rows_to_remove(0) - 1), arma::span::all);
      cur_row += height;
    }
    // now, check i'th row to remove to (i + 1)'th row to remove, until i =
    // penultimate row
    while (remove_ind < n_to_remove - 1)
    {
      arma::uword height = rows_to_remove[remove_ind + 1] -
          rows_to_remove[remove_ind] - 1;
      if (height > 0)
      {
        X_mod(arma::span(cur_row, cur_row + height - 1), arma::span::all) =
            X(arma::span(rows_to_remove[remove_ind] + 1,
            rows_to_remove[remove_ind + 1] - 1), arma::span::all);
        cur_row += height;
      }
      remove_ind++;
    }
    // now that i is last row to remove, check last row to remove to last row
    if (rows_to_remove[remove_ind] < n_rows - 1)
    {
      X_mod(arma::span(cur_row, n_to_keep - 1), arma::span::all) =
          X(arma::span(rows_to_remove[remove_ind] + 1, n_rows - 1),
          arma::span::all);
    }
  }
}

}; // namespace lcc
}; // namespace mlpack

#endif

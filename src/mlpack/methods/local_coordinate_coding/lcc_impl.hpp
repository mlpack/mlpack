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
    const size_t nAtoms,
    const double lambda) :
    nDims(data.n_rows),
    nAtoms(nAtoms),
    nPoints(data.n_cols),
    data(data),
    codes(nAtoms, nPoints),
    lambda(lambda)
{
  // Initialize the dictionary.
  DictionaryInitializer::Initialize(data, nAtoms, dictionary);
}

template<typename DictionaryInitializer>
void LocalCoordinateCoding<DictionaryInitializer>::DoLCC(
    const size_t nIterations)
{
  double lastObjVal = DBL_MAX;

  // Take the initial coding step, which has to happen before entering the main
  // loop.
  Log::Info << "Initial Coding Step." << std::endl;

  OptimizeCode();
  arma::uvec adjacencies = find(codes);

  Log::Info << "  Sparsity level: " << 100.0 * ((double)(adjacencies.n_elem)) /
      ((double)(nAtoms * nPoints)) << "%.\n";
  Log::Info << "  Objective value: " << Objective(adjacencies) << "."
      << std::endl;

  for (size_t t = 1; t <= nIterations; t++)
  {
    Log::Info << "Iteration " << t << " of " << nIterations << "." << std::endl;

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
    Log::Info << "  Sparsity level: " << 100.0 * ((double)(adjacencies.n_elem))
        / ((double)(nAtoms * nPoints)) << "%.\n";

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
  arma::mat matSqDists = repmat(trans(sum(square(dictionary))), 1, nPoints) +
      repmat(sum(square(data)), nAtoms, 1) - 2 * trans(dictionary) * data;

  arma::mat matInvSqDists = 1.0 / matSqDists;

  arma::mat dictionaryTD = trans(dictionary) * dictionary;
  arma::mat dictionaryPrimeTDPrime(dictionaryTD.n_rows, dictionaryTD.n_cols);

  for (size_t i = 0; i < nPoints; i++)
  {
    // report progress
    if ((i % 100) == 0)
    {
      Log::Debug << "Optimization at point " << i << "." << std::endl;
    }

    arma::vec w = matSqDists.unsafe_col(i);
    arma::vec invW = matInvSqDists.unsafe_col(i);
    arma::mat dictionaryPrime = dictionary * diagmat(invW);

    arma::mat dictionaryPrimeTDPrime = diagmat(invW) * dictionaryTD * diagmat(invW);

    //LARS lars;
    // do we still need 0.5 * lambda? yes, yes we do
    //lars.Init(dictionaryPrime.memptr(), data.colptr(i), nDims, nAtoms, true, 0.5 *
    //lambda); // apparently not as fast as using the below duo
    // this may change, depending on the dimensionality and sparsity

    // the duo
    /* lars.Init(dictionaryPrime.memptr(), data.colptr(i), nDims, nAtoms, false, 0.5 *
     * lambda); */
    /* lars.SetGram(dictionaryPrimeTDPrime.memptr(), nAtoms); */

    bool useCholesky = false;
    regression::LARS lars(useCholesky, dictionaryPrimeTDPrime, 0.5 * lambda);

    arma::vec beta;
    lars.Regress(dictionaryPrime, data.unsafe_col(i), beta, true);
    codes.col(i) = beta % invW;
  }
}

template<typename DictionaryInitializer>
void LocalCoordinateCoding<DictionaryInitializer>::OptimizeDictionary(
    arma::uvec adjacencies)
{
  // count number of atomic neighbors for each point x^i
  arma::uvec neighborCounts = arma::zeros<arma::uvec>(nPoints, 1);
  if (adjacencies.n_elem > 0)
  {
    // this gets the column index
    size_t curPointInd = (size_t) (adjacencies(0) / nAtoms);
    size_t curCount = 1;
    for (size_t l = 1; l < adjacencies.n_elem; l++)
    {
      if ((size_t) (adjacencies(l) / nAtoms) == curPointInd)
      {
        curCount++;
      }
      else
      {
        neighborCounts(curPointInd) = curCount;
        curPointInd = (size_t) (adjacencies(l) / nAtoms);
        curCount = 1;
      }
    }
    neighborCounts(curPointInd) = curCount;
  }

  // Build dataPrime := [X x^1 ... x^1 ... x^n ... x^n]
  // where each x^i is repeated for the number of neighbors x^i has.
  arma::mat dataPrime = arma::zeros(nDims, nPoints + adjacencies.n_elem);
  dataPrime(arma::span::all, arma::span(0, nPoints - 1)) = data;
  size_t curCol = nPoints;
  for (size_t i = 0; i < nPoints; i++)
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
  activeAtoms.reserve(nAtoms);
  for (size_t j = 0; j < nAtoms; j++)
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

  arma::uvec atomReverseLookup = arma::uvec(nAtoms);
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
      nPoints + adjacencies.n_elem);
  codesPrime(arma::span::all, arma::span(0, nPoints - 1)) = matActiveZ;

  arma::vec wSquared = arma::ones(nPoints + adjacencies.n_elem, 1);
  for (size_t l = 0; l < adjacencies.n_elem; l++)
  {
    size_t atomInd = adjacencies(l) % nAtoms;
    size_t pointInd = (size_t) (adjacencies(l) / nAtoms);
    codesPrime(atomReverseLookup(atomInd), nPoints + l) = 1.0;
    wSquared(nPoints + l) = codes(atomInd, pointInd);
  }

  wSquared.subvec(nPoints, wSquared.n_elem - 1) = lambda *
      abs(wSquared.subvec(nPoints, wSquared.n_elem - 1));

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
    dictionaryEstimate = arma::zeros(nDims, nAtoms);
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
    size_t atomInd = adjacencies(l) % nAtoms;
    size_t pointInd = (size_t) (adjacencies(l) / nAtoms);
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

/**
 * @file sparse_coding_impl.hpp
 * @author Nishant Mehta
 *
 * Implementation of Sparse Coding with Dictionary Learning using l1 (LASSO) or
 * l1+l2 (Elastic Net) regularization.
 */
#ifndef __MLPACK_METHODS_SPARSE_CODING_SPARSE_CODING_IMPL_HPP
#define __MLPACK_METHODS_SPARSE_CODING_SPARSE_CODING_IMPL_HPP

// In case it hasn't already been included.
#include "sparse_coding.hpp"

namespace mlpack {
namespace sparse_coding {

// TODO: parameterizable; options to methods?
#define OBJ_TOL 1e-2 // 1E-9
#define NEWTON_TOL 1e-6 // 1E-9

template<typename DictionaryInitializer>
SparseCoding<DictionaryInitializer>::SparseCoding(const arma::mat& data,
                                                  const size_t atoms,
                                                  const double lambda1,
                                                  const double lambda2) :
    atoms(atoms),
    data(data),
    codes(atoms, data.n_cols),
    lambda1(lambda1),
    lambda2(lambda2)
{
  // Initialize the dictionary.
  DictionaryInitializer::Initialize(data, atoms, dictionary);
}

template<typename DictionaryInitializer>
void SparseCoding<DictionaryInitializer>::Encode(const size_t maxIterations)
{
  double lastObjVal = DBL_MAX;

  Log::Info << "Initial Coding Step." << std::endl;

  OptimizeCode();
  arma::uvec adjacencies = find(codes);

  Log::Info << "  Sparsity level: "
      << 100.0 * ((double) (adjacencies.n_elem)) / ((double)
      (atoms * data.n_cols)) << "%" << std::endl;
  Log::Info << "  Objective value: " << Objective() << "." << std::endl;

  for (size_t t = 1; t != maxIterations; ++t)
  {
    Log::Info << "Iteration " << t << " of " << maxIterations << "."
        << std::endl;

    Log::Info << "Performing dictionary step... ";
    OptimizeDictionary(adjacencies);
    Log::Info << "objective value: " << Objective() << "." << std::endl;

    Log::Info << "Performing coding step..." << std::endl;
    OptimizeCode();
    adjacencies = find(codes);
    Log::Info << "  Sparsity level: "
        << 100.0 *
        ((double) (adjacencies.n_elem)) / ((double) (atoms * data.n_cols))
        << "%" << std::endl;

    double curObjVal = Objective();
    Log::Info << "  Objective value: " << curObjVal << "." << std::endl;

    double objValImprov = lastObjVal - curObjVal;
    Log::Info << "  Improvement: " << std::scientific << objValImprov << "."
        << std::endl;

    if (objValImprov < OBJ_TOL)
    {
      Log::Info << "Converged within tolerance " << OBJ_TOL << ".\n";
      break;
    }

    lastObjVal = curObjVal;
  }
}

template<typename DictionaryInitializer>
void SparseCoding<DictionaryInitializer>::OptimizeCode()
{
  // When using Cholesky version of LARS, this is correct even if lambda2 > 0.
  arma::mat matGram = trans(dictionary) * dictionary;
  // mat matGram;
  // if(lambda2 > 0) {
  //   matGram = trans(dictionary) * dictionary + lambda2 * eye(atoms, atoms);
  // }
  // else {
  //   matGram = trans(dictionary) * dictionary;
  // }

  for (size_t i = 0; i < data.n_cols; ++i)
  {
    // Report progress.
    if ((i % 100) == 0)
      Log::Debug << "Optimization at point " << i << "." << std::endl;

    bool useCholesky = true;
    regression::LARS lars(useCholesky, matGram, lambda1, lambda2);

    arma::vec beta;
    lars.Regress(dictionary, data.unsafe_col(i), beta, true);

    codes.col(i) = beta;
  }
}

template<typename DictionaryInitializer>
void SparseCoding<DictionaryInitializer>::OptimizeDictionary(
      const arma::uvec& adjacencies)
{
  // Count the number of atomic neighbors for each point x^i.
  arma::uvec neighborCounts = arma::zeros<arma::uvec>(data.n_cols, 1);

  if (adjacencies.n_elem > 0)
  {
    // This gets the column index.
    // TODO: is this integer division intentional?
    size_t curPointInd = (size_t) (adjacencies(0) / atoms);
    size_t curCount = 1;

    for (size_t l = 1; l < adjacencies.n_elem; ++l)
    {
      if ((size_t) (adjacencies(l) / atoms) == curPointInd)
      {
        ++curCount;
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

  // Handle the case of inactive atoms (atoms not used in the given coding).
  std::vector<size_t> inactiveAtoms;
  std::vector<size_t> activeAtoms;
  activeAtoms.reserve(atoms);

  for (size_t j = 0; j < atoms; ++j)
  {
    if (accu(codes.row(j) != 0) == 0)
      inactiveAtoms.push_back(j);
    else
      activeAtoms.push_back(j);
  }

  const size_t nActiveAtoms = activeAtoms.size();
  const size_t nInactiveAtoms = inactiveAtoms.size();

  // Efficient construction of Z restricted to active atoms.
  arma::mat matActiveZ;
  if (inactiveAtoms.empty())
  {
    matActiveZ = codes;
  }
  else
  {
    arma::uvec inactiveAtomsVec =
        arma::conv_to<arma::uvec>::from(inactiveAtoms);
    RemoveRows(codes, inactiveAtomsVec, matActiveZ);
  }

  arma::uvec atomReverseLookup(atoms);
  for (size_t i = 0; i < nActiveAtoms; ++i)
    atomReverseLookup(activeAtoms[i]) = i;

  if (nInactiveAtoms > 0)
  {
    Log::Info << "There are " << nInactiveAtoms
        << " inactive atoms. They will be re-initialized randomly.\n";
  }

  Log::Debug << "Solving Dual via Newton's Method.\n";

  arma::mat dictionaryEstimate;
  // Solve using Newton's method in the dual - note that the final dot
  // multiplication with inv(A) seems to be unavoidable. Although more
  // expensive, the code written this way (we use solve()) should be more
  // numerically stable than just using inv(A) for everything.
  arma::vec dualVars = arma::zeros<arma::vec>(nActiveAtoms);

  //vec dualVars = 1e-14 * ones<vec>(nActiveAtoms);

  // Method used by feature sign code - fails miserably here.  Perhaps the
  // MATLAB optimizer fmincon does something clever?
  //vec dualVars = 10.0 * randu(nActiveAtoms, 1);

  //vec dualVars = diagvec(solve(dictionary, data * trans(codes))
  //    - codes * trans(codes));
  //for (size_t i = 0; i < dualVars.n_elem; i++)
  //  if (dualVars(i) < 0)
  //    dualVars(i) = 0;

  bool converged = false;
  arma::mat codesXT = matActiveZ * trans(data);
  arma::mat codesZT = matActiveZ * trans(matActiveZ);

  for (size_t t = 1; !converged; ++t)
  {
    arma::mat A = codesZT + diagmat(dualVars);

    arma::mat matAInvZXT = solve(A, codesXT);

    arma::vec gradient = -(arma::sum(arma::square(matAInvZXT), 1) -
        arma::ones<arma::vec>(nActiveAtoms));

    arma::mat hessian = -(-2 * (matAInvZXT * trans(matAInvZXT)) % inv(A));

    arma::vec searchDirection = -solve(hessian, gradient);
    //vec searchDirection = -gradient;

    // Armijo line search.
    const double c = 1e-4;
    double alpha = 1.0;
    const double rho = 0.9;
    double sufficientDecrease = c * dot(gradient, searchDirection);

    /*
    {
      double sumDualVars = sum(dualVars);
      double fOld = -(-trace(trans(codesXT) * matAInvZXT) - sumDualVars);
      Log::Debug << "fOld = " << fOld << "." << std::endl;
      double fNew =
          -(-trace(trans(codesXT) * solve(codesZT +
          diagmat(dualVars + alpha * searchDirection), codesXT))
          - (sumDualVars + alpha * sum(searchDirection)) );
      Log::Debug << "fNew = " << fNew << "." << std::endl;
    }
    */

    double improvement;
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

    // End of Armijo line search code.

    dualVars += searchDirection;
    double normGradient = norm(gradient, 2);
    Log::Debug << "Newton Method iteration " << t << ":" << std::endl;
    Log::Debug << "  Gradient norm: " << std::scientific << normGradient
        << "." << std::endl;
    Log::Debug << "  Improvement: " << std::scientific << improvement << ".\n";

    if (improvement < NEWTON_TOL)
      converged = true;
  }

  if (inactiveAtoms.empty())
  {
    dictionaryEstimate = trans(solve(codesZT + diagmat(dualVars), codesXT));
  }
  else
  {
    arma::mat dictionaryActiveEstimate = trans(solve(codesZT +
        diagmat(dualVars), codesXT));
    dictionaryEstimate = arma::zeros(data.n_rows, atoms);

    for (size_t i = 0; i < nActiveAtoms; ++i)
      dictionaryEstimate.col(activeAtoms[i]) = dictionaryActiveEstimate.col(i);

    for (size_t i = 0; i < nInactiveAtoms; ++i)
    {
      // Make a new random atom estimate.
      dictionaryEstimate.col(inactiveAtoms[i]) =
          (data.col(math::RandInt(data.n_cols)) +
           data.col(math::RandInt(data.n_cols)) +
           data.col(math::RandInt(data.n_cols)));

      dictionaryEstimate.col(inactiveAtoms[i]) /=
          norm(dictionaryEstimate.col(inactiveAtoms[i]), 2);
    }
  }

  dictionary = dictionaryEstimate;
}

// Project each atom of the dictionary onto the unit ball.
template<typename DictionaryInitializer>
void SparseCoding<DictionaryInitializer>::ProjectDictionary()
{
  for (size_t j = 0; j < atoms; j++)
  {
    double normD_j = norm(dictionary.col(j), 2);
    if ((normD_j > 1) && (normD_j - 1.0 > 1e-9))
    {
      Log::Warn << "Norm exceeded 1 by " << std::scientific << normD_j - 1.0
          << ".  Shrinking...\n";
      dictionary.col(j) /= normD_j;
    }
  }
}

template<typename DictionaryInitializer>
double SparseCoding<DictionaryInitializer>::Objective()
{
  double l11NormZ = sum(sum(abs(codes)));
  double froNormResidual = norm(data - dictionary * codes, "fro");

  if (lambda2 > 0)
  {
    double froNormZ = norm(codes, "fro");
    return 0.5 *
      (froNormResidual * froNormResidual + lambda2 * froNormZ * froNormZ) +
      lambda1 * l11NormZ;
  }
  else
  {
    return 0.5 * froNormResidual * froNormResidual + lambda1 * l11NormZ;
  }
}

void RemoveRows(const arma::mat& X,
                const arma::uvec& rowsToRemove,
                arma::mat& modX)
{
  const size_t cols = X.n_cols;
  const size_t rows = X.n_rows;
  const size_t nRemove = rowsToRemove.n_elem;
  const size_t nKeep = rows - nRemove;

  if (nRemove == 0)
  {
    modX = X;
  }
  else
  {
    modX.set_size(nKeep, cols);

    size_t curRow = 0;
    size_t removeInd = 0;
    // First, check 0 to first row to remove.
    if (rowsToRemove(0) > 0)
    {
      // Note that this implies that n_rows > 1.
      size_t height = rowsToRemove(0);
      modX(arma::span(curRow, curRow + height - 1), arma::span::all) =
          X(arma::span(0, rowsToRemove(0) - 1), arma::span::all);
      curRow += height;
    }
    // Now, check i'th row to remove to (i + 1)'th row to remove, until i is the
    // penultimate row.
    while (removeInd < nRemove - 1)
    {
      size_t height = rowsToRemove[removeInd + 1] -
          rowsToRemove[removeInd] - 1;

      if (height > 0)
      {
        modX(arma::span(curRow, curRow + height - 1), arma::span::all) =
            X(arma::span(rowsToRemove[removeInd] + 1,
            rowsToRemove[removeInd + 1] - 1), arma::span::all);
        curRow += height;
      }

      removeInd++;
    }

    // Now that i is the last row to remove, check last row to remove to last
    // row.
    if (rowsToRemove[removeInd] < rows - 1)
    {
      modX(arma::span(curRow, nKeep - 1), arma::span::all) =
          X(arma::span(rowsToRemove[removeInd] + 1, rows - 1),
          arma::span::all);
    }
  }
}

}; // namespace sparse_coding
}; // namespace mlpack

#endif

/**
 * @file lars.cpp
 * @author Nishant Mehta (niche)
 *
 * Implementation of LARS and LASSO.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "lars.hpp"

using namespace mlpack;
using namespace mlpack::regression;

LARS::LARS(const bool useCholesky,
           const double lambda1,
           const double lambda2,
           const double tolerance) :
    matGram(&matGramInternal),
    useCholesky(useCholesky),
    lasso((lambda1 != 0)),
    lambda1(lambda1),
    elasticNet((lambda1 != 0) && (lambda2 != 0)),
    lambda2(lambda2),
    tolerance(tolerance)
{ /* Nothing left to do. */ }

LARS::LARS(const bool useCholesky,
           const arma::mat& gramMatrix,
           const double lambda1,
           const double lambda2,
           const double tolerance) :
    matGram(&gramMatrix),
    useCholesky(useCholesky),
    lasso((lambda1 != 0)),
    lambda1(lambda1),
    elasticNet((lambda1 != 0) && (lambda2 != 0)),
    lambda2(lambda2),
    tolerance(tolerance)
{ /* Nothing left to do */ }

void LARS::Train(const arma::mat& matX,
                 const arma::vec& y,
                 arma::vec& beta,
                 const bool transposeData)
{
  Timer::Start("lars_regression");

  // Clear any previous solution information.
  betaPath.clear();
  lambdaPath.clear();
  activeSet.clear();
  isActive.clear();
  ignoreSet.clear();
  isIgnored.clear();
  matUtriCholFactor.reset();

  // This matrix may end up holding the transpose -- if necessary.
  arma::mat dataTrans;
  // dataRef is row-major.
  const arma::mat& dataRef = (transposeData ? dataTrans : matX);
  if (transposeData)
    dataTrans = trans(matX);

  // Compute X' * y.
  arma::vec vecXTy = trans(dataRef) * y;

  // Set up active set variables.  In the beginning, the active set has size 0
  // (all dimensions are inactive).
  isActive.resize(dataRef.n_cols, false);

  // Set up ignores set variables. Initialized empty.
  isIgnored.resize(dataRef.n_cols, false);

  // Initialize yHat and beta.
  beta = arma::zeros(dataRef.n_cols);
  arma::vec yHat = arma::zeros(dataRef.n_rows);
  arma::vec yHatDirection(dataRef.n_rows);

  bool lassocond = false;

  // Compute the initial maximum correlation among all dimensions.
  arma::vec corr = vecXTy;
  double maxCorr = 0;
  size_t changeInd = 0;
  for (size_t i = 0; i < vecXTy.n_elem; ++i)
  {
    if (fabs(corr(i)) > maxCorr)
    {
      maxCorr = fabs(corr(i));
      changeInd = i;
    }
  }

  betaPath.push_back(beta);
  lambdaPath.push_back(maxCorr);

  // If the maximum correlation is too small, there is no reason to continue.
  if (maxCorr < lambda1)
  {
    lambdaPath[0] = lambda1;
    Timer::Stop("lars_regression");
    return;
  }

  // Compute the Gram matrix.  If this is the elastic net problem, we will add
  // lambda2 * I_n to the matrix.
  if (matGram->n_elem != dataRef.n_cols * dataRef.n_cols)
  {
    // In this case, matGram should reference matGramInternal.
    matGramInternal = trans(dataRef) * dataRef;

    if (elasticNet && !useCholesky)
      matGramInternal += lambda2 * arma::eye(dataRef.n_cols, dataRef.n_cols);
  }

  // Main loop.
  while (((activeSet.size() + ignoreSet.size()) < dataRef.n_cols) &&
         (maxCorr > tolerance))
  {
    // Compute the maximum correlation among inactive dimensions.
    maxCorr = 0;
    for (size_t i = 0; i < dataRef.n_cols; i++)
    {
      if ((!isActive[i]) && (!isIgnored[i]) && (fabs(corr(i)) > maxCorr))
      {
        maxCorr = fabs(corr(i));
        changeInd = i;
      }
    }

    if (!lassocond)
    {
      if (useCholesky)
      {
        // vec newGramCol = vec(activeSet.size());
        // for (size_t i = 0; i < activeSet.size(); i++)
        // {
        //   newGramCol[i] = dot(matX.col(activeSet[i]), matX.col(changeInd));
        // }
        // This is equivalent to the above 5 lines.
        arma::vec newGramCol = matGram->elem(changeInd * dataRef.n_cols +
            arma::conv_to<arma::uvec>::from(activeSet));

        CholeskyInsert((*matGram)(changeInd, changeInd), newGramCol);
      }

      // Add variable to active set.
      Activate(changeInd);
    }

    // Compute signs of correlations.
    arma::vec s = arma::vec(activeSet.size());
    for (size_t i = 0; i < activeSet.size(); i++)
      s(i) = corr(activeSet[i]) / fabs(corr(activeSet[i]));

    // Compute the "equiangular" direction in parameter space (betaDirection).
    // We use quotes because in the case of non-unit norm variables, this need
    // not be equiangular.
    arma::vec unnormalizedBetaDirection;
    double normalization;
    arma::vec betaDirection;
    if (useCholesky)
    {
      // Check for singularity.
      const double lastUtriElement = matUtriCholFactor(
          matUtriCholFactor.n_cols - 1, matUtriCholFactor.n_rows - 1);
      if (std::abs(lastUtriElement) > tolerance)
      {
        // Ok, no singularity.
        /**
         * Note that:
         * R^T R % S^T % S = (R % S)^T (R % S)
         * Now, for 1 the ones vector:
         * inv( (R % S)^T (R % S) ) 1
         *    = inv(R % S) inv((R % S)^T) 1
         *    = inv(R % S) Solve((R % S)^T, 1)
         *    = inv(R % S) Solve(R^T, s)
         *    = Solve(R % S, Solve(R^T, s)
         *    = s % Solve(R, Solve(R^T, s))
         */
        unnormalizedBetaDirection = solve(trimatu(matUtriCholFactor),
            solve(trimatl(trans(matUtriCholFactor)), s));

        normalization = 1.0 / sqrt(dot(s, unnormalizedBetaDirection));
        betaDirection = normalization * unnormalizedBetaDirection;
      }
      else
      {
        // Singularity, so remove variable from active set, add to ignores set,
        // and look for new variable to add.
        Log::Warn << "Encountered singularity when adding variable "
            << changeInd << " to active set; permanently removing."
            << std::endl;
        Deactivate(activeSet.size() - 1);
        Ignore(changeInd);
        CholeskyDelete(matUtriCholFactor.n_rows - 1);
        continue;
      }
    }
    else
    {
      arma::mat matGramActive = arma::mat(activeSet.size(), activeSet.size());
      for (size_t i = 0; i < activeSet.size(); i++)
        for (size_t j = 0; j < activeSet.size(); j++)
          matGramActive(i, j) = (*matGram)(activeSet[i], activeSet[j]);

      // Check for singularity.
      arma::mat matS = s * arma::ones<arma::mat>(1, activeSet.size());
      const bool solvedOk = solve(unnormalizedBetaDirection,
          matGramActive % trans(matS) % matS,
          arma::ones<arma::mat>(activeSet.size(), 1));
      if (solvedOk)
      {
        // Ok, no singularity.
        normalization = 1.0 / sqrt(sum(unnormalizedBetaDirection));
        betaDirection = normalization * unnormalizedBetaDirection % s;
      }
      else
      {
        // Singularity, so remove variable from active set, add to ignores set,
        // and look for new variable to add.
        Deactivate(activeSet.size() - 1);
        Ignore(changeInd);
        Log::Warn << "Encountered singularity when adding variable "
            << changeInd << " to active set; permanently removing."
            << std::endl;
        continue;
      }
    }

    // compute "equiangular" direction in output space
    ComputeYHatDirection(dataRef, betaDirection, yHatDirection);

    double gamma = maxCorr / normalization;

    // If not all variables are active.
    if ((activeSet.size() + ignoreSet.size()) < dataRef.n_cols)
    {
      // Compute correlations with direction.
      for (size_t ind = 0; ind < dataRef.n_cols; ind++)
      {
        if (isActive[ind] || isIgnored[ind])
          continue;

        double dirCorr = dot(dataRef.col(ind), yHatDirection);
        double val1 = (maxCorr - corr(ind)) / (normalization - dirCorr);
        double val2 = (maxCorr + corr(ind)) / (normalization + dirCorr);
        if ((val1 > 0) && (val1 < gamma))
          gamma = val1;
        if ((val2 > 0) && (val2 < gamma))
          gamma = val2;
      }
    }

    // Bound gamma according to LASSO.
    if (lasso)
    {
      lassocond = false;
      double lassoboundOnGamma = DBL_MAX;
      size_t activeIndToKickOut = -1;

      for (size_t i = 0; i < activeSet.size(); i++)
      {
        double val = -beta(activeSet[i]) / betaDirection(i);
        if ((val > 0) && (val < lassoboundOnGamma))
        {
          lassoboundOnGamma = val;
          activeIndToKickOut = i;
        }
      }

      if (lassoboundOnGamma < gamma)
      {
        gamma = lassoboundOnGamma;
        lassocond = true;
        changeInd = activeIndToKickOut;
      }
    }

    // Update the prediction.
    yHat += gamma * yHatDirection;

    // Update the estimator.
    for (size_t i = 0; i < activeSet.size(); i++)
    {
      beta(activeSet[i]) += gamma * betaDirection(i);
    }

    // Sanity check to make sure the kicked out dimension is actually zero.
    if (lassocond)
    {
      if (beta(activeSet[changeInd]) != 0)
        beta(activeSet[changeInd]) = 0;
    }

    betaPath.push_back(beta);

    if (lassocond)
    {
      // Index is in position changeInd in activeSet.
      if (useCholesky)
        CholeskyDelete(changeInd);

      Deactivate(changeInd);
    }

    corr = vecXTy - trans(dataRef) * yHat;
    if (elasticNet)
      corr -= lambda2 * beta;

    double curLambda = 0;
    for (size_t i = 0; i < activeSet.size(); i++)
      curLambda += fabs(corr(activeSet[i]));

    curLambda /= ((double) activeSet.size());

    lambdaPath.push_back(curLambda);

    // Time to stop for LASSO?
    if (lasso)
    {
      if (curLambda <= lambda1)
      {
        InterpolateBeta();
        break;
      }
    }
  }

  // Unfortunate copy...
  beta = betaPath.back();

  Timer::Stop("lars_regression");
}

void LARS::Predict(const arma::mat& points,
                   arma::vec& predictions,
                   const bool rowMajor) const
{
  // We really only need to store beta internally...
  if (rowMajor)
    predictions = points * betaPath.back();
  else
    predictions = (betaPath.back().t() * points).t();
}

// Private functions.
void LARS::Deactivate(const size_t activeVarInd)
{
  isActive[activeSet[activeVarInd]] = false;
  activeSet.erase(activeSet.begin() + activeVarInd);
}

void LARS::Activate(const size_t varInd)
{
  isActive[varInd] = true;
  activeSet.push_back(varInd);
}

void LARS::Ignore(const size_t varInd)
{
  isIgnored[varInd] = true;
  ignoreSet.push_back(varInd);
}

void LARS::ComputeYHatDirection(const arma::mat& matX,
                                const arma::vec& betaDirection,
                                arma::vec& yHatDirection)
{
  yHatDirection.fill(0);
  for (size_t i = 0; i < activeSet.size(); i++)
    yHatDirection += betaDirection(i) * matX.col(activeSet[i]);
}

void LARS::InterpolateBeta()
{
  int pathLength = betaPath.size();

  // interpolate beta and stop
  double ultimateLambda = lambdaPath[pathLength - 1];
  double penultimateLambda = lambdaPath[pathLength - 2];
  double interp = (penultimateLambda - lambda1)
      / (penultimateLambda - ultimateLambda);

  betaPath[pathLength - 1] = (1 - interp) * (betaPath[pathLength - 2])
      + interp * betaPath[pathLength - 1];

  lambdaPath[pathLength - 1] = lambda1;
}

void LARS::CholeskyInsert(const arma::vec& newX, const arma::mat& X)
{
  if (matUtriCholFactor.n_rows == 0)
  {
    matUtriCholFactor = arma::mat(1, 1);

    if (elasticNet)
      matUtriCholFactor(0, 0) = sqrt(dot(newX, newX) + lambda2);
    else
      matUtriCholFactor(0, 0) = norm(newX, 2);
  }
  else
  {
    arma::vec newGramCol = trans(X) * newX;
    CholeskyInsert(dot(newX, newX), newGramCol);
  }
}

void LARS::CholeskyInsert(double sqNormNewX, const arma::vec& newGramCol)
{
  int n = matUtriCholFactor.n_rows;

  if (n == 0)
  {
    matUtriCholFactor = arma::mat(1, 1);

    if (elasticNet)
      matUtriCholFactor(0, 0) = sqrt(sqNormNewX + lambda2);
    else
      matUtriCholFactor(0, 0) = sqrt(sqNormNewX);
  }
  else
  {
    arma::mat matNewR = arma::mat(n + 1, n + 1);

    if (elasticNet)
      sqNormNewX += lambda2;

    arma::vec matUtriCholFactork = solve(trimatl(trans(matUtriCholFactor)),
        newGramCol);

    matNewR(arma::span(0, n - 1), arma::span(0, n - 1)) = matUtriCholFactor;
    matNewR(arma::span(0, n - 1), n) = matUtriCholFactork;
    matNewR(n, arma::span(0, n - 1)).fill(0.0);
    matNewR(n, n) = sqrt(sqNormNewX - dot(matUtriCholFactork,
                                          matUtriCholFactork));

    matUtriCholFactor = matNewR;
  }
}

void LARS::GivensRotate(const arma::vec::fixed<2>& x,
                        arma::vec::fixed<2>& rotatedX,
                        arma::mat& matG)
{
  if (x(1) == 0)
  {
    matG = arma::eye(2, 2);
    rotatedX = x;
  }
  else
  {
    double r = norm(x, 2);
    matG = arma::mat(2, 2);

    double scaledX1 = x(0) / r;
    double scaledX2 = x(1) / r;

    matG(0, 0) = scaledX1;
    matG(1, 0) = -scaledX2;
    matG(0, 1) = scaledX2;
    matG(1, 1) = scaledX1;

    rotatedX = arma::vec(2);
    rotatedX(0) = r;
    rotatedX(1) = 0;
  }
}

void LARS::CholeskyDelete(const size_t colToKill)
{
  size_t n = matUtriCholFactor.n_rows;

  if (colToKill == (n - 1))
  {
    matUtriCholFactor = matUtriCholFactor(arma::span(0, n - 2),
                                          arma::span(0, n - 2));
  }
  else
  {
    matUtriCholFactor.shed_col(colToKill); // remove column colToKill
    n--;

    for (size_t k = colToKill; k < n; k++)
    {
      arma::mat matG;
      arma::vec::fixed<2> rotatedVec;
      GivensRotate(matUtriCholFactor(arma::span(k, k + 1), k), rotatedVec,
          matG);
      matUtriCholFactor(arma::span(k, k + 1), k) = rotatedVec;
      if (k < n - 1)
      {
        matUtriCholFactor(arma::span(k, k + 1), arma::span(k + 1, n - 1)) =
            matG * matUtriCholFactor(arma::span(k, k + 1),
            arma::span(k + 1, n - 1));
      }
    }

    matUtriCholFactor.shed_row(n);
  }
}

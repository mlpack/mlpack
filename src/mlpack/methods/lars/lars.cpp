/**
 * @file lars.cpp
 * @author Nishant Mehta (niche)
 *
 * Implementation of LARS and LASSO.
 */
#include "lars.hpp"

using namespace mlpack;
using namespace mlpack::regression;

LARS::LARS(const bool useCholesky,
           const double lambda1,
           const double lambda2,
           const double tolerance) :
    matGram(matGramInternal),
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
    matGram(gramMatrix),
    useCholesky(useCholesky),
    lasso((lambda1 != 0)),
    lambda1(lambda1),
    elasticNet((lambda1 != 0) && (lambda2 != 0)),
    lambda2(lambda2),
    tolerance(tolerance)
{ /* Nothing left to do */ }

void LARS::DoLARS(const arma::mat& matX, const arma::vec& y)
{
  // Compute X' * y.
  arma::vec vecXTy = trans(matX) * y;

  // Set up active set variables.
  nActive = 0;
  activeSet = std::vector<arma::uword>(0);
  isActive = std::vector<bool>(matX.n_cols);
  fill(isActive.begin(), isActive.end(), false);

  // Initialize yHat and beta.
  arma::vec beta = arma::zeros(matX.n_cols);
  arma::vec yHat = arma::zeros(matX.n_rows);
  arma::vec yHatDirection = arma::vec(matX.n_rows);

  bool lassocond = false;

  arma::vec corr = vecXTy;
  arma::vec absCorr = abs(corr);
  arma::uword changeInd;
  double maxCorr = absCorr.max(changeInd); // change_ind gets set here

  betaPath.push_back(beta);
  lambdaPath.push_back(maxCorr);

  // don't even start!
  if (maxCorr < lambda1)
  {
    lambdaPath[0] = lambda1;
    return;
  }

  // Compute the Gram matrix.  If this is the elastic net problem, we will add
  // lambda2 * I_n to the matrix.
  if (matGram.n_elem == 0)
  {
    // In this case, matGram should reference matGramInternal.
    matGramInternal = trans(matX) * matX;

    if (elasticNet && !useCholesky)
      matGramInternal += lambda2 * arma::eye(matX.n_cols, matX.n_cols);
  }

  // Main loop.
  while ((nActive < matX.n_cols) && (maxCorr > tolerance))
  {
    // explicit computation of max correlation, among inactive indices
    changeInd = -1;
    maxCorr = 0;
    for (arma::uword i = 0; i < matX.n_cols; i++)
    {
      if (!isActive[i])
      {
        if (fabs(corr(i)) > maxCorr)
        {
          maxCorr = fabs(corr(i));
          changeInd = i;
        }
      }
    }

    if (!lassocond)
    {
      // index is absolute index
      //printf("activating %d\n", changeInd);
      if (useCholesky)
      {
        // vec newGramCol = vec(nActive);
        // for (uword i = 0; i < nActive; i++)
        // {
        //   newGramCol[i] = dot(matX.col(activeSet[i]), matX.col(changeInd));
        // }
        // This is equivalent to the above 5 lines.
        arma::vec newGramCol = matGram.elem(changeInd * matX.n_cols +
            arma::conv_to<arma::uvec>::from(activeSet));

        //CholeskyInsert(matX.col(changeInd), newGramCol);
        CholeskyInsert(matGram(changeInd, changeInd), newGramCol);
      }

      // add variable to active set
      Activate(changeInd);
    }

    // compute signs of correlations
    arma::vec s = arma::vec(nActive);
    for (arma::uword i = 0; i < nActive; i++)
      s(i) = corr(activeSet[i]) / fabs(corr(activeSet[i]));

    // compute "equiangular" direction in parameter space (betaDirection)
    /* We use quotes because in the case of non-unit norm variables,
       this need not be equiangular. */
    arma::vec unnormalizedBetaDirection;
    double normalization;
    arma::vec betaDirection;
    if (useCholesky)
    {
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
      arma::mat matGramActive = arma::mat(nActive, nActive);
      for (arma::uword i = 0; i < nActive; i++)
        for (arma::uword j = 0; j < nActive; j++)
          matGramActive(i,j) = matGram(activeSet[i], activeSet[j]);

      arma::mat matS = s * arma::ones<arma::mat>(1, nActive);
      unnormalizedBetaDirection = solve(matGramActive % trans(matS) % matS,
          arma::ones<arma::mat>(nActive, 1));
      normalization = 1.0 / sqrt(sum(unnormalizedBetaDirection));
      betaDirection = normalization * unnormalizedBetaDirection % s;
    }

    // compute "equiangular" direction in output space
    ComputeYHatDirection(matX, betaDirection, yHatDirection);

    double gamma = maxCorr / normalization;

    // if not all variables are active
    if (nActive < matX.n_cols)
    {
      // compute correlations with direction
      for (arma::uword ind = 0; ind < matX.n_cols; ind++)
      {
        if (isActive[ind])
          continue;

        double dirCorr = dot(matX.col(ind), yHatDirection);
        double val1 = (maxCorr - corr(ind)) / (normalization - dirCorr);
        double val2 = (maxCorr + corr(ind)) / (normalization + dirCorr);
        if ((val1 > 0) && (val1 < gamma))
          gamma = val1;
        if ((val2 > 0) && (val2 < gamma))
          gamma = val2;
      }
    }

    // bound gamma according to LASSO
    if (lasso)
    {
      lassocond = false;
      double lassoboundOnGamma = DBL_MAX;
      arma::uword activeIndToKickOut = -1;

      for (arma::uword i = 0; i < nActive; i++)
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
        // printf("%d: gap = %e\tbeta(%d) = %e\n",
        //    activeSet[activeIndToKickOut],
        //    gamma - lassoboundOnGamma,
        //    activeSet[activeIndToKickOut],
        //    beta(activeSet[activeIndToKickOut]));
        gamma = lassoboundOnGamma;
        lassocond = true;
        changeInd = activeIndToKickOut;
      }
    }

    // update prediction
    yHat += gamma * yHatDirection;

    // update estimator
    for (arma::uword i = 0; i < nActive; i++)
    {
      beta(activeSet[i]) += gamma * betaDirection(i);
    }

    // sanity check to make sure the kicked out guy (or girl?) is actually zero
    if (lassocond)
    {
      if (beta(activeSet[changeInd]) != 0)
      {
        //printf("fixed from %e to 0\n", beta(activeSet[changeInd]));
        beta(activeSet[changeInd]) = 0;
      }
    }

    betaPath.push_back(beta);

    if (lassocond)
    {
      // index is in position changeInd in activeSet
      //printf("\t\tKICK OUT %d!\n", activeSet[changeInd]);

      if (useCholesky)
        CholeskyDelete(changeInd);

      Deactivate(changeInd);
    }

    corr = vecXTy - trans(matX) * yHat;
    if (elasticNet)
      corr -= lambda2 * beta;

    double curLambda = 0;
    for (arma::uword i = 0; i < nActive; i++)
      curLambda += fabs(corr(activeSet[i]));

    curLambda /= ((double)nActive);

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
}

void LARS::Solution(arma::vec& beta)
{
  beta = BetaPath().back();
}

// Private functions.
void LARS::Deactivate(arma::uword activeVarInd)
{
  nActive--;
  isActive[activeSet[activeVarInd]] = false;
  activeSet.erase(activeSet.begin() + activeVarInd);
}

void LARS::Activate(arma::uword varInd)
{
  nActive++;
  isActive[varInd] = true;
  activeSet.push_back(varInd);
}

void LARS::ComputeYHatDirection(const arma::mat& matX,
                                const arma::vec& betaDirection,
                                arma::vec& yHatDirection)
{
  yHatDirection.fill(0);
  for (arma::uword i = 0; i < nActive; i++)
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

void LARS::CholeskyDelete(arma::uword colToKill)
{
  arma::uword n = matUtriCholFactor.n_rows;

  if (colToKill == (n - 1))
  {
    matUtriCholFactor = matUtriCholFactor(arma::span(0, n - 2),
                                          arma::span(0, n - 2));
  }
  else
  {
    matUtriCholFactor.shed_col(colToKill); // remove column colToKill
    n--;

    for (arma::uword k = colToKill; k < n; k++)
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

/**
 * @file lars_impl.hpp
 * @author Nishant Mehta (niche)
 *
 * Implementation of LARS and LASSO.
 */
#ifndef __MLPACK_METHODS_LARS_LARS_IMPL_HPP
#define __MLPACK_METHODS_LARS_LARS_IMPL_HPP

// In case it hasn't been included.
#include "lars.hpp"

namespace mlpack {
namespace lars {

LARS::LARS(const arma::mat& matX,
           const arma::vec& y,
           const bool useCholesky) :
    matX(matX),
    y(y),
    useCholesky(useCholesky),
    lasso(false),
    elasticNet(false)
{ /* nothing left to do */ }

LARS::LARS(const arma::mat& matX,
           const arma::vec& y,
           const bool useCholesky,
           const double lambda1) :
    matX(matX),
    y(y),
    useCholesky(useCholesky),
    lasso(true),
    lambda1(lambda1),
    elasticNet(false),
    lambda2(0)
{ /* nothing left to do */ }

LARS::LARS(const arma::mat& matX,
           const arma::vec& y,
           const bool useCholesky,
           const double lambda1,
           const double lambda2) :
    matX(matX),
    y(y),
    useCholesky(useCholesky),
    lasso(true),
    lambda1(lambda1),
    elasticNet(true),
    lambda2(lambda2)
{ /* nothing left to do */ }

void LARS::SetGram(const arma::mat& matGram) {
  this->matGram = matGram;
}


void LARS::ComputeGram()
{
  if (elasticNet)
    matGram = trans(matX) * matX + lambda2 * arma::eye(matX.n_cols, matX.n_cols);
  else
    matGram = trans(matX) * matX;
}


void LARS::ComputeXty()
{
  matXTy = trans(matX) * y;
}


void LARS::UpdateX(const std::vector<int>& colInds, const arma::mat& matNewCols)
{
  for (arma::u32 i = 0; i < colInds.size(); i++)
    matX.col(colInds[i]) = matNewCols.col(i);

  if (!useCholesky)
    UpdateGram(colInds);

  UpdateXty(colInds);
}

void LARS::UpdateGram(const std::vector<int>& colInds)
{
  for (std::vector<int>::const_iterator i = colInds.begin();
      i != colInds.end(); ++i)
  {
    for (std::vector<int>::const_iterator j = colInds.begin();
        j != colInds.end(); ++j)
    {
      matGram(*i, *j) = dot(matX.col(*i), matX.col(*j));
    }
  }

  if (elasticNet)
  {
    for (std::vector<int>::const_iterator i = colInds.begin();
        i != colInds.end(); ++i)
    {
      matGram(*i, *i) += lambda2;
    }
  }
}

void LARS::UpdateXty(const std::vector<int>& colInds)
{
  for (std::vector<int>::const_iterator i = colInds.begin();
      i != colInds.end(); ++i)
    matXTy(*i) = dot(matX.col(*i), y);
}

void LARS::PrintGram()
{
  matGram.print("Gram arma::matrix");
}

void LARS::SetY(const arma::vec& y)
{
  this->y = y;
}

void LARS::PrintY()
{
  y.print();
}

const std::vector<arma::u32> LARS::ActiveSet()
{
  return activeSet;
}

const std::vector<arma::vec> LARS::BetaPath()
{
  return betaPath;
}

const std::vector<double> LARS::LambdaPath()
{
  return lambdaPath;
}

void LARS::SetDesiredLambda(double lambda1)
{
  this->lambda1 = lambda1;
}

void LARS::DoLARS()
{
  // compute Gram arma::matrix, XtY, and initialize active set varibles
  ComputeXty();
  if (!useCholesky && matGram.is_empty())
    ComputeGram();

  // set up active set variables
  nActive = 0;
  activeSet = std::vector<arma::u32>(0);
  isActive = std::vector<bool>(matX.n_cols);
  fill(isActive.begin(), isActive.end(), false);

  // initialize yHat and beta
  arma::vec beta = arma::zeros(matX.n_cols);
  arma::vec yHat = arma::zeros(matX.n_rows);
  arma::vec yHatDirection = arma::vec(matX.n_rows);

  bool lassocond = false;

  // used for elastic net
  if(!elasticNet)
  {
    lambda2 = 0; // just in case it is accidentally used, the code still will be correct
  }
  
  arma::vec corr = matXTy;
  arma::vec absCorr = abs(corr);
  arma::u32 changeInd;
  double maxCorr = absCorr.max(changeInd); // change_ind gets set here

  betaPath.push_back(beta);
  lambdaPath.push_back(maxCorr);
  
  // don't even start!
  if (maxCorr < lambda1)
  {
    lambdaPath[0] = lambda1;
    return;
  }

  //arma::u32 matX.n_rowsiterations_run = 0;
  // MAIN LOOP
  while ((nActive < matX.n_cols) && (maxCorr > EPS))
  {
    //matX.n_rowsiterations_run++;
    //printf("iteration %d\t", matX.n_rowsiterations_run);

    // explicit computation of max correlation, among inactive indices
    changeInd = -1;
    maxCorr = 0;
    for (arma::u32 i = 0; i < matX.n_cols; i++)
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
        arma::vec newGramCol = arma::vec(nActive);
        for (arma::u32 i = 0; i < nActive; i++)
          newGramCol[i] = dot(matX.col(activeSet[i]), matX.col(changeInd));

        CholeskyInsert(matX.col(changeInd), newGramCol);
      }

      // add variable to active set
      Activate(changeInd);
    }

    // compute signs of correlations
    arma::vec s = arma::vec(nActive);
    for (arma::u32 i = 0; i < nActive; i++)
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
       * Now, for 1 the ones arma::vector:
       * inv( (R % S)^T (R % S) ) 1
       *    = inv(R % S) inv((R % S)^T) 1
       *    = inv(R % S) Solve((R % S)^T, 1)
       *    = inv(R % S) Solve(R^T, s)
       *    = Solve(R % S, Solve(R^T, s)
       *    = s % Solve(R, Solve(R^T, s))
       */
      unnormalizedBetaDirection = solve(trimatu(utriCholFactor),
          solve(trimatl(trans(utriCholFactor)), s));

      normalization = 1.0 / sqrt(dot(s, unnormalizedBetaDirection));
      betaDirection = normalization * unnormalizedBetaDirection;
    }
    else
    {
      arma::mat matGramActive = arma::mat(nActive, nActive);
      for (arma::u32 i = 0; i < nActive; i++)
      {
        for (arma::u32 j = 0; j < nActive; j++)
        {
          matGramActive(i,j) = matGram(activeSet[i], activeSet[j]);
        }
      }

      arma::mat S = s * arma::ones<arma::mat>(1, nActive);
      unnormalizedBetaDirection =
          solve(matGramActive % trans(S) % S, arma::ones<arma::mat>(nActive, 1));
      normalization = 1.0 / sqrt(sum(unnormalizedBetaDirection));
      betaDirection = normalization * unnormalizedBetaDirection % s;
    }

    // compute "equiangular" direction in output space
    ComputeYHatDirection(betaDirection, yHatDirection);


    double gamma = maxCorr / normalization;

    // if not all variables are active
    if (nActive < matX.n_cols)
    {
      // compute correlations with direction
      for (arma::u32 ind = 0; ind < matX.n_cols; ind++)
      {
        if (isActive[ind])
        {
          continue;
        }

        double dirCorr = dot(matX.col(ind), yHatDirection);
        double val1 = (maxCorr - corr(ind)) / (normalization - dirCorr);
        double val2 = (maxCorr + corr(ind)) / (normalization + dirCorr);
        if ((val1 > 0) && (val1 < gamma))
          gamma = val1;
        if((val2 > 0) && (val2 < gamma))
          gamma = val2;
      }
    }

    // bound gamma according to LASSO
    if (lasso)
    {
      lassocond = false;
      double lassoboundOnGamma = DBL_MAX;
      arma::u32 activeIndToKickOut = -1;

      for (arma::u32 i = 0; i < nActive; i++)
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
        //printf("%d: gap = %e\tbeta(%d) = %e\n",
        //    activeSet[activeIndToKickOut],
        //    gamma - lassoBoundOnGamma,
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
    for (arma::u32 i = 0; i < nActive; i++)
    {
      beta(activeSet[i]) += gamma * betaDirection(i);
    }
    betaPath.push_back(beta);

    if (lassocond)
    {
      // index is in position changeInd in activeSet
      //printf("\t\tKICK OUT %d!\n", activeSet[changeInd]);
      if (beta(activeSet[changeInd]) != 0)
      {
        //printf("fixed from %e to 0\n", beta(activeSet[changeInd]));
        beta(activeSet[changeInd]) = 0;
      }

      if (useCholesky)
      {
        CholeskyDelete(changeInd);
      }

      Deactivate(changeInd);
    }

    corr = matXTy - trans(matX) * yHat;
    if (elasticNet)
    {
      corr -= lambda2 * beta;
    }
    double curLambda = 0;
    for (arma::u32 i = 0; i < nActive; i++)
    {
      curLambda += fabs(corr(activeSet[i]));
    }
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

void LARS::GetCholFactor(arma::mat& matR)
{
  matR = utriCholFactor;
}

void LARS::Deactivate(arma::u32 activeVarInd)
{
  nActive--;
  isActive[activeSet[activeVarInd]] = false;
  activeSet.erase(activeSet.begin() + activeVarInd);
}

void LARS::Activate(arma::u32 varInd)
{
  nActive++;
  isActive[varInd] = true;
  activeSet.push_back(varInd);
}

void LARS::ComputeYHatDirection(const arma::vec& betaDirection,
                                arma::vec& yHatDirection)
{
  yHatDirection.fill(0);
  for(arma::u32 i = 0; i < nActive; i++)
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
  if (utriCholFactor.n_rows == 0)
  {
    utriCholFactor = arma::mat(1, 1);
    if (elasticNet)
      utriCholFactor(0, 0) = sqrt(dot(newX, newX) + lambda2);
    else
      utriCholFactor(0, 0) = norm(newX, 2);
  }
  else
  {
    arma::vec newGramCol = trans(X) * newX;
    CholeskyInsert(newX, newGramCol);
  }
}

void LARS::CholeskyInsert(const arma::vec& newX, const arma::vec& newGramCol) {
  int n = utriCholFactor.n_rows;

  if (n == 0)
  {
    utriCholFactor = arma::mat(1, 1);
    if (elasticNet)
      utriCholFactor(0, 0) = sqrt(dot(newX, newX) + lambda2);
    else
      utriCholFactor(0, 0) = norm(newX, 2);
  }
  else
  {
    arma::mat matNewR = arma::mat(n + 1, n + 1);

    double sqNormNewX;
    if (elasticNet)
      sqNormNewX = dot(newX, newX) + lambda2;
    else
      sqNormNewX = dot(newX, newX);

    arma::vec utriCholFactork = solve(trimatl(trans(utriCholFactor)),
        newGramCol);

    matNewR(arma::span(0, n - 1), arma::span(0, n - 1)) = utriCholFactor;
    matNewR(arma::span(0, n - 1), n) = utriCholFactork;
    matNewR(n, arma::span(0, n - 1)).fill(0.0);
    matNewR(n, n) = sqrt(sqNormNewX - dot(utriCholFactork, utriCholFactork));

    utriCholFactor = matNewR;
  }
}

void LARS::GivensRotate(const arma::vec& x, arma::vec& rotatedX, arma::mat& G) 
{
  if (x(1) == 0)
  {
    G = arma::eye(2, 2);
    rotatedX = x;
  }
  else
  {
    double r = norm(x, 2);
    G = arma::mat(2, 2);

    double scaledX1 = x(0) / r;
    double scaledX2 = x(1) / r;

    G(0, 0) = scaledX1;
    G(1, 0) = -scaledX2;
    G(0, 1) = scaledX2;
    G(1, 1) = scaledX1;

    rotatedX = arma::vec(2);
    rotatedX(0) = r;
    rotatedX(1) = 0;
  }
}

void LARS::CholeskyDelete(arma::u32 colToKill)
{
  arma::u32 n = utriCholFactor.n_rows;

  if (colToKill == (n - 1))
  {
    utriCholFactor = utriCholFactor(arma::span(0, n - 2), arma::span(0, n - 2));
  }
  else
  {
    utriCholFactor.shed_col(colToKill); // remove column colToKill
    n--;

    for(arma::u32 k = colToKill; k < n; k++)
    {
      arma::mat G;
      arma::vec rotatedVec;
      GivensRotate(utriCholFactor(arma::span(k, k + 1), k), rotatedVec, G);
      utriCholFactor(arma::span(k, k + 1), k) = rotatedVec;
      if (k < n - 1)
      {
        utriCholFactor(arma::span(k, k + 1), arma::span(k + 1, n - 1)) = G *
            utriCholFactor(arma::span(k, k + 1), arma::span(k + 1, n - 1));
      }
    }
    utriCholFactor.shed_row(n);
  }
}

}; // namespace lars
}; // namespace mlpack

#endif

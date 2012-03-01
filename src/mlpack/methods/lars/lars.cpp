/**
 * @file lars.cpp
 * @author Nishant Mehta (niche)
 *
 * Implementation of LARS and LASSO.
 */

#include "lars.hpp"

// we are explicit with std:: to avoid confusing std::vector with arma::vec
// we use arma namespace too often to explicitly use arma:: everywhere
//using namespace std;
using namespace arma;
using namespace mlpack;
using namespace mlpack::regression;

LARS::LARS(const bool useCholesky) :
    useCholesky(useCholesky),
    lasso(false),
    elasticNet(false)
{ /* nothing left to do */ }

LARS::LARS(const bool useCholesky,
           const double lambda1) :
    useCholesky(useCholesky),
    lasso(true),
    lambda1(lambda1),
    elasticNet(false),
    lambda2(0)
{ /* nothing left to do */ }

LARS::LARS(const bool useCholesky,
           const double lambda1,
           const double lambda2) :
    useCholesky(useCholesky),
    lasso(true),
    lambda1(lambda1),
    elasticNet(true),
    lambda2(lambda2)
{ /* nothing left to do */ }

void LARS::SetGram(const mat& matGram)
{
  this->matGram = matGram;
}

void LARS::ComputeGram(const mat& matX)
{
  if (elasticNet)
  {
    matGram = trans(matX) * matX + lambda2 * eye(matX.n_cols, matX.n_cols);
  }
  else
  {
    matGram = trans(matX) * matX;
  }
}

void LARS::DoLARS(const mat& matX, const vec& y)
{
  // compute Xty
  vec vecXTy = trans(matX) * y;
  
  // compute Gram matrix
  if (!useCholesky && matGram.is_empty())
  {
    ComputeGram(matX);
  }

  // set up active set variables
  nActive = 0;
  activeSet = std::vector<u32>(0);
  isActive = std::vector<bool>(matX.n_cols);
  fill(isActive.begin(), isActive.end(), false);

  // initialize yHat and beta
  vec beta = zeros(matX.n_cols);
  vec yHat = zeros(matX.n_rows);
  vec yHatDirection = vec(matX.n_rows);

  bool lassocond = false;

  // used for elastic net
  if (!elasticNet)
  {
    lambda2 = 0; // just in case it is accidentally used, the code still will be correct
  }
  
  vec corr = vecXTy;
  vec absCorr = abs(corr);
  u32 changeInd;
  double maxCorr = absCorr.max(changeInd); // change_ind gets set here

  betaPath.push_back(beta);
  lambdaPath.push_back(maxCorr);
  
  // don't even start!
  if (maxCorr < lambda1)
  {
    lambdaPath[0] = lambda1;
    return;
  }

  //u32 iterations_run = 0;
  // MAIN LOOP
  while ((nActive < matX.n_cols) && (maxCorr > EPS))
  {
    //iterations_run++;
    //printf("iteration %d\t\n", iterations_run);

    // explicit computation of max correlation, among inactive indices
    changeInd = -1;
    maxCorr = 0;
    for (u32 i = 0; i < matX.n_cols; i++)
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
        vec newGramCol = vec(nActive);
        for (u32 i = 0; i < nActive; i++)
        {
          newGramCol[i] = dot(matX.col(activeSet[i]), matX.col(changeInd));
        }

        CholeskyInsert(matX.col(changeInd), newGramCol);
      }

      // add variable to active set
      Activate(changeInd);
    }

    // compute signs of correlations
    vec s = vec(nActive);
    for (u32 i = 0; i < nActive; i++)
    {
      s(i) = corr(activeSet[i]) / fabs(corr(activeSet[i]));
    }

    // compute "equiangular" direction in parameter space (betaDirection)
    /* We use quotes because in the case of non-unit norm variables,
       this need not be equiangular. */
    vec unnormalizedBetaDirection;
    double normalization;
    vec betaDirection;
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
      mat matGramActive = mat(nActive, nActive);
      for (u32 i = 0; i < nActive; i++)
      {
        for (u32 j = 0; j < nActive; j++)
        {
          matGramActive(i,j) = matGram(activeSet[i], activeSet[j]);
        }
      }

      mat matS = s * ones<mat>(1, nActive);
      unnormalizedBetaDirection =
          solve(matGramActive % trans(matS) % matS, ones<mat>(nActive, 1));
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
      for (u32 ind = 0; ind < matX.n_cols; ind++)
      {
        if (isActive[ind])
        {
          continue;
        }

        double dirCorr = dot(matX.col(ind), yHatDirection);
        double val1 = (maxCorr - corr(ind)) / (normalization - dirCorr);
        double val2 = (maxCorr + corr(ind)) / (normalization + dirCorr);
        if ((val1 > 0) && (val1 < gamma))
        {
          gamma = val1;
        }
        if ((val2 > 0) && (val2 < gamma))
        {
          gamma = val2;
        }
      }
    }

    // bound gamma according to LASSO
    if (lasso)
    {
      lassocond = false;
      double lassoboundOnGamma = DBL_MAX;
      u32 activeIndToKickOut = -1;

      for (u32 i = 0; i < nActive; i++)
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
    for (u32 i = 0; i < nActive; i++)
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
      {
        CholeskyDelete(changeInd);
      }

      Deactivate(changeInd);
    }

    corr = vecXTy - trans(matX) * yHat;
    if (elasticNet)
    {
      corr -= lambda2 * beta;
    }
    double curLambda = 0;
    for (u32 i = 0; i < nActive; i++)
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

void LARS::Solution(vec& beta)
{
  beta = BetaPath().back();
}



  ////////// private functions //////////

void LARS::Deactivate(u32 activeVarInd)
{
  nActive--;
  isActive[activeSet[activeVarInd]] = false;
  activeSet.erase(activeSet.begin() + activeVarInd);
}

void LARS::Activate(u32 varInd)
{
  nActive++;
  isActive[varInd] = true;
  activeSet.push_back(varInd);
}

 void LARS::ComputeYHatDirection(const mat& matX,
         const vec& betaDirection,
         vec& yHatDirection)
{
  yHatDirection.fill(0);
  for(u32 i = 0; i < nActive; i++)
  {
    yHatDirection += betaDirection(i) * matX.col(activeSet[i]);
  }
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

void LARS::CholeskyInsert(const vec& newX, const mat& X)
{
  if (matUtriCholFactor.n_rows == 0)
  {
    matUtriCholFactor = mat(1, 1);
    if (elasticNet)
    {
      matUtriCholFactor(0, 0) = sqrt(dot(newX, newX) + lambda2);
    }
    else
    {
      matUtriCholFactor(0, 0) = norm(newX, 2);
    }
  }
  else
  {
    vec newGramCol = trans(X) * newX;
    CholeskyInsert(newX, newGramCol);
  }
}

void LARS::CholeskyInsert(const vec& newX, const vec& newGramCol)
{
  int n = matUtriCholFactor.n_rows;

  if (n == 0)
  {
    matUtriCholFactor = mat(1, 1);
    if (elasticNet)
    {
      matUtriCholFactor(0, 0) = sqrt(dot(newX, newX) + lambda2);
    }
    else
    {
      matUtriCholFactor(0, 0) = norm(newX, 2);
    }
  }
  else
  {
    mat matNewR = mat(n + 1, n + 1);

    double sqNormNewX;
    if (elasticNet)
    {
      sqNormNewX = dot(newX, newX) + lambda2;
    }
    else
    {
      sqNormNewX = dot(newX, newX);
    }

    vec matUtriCholFactork = solve(trimatl(trans(matUtriCholFactor)),
        newGramCol);

    matNewR(span(0, n - 1), span(0, n - 1)) = matUtriCholFactor;
    matNewR(span(0, n - 1), n) = matUtriCholFactork;
    matNewR(n, span(0, n - 1)).fill(0.0);
    matNewR(n, n) = sqrt(sqNormNewX - dot(matUtriCholFactork,
        matUtriCholFactork));

    matUtriCholFactor = matNewR;
  }
}

void LARS::GivensRotate(const vec& x, vec& rotatedX, mat& matG) 
{
  if (x(1) == 0)
  {
    matG = eye(2, 2);
    rotatedX = x;
  }
  else
  {
    double r = norm(x, 2);
    matG = mat(2, 2);

    double scaledX1 = x(0) / r;
    double scaledX2 = x(1) / r;

    matG(0, 0) = scaledX1;
    matG(1, 0) = -scaledX2;
    matG(0, 1) = scaledX2;
    matG(1, 1) = scaledX1;

    rotatedX = vec(2);
    rotatedX(0) = r;
    rotatedX(1) = 0;
  }
}

void LARS::CholeskyDelete(u32 colToKill)
{
  u32 n = matUtriCholFactor.n_rows;

  if (colToKill == (n - 1))
  {
    matUtriCholFactor = matUtriCholFactor(span(0, n - 2), span(0, n - 2));
  }
  else
  {
    matUtriCholFactor.shed_col(colToKill); // remove column colToKill
    n--;

    for(u32 k = colToKill; k < n; k++)
    {
      mat matG;
      vec rotatedVec;
      GivensRotate(matUtriCholFactor(span(k, k + 1), k), rotatedVec, matG);
      matUtriCholFactor(span(k, k + 1), k) = rotatedVec;
      if (k < n - 1)
      {
        matUtriCholFactor(span(k, k + 1), span(k + 1, n - 1)) =
            matG * matUtriCholFactor(span(k, k + 1), span(k + 1, n - 1));
      }
    }
    matUtriCholFactor.shed_row(n);
  }
}

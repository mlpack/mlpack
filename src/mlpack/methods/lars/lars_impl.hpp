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

LARS::LARS(const arma::mat& data,
           const arma::vec& responses,
           const bool useCholesky) :
    data(data),
    responses(responses),
    useCholesky(useCholesky),
    lasso(false),
    elasticNet(false)
{ /* nothing left to do */ }

LARS::LARS(const arma::mat& data,
           const arma::vec& responses,
           const bool useCholesky,
           const double lambda1) :
    data(data),
    responses(responses),
    useCholesky(useCholesky),
    lasso(true),
    lambda1(lambda1),
    elasticNet(false),
    lambda2(0)
{ /* nothing left to do */ }

LARS::LARS(const arma::mat& data,
           const arma::vec& responses,
           const bool useCholesky,
           const double lambda1,
           const double lambda2) :
    data(data),
    responses(responses),
    useCholesky(useCholesky),
    lasso(true),
    lambda1(lambda1),
    elasticNet(true),
    lambda2(lambda2)
{ /* nothing left to do */ }

void LARS::SetGram(const arma::mat& Gram) {
  gram = gram;
}


void LARS::ComputeGram()
{
  if (elasticNet)
    gram = trans(data) * data + lambda2 * arma::eye(data.n_cols, data.n_cols);
  else
    gram = trans(data) * data;
}


void LARS::ComputeXty()
{
  xtResponses = trans(data) * responses;
}


void LARS::UpdateX(const std::vector<int>& col_inds, const arma::mat& new_cols)
{
  for (arma::u32 i = 0; i < col_inds.size(); i++)
    data.col(col_inds[i]) = new_cols.col(i);

  if (!useCholesky)
    UpdateGram(col_inds);

  UpdateXty(col_inds);
}

void LARS::UpdateGram(const std::vector<int>& col_inds)
{
  for (std::vector<int>::const_iterator i = col_inds.begin();
      i != col_inds.end(); ++i)
  {
    for (std::vector<int>::const_iterator j = col_inds.begin();
        j != col_inds.end(); ++j)
    {
      gram(*i, *j) = dot(data.col(*i), data.col(*j));
    }
  }

  if (elasticNet)
  {
    for (std::vector<int>::const_iterator i = col_inds.begin();
        i != col_inds.end(); ++i)
    {
      gram(*i, *i) += lambda2;
    }
  }
}

void LARS::UpdateXty(const std::vector<int>& col_inds)
{
  for (std::vector<int>::const_iterator i = col_inds.begin();
      i != col_inds.end(); ++i)
    xtResponses(*i) = dot(data.col(*i), responses);
}

void LARS::PrintGram()
{
  gram.print("Gram arma::matrix");
}

void LARS::SetY(const arma::vec& y)
{
  responses = y;
}

void LARS::PrintY()
{
  responses.print();
}

const std::vector<arma::u32> LARS::active_set()
{
  return activeSet;
}

const std::vector<arma::vec> LARS::beta_path()
{
  return betaPath;
}

const std::vector<double> LARS::lambda_path()
{
  return lambdaPath;
}

void LARS::SetDesiredLambda(double lambda_1)
{
  lambda1 = lambda_1;
}

void LARS::DoLARS()
{
  // compute Gram arma::matrix, XtY, and initialize active set varibles
  ComputeXty();
  if (!useCholesky && gram.is_empty())
    ComputeGram();

  // set up active set variables
  nActive = 0;
  activeSet = std::vector<arma::u32>(0);
  isActive = std::vector<bool>(data.n_cols);
  fill(isActive.begin(), isActive.end(), false);

  // initialize responseshat and beta
  arma::vec beta = arma::zeros(data.n_cols);
  arma::vec responseshat = arma::zeros(data.n_rows);
  arma::vec responseshat_direction = arma::vec(data.n_rows);

  bool lassocond = false;

  // used for elastic net
  if(!elasticNet)
  {
    lambda2 = 0; // just in case it is accidentally used, the code still will be correct
  }
  
  arma::vec corr = xtResponses;
  arma::vec abs_corr = abs(corr);
  arma::u32 change_ind;
  double max_corr = abs_corr.max(change_ind); // change_ind gets set here

  betaPath.push_back(beta);
  lambdaPath.push_back(max_corr);
  
  // don't even start!
  if (max_corr < lambda1)
  {
    lambdaPath[0] = lambda1;
    return;
  }

  //arma::u32 data.n_rowsiterations_run = 0;
  // MAIN LOOP
  while ((nActive < data.n_cols) && (max_corr > EPS))
  {
    //data.n_rowsiterations_run++;
    //printf("iteration %d\t", data.n_rowsiterations_run);

    // explicit computation of max correlation, among inactive indices
    change_ind = -1;
    max_corr = 0;
    for (arma::u32 i = 0; i < data.n_cols; i++)
    {
      if (!isActive[i])
      {
        if (fabs(corr(i)) > max_corr)
        {
          max_corr = fabs(corr(i));
          change_ind = i;
        }
      }
    }

    if (!lassocond)
    {
      // index is absolute index
      //printf("activating %d\n", change_ind);
      if (useCholesky)
      {
        arma::vec new_gramcol = arma::vec(nActive);
        for (arma::u32 i = 0; i < nActive; i++)
          new_gramcol[i] = dot(data.col(activeSet[i]), data.col(change_ind));

        CholeskyInsert(data.col(change_ind), new_gramcol);
      }

      // add variable to active set
      Activate(change_ind);
    }

    // compute signs of correlations
    arma::vec s = arma::vec(nActive);
    for (arma::u32 i = 0; i < nActive; i++)
      s(i) = corr(activeSet[i]) / fabs(corr(activeSet[i]));

    // compute "equiangular" direction in parameter space (beta_direction)
    /* We use quotes because in the case of non-unit norm variables,
       this need not be equiangular. */
    arma::vec unnormalized_beta_direction;
    double normalization;
    arma::vec beta_direction;
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
      unnormalized_beta_direction = solve(trimatu(utriCholFactor),
          solve(trimatl(trans(utriCholFactor)), s));

      normalization = 1.0 / sqrt(dot(s, unnormalized_beta_direction));
      beta_direction = normalization * unnormalized_beta_direction;
    }
    else
    {
      arma::mat gramactive = arma::mat(nActive, nActive);
      for (arma::u32 i = 0; i < nActive; i++)
      {
        for (arma::u32 j = 0; j < nActive; j++)
        {
          gramactive(i,j) = gram(activeSet[i], activeSet[j]);
        }
      }

      arma::mat S = s * arma::ones<arma::mat>(1, nActive);
      unnormalized_beta_direction =
          solve(gramactive % trans(S) % S, arma::ones<arma::mat>(nActive, 1));
      normalization = 1.0 / sqrt(sum(unnormalized_beta_direction));
      beta_direction = normalization * unnormalized_beta_direction % s;
    }

    // compute "equiangular" direction in output space
    ComputeYHatDirection(beta_direction, responseshat_direction);


    double gamma = max_corr / normalization;

    // if not all variables are active
    if (nActive < data.n_cols)
    {
      // compute correlations with direction
      for (arma::u32 ind = 0; ind < data.n_cols; ind++)
      {
        if (isActive[ind])
        {
          continue;
        }

        double dir_corr = dot(data.col(ind), responseshat_direction);
        double val1 = (max_corr - corr(ind)) / (normalization - dir_corr);
        double val2 = (max_corr + corr(ind)) / (normalization + dir_corr);
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
      double lassobound_on_gamma = DBL_MAX;
      arma::u32 active_ind_to_kick_out = -1;

      for (arma::u32 i = 0; i < nActive; i++)
      {
        double val = -beta(activeSet[i]) / beta_direction(i);
        if ((val > 0) && (val < lassobound_on_gamma))
        {
          lassobound_on_gamma = val;
          active_ind_to_kick_out = i;
        }
      }

      if (lassobound_on_gamma < gamma)
      {
        //printf("%d: gap = %e\tbeta(%d) = %e\n",
        //    activeSet[active_ind_to_kick_out],
        //    gamma - lassobound_odata.n_rowsgamma,
        //    activeSet[active_ind_to_kick_out],
        //    beta(activeSet[active_ind_to_kick_out]));
        gamma = lassobound_on_gamma;
        lassocond = true;
        change_ind = active_ind_to_kick_out;
      }
    }

    // update prediction
    responseshat += gamma * responseshat_direction;

    // update estimator
    for (arma::u32 i = 0; i < nActive; i++)
      beta(activeSet[i]) += gamma * beta_direction(i);
    betaPath.push_back(beta);

    if (lassocond)
    {
      // index is in position change_ind in active_set
      //printf("\t\tKICK OUT %d!\n", activeSet[change_ind]);
      if (beta(activeSet[change_ind]) != 0)
      {
        //printf("fixed from %e to 0\n", beta(activeSet[change_ind]));
        beta(activeSet[change_ind]) = 0;
      }

      if (useCholesky)
      {
        CholeskyDelete(change_ind);
      }

      Deactivate(change_ind);
    }

    corr = xtResponses - trans(data) * responseshat;
    if (elasticNet)
    {
      corr -= lambda2 * beta;
    }
    double cur_lambda = 0;
    for (arma::u32 i = 0; i < nActive; i++)
    {
      cur_lambda += fabs(corr(activeSet[i]));
    }
    cur_lambda /= ((double)nActive);

    lambdaPath.push_back(cur_lambda);

    // Time to stop for LASSO?
    if (lasso)
    {
      if (cur_lambda <= lambda1)
      {
        InterpolateBeta();
        break;
      }
    }
  }
}

void LARS::Solution(arma::vec& beta)
{
  beta = beta_path().back();
}

void LARS::GetCholFactor(arma::mat& R)
{
  R = utriCholFactor;
}

void LARS::Deactivate(arma::u32 active_var_ind)
{
  nActive--;
  isActive[activeSet[active_var_ind]] = false;
  activeSet.erase(activeSet.begin() + active_var_ind);
}

void LARS::Activate(arma::u32 var_ind)
{
  nActive++;
  isActive[var_ind] = true;
  activeSet.push_back(var_ind);
}

void LARS::ComputeYHatDirection(const arma::vec& beta_direction,
                                arma::vec& responseshat_direction)
{
  responseshat_direction.fill(0);
  for(arma::u32 i = 0; i < nActive; i++)
    responseshat_direction += beta_direction(i) * data.col(activeSet[i]);
}

void LARS::InterpolateBeta()
{
  int path_length = betaPath.size();

  // interpolate beta and stop
  double ultimate_lambda = lambdaPath[path_length - 1];
  double penultimate_lambda = lambdaPath[path_length - 2];
  double interp = (penultimate_lambda - lambda1)
      / (penultimate_lambda - ultimate_lambda);

  betaPath[path_length - 1] = (1 - interp) * (betaPath[path_length - 2])
      + interp * betaPath[path_length - 1];

  lambdaPath[path_length - 1] = lambda1;
}

void LARS::CholeskyInsert(const arma::vec& new_x, const arma::mat& X)
{
  if (utriCholFactor.n_rows == 0)
  {
    utriCholFactor = arma::mat(1, 1);
    if (elasticNet)
      utriCholFactor(0, 0) = sqrt(dot(new_x, new_x) + lambda2);
    else
      utriCholFactor(0, 0) = norm(new_x, 2);
  }
  else
  {
    arma::vec new_gramcol = trans(X) * new_x;
    CholeskyInsert(new_x, new_gramcol);
  }
}

void LARS::CholeskyInsert(const arma::vec& new_x, const arma::vec& new_gramcol) {
  int n = utriCholFactor.n_rows;

  if (n == 0)
  {
    utriCholFactor = arma::mat(1, 1);
    if (elasticNet)
      utriCholFactor(0, 0) = sqrt(dot(new_x, new_x) + lambda2);
    else
      utriCholFactor(0, 0) = norm(new_x, 2);
  }
  else
  {
    arma::mat new_R = arma::mat(n + 1, n + 1);

    double sq_norm_new_x;
    if (elasticNet)
      sq_norm_new_x = dot(new_x, new_x) + lambda2;
    else
      sq_norm_new_x = dot(new_x, new_x);

    arma::vec utriCholFactork = solve(trimatl(trans(utriCholFactor)),
        new_gramcol);

    new_R(arma::span(0, n - 1), arma::span(0, n - 1)) = utriCholFactor;
    new_R(arma::span(0, n - 1), n) = utriCholFactork;
    new_R(n, arma::span(0, n - 1)).fill(0.0);
    new_R(n, n) = sqrt(sq_norm_new_x - dot(utriCholFactork, utriCholFactork));

    utriCholFactor = new_R;
  }
}

void LARS::GivensRotate(const arma::vec& x, arma::vec& rotated_x, arma::mat& G) 
{
  if (x(1) == 0)
  {
    G = arma::eye(2, 2);
    rotated_x = x;
  }
  else
  {
    double r = norm(x, 2);
    G = arma::mat(2, 2);

    double scaled_x1 = x(0) / r;
    double scaled_x2 = x(1) / r;

    G(0, 0) = scaled_x1;
    G(1, 0) = -scaled_x2;
    G(0, 1) = scaled_x2;
    G(1, 1) = scaled_x1;

    rotated_x = arma::vec(2);
    rotated_x(0) = r;
    rotated_x(1) = 0;
  }
}

void LARS::CholeskyDelete(arma::u32 col_to_kill)
{
  arma::u32 n = utriCholFactor.n_rows;

  if (col_to_kill == (n - 1))
  {
    utriCholFactor = utriCholFactor(arma::span(0, n - 2), arma::span(0, n - 2));
  }
  else
  {
    utriCholFactor.shed_col(col_to_kill); // remove column col_to_kill
    n--;

    for(arma::u32 k = col_to_kill; k < n; k++)
    {
      arma::mat G;
      arma::vec rotated_vec;
      GivensRotate(utriCholFactor(arma::span(k, k + 1), k), rotated_vec, G);
      utriCholFactor(arma::span(k, k + 1), k) = rotated_vec;
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

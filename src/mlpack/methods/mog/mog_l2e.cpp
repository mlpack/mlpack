/**
 * @author Parikshit Ram (pram@cc.gatech.edu)
 * @file mog_l2e.cpp
 *
 * Implementation for L2 loss function, and
 * also some initial points generator
 *
 */
#include "mog_l2e.hpp"
#include "phi.hpp"
#include "kmeans.hpp"

using namespace mlpack;
using namespace gmm;

long double MoGL2E::L2Error(const arma::mat& data) {
  long double reg, fit, l2e;

  reg = RegularizationTerm_();
  fit = GoodnessOfFitTerm_(data);
  l2e = reg - (2 * fit) / data.n_cols;

  return l2e;
}

long double MoGL2E::L2Error(const arma::mat& data, arma::vec& gradients) {

  long double reg, fit, l2e;

  arma::vec g_reg, g_fit;
  reg = RegularizationTerm_(g_reg);
  fit = GoodnessOfFitTerm_(data, g_fit);

  gradients = g_reg - (2 * g_fit) / data.n_cols;

  l2e = reg - (2 * fit) / data.n_cols;

  return l2e;
}

long double MoGL2E::RegularizationTerm_() {
  arma::mat phi_mu, sum_covar;
  arma::vec x;
  long double reg, tmpVal;

  phi_mu.set_size(number_of_gaussians_, number_of_gaussians_);
  sum_covar.set_size(dimension_, dimension_);
  x = omega_;

  for (size_t k = 1; k < number_of_gaussians_; k++) {
    for (size_t j = 0; j < k; j++) {
      sum_covar = sigma_[k] + sigma_[j];

      tmpVal = phi(mu_[k], mu_[j], sum_covar);
      phi_mu(j, k) = tmpVal;
      phi_mu(k, j) = tmpVal;
    }
  }

  for(size_t k = 0; k < number_of_gaussians_; k++) {
    sum_covar = 2 * sigma_[k];

    phi_mu(k, k) = phi(mu_[k], mu_[k], sum_covar);
  }

  // Calculating the reg value
  reg = dot(x, x * phi_mu);

  return reg;
}

long double MoGL2E::RegularizationTerm_(arma::vec& g_reg) {
  arma::mat phi_mu, sum_covar;
  arma::vec x, y;
  long double reg, tmpVal;

  arma::vec df_dw, g_omega;
  std::vector<arma::vec> g_mu, g_sigma;
  std::vector<std::vector<arma::vec> > dp_d_mu, dp_d_sigma;

  phi_mu.set_size(number_of_gaussians_, number_of_gaussians_);
  sum_covar.set_size(dimension_, dimension_);
  x = omega_;

  g_mu.resize(number_of_gaussians_);
  g_sigma.resize(number_of_gaussians_);
  dp_d_mu.resize(number_of_gaussians_);
  dp_d_sigma.resize(number_of_gaussians_);
  for(size_t k = 0; k < number_of_gaussians_; k++){
    dp_d_mu[k].resize(number_of_gaussians_);
    dp_d_sigma[k].resize(number_of_gaussians_);
  }

  for(size_t k = 1; k < number_of_gaussians_; k++) {
    for(size_t j = 0; j < k; j++) {
      sum_covar = sigma_[k] * sigma_[j];

      std::vector<arma::mat> tmp_d_cov;
      arma::vec tmp_dp_d_sigma;

      tmp_d_cov.resize(dimension_ * (dimension_ + 1));

      for(size_t i = 0; i < (dimension_ * (dimension_ + 1) / 2); i++) {
        tmp_d_cov[i] = (d_sigma_[k])[i];
        tmp_d_cov[(dimension_ * (dimension_ + 1) / 2) + i] = (d_sigma_[j])[i];
      }

      tmpVal = phi(mu_[k], mu_[j], sum_covar, tmp_d_cov, dp_d_mu[j][k],
          tmp_dp_d_sigma);

      phi_mu(j, k) = tmpVal;
      phi_mu(k, j) = tmpVal;

      dp_d_mu[k][j] = -dp_d_mu[j][k];

      arma::vec tmp_dp_1(tmp_dp_d_sigma.n_elem / 2);
      arma::vec tmp_dp_2(tmp_dp_d_sigma.n_elem / 2);
      for (size_t i = 0; i < tmp_dp_1.n_elem; i++) {
        tmp_dp_1[i] = tmp_dp_d_sigma[i];
        tmp_dp_2[i] = tmp_dp_d_sigma[(dimension_ * (dimension_ + 1) / 2) + i];
      }

      dp_d_sigma[j][k] = tmp_dp_1;
      dp_d_sigma[k][j] = tmp_dp_2;
    }
  }

  for (size_t k = 0; k < number_of_gaussians_; k++) {
    sum_covar = 2 * sigma_[k];

    arma::vec junk;
    tmpVal = phi(mu_[k], mu_[k], sum_covar, d_sigma_[k], junk,
        dp_d_sigma[k][k]);

    phi_mu(k, k) = tmpVal;

    dp_d_mu[k][k].zeros(dimension_);
  }

  // Calculating the reg value
  reg = dot(x, x * phi_mu);

  // Calculating the g_omega values - a vector of size K-1
  df_dw = 2.0 * y;
  g_omega = d_omega_ * df_dw;

  // Calculating the g_mu values - K vectors of size D
  for (size_t k = 0; k < number_of_gaussians_; k++) {
    g_mu[k].zeros(dimension_);

    for (size_t j = 0; j < number_of_gaussians_; j++) {
      g_mu[k] += x[j] * dp_d_mu[j][k];
      g_mu[k] *= 2.0 * x[k];
    }

    // Calculating the g_sigma values - K vectors of size D(D+1)/2
    for (size_t k = 0; k < number_of_gaussians_; k++) {
      g_sigma[k].zeros((dimension_ * (dimension_ + 1)) / 2);
      for (size_t j = 0; j < number_of_gaussians_; j++)
        g_sigma[k] += x[k] * dp_d_sigma[j][k];
      g_sigma[k] *= 2.0 * x[k];
    }

    // Making the single gradient vector of size K*(D+1)*(D+2)/2 - 1
    arma::vec tmp_g_reg((number_of_gaussians_ * (dimension_ + 1) *
        (dimension_ * 2) / 2) - 1);
    size_t j = 0;
    for (size_t k = 0; k < g_omega.n_elem; k++)
      tmp_g_reg[k] = g_omega[k];
    j = g_omega.n_elem;

    for (size_t k = 0; k < number_of_gaussians_; k++) {
      for (size_t i = 0; i < dimension_; i++)
        tmp_g_reg[j + (k * dimension_) + i] = (g_mu[k])[i];

      for(size_t i = 0; i < (dimension_ * (dimension_ + 1) / 2); i++) {
        tmp_g_reg[j + (number_of_gaussians_ * dimension_)
            + k * (dimension_ * (dimension_ + 1) / 2)
            + i] = (g_sigma[k])[i];
      }
    }

    g_reg = tmp_g_reg;
  }

  return reg;
}

long double MoGL2E::GoodnessOfFitTerm_(const arma::mat& data) {
  long double fit;
  arma::mat phi_x(number_of_gaussians_, data.n_cols);
  arma::vec identity_vector;

  identity_vector.ones(data.n_cols);

  for (size_t k = 0; k < number_of_gaussians_; k++)
    for (size_t i = 0; i < data.n_cols; i++)
      phi_x(k, i) = phi(data.unsafe_col(i), mu_[k], sigma_[k]);

  fit = dot(omega_ * phi_x, identity_vector);

  return fit;
}

long double MoGL2E::GoodnessOfFitTerm_(const arma::mat& data,
                                       arma::vec& g_fit) {
  long double fit;
  arma::mat phi_x(number_of_gaussians_, data.n_cols);
  arma::vec weights, x, y, identity_vector;
  arma::vec g_omega,tmp_g_omega;
  std::vector<arma::vec> g_mu, g_sigma;

  weights = omega_;
  x.set_size(data.n_rows);
  identity_vector.ones(data.n_cols);

  g_mu.resize(number_of_gaussians_);
  g_sigma.resize(number_of_gaussians_);

  for(size_t k = 0; k < number_of_gaussians_; k++) {
    g_mu[k].zeros(dimension_);
    g_sigma[k].zeros(dimension_ * (dimension_ + 1) / 2);

    for (size_t i = 0; i < data.n_cols; i++) {
      arma::vec tmp_g_mu, tmp_g_sigma;
      phi_x(k, i) = phi(data.unsafe_col(i), mu_[k], sigma_[k], d_sigma_[k],
          tmp_g_mu, tmp_g_sigma);

      g_mu[k] += tmp_g_mu;
      g_sigma[k] = tmp_g_sigma;
    }

    g_mu[k] *= weights[k];
    g_sigma[k] *= weights[k];
  }

  fit = dot(weights * phi_x, identity_vector);

  // Calculating the g_omega
  tmp_g_omega = phi_x * identity_vector;
  g_omega = d_omega_ * tmp_g_omega;

  // Making the single gradient vector of size K*(D+1)*(D+2)/2
  arma::vec tmp_g_fit((number_of_gaussians_ * (dimension_ + 1) *
      (dimension_ * 2) / 2) - 1);
  size_t j = 0;
  for (size_t k = 0; k < g_omega.n_elem; k++)
    tmp_g_fit[k] = g_omega[k];
  j = g_omega.n_elem;
  for (size_t k = 0; k < number_of_gaussians_; k++) {
    for (size_t i = 0; i < dimension_; i++)
      tmp_g_fit[j + (k * dimension_) + i] = (g_mu[k])[i];

    for (size_t i = 0; i < (dimension_ * (dimension_ + 1) / 2); i++)
      tmp_g_fit[j + number_of_gaussians_ * dimension_
        + k * (dimension_ * (dimension_ + 1) / 2) + i] = (g_sigma[k])[i];
  }

  g_fit = tmp_g_fit;

  return fit;
}

void MoGL2E::MultiplePointsGenerator(arma::mat& points,
                                     const arma::mat& d,
                                     size_t number_of_components) {

  size_t i, j, x;

  for (i = 0; i < points.n_rows; i++)
    for (j = 0; j < points.n_cols - 1; j++)
      points(i, j) = (rand() % 20001) / 1000 - 10;

  for (i = 0; i < points.n_rows; i++) {
    for (j = 0; j < points.n_cols; j++) {
      arma::vec tmp_mu = d.col(rand() % d.n_cols);
      for (x = 0; x < d.n_rows; x++)
        points(i, number_of_components - 1 + (j * d.n_rows) + x) = tmp_mu[x];
    }
  }

  for (i = 0; i < points.n_rows; i++)
    for (j = 0; j < points.n_cols; j++)
      for (x = 0; x < (d.n_rows * (d.n_rows + 1) / 2); x++)
        points(i, (number_of_components * (d.n_rows + 1) - 1)
          + (j * (d.n_rows * (d.n_rows + 1) / 2)) + x) = (rand() % 501) / 100;

  return;
}

void MoGL2E::InitialPointGenerator(arma::vec& theta,
                                   const arma::mat& data,
                                   size_t k_comp) {
  std::vector<arma::vec> means;
  std::vector<arma::mat> covars;
  arma::vec weights;
  double noise;

  weights.set_size(k_comp);
  means.resize(k_comp);
  covars.resize(k_comp);

  theta.set_size(k_comp);

  for (size_t i = 0; i < k_comp; i++) {
    means[i].set_size(data.n_rows);
    covars[i].set_size(data.n_rows, data.n_rows);
  }

  KMeans(data, k_comp, means, covars, weights);

  for (size_t k = 0; k < k_comp - 1; k++) {
    noise = (double) (rand() % 10000) / (double) 1000;
    theta[k] = noise - 5;
  }

  for (size_t k = 0; k < k_comp; k++) {
    for (size_t j = 0; j < data.n_rows; j++)
      theta[k_comp - 1 + k * data.n_rows + j] = (means[k])[j];

    arma::mat u = chol(covars[k]);
    for(size_t j = 0; j < data.n_rows; j++)
      for(size_t i = 0; i < j + 1; i++)
        theta[k_comp - 1 + (k_comp * data.n_rows)
            + (k * data.n_rows * (data.n_rows + 1) / 2)
            + (j * (j + 1) / 2 + i)] = u(i, j) + ((rand() % 501) / 100);
  }
}

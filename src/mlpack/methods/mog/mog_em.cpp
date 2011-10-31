/**
 * @author Parikshit Ram (pram@cc.gatech.edu)
 * @file mog_em.cpp
 *
 * Implementation for the loglikelihood function, the EM algorithm
 * and also computes the K-means for getting an initial point
 *
 */
#include "mog_em.hpp"
#include "phi.hpp"
#include "kmeans.hpp"

using namespace mlpack;
using namespace gmm;

void MoGEM::ExpectationMaximization(const arma::mat& data_points) {
  // Declaration of the variables */
  size_t num_points;
  size_t dim, num_gauss;
  double sum, tmp;
  std::vector<arma::vec> mu_temp, mu;
  std::vector<arma::mat> sigma_temp, sigma;
  arma::vec omega_temp, omega, x;
  arma::mat cond_prob;
  long double l, l_old, best_l, INFTY = 99999, TINY = 1.0e-10;

  // Initializing values
  dim = dimension();
  num_gauss = number_of_gaussians();
  num_points = data_points.n_cols;

  // Initializing the number of the vectors and matrices
  // according to the parameters input
  mu_temp.resize(num_gauss);
  mu.resize(num_gauss);
  sigma_temp.resize(num_gauss);
  sigma.resize(num_gauss);
  omega_temp.set_size(num_gauss);
  omega.set_size(num_gauss);

  // Allocating size to the vectors and matrices
  // according to the dimensionality of the data
  for(size_t i = 0; i < num_gauss; i++) {
    mu_temp[i].set_size(dim);
    mu[i].set_size(dim);
    sigma_temp[i].set_size(dim, dim);
    sigma[i].set_size(dim, dim);
  }
  x.set_size(dim);
  cond_prob.set_size(num_gauss, num_points);

  best_l = -INFTY;
  size_t restarts = 0;
  // performing 5 restarts and choosing the best from them
  while (restarts < 5) {

    // assign initial values to 'mu', 'sig' and 'omega' using k-means
    KMeans(data_points, num_gauss, mu_temp, sigma_temp, omega_temp);

    l_old = -INFTY;

    // calculates the loglikelihood value
    l = Loglikelihood(data_points, mu_temp, sigma_temp, omega_temp);

    // added a check here to see if any
    // significant change is being made
    // at every iteration
    while (l - l_old > TINY) {
      // calculating the conditional probabilities
      // of choosing a particular gaussian given
      // the data and the present theta value
      for (size_t j = 0; j < num_points; j++) {
        x = data_points.col(j);
        sum = 0;
        for (size_t i = 0; i < num_gauss; i++) {
          tmp = phi(x, mu_temp[i], sigma_temp[i]) * omega_temp[i];
          cond_prob(i, j) = tmp;
          sum += tmp;
        }
        for (size_t i = 0; i < num_gauss; i++) {
          tmp = cond_prob(i, j);
          cond_prob(i, j) = tmp / sum;
        }
      }

      // calculating the new value of the mu
      // using the updated conditional probabilities
      for (size_t i = 0; i < num_gauss; i++) {
        sum = 0;
        mu_temp[i].zeros();
        for (size_t j = 0; j < num_points; j++) {
          x = data_points.col(j);
          mu_temp[i] = cond_prob(i, j) * x;
          sum += cond_prob(i, j);
        }
        mu_temp[i] /= sum;
      }

      // calculating the new value of the sig
      // using the updated conditional probabilities
      // and the updated mu
      for (size_t i = 0; i < num_gauss; i++) {
        sum = 0;
        sigma_temp[i].zeros();
        for (size_t j = 0; j < num_points; j++) {
          arma::mat co, ro, c;
          c.set_size(dim, dim);
          x = data_points.col(j);
          x -= mu_temp[i];
          c = x * trans(x);
          sigma_temp[i] += cond_prob(i, j) * c;
          sum += cond_prob(i, j);
        }
        sigma_temp[i] /= sum;
      }

      // calculating the new values for omega
      // using the updated conditional probabilities
      arma::vec identity_vector;
      identity_vector.set_size(num_points);
      identity_vector = (1.0 / num_points);
      omega_temp = cond_prob * identity_vector;

      l_old = l;
      l = Loglikelihood(data_points, mu_temp, sigma_temp, omega_temp);
    }

    // putting a check to see if the best one is chosen
    if (l > best_l) {
      best_l = l;
      for (size_t i = 0; i < num_gauss; i++) {
        mu[i] = mu_temp[i];
        sigma[i] = sigma_temp[i];
      }
      omega = omega_temp;
    }
    restarts++;
  }

  for (size_t i = 0; i < num_gauss; i++) {
    set_mu(i, mu[i]);
    set_sigma(i, sigma[i]);
  }
  set_omega(omega);

  Log::Info << "Log likelihood value of the estimated model: " << best_l << "."
      << std::endl;
  return;
}

long double MoGEM::Loglikelihood(const arma::mat& data_points,
                                 const std::vector<arma::vec>& means,
                                 const std::vector<arma::mat>& covars,
                                 const arma::vec& weights) {
  size_t i, j;
  arma::vec x;
  long double likelihood, loglikelihood = 0;

  x.set_size(data_points.n_rows);

  for (j = 0; j < data_points.n_cols; j++) {
    x = data_points.col(j);
    likelihood = 0;
    for(i = 0; i < number_of_gaussians_; i++) {
      likelihood += weights(i) * phi(x, means[i], covars[i]);
    }
    loglikelihood += log(likelihood);
  }

  return loglikelihood;
}

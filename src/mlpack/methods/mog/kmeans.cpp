/**
 * @author Parikshit Ram (pram@cc.gatech.edu)
 * @file kmeans.cpp
 *
 * Implementation for the K-means method for getting an initial point.
 *
 */
#include "kmeans.hpp"

namespace mlpack {
namespace gmm {

void KMeans(const arma::mat& data,
            const size_t value_of_k,
            std::vector<arma::vec>& means,
            std::vector<arma::mat>& covars,
            arma::vec& weights) {
  // Set size of vectors and matrices properly.
  means.resize(value_of_k);
  covars.resize(value_of_k);
  for (size_t i = 0; i < value_of_k; i++) {
    means[i].set_size(value_of_k);
    covars[i].set_size(value_of_k, value_of_k);
  }
  weights.set_size(value_of_k);

  std::vector<arma::vec> mu, mu_old;
  double* tmpssq = new double[value_of_k];
  double* sig = new double[value_of_k];
  double* sig_best = new double[value_of_k];
  size_t* y = new size_t[value_of_k];
  arma::vec x, diff;
  arma::mat ssq;
  size_t i, j, k, n, t, dim;
  double score, score_old, sum;

  n = data.n_cols;
  dim = data.n_rows;
  mu.resize(value_of_k);
  mu_old.resize(value_of_k);
  ssq.set_size(n, value_of_k);

  for (i = 0; i < value_of_k; i++) {
    mu[i].set_size(dim);
    mu_old[i].set_size(dim);
  }

  x.set_size(dim);
  diff.set_size(dim);

  score_old = 999999;

  // putting 5 random restarts to obtain the k-means
  for (i = 0; i < 5; i++) {
    t = -1;
    for (k = 0; k < value_of_k; k++){
      t = (t + 1 + (rand() % ((n - 1 - (value_of_k - k)) - (t + 1))));
      mu[k] = data.col(t);
      for(j = 0; j < n; j++) {
        x = data.col(j);
        diff = mu[k] - x;
        ssq(j, k) = dot(diff, diff);
      }
    }
    // This should be an Armadillo function, really.
    double min_val = DBL_MAX;
    for (i = 0; i < ssq.n_rows; i++) {
      for (k = 0; k < ssq.n_cols; k++) {
        if (ssq(i, k) < min_val) {
          min_val = ssq(i, k);
          y[i] = k;
        }
      }
    }

    do {
      for (k = 0; k < value_of_k; k++)
        mu_old[k] = mu[k];

      for(k = 0; k < value_of_k; k++) {
        size_t p = 0;
        mu[k].zeros();
        for (j = 0; j < n; j++) {
          x = data.col(j);
          if (y[j] == k) {
            mu[k] += x;
            p++;
          }
        }

        if (p != 0)
          mu[k] /= p;

        for (j = 0; j < n; j++) {
          x = data.col(j);
          diff = mu[k] - x;
          ssq(j, k) = dot(diff, diff);
        }
      }
      // This should be an Armadillo function, really.
      min_val = DBL_MAX;
      for (i = 0; i < ssq.n_rows; i++) {
        for (k = 0; k < ssq.n_cols; k++) {
          if (ssq(i, k) < min_val) {
            min_val = ssq(i, k);
            y[i] = k;
          }
        }
      }

      sum = 0;
      for(k = 0; k < value_of_k; k++) {
        diff = mu[k] - mu_old[k];
        sum += dot(diff, diff);
      }

    } while (sum != 0);

    for (k = 0; k < value_of_k; k++) {
      size_t p = 0;
      tmpssq[k] = 0;
      for (j = 0; j < n; j++) {
        if (y[j] == k) {
          tmpssq[k] += ssq(j, k);
          p++;
        }
      }
      sig[k] = sqrt(tmpssq[k] / p);
    }

    score = 0;
    for(k = 0; k < value_of_k; k++) {
      score += tmpssq[k];
    }
    score = score / n;

    if (score < score_old) {
      score_old = score;
      for(k = 0; k < value_of_k; k++){
        means[k] = mu[k];
        sig_best[k] = sig[k];
      }
    }
  }

  for (k = 0; k < value_of_k; k++) {
    x.fill(sig_best[k]);
    covars[k].diag() = x;
  }

  weights.fill(1.0 / value_of_k);

  delete[] tmpssq;
  delete[] sig;
  delete[] sig_best;
  delete[] y;
}

}; // namespace gmm
}; // namespace mlpack

/**
 * @file mixtureDST.h
 *
 * This file contains the class represent a mixture of Gaussian distribution
 * It maintains the priori cluster probabilities, means and covariance matrices
 * of Gaussian cluters in the mixture.
 *
 * It also provides accumulate  mechanism tho update the clusters
 * parameters (i.e mean and covariance) and the prior probabilities.
 * 
 */

#ifndef FASTLIB_MIXTURE_GAUSSIAN_H
#define FASTLIB_MIXTURE_GAUSSIAN_H

#include <fastlib/fastlib.h>
#include <armadillo>
#include "support.h"

class MixtureGauss {
  ////////////// Member variables //////////////////////////////////////
 private:
  /** List of means of clusters */
  std::vector<arma::vec> means;

  /** List of covariance matrices of clusters */
  std::vector<arma::mat> covs;

  /** Prior probabilities of the clusters */
  arma::vec prior;

  /** Inverse of covariance matrices */
  std::vector<arma::mat> inv_covs;

  /** Vector of constant in normal density formula */
  arma::vec det_covs;

  /** Accumulating means */
  std::vector<arma::vec> ACC_means;

  /** Accumulating covariance */
  std::vector<arma::mat> ACC_covs;

  /** Accumulating prior probability */
  arma::vec ACC_prior;

  /** The total denominator to divide by after accumulating complete*/
  double total;

 public:
  /** Init the mixture from file */
  void InitFromFile(const char* mean_fn, const char* covs_fn = NULL, const char* prior_fn = NULL);

  /** 
   * Init the mixture from profile, read from a list of matrices
   * start with the prior vector, follows by the mean and covariance
   * of each cluster.
   */
  void InitFromProfile(const std::vector<arma::mat>& matlst, int start, int N);

  /** Init with K clusters and dimension N */
  void Init(int K, int N);

  /** Init with K clusters and the data with label */
  void Init(int K, const arma::mat& data, const std::vector<int>& labels);

  /** Print the mixture to stdout for debugging */
  void print_mixture(const char* s) const;

  /** Generate a random vector from the mixture distribution */
  void generate(arma::vec& v) const;

  /** Get the value of density funtion at a given point */  
  double getPDF(const arma::vec& v) const;

  /** Get the value of density funtion of each cluster at a given point */  
  double getPDF(int cluster, const arma::vec& v) const;

  /** Get the prior probabilities vector */
  const arma::vec& get_prior() const { return prior; }

  /** Get the mean of certain cluster */
  const arma::vec& get_mean(int k) const { return means[k]; }

  /** Get the covariance of certain cluster */
  const arma::mat& get_cov(int k) const { return covs[k]; }

  /** Get the number of cluster */
  int n_clusters() const { return means.size(); }
  
  /** Get the dimension */
  int v_length() const { return means[0].n_elem; }

  
  /** Start accumulate by setting accumulate variable to zero */
  void start_accumulate() {
    total = 0;
    for (int i = 0; i < means.size(); i++) {
      ACC_means[i].zeros();
      ACC_covs[i].zeros();
      ACC_prior.zeros();
    }
  }

  /** Accumulate a vector v */
  void accumulate(const arma::vec& v) {
    double s = getPDF(v);
    for (int i = 0; i < means.size(); i++) {
      double p = getPDF(i, v) / s;
      ACC_prior[i] += p;
      ACC_means[i] += p * v;
      arma::vec d = means[i] - v;
      ACC_covs[i] += p * (d * trans(d));
    }
    total++;
  }

  /** Accumulate a vector into certain cluster */
  void accumulate_cluster(int i, const arma::vec& v) {
    ACC_means[i] += v;
    ACC_covs[i] += v * trans(v);
    ACC_prior[i]++;
    total++;
  }

  /** Accumulate a vector into certain cluster with weight */
  void accumulate(double p, int i, const arma::vec& v) {
    ACC_means[i] += p * v;
    ACC_covs[i] += p * (v * trans(v));
    ACC_prior[i] += p;
    total += p;
  }

  /** End the accumulate, calculate the new mean, covariance and prior */
  void end_accumulate_cluster() {
    for (int i = 0; i < means.size(); i++) 
      if (ACC_prior[i] != 0) {
        means[i] = ACC_means[i] / ACC_prior[i];

        ACC_covs[i] /= ACC_prior[i];
        ACC_covs[i] -= means[i] * trans(means[i]);
        covs[i] = ACC_covs[i];
	prior[i] = ACC_prior[i] / total;

        inv_covs[i] = inv(covs[i]);
	det_covs[i] = pow(2.0 * math::PI, -means[i].n_elem / 2.0) * pow(det(covs[i]), -0.5);
      }
  }

  /** End the accumulate, calculate the new mean, covariance and prior */  
  void end_accumulate() {
    for (int i = 0; i < means.size(); i++) {
      if (ACC_prior[i] != 0) {
        ACC_covs[i] /= ACC_prior[i];
        ACC_covs[i] -= means[i] * trans(means[i]);
	covs[i] = ACC_covs[i];
	prior[i] = ACC_prior[i] / total;

        inv_covs[i] = covs[i];
	det_covs[i] = pow(2.0 * math::PI, -means[i].n_elem / 2.0) * pow(det(covs[i]), -0.5);
      }
    }
  }
};

#endif

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
#include "fastlib/fastlib.h"
#include "support.h"
class MixtureGauss {
  ////////////// Member variables //////////////////////////////////////
 private:
  /** List of means of clusters */
  ArrayList<Vector> means;

  /** List of covariance matrices of clusters */
  ArrayList<Matrix> covs;

  /** Prior probabilities of the clusters */
  Vector prior;

  /** Inverse of covariance matrices */
  ArrayList<Matrix> inv_covs;

  /** Vector of constant in normal density formula */
  Vector det_covs;

  /** Accumulating means */
  ArrayList<Vector> ACC_means;

  /** Accumulating covariance */
  ArrayList<Matrix> ACC_covs;

  /** Accumulating prior probability */
  Vector ACC_prior;

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
  void InitFromProfile(const ArrayList<Matrix>& matlst, int start, int N);

  /** Init with K clusters and dimension N */
  void Init(int K, int N);

  /** Init with K clusters and the data with label */
  void Init(int K, const Matrix& data, const ArrayList<int>& labels);

  /** Print the mixture to stdout for debugging */
  void print_mixture(const char* s) const;

  /** Generate a random vector from the mixture distribution */
  void generate(Vector* v) const;

  /** Get the value of density funtion at a given point */  
  double getPDF(const Vector& v) const;

  /** Get the value of density funtion of each cluster at a given point */  
  double getPDF(int cluster, const Vector& v) const;

  /** Get the prior probabilities vector */
  const Vector& get_prior() const { return prior; }

  /** Get the mean of certain cluster */
  const Vector& get_mean(int k) const { return means[k]; }

  /** Get the covariance of certain cluster */
  const Matrix& get_cov(int k) const { return covs[k]; }

  /** Get the number of cluster */
  int n_clusters() const { return means.size(); }
  
  /** Get the dimension */
  int v_length() const { return means[0].length(); }

  
  /** Start accumulate by setting accumulate variable to zero */
  void start_accumulate() {
    total = 0;
    for (int i = 0; i < means.size(); i++) {
      ACC_means[i].SetZero();
      ACC_covs[i].SetZero();
      ACC_prior.SetZero();
    }
  }

  /** Accumulate a vector v */
  void accumulate(const Vector& v) {
    double s = getPDF(v);
    for (int i = 0; i < means.size(); i++) {
      double p = getPDF(i, v) / s;
      ACC_prior[i] += p;
      la::AddExpert(p, v, &ACC_means[i]);
      Vector d;
      la::SubInit(v, means[i], &d);
      Matrix D;
      D.AliasColVector(d);
      la::MulExpert(p, false, D, true, D, 1.0, &ACC_covs[i]);
    }
    total ++;
  }

  /** Accumulate a vector into certain cluster */
  void accumulate_cluster(int i, const Vector& v) {
    la::AddTo(v, &ACC_means[i]);
    Matrix V;
    V.AliasColVector(v);
    la::MulExpert(1.0, false, V, true, V, 1.0, &ACC_covs[i]);
    ACC_prior[i]++;
    total++;
  }

  /** Accumulate a vector into certain cluster with weight */
  void accumulate(double p, int i, const Vector& v) {
    la::AddExpert(p, v, &ACC_means[i]);
    Matrix V;
    V.AliasColVector(v);
    la::MulExpert(p, false, V, true, V, 1.0, &ACC_covs[i]);
    ACC_prior[i] += p;
    total += p;
  }

  /** End the accumulate, calculate the new mean, covariance and prior */
  void end_accumulate_cluster() {
    for (int i = 0; i < means.size(); i++) 
      if (ACC_prior[i] != 0) {
	la::ScaleOverwrite(1.0/ACC_prior[i], ACC_means[i], &means[i]);
	Matrix M;
	M.AliasColVector(means[i]);
	la::MulExpert(-1.0, false, M, true, M, 1.0/ACC_prior[i], &ACC_covs[i]);
	covs[i].CopyValues(ACC_covs[i]);
	prior[i] = ACC_prior[i]/total;

	double det = la::Determinant(covs[i]);
	la::InverseOverwrite(covs[i], &inv_covs[i]);
	det_covs[i] = pow(2.0*math::PI, -means[i].length()/2.0) * pow(det, -0.5);
      }
  }

  /** End the accumulate, calculate the new mean, covariance and prior */  
  void end_accumulate() {
    for (int i = 0; i < means.size(); i++) {
      if (ACC_prior[i] != 0) {
	la::ScaleOverwrite(1.0/ACC_prior[i], ACC_means[i], &means[i]);
	Matrix M;
	M.AliasColVector(means[i]);
	la::MulExpert(-1.0, false, M, true, M, 1.0/ACC_prior[i], &ACC_covs[i]);
	covs[i].CopyValues(ACC_covs[i]);
	prior[i] = ACC_prior[i]/total;

	double det = la::Determinant(covs[i]);
	la::InverseOverwrite(covs[i], &inv_covs[i]);
	det_covs[i] = pow(2.0*math::PI, -means[i].length()/2.0) * pow(det, -0.5);
      }
    }
  }
};
#endif

#ifndef FASTLIB_HMM_SUPPORT_H
#define FASTLIB_HMM_SUPPORT_H

#include <fastlib/fastlib.h>
#include <armadillo>

namespace hmm_support {
  /** Generate uniform random value in [0, 1] */
  double RAND_UNIFORM_01();

  /** Generate uniform random value in [a, b] */
  double RAND_UNIFORM(double a, double b);

  /** Generate normal (Gaussian) random value N(0, 1) */
  double RAND_NORMAL_01();

  /** Generate normal (Gaussian) random vector N(0, 1) */
  void RAND_NORMAL_01_INIT(int N, arma::vec& v);

  /** Generate normal (Gaussian) random vector N(mean, cov^2) */
  void RAND_NORMAL_INIT(const arma::vec& mean, const arma::mat& cov, arma::vec& v);

  /** Calculate quadratic form x'Ay */
  double MyMulExpert(const arma::vec& x, const arma::mat& A, const arma::vec& y);

  /** Compute normal density function */
  double NORMAL_DENSITY(const arma::vec& x, const arma::vec& mean, const arma::mat& inv_cov, double det_cov);

  /** Print a matrix to stdout */
  void print_matrix(const arma::mat& a, const char* msg);

  /** Print a matrix to a TextWriter object */
  void print_matrix(TextWriter& writer, const arma::mat& a, const char* msg, const char* format = "%f,");
  
  /** Print a vector to stdout */
  void print_vector(const arma::vec& a, const char* msg);

  /** Print a vector to a TextWriter object */
  void print_vector(TextWriter& writer, const arma::vec& a, const char* msg, const char* format = "%f,");

  /** Compute the centroids and label the samples by K-means algorithm */
  bool kmeans(const std::vector<arma::mat>& data, int num_clusters, 
	      std::vector<int>& labels_, std::vector<arma::vec>& centroids_, 
	      int max_iter = 1000, double error_thresh = 1e-3);

  bool kmeans(const arma::mat &data, int num_clusters, 
	      std::vector<int>& labels_, std::vector<arma::vec>& centroids_, 
	      int max_iter = 1000, double error_thresh = 1e-04);

  /** Convert a matrix in to an array list of vectors of its column */
  void mat2arrlst(arma::mat& a, std::vector<arma::vec>& seqs);
  
  /** Convert a matrix in to an array list of matrices of slice of its columns */
  void mat2arrlstmat(int N, arma::mat& a, std::vector<arma::mat>& seqs);

  /**
   * Load an array list of matrices from file where the matrices
   * are seperated by a line start with %
   */
  success_t load_matrix_list(const char* filename, std::vector<arma::mat>& matlst);

  /** 
   * Load an array list of vectors from file where the vectors
   * are seperated by a line start with %
   */
  success_t load_vector_list(const char* filename, std::vector<arma::vec>& veclst);
};

#endif


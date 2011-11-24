/**
 * @file infomax_ica.hpp
 * @author Chip Mappus
 *
 * Yet another infomax ICA implementation.
 *
 * Bell, A. and Sejnowski,T. (1995)
 * "An information maximisation approach to blind signal
 * separation."  Neural Computation. 1129-1159.
 *
 * For details:
 * http://www.cnl.salk.edu/~tony/ica.html
 *
 */
#ifndef __MLPACK_METHODS_INFOMAX_ICA_INFOMAX_ICA_HPP
#define __MLPACK_METHODS_INFOMAX_ICA_INFOMAX_ICA_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace infomax_ica {

class TestInfomaxICA; // forward reference

/**
 * Infomax ICA. Given an observation matrix and input parameters,
 * return the corresponding unmixming matrix, W.
 * Exmaple use:
 *
 * @code
 *   InfomaxICA *ica = new InfomaxICA(lambda, B, epsilon);
 *   Matrix west;
 *   ica->applyICA(dataset);
 *   ica->getUnmixing(west);
 * @endcode
 */
class InfomaxICA
{
 public:
  InfomaxICA();
  InfomaxICA(double lambda, size_t B, double epsilon);
  void applyICA(const arma::mat& dataset);
  void evaluateICA();
  void displayMatrix(const arma::mat& m);
  void displayVector(const arma::vec& m);
  void getUnmixing(arma::mat& w);
  void getSources(const arma::mat& dataset, arma::mat& s);
  void setLambda(const double lambda);
  void setB(const size_t b);
  void setEpsilon(const double epsilon);

  arma::mat sampleCovariance(const arma::mat& m);
  arma::mat sqrtm(const arma::mat& m);

 private:
  arma::mat w_;
  arma::mat data_;
  // learning rate
  double lambda_;
  // block size
  size_t b_;
  // epsilon for convergence
  double epsilon_;
  // utility functions
  void sphere(arma::mat& m);
  arma::mat subMeans(const arma::mat& m);
  arma::vec rowMean(const arma::mat& m);
  double w_delta(const arma::mat& w_prev, const arma::mat& w_pres);
};

}; // namespace infomax_ica
}; // namespace mlpack

#endif

/**
 * @file infomax_ica.h
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

#ifndef U_INFOMAX_ICA
#define U_INFOMAX_ICA

#include <math.h>
#include <limits.h>
#include <float.h>
#include <fastlib/fastlib.h>
#include <fastlib/fx/io.h>

class TestInfomaxICA; // forward reference

/* const fx_entry_doc infomax_ica_entries[] = {
  {"lambda", FX_PARAM, FX_DOUBLE, NULL,
   "  Learning rate for infomax method.\n"},
  {"B", FX_PARAM, FX_INT, NULL,
   "  Infomax data window size.\n"},
  {"epsilon", FX_PARAM, FX_DOUBLE, NULL,
   "  Infomax algorithm stop threshold.\n"},
  FX_ENTRY_DOC_DONE
}; */




/* const fx_module_doc infomax_ica_doc = {
  infomax_ica_entries, NULL,
  "Performs ICA decomposition using Infomax method.\n"
}; */

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

class InfomaxICA {

  friend class TestInfomaxICA;
  
 public:
  InfomaxICA();
  InfomaxICA(double lambda, index_t B, double epsilon);
  void applyICA(const arma::mat& dataset);
  void evaluateICA();
  void displayMatrix(const arma::mat& m);
  void displayVector(const arma::vec& m);
  void getUnmixing(arma::mat& w);
  void getSources(const arma::mat& dataset, arma::mat& s);
  void setLambda(const double lambda);
  void setB(const index_t b);
  void setEpsilon(const double epsilon);
  
  arma::mat sampleCovariance(const arma::mat& m);
  arma::mat sqrtm(const arma::mat& m);

 private:
  arma::mat w_;
  arma::mat data_;
  // learning rate 
  double lambda_;
  // block size
  index_t b_;
  // epsilon for convergence
  double epsilon_;
  // utility functions
  void sphere(arma::mat& m);
  arma::mat subMeans(const arma::mat& m);
  arma::vec rowMean(const arma::mat& m);
  double w_delta(const arma::mat& w_prev, const arma::mat& w_pres);
};

#endif

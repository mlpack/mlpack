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

class TestInfomaxICA; // forward reference

const fx_entry_doc infomax_ica_entries[] = {
  {"lambda", FX_PARAM, FX_DOUBLE, NULL,
   "  Learning rate for infomax method.\n"},
  {"B", FX_PARAM, FX_INT, NULL,
   "  Infomax data window size.\n"},
  {"epsilon", FX_PARAM, FX_DOUBLE, NULL,
   "  Infomax algorithm stop threshold.\n"},
  FX_ENTRY_DOC_DONE
};

const fx_module_doc infomax_ica_doc = {
  infomax_ica_entries, NULL,
  "Performs ICA decomposition using Infomax method.\n"
};

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
  InfomaxICA(double lambda, int B, double epsilon);
  void applyICA(const Matrix &dataset);
  void evaluateICA();
  void displayMatrix(const Matrix &m);
  void displayVector(const Vector &m);
  void getUnmixing(Matrix &w);
  void getSources(const Matrix &dataset, Matrix &s);
  void setLambda(const double lambda);
  void setB(const int b);
  void setEpsilon(const double epsilon);

 private:
  Matrix w_;
  Matrix data_;
  // learning rate 
  double lambda_;
  // block size
  int b_;
  // epsilon for convergence
  double epsilon_;
  // utility functions
  void expM(Matrix &m);
  void addOne(Matrix &m);
  void invertVals(Matrix &m);
  void vectorize(const Matrix &m, Vector &v);
  Matrix eye(index_t dim,double diagVal);
  Matrix sqrtm(const Matrix &m);
  void sphere(Matrix &m);
  Matrix subMeans(const Matrix &m);
  Vector rowMean(const Matrix &m);
  Matrix sampleCovariance(const Matrix &m);
  double w_delta(const Matrix &w_prev, const Matrix &w_pres);
};

#endif

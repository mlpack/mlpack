/**
 * @file infomaxICA.h
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
#include "fastlib/fastlib.h"

class TestInfomaxICA; // forward reference

/**
 * Infomax ICA. Given an observation matrix, return the
 * corresponding unmixming matrix, W. 
 */

class InfomaxICA {

  friend class TestInfomaxICA;
  
 public:
  InfomaxICA();
  InfomaxICA(double lambda, int B);
  Matrix& getUnmixing();
  Matrix getSources(const Matrix &m);
  void setLambda(double lambda);
  void setB(int b);
  void applyICA(const Dataset& dataset);
  void evaluateICA();
  void displayMatrix(const Matrix &m);
  void displayVector(const Vector &m);

 private:
  Matrix w_;
  Matrix data_;
  // learning rate 
  double lambda_;
  // block size
  int b_;
  // utility functions
  void expM(Matrix &m);
  void addOne(Matrix &m);
  void invertVals(Matrix &m);
  Matrix eye(index_t dim,double diagVal);
  Matrix sqrtm(Matrix &m);
  void sphere(Matrix &m);
  Matrix subMeans(Matrix &m);
  Vector rowMean(Matrix &m);
  Matrix sampleCovariance(Matrix &m);
};

#endif

#ifndef FASTLIB_SUPPORT_HMM_H
#define FASTLIB_SUPPORT_HMM_H

#include <fastlib/fastlib.h>

namespace supportHMM {
  void normalizeRows(Matrix* A);
  void cumulativeSum(const Matrix& A, Matrix* cA);
  double randUniform();
  double VectorSum(const Vector& v);
  void ScaleVector(Vector* v, double s);
  double NormalizeColumn(Matrix* A, index_t j);
  double ColumnSum(const Matrix& A, index_t j);
  void ScaleColumn(Matrix* A, index_t j, double s);
  void CopyLog(const Matrix& A, Matrix* B);
  double normDiff(const Matrix& A, const Matrix& B);
  double multxAy(const Vector& x, const Matrix& A, const Vector& y);
  double RandomNormal();
  void RandomNormal(int dim, Vector* v);
  void RandomNormal(const Vector& mean, const Matrix& cov, Vector* v);
  void RandomInit(int m, int n, Matrix* A);
  void printVector(FILE* f, const Vector& x);
  void printMatrix(FILE* f, const Matrix& x);
};

#endif

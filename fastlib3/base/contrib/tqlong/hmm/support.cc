#include "support.h"

namespace supportHMM {
  void cumulativeSum(const Matrix& A, Matrix* cA) {
    cA->Copy(A);
    for (index_t i = 0; i < cA->n_rows(); i++) 
      for (index_t j = 1; j < cA->n_cols(); j++)
	cA->ref(i, j) += cA->get(i, j-1);
  }

  void normalizeRows(Matrix* A) {
    for (index_t i = 0; i < A->n_rows(); i++) {
      double s = 0;
      for (index_t j = 0; j < A->n_cols(); j++) s += A->get(i, j);
      for (index_t j = 0; j < A->n_cols(); j++) A->ref(i, j) /= s;
    }
  }

  double randUniform() {
    return (double)rand() / RAND_MAX;
  }

  double VectorSum(const Vector& v) {
    double s = 0;
    for (index_t i = 0; i < v.length(); i++)
      s += v[i];
    return s;
  }

  void ScaleVector(Vector* v, double s) {
    for (index_t i = 0; i < v->length(); i++)
      (*v)[i] *= s;
  }

  double NormalizeColumn(Matrix* A, index_t j) {
    double s = ColumnSum(*A, j);
    ScaleColumn(A, j, 1.0/s);
    return s;
  }

  double ColumnSum(const Matrix& A, index_t j) {
    double s = 0.0;
    for (int i = 0; i < A.n_rows(); i++)
      s += A.get(i, j);
    return s;
  }

  void ScaleColumn(Matrix* A, index_t j, double s) {
    for (int i = 0; i < A->n_rows(); i++)
      A->ref(i, j) *= s;
  }

  void CopyLog(const Matrix& A, Matrix* B) {
    for (index_t i = 0; i < A.n_rows(); i++)
      for (index_t j = 0; j < A.n_cols(); j++)
	B->ref(i, j) = log(A.get(i, j));
  }

  double normDiff(const Matrix& A, const Matrix& B) {
    double norm = 0.0;
    for (index_t i = 0; i < A.n_rows(); i++)
      for (index_t j = 0; j < A.n_cols(); j++)
	if (norm < fabs(A.get(i, j)-B.get(i, j)))
	  norm = fabs(A.get(i, j)-B.get(i, j));
    return norm;
  }
	
  double multxAy(const Vector& x, const Matrix& A, const Vector& y) {
    Vector tmp;
    la::MulInit(A, y, &tmp);
    return la::Dot(x, tmp);
  }

  void printVector(FILE* f, const Vector& x) {
    for (index_t i = 0; i < x.length(); i++)
      fprintf(f, "%g,", x[i]);
    fprintf(f, "\n");
  }

  void printMatrix(FILE* f, const Matrix& x) {
    for (index_t i = 0; i < x.n_cols(); i++) {
      Vector x_col;
      x.MakeColumnVector(i, &x_col);
      printVector(f, x_col);
    }
  }

  double RandomNormal() {
    double r = 2, u, v;
    while (r > 1) {
      u = math::Random(-1, 1);
      v = math::Random(-1, 1);
      r = u*u+v*v;
    }
    return sqrt(-2*log(r)/r)*u;
  }

  void RandomNormal(int N, Vector* v) {
    double r, u, t;
    Vector& v_ = *v;
    v_.Init(N);
    for (int i = 0; i < N; i+=2) {
      r = 2;
      while (r > 1) {
	u = math::Random(-1, 1);
	t = math::Random(-1, 1);
	r = u*u+t*t;
      }
      v_[i] = sqrt(-2*log(r)/r)*u;
      if (i+1 < N) v_[i+1] = sqrt(-2*log(r)/r)*t;
    }
  }

  void RandomNormal(const Vector& mean, const Matrix& cov, Vector* v) {
    int N = mean.length();
    Vector v01;
    RandomNormal(N, &v01);
    la::MulInit(cov, v01, v);
    la::AddTo(mean, v);
  }

  void RandomInit(int m, int n, Matrix* A) {
    A->Init(n, n);
    for (index_t i = 0; i < n; i++)
      for (index_t j = 0; j < n; j++)
	A->ref(i, j) = math::Random();
  }
}


#include "la.h"
#include "matrix.h"

#include "base/test.h"

TEST_SUITE_BEGIN(la)

void MakeCountMatrix(index_t n_rows, index_t n_cols, Matrix *m) {
  m->Init(n_rows, n_cols);
  
  for (index_t c = 0; c < n_cols; c++) {
    for (index_t r = 0; r < n_rows; r++) {
      m->set(r, c, r + c);
    }
  }
}

void MakeConstantMatrix(index_t n_rows, index_t n_cols, double v, Matrix *m) {
  m->Init(n_rows, n_cols);
  
  for (index_t c = 0; c < n_cols; c++) {
    for (index_t r = 0; r < n_rows; r++) {
      m->set(r, c, v);
    }
  }
}

void TestMatrixSimpleMath() {
  Matrix m1;
  Matrix m2;
  Matrix m3;
  Matrix m4;
  Matrix m5;
  
  MakeCountMatrix(3, 4, &m1);
  MakeCountMatrix(3, 4, &m2);
  la::MatrixAddTo(m2, &m1);
  la::MatrixAddInit(m1, m2, &m3);
  TEST_ASSERT(m3.get(0, 0) == 0);
  TEST_ASSERT(m3.get(2, 3) == (2+3)*3);
  la::MatrixAddTo(-1.0, m2, &m1);
  TEST_ASSERT(m1.get(0, 0) == 0);
  TEST_ASSERT(m1.get(2, 3) == (2+3));
  la::MatrixScale(4.0, &m1);
  TEST_ASSERT(m1.get(2, 3) == (2+3)*4);
  MakeConstantMatrix(3, 4, 7.0, &m4);
  la::MatrixAddInit(m1, m4, &m5);
  TEST_ASSERT(m5.get(2, 3) == (2+3)*4 + 7.0);
  TEST_ASSERT(m5.get(1, 3) == (1+3)*4 + 7.0);
  TEST_ASSERT(m5.get(1, 0) == (1+0)*4 + 7.0);
}

void MakeCountVector(index_t n, Vector *v) {
  v->Init(n);
  
  for (index_t c = 0; c < n; c++) {
    (*v)[c] = c;
  }
}

void MakeConstantVector(index_t n, double d, Vector *v) {
  v->Init(n);
  
  for (index_t c = 0; c < n; c++) {
    (*v)[c] = d;
  }
}

void TestVectorSimpleMath() {
  Vector v1;
  Vector v2;
  Vector v3;
  Vector v4;
  Vector v5;
  
  MakeCountVector(6, &v1);
  MakeCountVector(6, &v2);
  la::VectorAddTo(v2, &v1);
  la::VectorAddInit(v1, v2, &v3);
  TEST_ASSERT(v3[0] == 0);
  TEST_ASSERT(v3[5] == (5)*3);
  la::VectorAddTo(-1.0, v2, &v1);
  TEST_ASSERT(v1[0] == 0);
  TEST_ASSERT(v1[5] == (5));
  la::VectorScale(4.0, &v1);
  TEST_ASSERT(v1[5] == (5)*4);
  MakeConstantVector(6, 7.0, &v4);
  la::VectorAddInit(v1, v4, &v5);
  TEST_ASSERT(v5[5] == (5)*4 + 7.0);
  TEST_ASSERT(v5[4] == (4)*4 + 7.0);
  TEST_ASSERT(v5[1] == (1)*4 + 7.0);
}

void TestDistance() {
  Vector v1;
  Vector v2;
  
  v1.Init(3);
  v2.Init(3);
  
  v1[0] = 0; v1[1] = 2; v1[2] = 0;
  v2[0] = 2; v2[1] = 1; v2[2] = 0;
  TEST_ASSERT(la::DistanceSqEuclidean(v1, v2) == 5.0);

  v1[0] = 0; v1[1] = 2; v1[2] = 0;
  v2[0] = -2; v2[1] = 1; v2[2] = 0;
  TEST_ASSERT(la::DistanceSqEuclidean(v1, v2) == 5.0);

  v1[0] = -2; v1[1] = 1; v1[2] = 0;
  v2[0] = -2; v2[1] = 1; v2[2] = 0;
  TEST_ASSERT(la::DistanceSqEuclidean(v1, v2) == 0.0);

  v1[0] = -2; v1[1] = DBL_NAN; v1[2] = 0;
  v2[0] = -2; v2[1] = 1; v2[2] = 0;
  TEST_ASSERT(isnan(la::DistanceSqEuclidean(v1, v2)));
}

void TestVector() {
  Vector v1;
  const Vector *v_const;
  Vector v2;
  Vector v3;
  Vector v4;
  Vector v5;
  Vector v6;
  Vector v7;
  Vector v8;
  Vector v9;
  
  MakeCountVector(10, &v1);
  TEST_ASSERT(v1.length() == 10);
  TEST_ASSERT(v1.ptr()[3] == v1[3]);
  v_const = &v1;
  TEST_ASSERT(v_const->ptr()[3] == (*v_const)[3]);
  
  
  v2.Alias(v1);
  TEST_ASSERT(v2[9] == 9);
  TEST_ASSERT(v1.ptr() == v2.ptr());
  v2.MakeSubvector(2, 5, &v3);
  TEST_ASSERT(v3.length() == 5);
  TEST_ASSERT(v3[4] == 6);
  TEST_ASSERT(v3.ptr() != v2.ptr());
  v4.Copy(v3);
  TEST_ASSERT(v4.length() == 5);
  TEST_ASSERT(v4[4] == 6);
  v5.Init(21);
  v5.SetZero();
  TEST_ASSERT(v5[20] == 0.0);
  v6.Alias(v1.ptr(), v1.length());
  TEST_ASSERT(v6[9] == 9);
  TEST_ASSERT(v6[3] == 3);
  v7.Own(&v1);
  TEST_ASSERT(v7[9] == 9);
  TEST_ASSERT(v7[3] == 3);
  v8.WeakCopy(v1);
  TEST_ASSERT(v8[9] == 9);
  TEST_ASSERT(v8[3] == 3);
  MakeConstantVector(10, 3.5, &v9);
  TEST_ASSERT(v9[0] == 3.5);
  v9.SwapValues(&v1);
  TEST_ASSERT(v9[0] == 0.0);
  TEST_ASSERT(v1[0] == 3.5);
  TEST_ASSERT(v2[0] == 3.5);
  TEST_ASSERT(v3[0] == 3.5);
  TEST_ASSERT(v4[0] != 3.5);
  TEST_ASSERT(v6[0] == 3.5);
  TEST_ASSERT(v7[0] == 3.5);
  TEST_ASSERT(v8[0] == 3.5);
  v8.SetZero();
  TEST_ASSERT(v1[0] == 0.0);
}

void TestMatrix() {
  Matrix m1;
  const Matrix *m_const;
  Matrix m2;
  Matrix m3;
  Matrix m4;
  Matrix m5;
  Matrix m6;
  Matrix m7;
  Matrix m8;
  Matrix m9;
  
  MakeCountMatrix(13, 10, &m1);
  TEST_ASSERT(m1.n_cols() == 10);
  TEST_ASSERT(m1.n_rows() == 13);
  TEST_ASSERT(m1.ptr()[3] == m1.get(3, 0));
  m_const = &m1;
  TEST_ASSERT(m_const->ptr()[3] == (*m_const).get(3, 0));

  Vector v1, v2;  
  m1.MakeColumnVector(0, &v1);
  m1.MakeColumnVector(1, &v2);
  TEST_ASSERT(v1[12] == 12);
  TEST_ASSERT(v2[12] == 13);
  
  m2.Alias(m1);
  TEST_ASSERT(m2.get(9, 0) == 9);
  TEST_ASSERT(m1.ptr() == m2.ptr());
  m2.MakeColumnSlice(2, 5, &m3);
  TEST_ASSERT(m3.n_cols() == 5);
  TEST_ASSERT(m3.get(4, 0) == 6);
  TEST_ASSERT(m3.ptr() != m2.ptr());
  m4.Copy(m3);
  TEST_ASSERT(m4.n_cols() == 5);
  TEST_ASSERT(m4.get(4, 0) == 6);
  m5.Init(21, 21);
  m5.SetZero();
  TEST_ASSERT(m5.get(20, 0) == 0.0);
  m6.Alias(m1.ptr(), m1.n_rows(), m1.n_cols());
  TEST_ASSERT(m6.get(9, 0) == 9);
  TEST_ASSERT(m6.get(3, 0) == 3);
  m7.Own(&m1);
  TEST_ASSERT(m7.get(9, 0) == 9);
  TEST_ASSERT(m7.get(3, 0) == 3);
  m8.WeakCopy(m1);
  TEST_ASSERT(m8.get(9, 0) == 9);
  TEST_ASSERT(m8.get(3, 0) == 3);
  MakeConstantMatrix(13, 10, 3.5, &m9);
  TEST_ASSERT(m9.get(0, 0) == 3.5);
  m9.SwapValues(&m1);
  TEST_ASSERT(m9.get(0, 0) == 0.0);
  TEST_ASSERT(m1.get(0, 0) == 3.5);
  TEST_ASSERT(m2.get(0, 0) == 3.5);
  TEST_ASSERT(m3.get(0, 0) == 3.5);
  TEST_ASSERT(m4.get(0, 0) != 3.5);
  TEST_ASSERT(m6.get(0, 0) == 3.5);
  TEST_ASSERT(m7.get(0, 0) == 3.5);
  TEST_ASSERT(m8.get(0, 0) == 3.5);
  m8.SetZero();
  TEST_ASSERT(m1.get(0, 0) == 0.0);
  
  m8.ref(3, 4) = 21.75;
  TEST_ASSERT(m8.get(3, 4) == 21.75);
}

TEST_SUITE_END(la,
    TestMatrixSimpleMath,
    TestVectorSimpleMath,
    TestDistance,
    TestMatrix,
    TestVector)

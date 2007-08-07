/**
 * @file uselapack_test.cc
 *
 * Tests for LAPACK integration.
 */

#include "base/test.h"

#include "uselapack.h"

TEST_SUITE_BEGIN(uselapack);

/**
 * Creates a matrix locally.
 * The matrix cotents are column-major.
 */
#define MAKE_MATRIX_TRANS(name, n_rows, n_cols, contents ...) \
    double name ## _values [] = { contents }; \
    DEBUG_ASSERT(sizeof(name ## _values) / sizeof(double) == n_rows * n_cols); \
    Matrix name; \
    name.Alias(name ## _values, (n_rows), (n_cols));

/**
 * Creates a vector locally.
 * The matrix cotents are column-major.
 */
#define MAKE_VECTOR(name, length, contents ...) \
    double name ## _values [] = { contents }; \
    DEBUG_ASSERT(sizeof(name ## _values) / sizeof(double) == (length)); \
    Vector name; \
    name.Alias(name ## _values, (length));

bool VectorApproxEqual(const Vector& a, const Vector& b,
     double eps) {
  if (a.length() != b.length()) {
    fprintf(stderr, "XXX Size mismatch.\n");
    return false;
  }
  
  int wrong = 0;
  double max_diff = 0;
  
  for (index_t i = 0; i < a.length(); i++) {
    double diff = fabs(a.get(i) - b.get(i));
    max_diff = max(max_diff, diff);
    if (!(diff <= eps)) {
      wrong++;
      if (wrong <= 3) {
        fprintf(stderr, "XXX Mismatch (index %d) zero-based (%e)\n",
            i, diff);
      }
    }
  }
  
  if (wrong) {
    fprintf(stderr, "XXX Total %d mismatches, max diff %e.\n",
        wrong, max_diff);
  }
  
  return wrong == 0;
}

void AssertApproxVector(const Vector& a, const Vector& b, double eps) {
  if (!VectorApproxEqual(a, b, eps)) {
    a.PrintDebug("a");
    b.PrintDebug("b");
    abort();
  }
  //fprintf(stderr, "... Correct vector!\n");
}

bool MatrixApproxEqual(const Matrix& a, const Matrix& b,
     double eps) {
  if (a.n_rows() != b.n_rows() || a.n_cols() != b.n_cols()) {
    fprintf(stderr, "XXX Size mismatch.\n");
    return false;
  }
  
  int wrong = 0;
  double max_diff = 0;
  
  for (index_t c = 0; c < a.n_cols(); c++) {
    for (index_t r = 0; r < a.n_rows(); r++) {
      double diff = fabs(a.get(r, c) - b.get(r, c));
      max_diff = max(max_diff, diff);
      if (!(diff <= eps)) {
        wrong++;
        if (wrong <= 3) {
          fprintf(stderr, "XXX Mismatch (%d, %d) zero-based (%e)\n",
              r, c, diff);
        }
      }
    }
  }
  
  if (wrong) {
    fprintf(stderr, "XXX Total %d mismatches, max diff %e.\n",
        wrong, max_diff);
  }
  
  return wrong == 0;
}

void AssertApproxMatrix(const Matrix& a, const Matrix& b,
    double eps) {
  if (!MatrixApproxEqual(a, b, eps)) {
    a.PrintDebug("a");
    b.PrintDebug("b");
    abort();
  }
  //fprintf(stderr, "... Correct matrix!\n");
}

void AssertExactMatrix(const Matrix& a, const Matrix& b) {
  AssertApproxMatrix(a, b, 0);
}

void AssertApproxTransMatrix(const Matrix& a, const Matrix& b,
    double eps) {
  Matrix a_trans;
  la::TransposeInit(a, &a_trans);
  if (!MatrixApproxEqual(a_trans, b, eps)) {
    a_trans.PrintDebug("a_trans");
    b.PrintDebug("b");
    abort();
  }
}

void TestVectorDot() {
  MAKE_VECTOR(a, 4,    2, 1, 4, 5);
  MAKE_VECTOR(b, 4,    3, 0, 2, -1);
  
  //TEST_DOUBLE_EXACT(F77_FUNC(ddot)(4, a.ptr(), 1, a.ptr(), 1),  4+1+16+25);
  TEST_DOUBLE_EXACT(la::Dot(a, a), 4+1+16+25);
  TEST_DOUBLE_EXACT(la::Dot(a, b), 6+0+8-5);
  TEST_DOUBLE_APPROX(la::LengthEuclidean(b), sqrt(9+0+4+1), 1.0e-8);
  TEST_DOUBLE_APPROX(la::LengthEuclidean(a), sqrt(4+1+16+25), 1.0e-8);
}

// ---- INCLUDED FROM ORIGINAL LA TEST -----
// ----
// ----
// ---- (Except distance tests are omitted)

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

/** Tests level 1 BLAS-ish stuff. */
void TestMatrixSimpleMath() {
  Matrix m1;
  Matrix m2;
  Matrix m3;
  Matrix m4;
  Matrix m5;
  
  MakeCountMatrix(3, 4, &m1);
  MakeCountMatrix(3, 4, &m2);
  la::AddTo(m2, &m1);
  la::AddInit(m1, m2, &m3);
  TEST_ASSERT(m3.get(0, 0) == 0);
  TEST_ASSERT(m3.get(2, 3) == (2+3)*3);
  la::AddExpert(-1.0, m2, &m1);
  TEST_ASSERT(m1.get(0, 0) == 0);
  TEST_ASSERT(m1.get(2, 3) == (2+3));
  la::Scale(4.0, &m1);
  TEST_ASSERT(m1.get(2, 3) == (2+3)*4);
  MakeConstantMatrix(3, 4, 7.0, &m4);
  la::AddInit(m1, m4, &m5);
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

/** Tests level 1 BLAS-ish stuff. */
void TestVectorSimpleMath() {
  Vector v1;
  Vector v2;
  Vector v3;
  Vector v4;
  Vector v5;
  
  MakeCountVector(6, &v1);
  MakeCountVector(6, &v2);
  la::AddTo(v2, &v1);
  la::AddInit(v1, v2, &v3);
  TEST_ASSERT(v3[0] == 0);
  TEST_ASSERT(v3[5] == (5)*3);
  la::AddExpert(-1.0, v2, &v1);
  TEST_ASSERT(v1[0] == 0);
  TEST_ASSERT(v1[5] == (5));
  la::Scale(4.0, &v1);
  TEST_ASSERT(v1[5] == (5)*4);
  MakeConstantVector(6, 7.0, &v4);
  la::AddInit(v1, v4, &v5);
  TEST_ASSERT(v5[5] == (5)*4 + 7.0);
  TEST_ASSERT(v5[4] == (4)*4 + 7.0);
  TEST_ASSERT(v5[1] == (1)*4 + 7.0);
}

/** Tests aliases and copies */
void TestVector() {
  Vector v1;
  const Vector *v_const;
  Vector v2;
  Vector v3;
  Vector v4;
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
  SmallVector<21> v5;
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
  TEST_ASSERT(v1[0] == 0.0);
  v9.SwapValues(&v1);
  TEST_DOUBLE_EXACT(v1[0], 3.5);
  TEST_ASSERT(v9[0] == 0.0);
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
  SmallMatrix<21, 21> m5;
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

// ---- -------- ----
// ---- -------- ----
// ---- -------- ----
// ---- NEW TESTS ----
// ---- -------- ----
// ---- -------- ----
// ---- -------- ----

void TestMultiply() {
  MAKE_MATRIX_TRANS(a, 3, 3,
      3, 1, 4,
      1, 5, 9,
      2, 6, 5);
  MAKE_MATRIX_TRANS(b, 3, 4,
      3, 5, 8,
      9, 7, 9,
      3, 2, 3,
      8, 4, 6);
  
  MAKE_MATRIX_TRANS(product_expect, 3, 4,
      30, 76, 97,
      52, 98, 144,
      17, 31, 45,
      40, 64, 98);
  
  Matrix product_actual;
  
  // product_actual is uninitialized
  la::MulInit(a, b, &product_actual);
  AssertExactMatrix(product_expect, product_actual);
  
  product_actual.SetZero();
  la::MulOverwrite(a, b, &product_actual);
  AssertExactMatrix(product_expect, product_actual);
  
  MAKE_MATRIX_TRANS(product_expect_transa, 3, 4,
      46, 100, 76,
      70, 125, 105,
      23, 40, 33,
      52, 82, 70);
  
  Matrix product_actual_transa;
  la::MulTransAInit(a, b, &product_actual_transa);
  AssertExactMatrix(product_expect_transa, product_actual_transa);

  Matrix a_t, b_t;
  la::TransposeInit(a, &a_t);
  la::TransposeInit(b, &b_t);
  
  product_actual_transa.Destruct();
  la::MulTransBInit(a_t, b_t, &product_actual_transa);
  AssertExactMatrix(product_expect_transa, product_actual_transa);
  
  product_actual.SetZero();
  la::MulTransAOverwrite(a, b, &product_actual_transa);
  AssertExactMatrix(product_expect_transa, product_actual_transa);

  // test matrix-vector multiplication
  MAKE_VECTOR(v1, 3,     9, 1, 2);
  MAKE_VECTOR(a_v1, 3,   32, 26, 55);
  MAKE_VECTOR(v1_a, 3,   36, 32, 34);
  MAKE_VECTOR(v2, 3,     2, 3, 4);
  MAKE_VECTOR(a_v2, 3,   17, 41, 55);
  MAKE_VECTOR(v2_a, 3,   25, 53, 42);
  
  Vector a_v1_actual;
  la::MulInit(a, v1, &a_v1_actual);
  AssertApproxVector(a_v1, a_v1_actual, 0);
  
  Vector a_v2_actual;
  a_v2_actual.Init(3);
  la::MulOverwrite(a, v2, &a_v2_actual);
  AssertApproxVector(a_v2, a_v2_actual, 0);
  
  Vector v1_a_actual;
  la::MulInit(v1, a, &v1_a_actual);
  AssertApproxVector(v1_a, v1_a_actual, 0);
  
  Vector v2_a_actual;
  v2_a_actual.Init(3);
  la::MulOverwrite(v2, a, &v2_a_actual);
  AssertApproxVector(v2_a, v2_a_actual, 0);
  
  // Test non-square matrices (we had some bad debug checks)
  MAKE_VECTOR(v3, 4,     1, 2, 3, 4);
  MAKE_VECTOR(b_v3, 3,   62, 41, 59);
  MAKE_VECTOR(v1_b, 4,   48, 106, 35, 88);
  
  SmallVector<3> b_v3_actual;
  
  la::MulOverwrite(b, v3, &b_v3_actual);
  AssertApproxVector(b_v3, b_v3_actual, 0);

  Vector v1_b_actual;
  la::MulInit(v1, b, &v1_b_actual);
  AssertApproxVector(v1_b, v1_b_actual, 0);
  
}

void TestInverse() {
  MAKE_MATRIX_TRANS(a, 3, 3,
      .5, 0, 0,
      0, 1, 0,
      0, 0, 2);
  MAKE_MATRIX_TRANS(a_inv_expect, 3, 3,
      2, 0, 0,
      0, 1, 0,
      0, 0, .5);

  MAKE_MATRIX_TRANS(b, 3, 3,
      3, 1, 4,
      1, 5, 9,
      2, 6, 5);
  MAKE_MATRIX_TRANS(b_inv_expect, 3, 3,
      0.3222222, -0.2111111, 0.1222222,
      -0.1444444, -0.0777778, 0.2555556,
      0.0444444, 0.1777778, -0.1555556);
  MAKE_MATRIX_TRANS(c, 3, 3,
      1, 0, 0,
      0, 1, 0,
      0, 0, 0);
  
  Matrix a_inv_actual;
  
  TEST_ASSERT(PASSED(la::InverseInit(a, &a_inv_actual)));
  AssertExactMatrix(a_inv_expect, a_inv_actual);
  
  Matrix b_inv_actual;
  
  b_inv_actual.Init(3, 3);
  TEST_ASSERT(PASSED(la::InverseOverwrite(b, &b_inv_actual)));
  AssertApproxMatrix(b_inv_expect, b_inv_actual, 1.0e-5);
  
  Matrix c_inv_actual;
  // Try inverting a 3x3 rank-3 matrix
  TEST_ASSERT(!PASSED(la::InverseInit(c, &c_inv_actual)));
  
  // Try inverting a 3x3 rank-3 matrix
  TEST_ASSERT(PASSED(la::Inverse(&b)));
  AssertApproxMatrix(b, b_inv_actual, 1.0e-5);
}

void TestDeterminant() {
  MAKE_MATRIX_TRANS(a, 3, 3,
      3, 1, 4,
      1, 5, 9,
      2, 6, 5);
  MAKE_MATRIX_TRANS(b, 3, 3,
      -3, 5, -8,
      9, -7, 9,
      2, 6, 5);
  MAKE_MATRIX_TRANS(c, 3, 3,
      -3, -5, -8,
      9, -7, 9,
      -2, 6, 5);
  MAKE_MATRIX_TRANS(d, 3, 3,
      31, 41, 59,
      26, 53, 58,
      97, 93, 23);
  
  int sign;
  
  TEST_DOUBLE_APPROX(-90.0, la::Determinant(a), 1.0e-7);
  TEST_DOUBLE_APPROX(log(90.0), la::DeterminantLog(a, &sign), 1.0e-7);
  DEBUG_ASSERT_MSG(sign == -1, "%d", sign);
  TEST_DOUBLE_APPROX(-412.0, la::Determinant(b), 1.0e-7);
  TEST_DOUBLE_APPROX(262.0, la::Determinant(c), 1.0e-7);
  TEST_DOUBLE_APPROX(log(262.0), la::DeterminantLog(c, &sign), 1.0e-7);
  DEBUG_ASSERT_MSG(sign == 1, "%d", sign);
  TEST_DOUBLE_APPROX(-8.3934e4, la::Determinant(d), 1.0e-7);
}

void TestQR() {
  MAKE_MATRIX_TRANS(a, 3, 3,
      3, 1, 4,
      1, 5, 9,
      2, 6, 5);
  MAKE_MATRIX_TRANS(a_q_expect, 3, 3,
      -0.58835, -0.19612, -0.78446,
       0.71472, -0.57986, -0.39107,
      -0.37819, -0.79076, 0.48133);
  MAKE_MATRIX_TRANS(a_r_expect, 3, 3,
      -5.09902, 0.00000, 0.00000,
      -8.62911, -5.70425, 0.00000,
      -6.27572, -4.00511, -3.09426);
  
  MAKE_MATRIX_TRANS(b, 3, 4,
      3, 5, 8,
      9, 7, 9,
      3, 2, 3,
      8, 4, 6);
  MAKE_MATRIX_TRANS(b_q_expect, 3, 3,
      -0.303046, -0.505076, -0.808122,
       0.929360, 0.030979, -0.367872,
       0.210838, -0.862519, 0.460010);
  MAKE_MATRIX_TRANS(b_r_expect, 3, 4,
       -9.89949, 0.00000, 0.00000,
      -13.53604, 5.27025, 0.00000,
       -4.34366, 1.74642, 0.28751,
       -9.29340, 5.35157, 0.99669);

  MAKE_MATRIX_TRANS(c, 4, 3,
      3, 9, 3, 8,
      5, 7, 2, 4,
      8, 9, 3, 6);
  MAKE_MATRIX_TRANS(c_q_expect, 4, 3,
     -0.234978, -0.704934, -0.234978, -0.626608,
      0.846774, 0.175882, -0.039891, -0.500449,
     -0.464365, 0.686138, -0.180138, -0.530217);
   //0.110115, 0.036705, -0.954329, 0.275287
  MAKE_MATRIX_TRANS(c_r_expect, 3, 3,
     -12.76715, 0.00000, 0.00000,
      -9.08582, 3.38347, 0.00000,
     -12.68882, 5.23476, -1.26139);

  Matrix a_q_actual;
  Matrix a_r_actual;
  
  TEST_ASSERT(PASSED(la::QRInit(a, &a_q_actual, &a_r_actual)));
  AssertApproxMatrix(a_q_expect, a_q_actual, 1.0e-5);
  AssertApproxMatrix(a_r_expect, a_r_actual, 1.0e-5);
  
  Matrix b_q_actual;
  Matrix b_r_actual;
  
  TEST_ASSERT(PASSED(la::QRInit(b, &b_q_actual, &b_r_actual)));
  AssertApproxMatrix(b_q_expect, b_q_actual, 1.0e-5);
  AssertApproxMatrix(b_r_expect, b_r_actual, 1.0e-5);
  
  Matrix c_q_actual;
  Matrix c_r_actual;
  
  TEST_ASSERT(PASSED(la::QRInit(c, &c_q_actual, &c_r_actual)));
  AssertApproxMatrix(c_q_expect, c_q_actual, 1.0e-5);
  AssertApproxMatrix(c_r_expect, c_r_actual, 1.0e-5);
}


void TestEigen() {
  MAKE_MATRIX_TRANS(a, 3, 3,
      3, 1, 4,
      1, 5, 9,
      2, 6, 5);
  MAKE_MATRIX_TRANS(a_eigenvectors_expect, 3, 3,
     -0.212480, -0.912445, -0.172947,
     -0.599107, 0.408996, -0.592976,
     -0.771960, -0.012887, 0.786428);
  MAKE_VECTOR(a_eigenvalues_real_expect, 3,
      13.08576, 2.58001, -2.66577);
  MAKE_VECTOR(a_eigenvalues_imag_expect, 3,
      0, 0, 0);

  MAKE_MATRIX_TRANS(b, 2, 2,
      3, 4,
      -2, -1);
  MAKE_MATRIX_TRANS(b_eigenvectors_real_expect, 2, 2,
      0.40825, 0.81650,
      0.40825, 0.81650);
  MAKE_MATRIX_TRANS(b_eigenvectors_imag_expect, 2, 2,
      0.40825, 0.0,
      -0.40825, 0.0);
  MAKE_VECTOR(b_eigenvalues_real_expect, 2,
     1.0, 1.0);
  MAKE_VECTOR(b_eigenvalues_imag_expect, 2,
     2.0, -2.0);

  Matrix a_eigenvectors_actual;
  Vector a_eigenvalues_actual;
  
  TEST_ASSERT(PASSED(la::EigenvectorsInit(
      a, &a_eigenvalues_actual, &a_eigenvectors_actual)));
  AssertApproxVector(a_eigenvalues_real_expect, a_eigenvalues_actual, 1.0e-5);
  AssertApproxTransMatrix(a_eigenvectors_expect, a_eigenvectors_actual, 1.0e-5);

  Vector a_eigenvalues_real_actual;
  Vector a_eigenvalues_imag_actual;
  TEST_ASSERT(PASSED(la::EigenvaluesInit(
      a, &a_eigenvalues_real_actual, &a_eigenvalues_imag_actual)));
  AssertApproxVector(a_eigenvalues_real_expect, a_eigenvalues_real_actual, 1.0e-5);
  AssertApproxVector(a_eigenvalues_imag_expect, a_eigenvalues_imag_actual, 0.0);

  Vector a_eigenvalues_actual_2;
  TEST_ASSERT(PASSED(la::EigenvaluesInit(
      a, &a_eigenvalues_actual_2)));
  AssertApproxVector(a_eigenvalues_real_expect, a_eigenvalues_actual_2, 1.0e-5);
  
  // complex eigenvalues

  /*
   * This function no longer fails on imaginary, but sets them to NaN
   */
  //Matrix b_eigenvectors_actual;
  //Vector b_eigenvalues_actual;
  //TEST_ASSERT(!PASSED(la::EigenvectorsInit(
  //    b, &b_eigenvalues_actual, &b_eigenvectors_actual)));

  Matrix b_eigenvectors_real_actual;
  Matrix b_eigenvectors_imag_actual;
  Vector b_eigenvalues_real_actual;
  Vector b_eigenvalues_imag_actual;
  TEST_ASSERT(PASSED(la::EigenvectorsInit(
      b, &b_eigenvalues_real_actual, &b_eigenvalues_imag_actual,
      &b_eigenvectors_real_actual, &b_eigenvectors_imag_actual)));
  AssertApproxVector(b_eigenvalues_real_expect, b_eigenvalues_real_actual, 1.0e-5);
  AssertApproxMatrix(b_eigenvectors_real_expect, b_eigenvectors_real_actual, 1.0e-5);
  AssertApproxVector(b_eigenvalues_imag_expect, b_eigenvalues_imag_actual, 1.0e-5);
  AssertApproxMatrix(b_eigenvectors_imag_expect, b_eigenvectors_imag_actual, 1.0e-5);
}

void TrySchur(const Matrix &orig) {
  Matrix z;
  Matrix t;
  Vector eigen_real;
  Vector eigen_imag;
  
  la::SchurInit(orig, &eigen_real, &eigen_imag, &t, &z);
  
  Matrix z_trans;
  la::TransposeInit(z, &z_trans);
  Matrix tmp;
  la::MulInit(t, z_trans, &tmp);
  Matrix result;
  la::MulInit(z, tmp, &result);
  
  AssertApproxMatrix(orig, result, 1.0e-8);
  
  /*
   * This test now fails because Schur finds real components while 
   * Eigenvectors on 3 args only finds true real eigenvalues
   */
  //Vector eigen_real_2;
  //Matrix eigenvectors_2;
  //la::EigenvectorsInit(orig, &eigen_real_2, &eigenvectors_2);
  //AssertApproxVector(eigen_real_2, eigen_real, 1.0e-8);
}

void TestSchur() {
  MAKE_MATRIX_TRANS(a, 3, 3,
     3, 1, 4,
     1, 5, 9,
     2, 6, 5);
  MAKE_MATRIX_TRANS(b, 5, 5,
     3, 1, 4, 1, 5,
     9, 2, 6, 5, 3,
     5, 8, 9, 7, 9,
     3, 2, 3, 8, 4,
     6, 2, 6, 4, 3);
  
  TrySchur(a);
  TrySchur(b);
}

void AssertProperSVD(const Matrix& orig,
    const Vector &s, const Matrix& u, const Matrix& vt) {
  Matrix s_matrix;
  s_matrix.Init(s.length(), s.length());
  s_matrix.SetDiagonal(s);
  Matrix tmp;
  la::MulInit(u, s_matrix, &tmp);
  Matrix result;
  la::MulInit(tmp, vt, &result);
  AssertApproxMatrix(result, orig, 1.0e-8);
}

void TrySVD(const Matrix& orig) {
  Vector s;
  Matrix u;
  Matrix vt;
  la::SVDInit(orig, &s, &u, &vt);
  AssertProperSVD(orig, s, u, vt);
}

void TestSVD() {
  MAKE_MATRIX_TRANS(a, 3, 3,
      3, 1, 4,
      1, 5, 9,
      2, 6, 5);
  MAKE_MATRIX_TRANS(a_u_expect, 3, 3,
    -0.21141,  -0.55393,  -0.80528,
     0.46332,  -0.78225,   0.41645,
    -0.86060,  -0.28506,   0.42202);
  MAKE_VECTOR(a_s_expect, 3,
    13.58236, 2.84548, 2.32869);
  MAKE_MATRIX_TRANS(a_vt_expect, 3, 3,
    -0.32463,   0.79898,  -0.50620,
    -0.75307,   0.10547,   0.64943,
    -0.57227,  -0.59203,  -0.56746);
  MAKE_MATRIX_TRANS(b, 3, 10,
      3, 1, 4,
      1, 5, 9,
      2, 6, 5,
      3, 5, 8,
      9, 7, 9,
      3, 2, 3,
      8, 4, 6,
      2, 6, 4,
      3, 3, 8,
      3, 2, 7);
  MAKE_MATRIX_TRANS(c, 9, 3,
      3, 1, 4, 1, 5, 9, 2, 6, 5,
      3, 5, 8, 9, 7, 9, 3, 2, 3,
      8, 4, 6, 2, 6, 4, 3, 3, 8);
  MAKE_MATRIX_TRANS(d, 3, 3,
      0, 1, 0,
      -1, 0, 0,
      0, 0, 1);

  Matrix a_u_actual;
  Vector a_s_actual;
  Matrix a_vt_actual;

  la::SVDInit(a, &a_s_actual, &a_u_actual, &a_vt_actual);
  AssertProperSVD(a, a_s_actual, a_u_actual, a_vt_actual);
  AssertApproxVector(a_s_expect, a_s_actual, 1.0e-5);
  AssertApproxMatrix(a_u_expect, a_u_actual, 1.0e-5);
  AssertApproxMatrix(a_vt_expect, a_vt_actual, 1.0e-5);

  Vector a_s_actual_2;
  la::SVDInit(a, &a_s_actual_2);
  AssertApproxVector(a_s_expect, a_s_actual_2, 1.0e-5);

  TrySVD(b);
  TrySVD(c);
  TrySVD(d);

  // let's try a big, but asymmetric, one
  Matrix e;
  e.Init(3000, 10);
  for (index_t j = 0; j < e.n_cols(); j++) {
    for (index_t i = 0; i < e.n_rows(); i++) {
      e.set(i, j, rand() * 1.0 / RAND_MAX);
    }
  }

  TrySVD(e);
}

void TryCholesky(const Matrix &orig) {
  Matrix u;
  TEST_ASSERT(PASSED(la::CholeskyInit(orig, &u)));
  Matrix result;
  la::MulTransAInit(u, u, &result);
  AssertApproxMatrix(orig, result, 1.0e-8);
}

void TestCholesky() {
  MAKE_MATRIX_TRANS(a, 3, 3,
      1, 0, 0,
      0, 2, 0,
      0, 0, 3);
  MAKE_MATRIX_TRANS(b, 4, 4,
    9.00,   0.60,  -0.30,   1.50,
    0.60,  16.04,   1.18,  -1.50,
   -0.30,   1.18,   4.10,  -0.57,
    1.50,  -1.50,  -0.57,  25.45);
  TryCholesky(a);
  TryCholesky(b);
}

void TrySolveMatrix(const Matrix& a, const Matrix& b) {
  Matrix x;
  TEST_ASSERT(PASSED(la::SolveInit(a, b, &x)));
  Matrix result;
  la::MulInit(a, x, &result);
  AssertApproxMatrix(b, result, 1.0e-8);
}

void TrySolveVector(const Matrix& a, const Vector& b) {
  Vector x;
  la::SolveInit(a, b, &x);
  Vector result;
  la::MulInit(a, x, &result);
  AssertApproxVector(b, result, 1.0e-8);
}

void TestSolve() {
  MAKE_MATRIX_TRANS(a, 3, 3,
     3, 1, 4,
     1, 5, 9,
     2, 6, 5);
  MAKE_MATRIX_TRANS(a_vectors, 3, 5,
     1, 2, 3,
     4, 5, 2,
     1, 6, 3,
     2, 1, 8,
     4, 2, 6);
  MAKE_VECTOR(a_vector_1, 3,   3, 1, 2);
  MAKE_VECTOR(a_vector_2, 3,   2, 4, 6);
  MAKE_VECTOR(a_vector_3, 3,   2, 4, 6);
  MAKE_VECTOR(a_vector_4, 3,   5, 7, 8);
  MAKE_MATRIX_TRANS(b, 5, 5,
     3, 1, 4, 1, 5,
     9, 2, 6, 5, 3,
     5, 8, 9, 7, 9,
     3, 2, 3, 8, 4,
     6, 2, 6, 4, 3);
  
  TrySolveMatrix(a, a_vectors);
  TrySolveVector(a, a_vector_1);
  TrySolveVector(a, a_vector_2);
  TrySolveVector(a, a_vector_3);
  TrySolveVector(a, a_vector_4);
}

TEST_SUITE_END(uselapack,
    TestVector,
    TestMatrix,
    TestVectorDot,
    TestVectorSimpleMath,
    TestMatrixSimpleMath,
    TestMultiply,
    TestInverse,
    TestDeterminant,
    TestQR,
    TestEigen,
    TestSchur,
    TestSVD,
    TestCholesky,
    TestSolve,
    );

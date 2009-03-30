/**
 * @file uselapack_test.h
 *
 * Tests for LAPACK integration.
 */

#include "fastlib/base/test.h"

#include "uselapack.h"

#include "la.h"

/**
 * Creates a matrix locally.
 * The matrix cotents are column-major.
 */
#define MAKE_MATRIX_TRANS(Precision, name, n_rows, n_cols, contents ...) \
    Precision name ## _values [] = { contents }; \
    DEBUG_ASSERT(sizeof(name ## _values) / sizeof(Precision) == n_rows * n_cols); \
    GenMatrix<Precision, false> name; \
    name.Alias(name ## _values, (n_rows), (n_cols));

/**
 * Creates a vector locally.
 * The matrix cotents are column-major.
 */
#define MAKE_VECTOR(Precision, name, length, contents ...) \
    Precision name ## _values [] = { contents }; \
    DEBUG_ASSERT(sizeof(name ## _values) / sizeof(Precision) == (length)); \
    GenMatrix<Precision, true> name; \
    name.Alias(name ## _values, (length));
template<typename Precision>
bool VectorApproxEqual(const GenMatrix<Precision, true>& a, 
                       const GenMatrix<Precision, true>& b,
                       Precision eps) {
  if (a.length() != b.length()) {
    fprintf(stderr, "XXX Size mismatch.\n");
    return false;
  }
  
  int wrong = 0;
  Precision max_diff = 0;
  
  for (index_t i = 0; i < a.length(); i++) {
    Precision diff = fabs(a.get(i) - b.get(i));
    max_diff = std::max(max_diff, diff);
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
template<typename Precision>
void AssertApproxVector(const GenMatrix<Precision, true>& a, 
                        const GenMatrix<Precision, true>& b, Precision eps) {
  if (!VectorApproxEqual(a, b, eps)) {
    a.PrintDebug("a");
    b.PrintDebug("b");
    abort();
  }
  //fprintf(stderr, "... Correct vector!\n");
}

template<typename Precision>
bool MatrixApproxEqual(const GenMatrix<Precision, false>& a, 
                       const GenMatrix<Precision, false>& b,
     Precision eps) {
  if (a.n_rows() != b.n_rows() || a.n_cols() != b.n_cols()) {
    fprintf(stderr, "XXX Size mismatch.\n");
    return false;
  }
  
  int wrong = 0;
  Precision max_diff = 0;
  
  for (index_t c = 0; c < a.n_cols(); c++) {
    for (index_t r = 0; r < a.n_rows(); r++) {
      Precision diff = fabs(a.get(r, c) - b.get(r, c));
      max_diff = std::max(max_diff, diff);
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

template<typename Precision>
void AssertApproxMatrix(const GenMatrix<Precision, false>& a, 
                        const GenMatrix<Precision, false>& b,
    Precision eps) {
  if (!MatrixApproxEqual(a, b, eps)) {
    a.PrintDebug("a");
    b.PrintDebug("b");
    abort();
  }
  //fprintf(stderr, "... Correct matrix!\n");
}

template<typename Precision>
void AssertExactMatrix(const GenMatrix<Precision, false>& a, 
                       const GenMatrix<Precision, false>& b) {
  AssertApproxMatrix<Precision>(a, b, 0);
}

template<typename Precision>
void AssertApproxTransMatrix(const GenMatrix<Precision, false>& a, 
                             const GenMatrix<Precision, false>& b,
                             Precision eps) {
  GenMatrix<Precision, false> a_trans;
  la::TransposeInit<Precision>(a, &a_trans);
  if (!MatrixApproxEqual(a_trans, b, eps)) {
    a_trans.PrintDebug("a_trans");
    b.PrintDebug("b");
    abort();
  }
}

template<typename Precision>
void TestVectorDot() {
  MAKE_VECTOR(Precision, a, 4,    2, 1, 4, 5);
  MAKE_VECTOR(Precision, b, 4,    3, 0, 2, -1);
  
  //TEST_DOUBLE_EXACT(F77_FUNC(ddot)(4, a.ptr(), 1, a.ptr(), 1),  4+1+16+25);
  TEST_DOUBLE_EXACT(la::Dot<Precision>(a, a), 4+1+16+25);
  TEST_DOUBLE_EXACT(la::Dot<Precision>(a, b), 6+0+8-5);
  TEST_DOUBLE_APPROX(la::LengthEuclidean<Precision>(b), sqrt(9+0+4+1), 1.0e-5);
  TEST_DOUBLE_APPROX(la::LengthEuclidean<Precision>(a), sqrt(4+1+16+25), 1.0e-5);
}

// ---- INCLUDED FROM ORIGINAL LA TEST -----
// ----
// ----
// ---- (Except distance tests are omitted)
template<typename Precision>
void MakeCountMatrix(index_t n_rows, index_t n_cols, GenMatrix<Precision, false> *m) {
  m->Init(n_rows, n_cols);
  
  for (index_t c = 0; c < n_cols; c++) {
    for (index_t r = 0; r < n_rows; r++) {
      m->set(r, c, r + c);
    }
  }
}

template<typename Precision>
void MakeConstantMatrix(index_t n_rows, index_t n_cols, Precision v, GenMatrix<Precision, false> *m) {
  m->Init(n_rows, n_cols);
  
  for (index_t c = 0; c < n_cols; c++) {
    for (index_t r = 0; r < n_rows; r++) {
      m->set(r, c, v);
    }
  }
}

/** Tests level 1 BLAS-ish stuff. */
template<typename Precision>
void TestMatrixSimpleMath() {
  GenMatrix<Precision, false> m1;
  GenMatrix<Precision, false> m2;
  GenMatrix<Precision, false> m3;
  GenMatrix<Precision, false> m4;
  GenMatrix<Precision, false> m5;
  
  MakeCountMatrix<Precision>(3, 4, &m1);
  MakeCountMatrix<Precision>(3, 4, &m2);
  la::AddTo<Precision>(m2, &m1);
  la::AddInit<Precision>(m1, m2, &m3);
  TEST_ASSERT(m3.get(0, 0) == 0);
  TEST_ASSERT(m3.get(2, 3) == (2+3)*3);
  la::AddExpert<Precision>(-1.0, m2, &m1);
  TEST_ASSERT(m1.get(0, 0) == 0);
  TEST_ASSERT(m1.get(2, 3) == (2+3));
  la::Scale<Precision>(4.0, &m1);
  TEST_ASSERT(m1.get(2, 3) == (2+3)*4);
  MakeConstantMatrix<Precision>(3, 4, 7.0, &m4);
  la::AddInit<Precision>(m1, m4, &m5);
  TEST_ASSERT(m5.get(2, 3) == (2+3)*4 + 7.0);
  TEST_ASSERT(m5.get(1, 3) == (1+3)*4 + 7.0);
  TEST_ASSERT(m5.get(1, 0) == (1+0)*4 + 7.0);
}

template<typename Precision>
void MakeCountVector(index_t n, GenMatrix<Precision, true> *v) {
  v->Init(n);
  
  for (index_t c = 0; c < n; c++) {
    (*v)[c] = c;
  }
}

template<typename Precision>
void MakeConstantVector(index_t n, Precision d, GenMatrix<Precision, true> *v) {
  v->Init(n);
  
  for (index_t c = 0; c < n; c++) {
    (*v)[c] = d;
  }
}

/** Tests level 1 BLAS-ish stuff. */
template<typename Precision>
void TestVectorSimpleMath() {
  GenMatrix<Precision, true> v1;
  GenMatrix<Precision, true> v2;
  GenMatrix<Precision, true> v3;
  GenMatrix<Precision, true> v4;
  GenMatrix<Precision, true> v5;
  
  MakeCountVector<Precision>(6, &v1);
  MakeCountVector<Precision>(6, &v2);
  la::AddTo<Precision>(v2, &v1);
  la::AddInit<Precision>(v1, v2, &v3);
  TEST_ASSERT(v3[0] == 0);
  TEST_ASSERT(v3[5] == (5)*3);
  la::AddExpert<Precision>(-1.0, v2, &v1);
  TEST_ASSERT(v1[0] == 0);
  TEST_ASSERT(v1[5] == (5));
  la::Scale<Precision>(4.0, &v1);
  TEST_ASSERT(v1[5] == (5)*4);
  MakeConstantVector<Precision>(6, 7.0, &v4);
  la::AddInit<Precision>(v1, v4, &v5);
  TEST_ASSERT(v5[5] == (5)*4 + 7.0);
  TEST_ASSERT(v5[4] == (4)*4 + 7.0);
  TEST_ASSERT(v5[1] == (1)*4 + 7.0);
}

/** Tests aliases and copies */
template<typename Precision>
void TestVector() {
  GenMatrix<Precision, true> v1;
  const GenMatrix<Precision, true> *v_const;
  GenMatrix<Precision, true> v2;
  GenMatrix<Precision, true> v3;
  GenMatrix<Precision, true> v4;
  GenMatrix<Precision, true> v6;
  GenMatrix<Precision, true> v7;
  GenMatrix<Precision, true> v8;
  GenMatrix<Precision, true> v9;
  
  MakeCountVector<Precision>(10, &v1);
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
  SmallVector<Precision, 21> v5;
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
  MakeConstantVector<Precision>(10, 3.5, &v9);
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

template<typename Precision>
void TestMatrix() {
  GenMatrix<Precision, false> m1;
  const GenMatrix<Precision, false> *m_const;
  GenMatrix<Precision, false> m2;
  GenMatrix<Precision, false> m3;
  GenMatrix<Precision, false> m4;
  GenMatrix<Precision, false> m6;
  GenMatrix<Precision, false> m7;
  GenMatrix<Precision, false> m8;
  GenMatrix<Precision, false> m9;
  
  MakeCountMatrix<Precision>(13, 10, &m1);
  TEST_ASSERT(m1.n_cols() == 10);
  TEST_ASSERT(m1.n_rows() == 13);
  TEST_ASSERT(m1.ptr()[3] == m1.get(3, 0));
  m_const = &m1;
  TEST_ASSERT(m_const->ptr()[3] == (*m_const).get(3, 0));

  GenMatrix<Precision, true> v1, v2;  
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
  SmallMatrix<Precision, 21, 21> m5;
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
  MakeConstantMatrix<Precision>(13, 10, 3.5, &m9);
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
template<typename Precision>
void TestMultiply() {
  MAKE_MATRIX_TRANS(Precision, a, 3, 3,
      3, 1, 4,
      1, 5, 9,
      2, 6, 5);

  MAKE_MATRIX_TRANS(Precision, b, 3, 4,
      3, 5, 8,
      9, 7, 9,
      3, 2, 3,
      8, 4, 6);
  
  MAKE_MATRIX_TRANS(Precision, product_expect, 3, 4,
      30, 76, 97,
      52, 98, 144,
      17, 31, 45,
      40, 64, 98);
  
  GenMatrix<Precision, false> product_actual;
  
  // product_actual is uninitialized
  la::MulInit<Precision>(a, b, &product_actual);
  AssertExactMatrix<Precision>(product_expect, product_actual);
  
  product_actual.SetZero();
  la::MulOverwrite<Precision>(a, b, &product_actual);
  AssertExactMatrix<Precision>(product_expect, product_actual);
  
  MAKE_MATRIX_TRANS(Precision, product_expect_transa, 3, 4,
      46, 100, 76,
      70, 125, 105,
      23, 40, 33,
      52, 82, 70);
  
  GenMatrix<Precision, false> product_actual_transa;
  la::MulTransAInit<Precision>(a, b, &product_actual_transa);
  AssertExactMatrix<Precision>(product_expect_transa, product_actual_transa);

  GenMatrix<Precision, false> a_t, b_t;
  la::TransposeInit<Precision>(a, &a_t);
  la::TransposeInit<Precision>(b, &b_t);
  
  product_actual_transa.Destruct();
  la::MulTransBInit<Precision>(a_t, b_t, &product_actual_transa);
  AssertExactMatrix<Precision>(product_expect_transa, product_actual_transa);
  
  product_actual.SetZero();
  la::MulTransAOverwrite<Precision>(a, b, &product_actual_transa);
  AssertExactMatrix<Precision>(product_expect_transa, product_actual_transa);

  // test matrix-vector multiplication
  MAKE_VECTOR(Precision, v1, 3,     9, 1, 2);
  MAKE_VECTOR(Precision, a_v1, 3,   32, 26, 55);
  MAKE_VECTOR(Precision, v1_a, 3,   36, 32, 34);
  MAKE_VECTOR(Precision, v2, 3,     2, 3, 4);
  MAKE_VECTOR(Precision, a_v2, 3,   17, 41, 55);
  MAKE_VECTOR(Precision, v2_a, 3,   25, 53, 42);
  
  GenMatrix<Precision, true> a_v1_actual;
  la::MulInit<Precision>(a, v1, &a_v1_actual);
  AssertApproxVector<Precision>(a_v1, a_v1_actual, 0);
  
  GenMatrix<Precision, true> a_v2_actual;
  a_v2_actual.Init(3);
  la::MulOverwrite<Precision>(a, v2, &a_v2_actual);
  AssertApproxVector<Precision>(a_v2, a_v2_actual, 0);
  
  GenMatrix<Precision, true> v1_a_actual;
  la::MulInit<Precision>(v1, a, &v1_a_actual);
  AssertApproxVector<Precision>(v1_a, v1_a_actual, 0);
  
  GenMatrix<Precision, true> v2_a_actual;
  v2_a_actual.Init(3);
  la::MulOverwrite<Precision>(v2, a, &v2_a_actual);
  AssertApproxVector<Precision>(v2_a, v2_a_actual, 0);
  
  // Test non-square matrices (we had some bad debug checks)
  MAKE_VECTOR(Precision, v3, 4,     1, 2, 3, 4);
  MAKE_VECTOR(Precision, b_v3, 3,   62, 41, 59);
  MAKE_VECTOR(Precision, v1_b, 4,   48, 106, 35, 88);
  
  SmallVector<Precision, 3> b_v3_actual;
  
  la::MulOverwrite<Precision>(b, v3, &b_v3_actual);
  AssertApproxVector<Precision>(b_v3, b_v3_actual, 0);

  GenMatrix<Precision, true> v1_b_actual;
  la::MulInit<Precision>(v1, b, &v1_b_actual);
  AssertApproxVector<Precision>(v1_b, v1_b_actual, 0);
  
}

template<typename Precision>
void TestInverse() {
  MAKE_MATRIX_TRANS(Precision, a, 3, 3,
      .5, 0, 0,
      0, 1, 0,
      0, 0, 2);
  MAKE_MATRIX_TRANS(Precision, a_inv_expect, 3, 3,
      2, 0, 0,
      0, 1, 0,
      0, 0, .5);

  MAKE_MATRIX_TRANS(Precision, b, 3, 3,
      3, 1, 4,
      1, 5, 9,
      2, 6, 5);
  MAKE_MATRIX_TRANS(Precision, b_inv_expect, 3, 3,
      0.3222222, -0.2111111, 0.1222222,
      -0.1444444, -0.0777778, 0.2555556,
      0.0444444, 0.1777778, -0.1555556);
  MAKE_MATRIX_TRANS(Precision, c, 3, 3,
      1, 0, 0,
      0, 1, 0,
      0, 0, 0);
  
  GenMatrix<Precision, false> a_inv_actual;
  
  TEST_ASSERT(PASSED(la::InverseInit<Precision>(a, &a_inv_actual)));
  AssertExactMatrix<Precision>(a_inv_expect, a_inv_actual);
  
  GenMatrix<Precision, false> b_inv_actual;
  
  b_inv_actual.Init(3, 3);
  TEST_ASSERT(PASSED(la::InverseOverwrite<Precision>(b, &b_inv_actual)));
  AssertApproxMatrix<Precision>(b_inv_expect, b_inv_actual, 1.0e-5);
  
  GenMatrix<Precision, false> c_inv_actual;
  // Try inverting a 3x3 rank-3 matrix
  TEST_ASSERT(!PASSED(la::InverseInit<Precision>(c, &c_inv_actual)));
  
  // Try inverting a 3x3 rank-3 matrix
  TEST_ASSERT(PASSED(la::Inverse<Precision>(&b)));
  AssertApproxMatrix<Precision>(b, b_inv_actual, 1.0e-5);
}

template<typename Precision>
void TestDeterminant() {
  MAKE_MATRIX_TRANS(Precision, a, 3, 3,
      3, 1, 4,
      1, 5, 9,
      2, 6, 5);
  MAKE_MATRIX_TRANS(Precision, b, 3, 3,
      -3, 5, -8,
      9, -7, 9,
      2, 6, 5);
  MAKE_MATRIX_TRANS(Precision, c, 3, 3,
      -3, -5, -8,
      9, -7, 9,
      -2, 6, 5);
  MAKE_MATRIX_TRANS(Precision, d, 3, 3,
      31, 41, 59,
      26, 53, 58,
      97, 93, 23);
  
  int sign;
  
  TEST_DOUBLE_APPROX(-90.0, la::Determinant<Precision>(a), 1.0e-4);
  TEST_DOUBLE_APPROX(log(90.0), la::DeterminantLog<Precision>(a, &sign), 1.0e-4);
  DEBUG_ASSERT_MSG(sign == -1, "%d", sign);
  TEST_DOUBLE_APPROX(-412.0, la::Determinant<Precision>(b), 1.0e-4);
  TEST_DOUBLE_APPROX(262.0, la::Determinant<Precision>(c), 1.0e-4);
  TEST_DOUBLE_APPROX(log(262.0), la::DeterminantLog<Precision>(c, &sign), 1.0e-4);
  DEBUG_ASSERT_MSG(sign == 1, "%d", sign);
  TEST_DOUBLE_APPROX(-8.3934e4, la::Determinant<Precision>(d), 1.0e-3);
}

template<typename Precision>
void TestQR() {
  MAKE_MATRIX_TRANS(Precision, a, 3, 3,
      3, 1, 4,
      1, 5, 9,
      2, 6, 5);
  MAKE_MATRIX_TRANS(Precision, a_q_expect, 3, 3,
      -0.58835, -0.19612, -0.78446,
       0.71472, -0.57986, -0.39107,
      -0.37819, -0.79076, 0.48133);
  MAKE_MATRIX_TRANS(Precision, a_r_expect, 3, 3,
      -5.09902, 0.00000, 0.00000,
      -8.62911, -5.70425, 0.00000,
      -6.27572, -4.00511, -3.09426);
  
  MAKE_MATRIX_TRANS(Precision, b, 3, 4,
      3, 5, 8,
      9, 7, 9,
      3, 2, 3,
      8, 4, 6);
  MAKE_MATRIX_TRANS(Precision, b_q_expect, 3, 3,
      -0.303046, -0.505076, -0.808122,
       0.929360, 0.030979, -0.367872,
       0.210838, -0.862519, 0.460010);
  MAKE_MATRIX_TRANS(Precision, b_r_expect, 3, 4,
       -9.89949, 0.00000, 0.00000,
      -13.53604, 5.27025, 0.00000,
       -4.34366, 1.74642, 0.28751,
       -9.29340, 5.35157, 0.99669);

  MAKE_MATRIX_TRANS(Precision, c, 4, 3,
      3, 9, 3, 8,
      5, 7, 2, 4,
      8, 9, 3, 6);
  MAKE_MATRIX_TRANS(Precision, c_q_expect, 4, 3,
     -0.234978, -0.704934, -0.234978, -0.626608,
      0.846774, 0.175882, -0.039891, -0.500449,
     -0.464365, 0.686138, -0.180138, -0.530217);
   //0.110115, 0.036705, -0.954329, 0.275287
  MAKE_MATRIX_TRANS(Precision, c_r_expect, 3, 3,
     -12.76715, 0.00000, 0.00000,
      -9.08582, 3.38347, 0.00000,
     -12.68882, 5.23476, -1.26139);

  GenMatrix<Precision, false> a_q_actual;
  GenMatrix<Precision, false> a_r_actual;
  GenMatrix<Precision, false> a_q_r_actual;
  
  TEST_ASSERT(PASSED(la::QRInit<Precision>(a, &a_q_actual, &a_r_actual)));
  la::MulInit<Precision>(a_q_actual, a_r_actual, &a_q_r_actual);
  AssertApproxMatrix<Precision>(a, a_q_r_actual, 1.0e-5);
  
  GenMatrix<Precision, false> b_q_actual;
  GenMatrix<Precision, false> b_r_actual;
  GenMatrix<Precision, false> b_q_r_actual;
  
  TEST_ASSERT(PASSED(la::QRInit<Precision>(b, &b_q_actual, &b_r_actual)));
  la::MulInit<Precision>(b_q_actual, b_r_actual, &b_q_r_actual);
  AssertApproxMatrix<Precision>(b, b_q_r_actual, 1.0e-5);
  
  GenMatrix<Precision, false> c_q_actual;
  GenMatrix<Precision, false> c_r_actual;
  GenMatrix<Precision, false> c_q_r_actual;
  
  TEST_ASSERT(PASSED(la::QRInit<Precision>(c, &c_q_actual, &c_r_actual)));
  la::MulInit<Precision>(c_q_actual, c_r_actual, &c_q_r_actual);
  AssertApproxMatrix<Precision>(c, c_q_r_actual, 1.0e-5);
}

template<typename Precision>
void TestEigen() {
  MAKE_MATRIX_TRANS(Precision, a, 3, 3,
      3, 1, 4,
      1, 5, 9,
      2, 6, 5);
  MAKE_MATRIX_TRANS(Precision, a_eigenvectors_expect, 3, 3,
     0.212480, 0.912445, 0.172947,
     0.599107, -0.408996, 0.592976,
     0.771960, 0.012887,-0.786428);
  MAKE_VECTOR(Precision, a_eigenvalues_real_expect, 3,
      13.08576, 2.58001, -2.66577);
  MAKE_VECTOR(Precision, a_eigenvalues_imag_expect, 3,
      0, 0, 0);

  MAKE_MATRIX_TRANS(Precision, b, 2, 2,
      3, 4,
      -2, -1);
  MAKE_MATRIX_TRANS(Precision, b_eigenvectors_real_expect, 2, 2,
      0.40825, 0.81650,
      0.40825, 0.81650);
  MAKE_MATRIX_TRANS(Precision, b_eigenvectors_imag_expect, 2, 2,
      0.40825, 0.0,
      -0.40825, 0.0);
  MAKE_VECTOR(Precision, b_eigenvalues_real_expect, 2,
     1.0, 1.0);
  MAKE_VECTOR(Precision, b_eigenvalues_imag_expect, 2,
     2.0, -2.0);

  GenMatrix<Precision, false> a_eigenvectors_actual;
  GenMatrix<Precision, true> a_eigenvalues_actual;
  
  TEST_ASSERT(PASSED(la::EigenvectorsInit<Precision>(
      a, &a_eigenvalues_actual, &a_eigenvectors_actual)));
  AssertApproxVector<Precision>(a_eigenvalues_real_expect, a_eigenvalues_actual, 1.0e-3);
  for(index_t i=0; i<a_eigenvectors_actual.n_cols(); i++) {
    double  sign=0;
    if (a_eigenvectors_actual.get(0,i) > 0) {
      sign=1;
    } else {
      sign=-1;
    }
    for(index_t j=0; j<a_eigenvectors_actual.n_rows(); j++) {
      a_eigenvectors_actual.set(j, i, 
          a_eigenvectors_actual.get(j,i)*sign);
    }
  }
  AssertApproxTransMatrix<Precision>(a_eigenvectors_expect, a_eigenvectors_actual, 1.0e-3);

  GenMatrix<Precision, true> a_eigenvalues_real_actual;
  GenMatrix<Precision, true> a_eigenvalues_imag_actual;
  TEST_ASSERT(PASSED(la::EigenvaluesInit(
      a, &a_eigenvalues_real_actual, &a_eigenvalues_imag_actual)));
  AssertApproxVector<Precision>(a_eigenvalues_real_expect, a_eigenvalues_real_actual, 1.0e-3);
  AssertApproxVector<Precision>(a_eigenvalues_imag_expect, a_eigenvalues_imag_actual, 0.0);

  GenMatrix<Precision, true> a_eigenvalues_actual_2;
  TEST_ASSERT(PASSED(la::EigenvaluesInit(
      a, &a_eigenvalues_actual_2)));
  AssertApproxVector<Precision>(a_eigenvalues_real_expect, a_eigenvalues_actual_2, 1.0e-3);
  
  // complex eigenvalues

  /*
   * This function no longer fails on imaginary, but sets them to NaN
   */
  //Matrix b_eigenvectors_actual;
  //Vector b_eigenvalues_actual;
  //TEST_ASSERT(!PASSED(la::EigenvectorsInit(
  //    b, &b_eigenvalues_actual, &b_eigenvectors_actual)));

  GenMatrix<Precision, false> b_eigenvectors_real_actual;
  GenMatrix<Precision, false> b_eigenvectors_imag_actual;
  GenMatrix<Precision, true> b_eigenvalues_real_actual;
  GenMatrix<Precision, true> b_eigenvalues_imag_actual;
  TEST_ASSERT(PASSED(la::EigenvectorsInit<Precision>(
      b, &b_eigenvalues_real_actual, &b_eigenvalues_imag_actual,
      &b_eigenvectors_real_actual, &b_eigenvectors_imag_actual)));
  AssertApproxVector<Precision>(b_eigenvalues_real_expect, b_eigenvalues_real_actual, 1.0e-3);
  AssertApproxMatrix<Precision>(b_eigenvectors_real_expect, b_eigenvectors_real_actual, 1.0e-3);
  AssertApproxVector<Precision>(b_eigenvalues_imag_expect, b_eigenvalues_imag_actual, 1.0e-3);
  AssertApproxMatrix<Precision>(b_eigenvectors_imag_expect, b_eigenvectors_imag_actual, 1.0e-3);
}

template<typename Precision>
void TrySchur(const GenMatrix<Precision, false> &orig) {
  GenMatrix<Precision, false> z;
  GenMatrix<Precision, false> t;
  GenMatrix<Precision, true> eigen_real;
  GenMatrix<Precision, true> eigen_imag;
  
  la::SchurInit<Precision>(orig, &eigen_real, &eigen_imag, &t, &z);
  
  GenMatrix<Precision, false> z_trans;
  la::TransposeInit<Precision>(z, &z_trans);
  GenMatrix<Precision, false> tmp;
  la::MulInit<Precision>(t, z_trans, &tmp);
  GenMatrix<Precision, false> result;
  la::MulInit<Precision>(z, tmp, &result);
  
  AssertApproxMatrix<Precision>(orig, result, 1.0e-4);
  
  /*
   * This test now fails because Schur finds real components while 
   * Eigenvectors on 3 args only finds true real eigenvalues
   */
  //Vector eigen_real_2;
  //Matrix eigenvectors_2;
  //la::EigenvectorsInit(orig, &eigen_real_2, &eigenvectors_2);
  //AssertApproxVector(eigen_real_2, eigen_real, 1.0e-8);
}

template<typename Precision>
void TestSchur() {
  MAKE_MATRIX_TRANS(Precision, a, 3, 3,
     3, 1, 4,
     1, 5, 9,
     2, 6, 5);
  MAKE_MATRIX_TRANS(Precision, b, 5, 5,
     3, 1, 4, 1, 5,
     9, 2, 6, 5, 3,
     5, 8, 9, 7, 9,
     3, 2, 3, 8, 4,
     6, 2, 6, 4, 3);
  
  TrySchur<Precision>(a);
  TrySchur<Precision>(b);
}

template<typename Precision>
void AssertProperSVD(const GenMatrix<Precision, false>& orig,
    const GenMatrix<Precision, true> &s, const GenMatrix<Precision, false>& u, 
    const GenMatrix<Precision, false>& vt) {
  GenMatrix<Precision, false> s_matrix;
  s_matrix.Init(s.length(), s.length());
  s_matrix.SetDiagonal(s);
  GenMatrix<Precision, false> tmp;
  la::MulInit<Precision>(u, s_matrix, &tmp);
  GenMatrix<Precision, false> result;
  la::MulInit<Precision>(tmp, vt, &result);
  AssertApproxMatrix<Precision>(result, orig, 1.0e-4);
}

template<typename Precision>
void TrySVD(const GenMatrix<Precision, false>& orig) {
  GenMatrix<Precision, true> s;
  GenMatrix<Precision, false> u;
  GenMatrix<Precision, false> vt;
  la::SVDInit<Precision>(orig, &s, &u, &vt);
  AssertProperSVD<Precision>(orig, s, u, vt);
}

template<typename Precision>
void TestSVD() {
  MAKE_MATRIX_TRANS(Precision, a, 3, 3,
      3, 1, 4,
      1, 5, 9,
      2, 6, 5);
  MAKE_MATRIX_TRANS(Precision, a_u_expect, 3, 3,
    -0.21141,  -0.55393,  -0.80528,
     0.46332,  -0.78225,   0.41645,
    -0.86060,  -0.28506,   0.42202);
  MAKE_VECTOR(Precision, a_s_expect, 3,
    13.58236, 2.84548, 2.32869);
  MAKE_MATRIX_TRANS(Precision, a_vt_expect, 3, 3,
    -0.32463,   0.79898,  -0.50620,
    -0.75307,   0.10547,   0.64943,
    -0.57227,  -0.59203,  -0.56746);
  MAKE_MATRIX_TRANS(Precision, b, 3, 10,
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
  MAKE_MATRIX_TRANS(Precision, c, 9, 3,
      3, 1, 4, 1, 5, 9, 2, 6, 5,
      3, 5, 8, 9, 7, 9, 3, 2, 3,
      8, 4, 6, 2, 6, 4, 3, 3, 8);
  MAKE_MATRIX_TRANS(Precision, d, 3, 3,
      0, 1, 0,
      -1, 0, 0,
      0, 0, 1);

  GenMatrix<Precision, false> a_u_actual;
  GenMatrix<Precision, true> a_s_actual;
  GenMatrix<Precision, false> a_vt_actual;

  la::SVDInit<Precision>(a, &a_s_actual, &a_u_actual, &a_vt_actual);
  AssertProperSVD<Precision>(a, a_s_actual, a_u_actual, a_vt_actual);
  AssertApproxVector<Precision>(a_s_expect, a_s_actual, 1.0e-3);
  AssertApproxMatrix<Precision>(a_u_expect, a_u_actual, 1.0e-3);
  AssertApproxMatrix<Precision>(a_vt_expect, a_vt_actual, 1.0e-3);

  GenMatrix<Precision, true> a_s_actual_2;
  la::SVDInit<Precision>(a, &a_s_actual_2);
  AssertApproxVector<Precision>(a_s_expect, a_s_actual_2, 1.0e-3);

  TrySVD<Precision>(b);
  TrySVD<Precision>(c);
  TrySVD<Precision>(d);

  // let's try a big, but asymmetric, one
  GenMatrix<Precision, false> e;
  e.Init(3000, 10);
  for (index_t j = 0; j < e.n_cols(); j++) {
    for (index_t i = 0; i < e.n_rows(); i++) {
      e.set(i, j, rand() * 1.0 / RAND_MAX);
    }
  }

  TrySVD<Precision>(e);
}

template<typename Precision>
void TryCholesky(const GenMatrix<Precision, false> &orig) {
  GenMatrix<Precision, false> u;
  TEST_ASSERT(PASSED(la::CholeskyInit<Precision>(orig, &u)));
  GenMatrix<Precision, false> result;
  la::MulTransAInit<Precision>(u, u, &result);
  AssertApproxMatrix<Precision>(orig, result, 1.0e-3);
}

template<typename Precision>
void TestCholesky() {
  MAKE_MATRIX_TRANS(Precision, a, 3, 3,
      1, 0, 0,
      0, 2, 0,
      0, 0, 3);
  MAKE_MATRIX_TRANS(Precision, b, 4, 4,
    9.00,   0.60,  -0.30,   1.50,
    0.60,  16.04,   1.18,  -1.50,
   -0.30,   1.18,   4.10,  -0.57,
    1.50,  -1.50,  -0.57,  25.45);
  TryCholesky<Precision>(a);
  TryCholesky<Precision>(b);
}

template<typename Precision>
void TrySolveMatrix(const GenMatrix<Precision, false>& a, const GenMatrix<Precision, false>& b) {
  GenMatrix<Precision, false> x;
  TEST_ASSERT(PASSED(la::SolveInit(a, b, &x)));
  GenMatrix<Precision, false> result;
  la::MulInit<Precision>(a, x, &result);
  AssertApproxMatrix<Precision>(b, result, 1.0e-3);
}

template<typename Precision>
void TrySolveVector(const GenMatrix<Precision, false>& a, 
                    const GenMatrix<Precision, true>& b) {
  GenMatrix<Precision, true> x;
  la::SolveInit<Precision>(a, b, &x);
  GenMatrix<Precision, true> result;
  la::MulInit<Precision>(a, x, &result);
  AssertApproxVector<Precision>(b, result, 1.0e-3);
}

template<typename Precision>
void TestSolve() {
  MAKE_MATRIX_TRANS(Precision, a, 3, 3,
     3, 1, 4,
     1, 5, 9,
     2, 6, 5);
  MAKE_MATRIX_TRANS(Precision, a_vectors, 3, 5,
     1, 2, 3,
     4, 5, 2,
     1, 6, 3,
     2, 1, 8,
     4, 2, 6);
  MAKE_VECTOR(Precision, a_vector_1, 3,   3, 1, 2);
  MAKE_VECTOR(Precision, a_vector_2, 3,   2, 4, 6);
  MAKE_VECTOR(Precision, a_vector_3, 3,   2, 4, 6);
  MAKE_VECTOR(Precision, a_vector_4, 3,   5, 7, 8);
  MAKE_MATRIX_TRANS(Precision, b, 5, 5,
     3, 1, 4, 1, 5,
     9, 2, 6, 5, 3,
     5, 8, 9, 7, 9,
     3, 2, 3, 8, 4,
     6, 2, 6, 4, 3);
  
  TrySolveMatrix<Precision>(a, a_vectors);
  TrySolveVector<Precision>(a, a_vector_1);
  TrySolveVector<Precision>(a, a_vector_2);
  TrySolveVector<Precision>(a, a_vector_3);
  TrySolveVector<Precision>(a, a_vector_4);
}

/**
 * Writen by Nick to Test LeastSquareFit
 */
template<typename Precision>
void TestLeastSquareFit() {
  GenMatrix<Precision, false> x;
  GenMatrix<Precision, false> y;
  GenMatrix<Precision, false> a;
  x.Init(3,2);
  x.set(0, 0, 1.0);
  x.set(0, 1, -1.0);
  x.set(1, 0, 0.33);
  x.set(1, 1, 0.44);
  x.set(2, 0, 1.5);
  x.set(2, 1, -0.2);
  y.Init(3, 2);
  y.set(0, 0, 1.5);
  y.set(0, 1, -2.0);
  y.set(1, 0, -0.3);
  y.set(1, 1, 4.0);
  y.set(2, 0, 0.2);
  y.set(2, 1, -0.4);
  la::LeastSquareFit<Precision>(y, x, &a);
  GenMatrix<Precision, false> true_a;
  true_a.Init(2, 2);
  true_a.set(0, 0, 0.0596);
  true_a.set(0, 1, 1.0162);
  true_a.set(1, 0, -1.299);
  true_a.set(1, 1, 4.064);
  for (index_t i=0; i<2; i++) {
    for(index_t j=0; j<2; j++) {
      TEST_DOUBLE_APPROX(true_a.get(i,j), a.get(i, j), 0.001);
    }
  }
  
    

}



/* MLPACK 0.2
 *
 * Copyright (c) 2008, 2009 Alexander Gray,
 *                          Garry Boyer,
 *                          Ryan Riegel,
 *                          Nikolaos Vasiloglou,
 *                          Dongryeol Lee,
 *                          Chip Mappus, 
 *                          Nishant Mehta,
 *                          Hua Ouyang,
 *                          Parikshit Ram,
 *                          Long Tran,
 *                          Wee Chin Wong
 *
 * Copyright (c) 2008, 2009 Georgia Institute of Technology
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
 * 02110-1301, USA.
 */
/**
 * @file uselapack.cc
 *
 * implementstion of the LAPACK wrappers.
 */

#include "uselapack.h"
#include "la.h"
//#include "uselapack.h"
//#include "la.h"

int la::dgetri_block_size;
int la::dgeqrf_block_size;
int la::dorgqr_block_size;
int la::dgeqrf_dorgqr_block_size;

namespace la {

  struct zzzLapackInit {
    zzzLapackInit() {
      double fake_matrix[64];
      double fake_workspace;
      double fake_vector;
      f77_integer fake_pivots;
      f77_integer fake_info;
      
      /* TODO: This may want to be ilaenv */
      F77_FUNC(dgetri)(1, fake_matrix, 1, &fake_pivots, &fake_workspace,
          -1, &fake_info);
      la::dgetri_block_size = int(fake_workspace);
      
      F77_FUNC(dgeqrf)(1, 1, fake_matrix, 1, &fake_vector, &fake_workspace, -1,
          &fake_info);
      la::dgeqrf_block_size = int(fake_workspace);
      
      F77_FUNC(dorgqr)(1, 1, 1, fake_matrix, 1, &fake_vector, &fake_workspace, -1,
          &fake_info);
      la::dorgqr_block_size = int(fake_workspace);
      
      la::dgeqrf_dorgqr_block_size =
          std::max(la::dgeqrf_block_size, la::dorgqr_block_size);
    }
  };
};

success_t la::PLUInit(const Matrix &A,
    std::vector<f77_integer>& pivots, Matrix *L, Matrix *U) {
  index_t m = A.n_rows();
  index_t n = A.n_cols();
  success_t success;

  if (m > n) {
    pivots.reserve(n);
    L->Copy(A);
    U->Init(n, n);
    success = PLUExpert(&pivots.front(), L); // undefined?

    if (!PASSED(success)) {
      return success;
    }

    for (index_t j = 0; j < n; j++) {
      double *lcol = L->GetColumnPtr(j);
      double *ucol = U->GetColumnPtr(j);

      mem::Copy(ucol, lcol, j + 1);
      mem::Zero(ucol + j + 1, n - j - 1);
      mem::Zero(lcol, j);
      lcol[j] = 1.0;
    }
  } else {
    pivots.reserve(m);
    L->Init(m, m);
    U->Copy(A);
    success = PLUExpert(&pivots.front(), U); // undefined

    if (!PASSED(success)) {
      return success;
    }

    for (index_t j = 0; j < m; j++) {
      double *lcol = L->GetColumnPtr(j);
      double *ucol = U->GetColumnPtr(j);

      mem::Zero(lcol, j);
      lcol[j] = 1.0;
      mem::Copy(lcol + j + 1, ucol + j + 1, m - j - 1);
      mem::Zero(ucol + j + 1, m - j - 1);
    }
  }

  return success;
}

success_t la::Inverse(Matrix *A) {
  f77_integer pivots[A->n_rows()];

  success_t success = PLUExpert(pivots, A);

  if (!PASSED(success)) {
    return success;
  }

  return InverseExpert(pivots, A);
}
  

success_t la::InverseOverwrite(const Matrix &A, Matrix *B) {
  f77_integer pivots[A.n_rows()];

  if (likely(A.ptr() != B->ptr())) {
    B->CopyValues(A);
  }
  success_t success = PLUExpert(pivots, B);

  if (!PASSED(success)) {
    return success;
  }

  return InverseExpert(pivots, B);
}

long double la::Determinant(const Matrix &A) {
  DEBUG_MATSQUARE(A);
  int n = A.n_rows();
  f77_integer pivots[n];
  Matrix LU;

  LU.Copy(A);
  PLUExpert(pivots, &LU);

  long double det = 1.0;

  for (index_t i = 0; i < n; i++) {
    if (pivots[i] != i+1) {
      // pivoting occured (note FORTRAN has 1-based indexing)
      det = -det;
    }
    det *= LU.get(i, i);
  }

  return det;
}

double la::DeterminantLog(const Matrix &A, int *sign_out) {
  DEBUG_MATSQUARE(A);
  int n = A.n_rows();
  f77_integer pivots[n];
  Matrix LU;

  LU.Copy(A);
  PLUExpert(pivots, &LU);

  double log_det = 0.0;
  int sign_det = 1;

  for (index_t i = 0; i < n; i++) {
    if (pivots[i] != i+1) {
      // pivoting occured (note FORTRAN has one-based indexing)
      sign_det = -sign_det;
    }

    double value = LU.get(i, i);

    if (value < 0) {
      sign_det = -sign_det;
      value = -value;
    } else if (!(value > 0)) {
      sign_det = 0;
      log_det = DBL_NAN;
      break;
    }

    log_det += log(value);
  }

  if (sign_out) {
    *sign_out = sign_det;
  }

  return log_det;
}

/*
Replaced this with a non-querying version that uses cached block sizes.

success_t la::QRExpert(Matrix *A_in_Q_out, Matrix *R) {
  f77_integer info;
  f77_integer m = A_in_Q_out->n_rows();
  f77_integer n = A_in_Q_out->n_cols();
  f77_integer k = std::min(m, n);
  double d; // for querying optimal work size
  double tau[k];

  // Obtain both Q and R in A_in_Q_out
  F77_FUNC(dgeqrf)(m, n, A_in_Q_out->ptr(), m,
      tau, &d, -1, &info);
  {
    f77_integer lwork = (f77_integer)d;
    double work[lwork];

    F77_FUNC(dgeqrf)(m, n, A_in_Q_out->ptr(), m,
        tau, work, lwork, &info);
  }
  
  R->SetZero();
  if (info != 0) {
    return SUCCESS_FROM_LAPACK(info);
  }

  // Extract R
  for (index_t j = 0; j < n; j++) {
    mem::Copy(R->GetColumnPtr(j), A_in_Q_out->GetColumnPtr(j),
        std::min(j + 1, k));
  }
  
  // Fix Q
  F77_FUNC(dorgqr)(m, k, k, A_in_Q_out->ptr(), m,
      tau, &d, -1, &info);
  {
    f77_integer lwork = (f77_integer)d;
    double work[lwork];

    F77_FUNC(dorgqr)(m, k, k, A_in_Q_out->ptr(), m,
        tau, work, lwork, &info);
  }

  return SUCCESS_FROM_LAPACK(info);
}
*/

success_t la::QRExpert(Matrix *A_in_Q_out, Matrix *R) {
  f77_integer info;
  f77_integer m = A_in_Q_out->n_rows();
  f77_integer n = A_in_Q_out->n_cols();
  f77_integer k = std::min(m, n);
  f77_integer lwork = n * dgeqrf_dorgqr_block_size;
  double tau[k + lwork];
  double *work = tau + k;

  // Obtain both Q and R in A_in_Q_out
  F77_FUNC(dgeqrf)(m, n, A_in_Q_out->ptr(), m,
     tau, work, lwork, &info);

  if (info != 0) {
    return SUCCESS_FROM_LAPACK(info);
  }

  // Extract R
  for (index_t j = 0; j < n; j++) {
    double *r_col = R->GetColumnPtr(j);
    double *q_col = A_in_Q_out->GetColumnPtr(j);
    int i = std::min(j + 1, index_t(k));
    mem::Copy(r_col, q_col, i);
    mem::Zero(r_col + i, k - i);
  }

  // Fix Q
  F77_FUNC(dorgqr)(m, k, k, A_in_Q_out->ptr(), m,
      tau, work, lwork, &info);

  return SUCCESS_FROM_LAPACK(info);
}

success_t la::QRInit(const Matrix &A, Matrix *Q, Matrix *R) {
  index_t k = std::min(A.n_rows(), A.n_cols());
  Q->Copy(A);
  R->Init(k, A.n_cols());
  success_t success = QRExpert(Q, R);
  Q->ResizeNoalias(k);

  return success;
}

success_t la::SchurExpert(Matrix *A_in_T_out,
    double *w_real, double *w_imag, double *Z) {
  DEBUG_MATSQUARE(*A_in_T_out);
  f77_integer info;
  f77_integer n = A_in_T_out->n_rows();
  f77_integer sdim;
  const char *job = Z ? "V" : "N";
  double d; // for querying optimal work size

  F77_FUNC(dgees)(job, "N", NULL,
      n, A_in_T_out->ptr(), n, &sdim, w_real, w_imag,
      Z, n, &d, -1, NULL, &info);
  {
    f77_integer lwork = (f77_integer)d;
    double work[lwork];

    F77_FUNC(dgees)(job, "N", NULL,
        n, A_in_T_out->ptr(), n, &sdim, w_real, w_imag,
        Z, n, work, lwork, NULL, &info);
  }

  return SUCCESS_FROM_LAPACK(info);
}

success_t la::EigenExpert(Matrix *A_garbage,
    double *w_real, double *w_imag, double *V_raw) {
  DEBUG_MATSQUARE(*A_garbage);
  f77_integer info;
  f77_integer n = A_garbage->n_rows();
  const char *job = V_raw ? "V" : "N";
  double d; // for querying optimal work size

  F77_FUNC(dgeev)("N", job, n, A_garbage->ptr(), n,
      w_real, w_imag, NULL, 1, V_raw, n, &d, -1, &info);
  {
    f77_integer lwork = (f77_integer)d;
    double work[lwork];

    F77_FUNC(dgeev)("N", job, n, A_garbage->ptr(), n,
        w_real, w_imag, NULL, 1, V_raw, n, work, lwork, &info);
  }

  return SUCCESS_FROM_LAPACK(info);
}

success_t la::EigenvaluesInit(const Matrix &A, Vector *w) {
  DEBUG_MATSQUARE(A);
  int n = A.n_rows();
  w->Init(n);
  double w_imag[n];

  Matrix tmp;
  tmp.Copy(A);
  success_t success = SchurExpert(&tmp, w->ptr(), w_imag, NULL);

  if (!PASSED(success)) {
    return success;
  }

  for (index_t j = 0; j < n; j++) {
    if (unlikely(w_imag[j] != 0.0)) {
      (*w)[j] = DBL_NAN;
    }
  }

  return success;
}

success_t la::EigenvectorsInit(const Matrix &A,
    Vector *w_real, Vector *w_imag, Matrix *V_real, Matrix *V_imag) {
  DEBUG_MATSQUARE(A);
  index_t n = A.n_rows();
  w_real->Init(n);
  w_imag->Init(n);
  V_real->Init(n, n);
  V_imag->Init(n, n);

  Matrix tmp;
  tmp.Copy(A);
  success_t success = EigenExpert(&tmp,
      w_real->ptr(), w_imag->ptr(), V_real->ptr());

  if (!PASSED(success)) {
    return success;
  }

  V_imag->SetZero();
  for (index_t j = 0; j < n; j++) {
    if (unlikely(w_imag->get(j) != 0.0)) {
      double *r_cur = V_real->GetColumnPtr(j);
      double *r_next = V_real->GetColumnPtr(j+1);
      double *i_cur = V_imag->GetColumnPtr(j);
      double *i_next = V_imag->GetColumnPtr(j+1);

      for (index_t i = 0; i < n; i++) {
        i_next[i] = -(i_cur[i] = r_next[i]);
        r_next[i] = r_cur[i];
      }

      j++; // skip paired column
    }
  }

  return success;
}

success_t la::EigenvectorsInit(const Matrix &A, Vector *w, Matrix *V) {
  DEBUG_MATSQUARE(A);
  index_t n = A.n_rows();
  w->Init(n);
  double w_imag[n];
  V->Init(n, n);

  Matrix tmp;
  tmp.Copy(A);
  success_t success = EigenExpert(&tmp, w->ptr(), w_imag, V->ptr());

  if (!PASSED(success)) {
    return success;
  }

  for (index_t j = 0; j < n; j++) {
    if (unlikely(w_imag[j] != 0.0)) {
      (*w)[j] = DBL_NAN;
    }
  }

  return success;
}

success_t la::GenEigenSymmetric(int itype, Matrix *A_eigenvec, Matrix *B_chol, double *w) {
  DEBUG_MATSQUARE(*A_eigenvec);
  DEBUG_MATSQUARE(*B_chol);
  DEBUG_ASSERT(A_eigenvec->n_rows()==B_chol->n_rows());
  f77_integer itype_f77 = itype;
  f77_integer info;
  f77_integer n = A_eigenvec->n_rows();
  const char *job = "V"; // Compute eigenvalues and eigenvectors.
  double d; // for querying optimal work size

  F77_FUNC(dsygv)(&itype_f77, job, "U", n, A_eigenvec->ptr(), n, B_chol->ptr(), n, w,
      &d, -1, &info);
  {
    f77_integer lwork = (f77_integer)d;
    double work[lwork];

    F77_FUNC(dsygv)(&itype_f77, job, "U", n, A_eigenvec->ptr(), n, B_chol->ptr(), n, w,
      work, lwork, &info);
  }

  return SUCCESS_FROM_LAPACK(info);
}

success_t la::GenEigenNonSymmetric(Matrix *A_garbage, Matrix *B_garbage,
    double *alpha_real, double *alpha_imag, double *beta, double *V_raw) {
  DEBUG_MATSQUARE(*A_garbage);
  DEBUG_MATSQUARE(*B_garbage);
  DEBUG_ASSERT(A_garbage->n_rows()==B_garbage->n_rows());
  f77_integer info;
  f77_integer n = A_garbage->n_rows();
  const char *job = V_raw ? "V" : "N";
  double d; // for querying optimal work size

  F77_FUNC(dgegv)("N", job, n, A_garbage->ptr(), n, B_garbage->ptr(), n, 
      alpha_real, alpha_imag, beta, NULL, 1, V_raw, n, &d, -1, &info);
  {
    f77_integer lwork = (f77_integer)d;
    double work[lwork];

    F77_FUNC(dgegv)("N", job, n, A_garbage->ptr(), n, B_garbage->ptr(), n, 
      alpha_real, alpha_imag, beta, NULL, 1, V_raw, n, work, lwork, &info);
  }

  return SUCCESS_FROM_LAPACK(info);
}

/*
DGESDD is supposed to be faster, although I haven't actually found this
to be the case.

success_t la::SVDExpert(Matrix* A_garbage, double *s, double *U, double *VT) {
  f77_integer info;
  f77_integer m = A_garbage->n_rows();
  f77_integer n = A_garbage->n_cols();
  f77_integer k = std::min(m, n);
  const char *job_u = U ? (U == A_garbage->ptr ? "O" : "S") : "N";
  const char *job_v = VT ? "S" : "N";
  double d; // for querying optimal work size

  F77_FUNC(dgesvd)(job_u, job_v, m, n, A_garbage->ptr(), m,
      s, U, m, VT, k, &d, -1, &info);
  {
    f77_integer lwork = (f77_integer)lwork_dbl;
    double work[lwork];

    F77_FUNC(dgesvd)(job_u, job_v, m, n, A_garbage->ptr(), m,
        s, U, m, VT, k, work, lwork, &info);
  }

  return SUCCESS_FROM_LAPACK(info);
}
*/

success_t la::SVDExpert(Matrix* A_garbage, double *s, double *U, double *VT) {
  DEBUG_ASSERT_MSG((U == NULL) == (VT == NULL),
                   "You must fill both U and VT or neither.");
  f77_integer info;
  f77_integer m = A_garbage->n_rows();
  f77_integer n = A_garbage->n_cols();
  f77_integer k = std::min(m, n);
  f77_integer iwork[8 * k];
  const char *job = U ? "S" : "N";
  double d; // for querying optimal work size

  F77_FUNC(dgesdd)(job, m, n, A_garbage->ptr(), m,
      s, U, m, VT, k, &d, -1, iwork, &info);
  {
    f77_integer lwork = (f77_integer)d;
    // work for DGESDD can be large, we really do need to malloc it
    double *work = mem::Alloc<double>(lwork);

    F77_FUNC(dgesdd)(job, m, n, A_garbage->ptr(), m,
        s, U, m, VT, k, work, lwork, iwork, &info);

    mem::Free(work);
  }

  return SUCCESS_FROM_LAPACK(info);
}

success_t la::Cholesky(Matrix *A_in_U_out) {
  DEBUG_MATSQUARE(*A_in_U_out);
  f77_integer info;
  f77_integer n = A_in_U_out->n_rows();

  F77_FUNC(dpotrf)("U", n, A_in_U_out->ptr(), n, &info);

  /* set the garbage part of the matrix to 0. */
  for (f77_integer j = 0; j < n; j++) {
    mem::Zero(A_in_U_out->GetColumnPtr(j) + j + 1, n - j - 1);
  }

  return SUCCESS_FROM_LAPACK(info);
}

static la::zzzLapackInit lapack_initializer;

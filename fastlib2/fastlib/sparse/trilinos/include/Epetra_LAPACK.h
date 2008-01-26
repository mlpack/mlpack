
//@HEADER
/*
************************************************************************

              Epetra: Linear Algebra Services Package 
                Copyright (2001) Sandia Corporation

Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
license for use of this work by or on behalf of the U.S. Government.

This library is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation; either version 2.1 of the
License, or (at your option) any later version.
 
This library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.
 
You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
USA
Questions? Contact Michael A. Heroux (maherou@sandia.gov) 

************************************************************************
*/
//@HEADER

#ifndef EPETRA_LAPACK_H
#define EPETRA_LAPACK_H

//! Epetra_LAPACK:  The Epetra LAPACK Wrapper Class.
/*! The Epetra_LAPACK class is a wrapper that encapsulates LAPACK
    (Linear Algebra Package).  LAPACK provides portable, high-
    performance implementations of linear, eigen, SVD, etc solvers.

    The standard LAPACK interface is Fortran-specific.  Unfortunately, the 
    interface between C++ and Fortran is not standard across all computer
    platforms.  The Epetra_LAPACK class provides C++ wrappers for the LAPACK
    kernels in order to insulate the rest of Epetra from the details of C++ to Fortran
    translation.
    A Epetra_LAPACK object is essentially nothing, but allows access to the LAPACK wrapper
    functions.
  
    Epetra_LAPACK is a serial interface only.  This is appropriate since the standard 
    LAPACK are only specified for serial execution (or shared memory parallel).
*/

#include "Epetra_Object.h"

class Epetra_LAPACK {
    
  public:
    //! @name Constructors/destructors
  //@{ 
  //! Epetra_LAPACK Constructor.
  /*! Builds an instance of a serial LAPACK object.
   */
  Epetra_LAPACK(void);


  //! Epetra_LAPACK Copy Constructor.
  /*! Makes an exact copy of an existing Epetra_LAPACK instance.
  */
  Epetra_LAPACK(const Epetra_LAPACK& LAPACK);

  //! Epetra_LAPACK Destructor.
  virtual ~Epetra_LAPACK(void);
  //@}


  //! @name Symmetric Positive Definite linear system routines
  //@{ 
  
  //! Epetra_LAPACK factorization for positive definite matrix (SPOTRF)
  void POTRF( const char UPLO, const int N, float * A, const int LDA, int * INFO) const;
  //! Epetra_LAPACK factorization for positive definite matrix (DPOTRF)
  void POTRF( const char UPLO, const int N, double * A, const int LDA, int * INFO) const;

  //! Epetra_LAPACK solve (after factorization) for positive definite matrix (SPOTRS)
  void POTRS( const char UPLO, const int N, const int NRHS, const float * A, const int LDA, float * X, const int LDX, int * INFO) const;
  //! Epetra_LAPACK solve (after factorization) for positive definite matrix (DPOTRS)
  void POTRS( const char UPLO, const int N, const int NRHS, const double * A, const int LDA, double * X, const int LDX, int * INFO) const;

  //! Epetra_LAPACK inversion  for positive definite matrix (SPOTRI)
  void POTRI( const char UPLO, const int N, float * A, const int LDA, int * INFO) const;
  //! Epetra_LAPACK inversion  for positive definite matrix (DPOTRI)
  void POTRI( const char UPLO, const int N, double * A, const int LDA, int * INFO) const;

  //! Epetra_LAPACK condition number estimator for positive definite matrix (SPOCON)
  void POCON( const char UPLO, const int N, const float * A, const int LDA, const float ANORM, 
			  float * RCOND, float * WORK, int * IWORK, int * INFO) const;
  //! Epetra_LAPACK condition number estimator for positive definite matrix (DPOCON)
  void POCON( const char UPLO, const int N, const double * A, const int LDA, const double ANORM, 
			  double * RCOND, double * WORK, int * IWORK, int * INFO) const;

  //! Epetra_LAPACK factor and solve for positive definite matrix (SPOSV)
  void POSV( const char UPLO, const int N, const int NRHS, float * A, const int LDA, float * X, const int LDX, int * INFO) const;
  //! Epetra_LAPACK factor and solve for positive definite matrix (DPOSV)
  void POSV( const char UPLO, const int N, const int NRHS, double * A, const int LDA, double * X, const int LDX, int * INFO) const;

  //! Epetra_LAPACK equilibration for positive definite matrix (SPOEQU)
  void POEQU(const int N, const float * A, const int LDA, float * S, float * SCOND, float * AMAX, int * INFO) const;
  //! Epetra_LAPACK equilibration for positive definite matrix (DPOEQU)
  void POEQU(const int N, const double * A, const int LDA, double * S, double * SCOND, double * AMAX, int * INFO) const;

  //! Epetra_LAPACK solve driver for positive definite matrix (SPOSVX)
  void PORFS(const char UPLO, const int N, const int NRHS, const float * A, const int LDA, const float * AF, const int LDAF, 
	     const float * B, const int LDB, float * X, const int LDX, 
	     float * FERR, float * BERR, float * WORK, int * IWORK, int * INFO) const;
  //! Epetra_LAPACK solve driver for positive definite matrix (DPOSVX)
  void PORFS(const char UPLO, const int N, const int NRHS, const double * A, const int LDA, const double * AF, const int LDAF, 
	     const double * B, const int LDB, double * X, const int LDX,
	     double * FERR, double * BERR, double * WORK, int * IWORK, int * INFO) const;

  //! Epetra_LAPACK solve driver for positive definite matrix (SPOSVX)
  void POSVX(const char FACT, const char UPLO, const int N, const int NRHS, float * A, const int LDA, float * AF, const int LDAF, 
	     const char EQUED, float * S, float * B, const int LDB, float * X, const int LDX, float * RCOND, 
	     float * FERR, float * BERR, float * WORK, int * IWORK, int * INFO) const;
  //! Epetra_LAPACK solve driver for positive definite matrix (DPOSVX)
  void POSVX(const char FACT, const char UPLO, const int N, const int NRHS, double * A, const int LDA, double * AF, const int LDAF, 
	     const char EQUED, double * S, double * B, const int LDB, double * X, const int LDX, double * RCOND, 
	     double * FERR, double * BERR, double * WORK, int * IWORK, int * INFO) const;
  //@}

  //! @name General linear system routines
  //@{ 

  //! Epetra_LAPACK simple driver to solve least-squares systems
  void GELS( const char TRANS, const int M, const int N, const int NRHS, double* A, const int LDA, 
	  double* B, const int LDB, double* WORK, const int LWORK, int * INFO) const;
  //! Epetra_LAPACK factorization for general matrix (SGETRF)
  void GETRF( const int M, const int N, float * A, const int LDA, int * IPIV, int * INFO) const;
  //! Epetra_LAPACK factorization for general matrix (DGETRF)
  void GETRF( const int M, const int N, double * A, const int LDA, int * IPIV, int * INFO) const;

  //! Epetra_LAPACK QR factorization for general matrix (SGEQRF)
  void GEQRF( const int M, const int N,  float * A, const int LDA,  float * TAU,  float * WORK, const int lwork, int * INFO) const;
  //! Epetra_LAPACK factorization for general matrix (DGEQRF)
  void GEQRF( const int M, const int N, double * A, const int LDA, double * TAU, double * WORK, const int lwork, int * INFO) const;

  //! Epetra_LAPACK solve (after factorization) for general matrix (SGETRS)
  void GETRS( const char TRANS, const int N, const int NRHS, const float * A, const int LDA, const int * IPIV, float * X, const int LDX, int * INFO) const;
  //! Epetra_LAPACK solve (after factorization) for general matrix (DGETRS)
  void GETRS( const char TRANS, const int N, const int NRHS, const double * A, const int LDA, const int * IPIV, double * X, const int LDX, int * INFO) const;

  //! Epetra_LAPACK inversion  for general matrix (SGETRI)
  void GETRI( const int N, float * A, const int LDA, int * IPIV, float * WORK, const int * LWORK, int * INFO) const;
  //! Epetra_LAPACK inversion  for general matrix (DGETRI)
  void GETRI( const int N, double * A, const int LDA, int * IPIV, double * WORK, const int * LWORK, int * INFO) const;

  //! Epetra_LAPACK condition number estimator for general matrix (SGECON)
  void GECON( const char NORM, const int N, const float * A, const int LDA, const float ANORM, 
			  float * RCOND, float * WORK, int * IWORK, int * INFO) const;
  //! Epetra_LAPACK condition number estimator for general matrix (DGECON)
  void GECON( const char NORM, const int N, const double * A, const int LDA, const double ANORM, 
			  double * RCOND, double * WORK, int * IWORK, int * INFO) const;

  //! Epetra_LAPACK factor and solve for general matrix (SGESV)
  void GESV( const int N, const int NRHS, float * A, const int LDA, int * IPIV, float * X, const int LDX, int * INFO) const;
  //! Epetra_LAPACK factor and solve for general matrix (DGESV)
  void GESV( const int N, const int NRHS, double * A, const int LDA, int * IPIV, double * X, const int LDX, int * INFO) const;

  //! Epetra_LAPACK equilibration for general matrix (SGEEQU)
  void GEEQU(const int M, const int N, const float * A, const int LDA, float * R, float * C, float * ROWCND, float * COLCND, float * AMAX, int * INFO) const;
  //! Epetra_LAPACK equilibration for general matrix (DGEEQU)
  void GEEQU(const int M, const int N, const double * A, const int LDA, double * R, double * C, double * ROWCND, double * COLCND, double * AMAX, int * INFO) const;

  //! Epetra_LAPACK Refine solution (GERFS)
  void GERFS(const char TRANS, const int N, const int NRHS, const float * A, const int LDA, const float * AF, const int LDAF, 
	     const int * IPIV, const float * B, const int LDB, float * X, const int LDX, 
	     float * FERR, float * BERR, float * WORK, int * IWORK, int * INFO) const;
  //! Epetra_LAPACK Refine solution (GERFS)
  void GERFS(const char TRANS, const int N, const int NRHS, const double * A, const int LDA, const double * AF, const int LDAF, 
	     const int * IPIV, const double * B, const int LDB, double * X, const int LDX,
	     double * FERR, double * BERR, double * WORK, int * IWORK, int * INFO) const;

  //! Epetra_LAPACK solve driver for general matrix (SGESVX)
  void GESVX(const char FACT, const char TRANS, const int N, const int NRHS, float * A, const int LDA, float * AF, const int LDAF, int * IPIV, 
	     const char EQUED, float * R, float * C, float * B, const int LDB, float * X, const int LDX, float * RCOND, 
	     float * FERR, float * BERR, float * WORK, int * IWORK, int * INFO) const;
  //! Epetra_LAPACK solve driver for general matrix (DGESVX)
  void GESVX(const char FACT, const char TRANS, const int N, const int NRHS, double * A, const int LDA, double * AF, const int LDAF, int * IPIV, 
	     const char EQUED, double * R, double * C, double * B, const int LDB, double * X, const int LDX, double * RCOND, 
	     double * FERR, double * BERR, double * WORK, int * IWORK, int * INFO) const;


  //! Epetra_LAPACK wrapper for reduction to Hessenberg form (SGEHRD)
  void GEHRD(const int N, const int ILO, const int IHI, float * A, const int LDA, float * TAU, float * WORK, const int LWORK, int * INFO) const;
  //! Epetra_LAPACK wrapper for reduction to Hessenberg form (DGEHRD)
  void GEHRD(const int N, const int ILO, const int IHI, double * A, const int LDA, double * TAU, double * WORK, const int LWORK, int * INFO) const;
  //@}

  //! @name Hessenberg routines
  //@{ 
  //! Epetra_LAPACK wrapper for computing the eigenvalues of a real upper Hessenberg matrix (SHSEQR)
  void HSEQR( const char JOB, const char COMPZ, const int N, const int ILO, const int IHI, float * H, const int LDH, float * WR, float * WI,
	      float * Z, const int LDZ, float * WORK, const int LWORK, int * INFO) const;
  //! Epetra_LAPACK wrapper for computing the eigenvalues of a real upper Hessenberg matrix (DHSEQR)
  void HSEQR( const char JOB, const char COMPZ, const int N, const int ILO, const int IHI, double * H, const int LDH, double * WR, double * WI,
	      double * Z, const int LDZ, double * WORK, const int LWORK, int * INFO) const;
  //@}

  //! @name Orthogonal matrix routines
  //@{ 
  //! Epetra_LAPACK wrapper for generating a m x n real matrix Q with orthonormal columns, defined as the product of k elementary reflectors. (SORGQR)
  void ORGQR( const int M, const int N, const int K, float * A, const int LDA, float * TAU, float * WORK, const int LWORK, int * INFO) const;
  //! Epetra_LAPACK wrapper for generating a m x n real matrix Q with orthonormal columns, defined as the product of k elementary reflectors. (DORGQR)
  void ORGQR( const int M, const int N, const int K, double * A, const int LDA, double * TAU, double * WORK, const int LWORK, int * INFO) const;

  //! Epetra_LAPACK wrapper for generating a real orthogonal matrix Q defined by elementary reflectors. (SORGHR)
  void ORGHR( const int N, const int ILO, const int IHI, float * A, const int LDA, float * TAU, float * WORK, const int LWORK, int * INFO) const;
  //! Epetra_LAPACK wrapper for generating a real orthogonal matrix Q defined by elementary reflectors. (DORGHR)
  void ORGHR( const int N, const int ILO, const int IHI, double * A, const int LDA, double * TAU, double * WORK, const int LWORK, int * INFO) const;

  //! Epetra_LAPACK wrapper for applying an orthogonal matrix in-place (SORMHR)
  void ORMHR( const char SIDE, const char TRANS, const int M, const int N, const int ILO, const int IHI, const float * A, const int LDA, 
	      const float * TAU, float * C,
	      const int LDC, float * WORK, const int LWORK, int * INFO) const;
  //! Epetra_LAPACK wrapper for applying an orthogonal matrix in-place (DORMHR)
  void ORMHR( const char SIDE, const char TRANS, const int M, const int N, const int ILO, const int IHI, const double * A, const int LDA, 
	      const double * TAU, double * C,
	      const int LDC, double * WORK, const int LWORK, int * INFO) const;
  //! Epetra_LAPACK for forming the triangular factor of a product of elementary Householder reflectors (SLARFT).
  void LARFT( const char DIRECT, const char STOREV, const int N, const int K, double * V, const int LDV, double * TAU, double * T, const int LDT) const;
  //! Epetra_LAPACK for forming the triangular factor of a product of elementary Householder reflectors (DLARFT).
  void LARFT( const char DIRECT, const char STOREV, const int N, const int K, float * V, const int LDV, float * TAU, float * T, const int LDT) const;
  //@}

  //! @name Triangular matrix routines
  //@{ 

  //! Epetra_LAPACK wrapper for computing eigenvectors of a quasi-triangular/triagnular matrix (STREVC)
  /*! \warning HOWMNY = 'S" is not supported.
   */
  void TREVC( const char SIDE, const char HOWMNY, int * SELECT, const int N, const float * T, const int LDT, float *VL, const int LDVL,
	      float * VR, const int LDVR, const int MM, int * M, float * WORK, int * INFO) const;
  //! Epetra_LAPACK wrapper for computing eigenvectors of a quasi-triangular/triagnular matrix (DTREVC)
  /*! \warning HOWMNY = 'S" is not supported.
   */
  void TREVC( const char SIDE, const char HOWMNY, int * SELECT, const int N, const double * T, const int LDT, double *VL, const int LDVL,
	      double * VR, const int LDVR, const int MM, int  *M, double * WORK, int * INFO) const;

  //! Epetra_LAPACK wrapper for reordering the real-Schur/Schur factorization of a matrix (STREXC)
  void TREXC( const char COMPQ, const int N, float * T, const int LDT, float * Q, const int LDQ, int IFST, int ILST, 
	      float * WORK, int * INFO) const;
  //! Epetra_LAPACK wrapper for reordering the real-Schur/Schur factorization of a matrix (DTREXC)
  void TREXC( const char COMPQ, const int N, double * T, const int LDT, double * Q, const int LDQ, int IFST, int ILST, 
	      double * WORK, int * INFO) const;
  //@}

  //! @name Singular Value Decomposition matrix routines
  //@{ 

  //! Epetra_LAPACK wrapper for computing the singular value decomposition (SGESVD)
  void GESVD( const char JOBU, const char JOBVT, const int M, const int N, float * A, const int LDA, float * S, float * U,
	      const int LDU, float * VT, const int LDVT, float * WORK, const int * LWORK, int * INFO) const;
  //! Epetra_LAPACK wrapper for computing the singular value decomposition (DGESVD)
  void GESVD( const char JOBU, const char JOBVT, const int M, const int N, double * A, const int LDA, double * S, double * U,
	      const int LDU, double * VT, const int LDVT, double * WORK, const int * LWORK, int * INFO) const;

  //! Epetra_LAPACK wrapper to compute the generalized singular value decomposition (GSVD) of an M-by-N real matrix A and P-by-N real matrix B
  void GGSVD(const char JOBU, const char JOBV, const char JOBQ, const int M, const int N, const int P, int * K, int * L,  double* A,  const int LDA,  double* B,  const int LDB,
                          double* ALPHA,  double* BETA,  double* U,  const int LDU, double* V, const int LDV, double* Q, const int LDQ, double* WORK, int* IWORK,
                          int* INFO) const;
  //! Epetra_LAPACK wrapper to compute the generalized singular value decomposition (GSVD) of an M-by-N real matrix A and P-by-N real matrix B
  void GGSVD(const char JOBU, const char JOBV, const char JOBQ, const int M, const int N, const int P, int * K, int * L,  float* A,  const int LDA,  float* B,  const int LDB,
                          float* ALPHA,  float* BETA,  float* U,  const int LDU, float* V, const int LDV, float* Q, const int LDQ, float* WORK, int* IWORK,
                          int* INFO) const;
   //@}

   //! @name Eigenvalue/Eigenvector routines
  //@{ 
  //! Epetra_LAPACK wrapper to compute for an N-by-N real nonsymmetric matrix A, the eigenvalues and, optionally, the left and/or right eigenvectors
  void GEEV(const char JOBVL, const char JOBVR, const int N, double* A, const int LDA, double* WR, double* WI, 
			double* VL, const int LDVL, double* VR, const int LDVR, double* WORK, const int LWORK, int* INFO) const;
  //! Epetra_LAPACK wrapper to compute for an N-by-N real nonsymmetric matrix A, the eigenvalues and, optionally, the left and/or right eigenvectors
  void GEEV(const char JOBVL, const char JOBVR, const int N, float* A, const int LDA, float* WR, float* WI, 
			float* VL, const int LDVL, float* VR, const int LDVR, float* WORK, const int LWORK, int* INFO) const;

  //! Epetra_LAPACK wrapper to compute all the eigenvalues and, optionally, eigenvectors of a real symmetric matrix A in packed storage
  void SPEV(const char JOBZ, const char UPLO, const int N, double* AP, double* W, double* Z, int LDZ, double* WORK, int* INFO) const;
  //! Epetra_LAPACK wrapper to compute all the eigenvalues and, optionally, eigenvectors of a real symmetric matrix A in packed storage
  void SPEV(const char JOBZ, const char UPLO, const int N, float* AP, float* W, float* Z, int LDZ, float* WORK, int* INFO) const;

  //! Epetra_LAPACK wrapper to compute all the eigenvalues and, optionally, the eigenvectors of a real generalized symmetric-definite eigenproblem, of the form A*x=(lambda)*B*x, A*Bx=(lambda)*x, or B*A*x=(lambda)*x
  void SPGV(const int ITYPE, const char JOBZ, const char UPLO, const int N, double* AP, double* BP, double* W, double* Z, const int LDZ, double* WORK, int* INFO) const;
  //! Epetra_LAPACK wrapper to compute all the eigenvalues and, optionally, the eigenvectors of a real generalized symmetric-definite eigenproblem, of the form A*x=(lambda)*B*x, A*Bx=(lambda)*x, or B*A*x=(lambda)*x
  void SPGV(const int ITYPE, const char JOBZ, const char UPLO, const int N, float* AP, float* BP, float* W, float* Z, const int LDZ, float* WORK, int* INFO) const;

  //! Epetra_LAPACK wrapper to compute all eigenvalues and, optionally, eigenvectors of a real symmetric matrix A
  void SYEV(const char JOBZ, const char UPLO, const int N, double* A, const int LDA, double* W, double* WORK, const int LWORK, int* INFO) const;
  //! Epetra_LAPACK wrapper to compute all eigenvalues and, optionally, eigenvectors of a real symmetric matrix A
  void SYEV(const char JOBZ, const char UPLO, const int N, float* A, const int LDA, float* W, float* WORK, const int LWORK, int* INFO) const;

  //! Epetra_LAPACK wrapper to compute  all  eigenvalues and, optionally, eigenvectors of a real symmetric matrix A
  void SYEVD(const char JOBZ, const char UPLO,  const int N,  double* A,  const int LDA,  double* W,  
	     double* WORK,  const int LWORK,  int* IWORK, const int LIWORK, int* INFO) const;
  //! Epetra_LAPACK wrapper to compute  all  eigenvalues and, optionally, eigenvectors of a real symmetric matrix A
  void SYEVD(const char JOBZ, const char UPLO,  const int N,  float* A,  const int LDA,  float* W,  
	     float* WORK,  const int LWORK,  int* IWORK, const int LIWORK, int* INFO) const;

  //! Epetra_LAPACK wrapper to compute selected eigenvalues and, optionally, eigenvectors of a real symmetric matrix A
  void SYEVX(const char JOBZ, const char RANGE, const char UPLO,  const int N,  double* A,  const int LDA,  
	     const double* VL,  const double* VU,  const int* IL,  const int* IU,
	     const double ABSTOL,  int * M,  double* W,  double* Z,  const int LDZ, double* WORK, 
	     const int LWORK, int* IWORK, int* IFAIL,
	     int* INFO) const;
  //! Epetra_LAPACK wrapper to compute selected eigenvalues and, optionally, eigenvectors of a real symmetric matrix A
  void SYEVX(const char JOBZ, const char RANGE, const char UPLO,  const int N,  float* A,  const int LDA,  
	     const float* VL,  const float* VU,  const int* IL,  const int* IU,
	     const float ABSTOL,  int * M,  float* W,  float* Z,  const int LDZ, float* WORK, 
	     const int LWORK, int* IWORK, int* IFAIL,
	     int* INFO) const;

  //! Epetra_LAPACK wrapper to compute all the eigenvalues, and optionally, the eigenvectors of a real generalized symmetric-definite eigenproblem, of the form A*x=(lambda)*B*x, A*Bx=(lambda)*x, or B*A*x=(lambda)*x
  void SYGV(const int ITYPE, const char JOBZ, const char UPLO, const int N, double* A, const int LDA, double* B, 
	    const int LDB, double* W, double* WORK, const int LWORK, int* INFO) const;
  //! Epetra_LAPACK wrapper to compute all the eigenvalues, and optionally, the eigenvectors of a real generalized symmetric-definite eigenproblem, of the form A*x=(lambda)*B*x, A*Bx=(lambda)*x, or B*A*x=(lambda)*x
  void SYGV(const int ITYPE, const char JOBZ, const char UPLO, const int N, float* A, const int LDA, float* B, 
	    const int LDB, float* W, float* WORK, const int LWORK, int* INFO) const;

  //! Epetra_LAPACK wrapper to compute selected eigenvalues, and optionally, eigenvectors of a  real generalized symmetric-definite eigenproblem, of the form A*x=(lambda)*B*x, A*Bx=(lambda)*x, or B*A*x=(lambda)*x
  void SYGVX(const int ITYPE, const char JOBZ, const char RANGE, const char UPLO, const int N, 
	     double* A, const int LDA, double* B, const int LDB, const double* VL, const double* VU,
	     const int* IL, const int* IU, const double ABSTOL, int* M, double* W, double* Z, 
	     const int LDZ,  double* WORK,  const int LWORK,  int* IWORK,
	     int* IFAIL, int* INFO) const;
  //! Epetra_LAPACK wrapper to compute selected eigenvalues, and optionally, eigenvectors of a  real generalized symmetric-definite eigenproblem, of the form A*x=(lambda)*B*x, A*Bx=(lambda)*x, or B*A*x=(lambda)*x
  void SYGVX(const int ITYPE, const char JOBZ, const char RANGE, const char UPLO, const int N, 
	     float* A, const int LDA, float* B, const int LDB, const float* VL, const float* VU,
	     const int* IL, const int* IU, const float ABSTOL, int* M, float* W, float* Z, 
	     const int LDZ,  float* WORK,  const int LWORK,  int* IWORK,
	     int* IFAIL, int* INFO) const;

  //! Epetra_LAPACK wrapper to compute selected eigenvalues and, optionally, eigenvectors of a real symmetric matrix T
  void SYEVR(const char JOBZ, const char RANGE, const char UPLO,  const int N,  double* A,  const int LDA,  const double* VL,  const double* VU,  const int *IL,  const int *IU,
                          const double ABSTOL,  int* M,  double* W,  double* Z, const int LDZ, int* ISUPPZ, double* WORK, const int LWORK, int* IWORK,
                          const int LIWORK, int* INFO) const;
  //! Epetra_LAPACK wrapper to compute selected eigenvalues and, optionally, eigenvectors of a real symmetric matrix T
  void SYEVR(const char JOBZ, const char RANGE, const char UPLO,  const int N,  float* A,  const int LDA,  
	     const float* VL,  const float* VU,  const int *IL,  const int *IU,
	     const float ABSTOL,  int* M,  float* W,  float* Z, const int LDZ, int* ISUPPZ, 
	     float* WORK, const int LWORK, int* IWORK,
	     const int LIWORK, int* INFO) const;

  //! Epetra_LAPACK wrapper to compute for an N-by-N real nonsymmetric matrix A, the eigenvalues and, optionally, the left and/or right eigenvectors
  void GEEVX(const char BALANC, const char JOBVL, const char JOBVR, const char SENSE, const int N, double* A, const int LDA, double* WR, double* WI,  double* VL,
	     const int LDVL,  double* VR,  const int LDVR,  int* ILO,  int* IHI,  double* SCALE, double* ABNRM, double* RCONDE,
	     double* RCONDV, double* WORK, const int LWORK, int* IWORK, int* INFO) const;
  //! Epetra_LAPACK wrapper to compute for an N-by-N real nonsymmetric matrix A, the eigenvalues and, optionally, the left and/or right eigenvectors
  void GEEVX(const char BALANC, const char JOBVL, const char JOBVR, const char SENSE, const int N, float* A, const int LDA, float* WR, float* WI,  float* VL,
	     const int LDVL,  float* VR,  const int LDVR,  int* ILO,  int* IHI,  float* SCALE, float* ABNRM, float* RCONDE,
	     float* RCONDV, float* WORK, const int LWORK, int* IWORK, int* INFO) const;

  //! Epetra_LAPACK wrapper to compute the singular value decomposition (SVD) of a real M-by-N matrix A, optionally computing the left and right singular vectors
  void GESDD(const char JOBZ, const int M, const int N, double* A, const int LDA,  double* S,  double* U,  const int LDU,  double* VT,  const int LDVT,  double* WORK,
	     const int LWORK, int* IWORK, int* INFO) const;
  //! Epetra_LAPACK wrapper to 
  void GESDD(const char JOBZ, const int M, const int N, float* A, const int LDA,  float* S,  float* U,  const int LDU,  float* VT,  const int LDVT,  float* WORK,
	     const int LWORK, int* IWORK, int* INFO) const;
  //! Epetra_LAPACK wrapper to compute for a pair of N-by-N real nonsymmetric matrices (A,B) the generalized eigenvalues, and optionally, the left and/or right generalized eigenvectors.

  void GGEV(const char JOBVL,  const char JOBVR,  const int N,  double* A,  const int LDA,  double* B, const int LDB, double* ALPHAR, double* ALPHAI,
	    double* BETA, double* VL, const int LDVL, double* VR, const int LDVR, double* WORK, const int LWORK, int* INFO) const;
  //! Epetra_LAPACK wrapper to compute for a pair of N-by-N real nonsymmetric matrices (A,B) the generalized eigenvalues, and optionally, the left and/or right generalized eigenvectors.
  void GGEV(const char JOBVL,  const char JOBVR,  const int N,  float* A,  const int LDA,  float* B, const int LDB, float* ALPHAR, float* ALPHAI,
	    float* BETA, float* VL, const int LDVL, float* VR, const int LDVR, float* WORK, const int LWORK, int* INFO) const;

    //@}

    //! @name Linear Least Squares
  //@{ 
  //! Epetra_LAPACK wrapper to solve the linear equality-constrained least squares (LSE) problem
  void GGLSE(const int M, const int N, const int P, double* A, const int LDA, double* B, const int LDB, 
	     double* C, double* D, double* X, double* WORK, const int LWORK, int* INFO) const;
  //! Epetra_LAPACK wrapper to solve the linear equality-constrained least squares (LSE) problem
  void GGLSE(const int M, const int N, const int P, float* A, const int LDA, float* B, const int LDB, 
	     float* C, float* D, float* X, float* WORK, const int LWORK, int* INFO) const;
    //@}

    //! @name Machine characteristics routines
  //@{ 
  //! Epetra_LAPACK wrapper for DLAMCH routine.  On out, T holds machine double precision floating point characteristics.  This information is returned by the Lapack routine.
  void LAMCH ( const char CMACH, float & T) const;
  //! Epetra_LAPACK wrapper for SLAMCH routine.  On out, T holds machine single precision floating point characteristics.  This information is returned by the Lapack routine.
  void LAMCH ( const char CMACH, double & T) const;
  //@}

};

// Epetra_LAPACK constructor
inline Epetra_LAPACK::Epetra_LAPACK(void){}
// Epetra_LAPACK constructor
inline Epetra_LAPACK::Epetra_LAPACK(const Epetra_LAPACK& LAPACK){(void)LAPACK;}
// Epetra_LAPACK destructor
inline Epetra_LAPACK::~Epetra_LAPACK(){}

#endif /* EPETRA_LAPACK_H */

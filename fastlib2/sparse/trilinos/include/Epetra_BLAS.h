
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

#ifndef EPETRA_BLAS_H
#define EPETRA_BLAS_H

#include "Epetra_Object.h"

//! Epetra_BLAS:  The Epetra BLAS Wrapper Class.
/*! The Epetra_BLAS class is a wrapper that encapsulates the BLAS
    (Basic Linear Algebra Subprograms).  The BLAS provide portable, high-
    performance implementations of kernels such as dense vectoer multiplication,
    dot products, dense matrix-vector multiplication and dense matrix-matrix
    multiplication.

    The standard BLAS interface is Fortran-specific.  Unfortunately, the 
    interface between C++ and Fortran is not standard across all computer
    platforms.  The Epetra_BLAS class provides C++ wrappers for the BLAS
    kernels in order to insulate the rest of Epetra from the details of C++ to Fortran
    translation.
    A Epetra_BLAS object is essentially nothing, but allows access to the BLAS wrapper
    functions.
  
    Epetra_BLAS is a serial interface only.  This is appropriate since the standard 
    BLAS are only specified for serial execution (or shared memory parallel).
*/


class Epetra_BLAS {
    
  public:
    //! @name Constructors/Destructor
  //@{ 
   //! Epetra_BLAS Constructor.
  /*! Builds an instance of a serial BLAS object.
   */
  Epetra_BLAS(void);


  //! Epetra_BLAS Copy Constructor.
  /*! Makes an exact copy of an existing Epetra_BLAS instance.
  */
  Epetra_BLAS(const Epetra_BLAS& BLAS);

  //! Epetra_BLAS Destructor.
  virtual ~Epetra_BLAS(void);
  //@}
  
  //! @name Level 1 BLAS
  //@{ 
  //! Epetra_BLAS one norm function (SASUM).
  float ASUM(const int N, const float * X, const int INCX = 1) const;
  //! Epetra_BLAS one norm function (DASUM).
  double ASUM(const int N, const double * X, const int INCX = 1) const;

  //! Epetra_BLAS dot product function (SDOT).
  float DOT(const int N, const float * X, const float * Y, const int INCX = 1, const int INCY = 1) const;
  //! Epetra_BLAS dot product function (DDOT).
  double DOT(const int N, const double * X, const double * Y, const int INCX = 1, const int INCY = 1) const;

  //! Epetra_BLAS norm function (SNRM2).
  float NRM2(const int N, const float * X, const int INCX = 1) const;
  //! Epetra_BLAS norm function (DNRM2).
  double NRM2(const int N, const double * X, const int INCX = 1) const;

  //! Epetra_BLAS vector scale function (SSCAL)
  void SCAL( const int N, const float ALPHA, float * X, const int INCX = 1) const;
  //! Epetra_BLAS vector scale function (DSCAL)
  void SCAL( const int N, const double ALPHA, double * X, const int INCX = 1) const;

  //! Epetra_BLAS vector copy function (SCOPY)
  void COPY( const int N, const float * X, float * Y, const int INCX = 1, const int INCY = 1) const;
  //! Epetra_BLAS vector scale function (DCOPY)
  void COPY( const int N, const double * X, double * Y, const int INCX = 1, const int INCY = 1) const;

  //! Epetra_BLAS arg maximum of absolute value function (ISAMAX)
  int IAMAX( const int N, const float * X, const int INCX = 1) const;
  //! Epetra_BLAS arg maximum of absolute value function (IDAMAX)
  int IAMAX( const int N, const double * X, const int INCX = 1) const;

  //! Epetra_BLAS vector update function (SAXPY)
  void AXPY( const int N, const float ALPHA, const float * X, float * Y, const int INCX = 1, const int INCY = 1) const;
  //! Epetra_BLAS vector update function (DAXPY)
  void AXPY( const int N, const double ALPHA, const double * X, double * Y, const int INCX = 1, const int INCY = 1) const;
  //@}

  //! @name Level 2 BLAS
  //@{ 
  //! Epetra_BLAS matrix-vector multiply function (SGEMV)
  void GEMV(const char TRANS, const int M, const int N,
         const float ALPHA, const float * A, const int LDA, const float * X,
         const float BETA, float * Y, const int INCX = 1, const int INCY = 1) const;
  //! Epetra_BLAS matrix-vector multiply function (DGEMV)
  void GEMV(const char TRANS, const int M, const int N,
         const double ALPHA, const double * A, const int LDA, const double * X,
         const double BETA, double * Y, const int INCX = 1, const int INCY = 1) const;
  //@}


  //! @name Level 3 BLAS
  //@{ 
  //! Epetra_BLAS matrix-matrix multiply function (SGEMM)
  void GEMM(const char TRANSA, const char TRANSB, const int M, const int N, const int K,
	    const float ALPHA, const float * A, const int LDA, const float * B,
	    const int LDB, const float BETA, float * C, const int LDC) const;
  //! Epetra_BLAS matrix-matrix multiply function (DGEMM)
  void GEMM(const char TRANSA, const char TRANSB, const int M, const int N, const int K,
	    const double ALPHA, const double * A, const int LDA, const double * B,
	    const int LDB, const double BETA, double * C, const int LDC) const;

  //! Epetra_BLAS symmetric matrix-matrix multiply function (SSYMM)
  void SYMM(const char SIDE, const char UPLO, const int M, const int N,
	    const float ALPHA, const float * A, const int LDA, const float * B,
	    const int LDB, const float BETA, float * C, const int LDC) const;
  //! Epetra_BLAS matrix-matrix multiply function (DSYMM)
  void SYMM(const char SIDE, const char UPLO, const int M, const int N,
	    const double ALPHA, const double * A, const int LDA, const double * B,
	    const int LDB, const double BETA, double * C, const int LDC) const;

  //! Epetra_BLAS triangular matrix-matrix multiply function (STRMM)
  void TRMM(const char SIDE, const char UPLO, const char TRANSA, const char DIAG, const int M, const int N,
	    const float ALPHA, const float * A, const int LDA, float * B,
	    const int LDB) const;
  //! Epetra_BLAS triangular matrix-matrix multiply function (DTRMM)
  void TRMM(const char SIDE, const char UPLO, const char TRANSA, const char DIAG, const int M, const int N,
	    const double ALPHA, const double * A, const int LDA, double * B,
	    const int LDB) const;
  //@}
};

// Epetra_BLAS constructor
inline Epetra_BLAS::Epetra_BLAS(void){}
// Epetra_BLAS constructor
inline Epetra_BLAS::Epetra_BLAS(const Epetra_BLAS& BLAS){(void)BLAS;}
// Epetra_BLAS destructor
inline Epetra_BLAS::~Epetra_BLAS(){}

#endif /* EPETRA_BLAS_H */

// @HEADER
// ***********************************************************************
// 
//                    Teuchos: Common Tools Package
//                 Copyright (2004) Sandia Corporation
// 
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
// 
// This library is free software; you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as
// published by the Free Software Foundation; either version 2.1 of the
// License, or (at your option) any later version.
//  
// This library is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//  
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
// USA
// Questions? Contact Michael A. Heroux (maherou@sandia.gov) 
// 
// ***********************************************************************
// @HEADER

#ifndef _TEUCHOS_LAPACK_HPP_
#define _TEUCHOS_LAPACK_HPP_

/*! \file Teuchos_LAPACK.hpp
    \brief Templated interface class to LAPACK routines.
*/
/** \example LAPACK/cxx_main.cpp
    This is an example of how to use the Teuchos::LAPACK class.
*/

/* for INTEL_CXML, the second arg may need to be changed to 'one'.  If so
the appropriate declaration of one will need to be added back into
functions that include the macro:
*/
#if defined (INTEL_CXML)
        unsigned int one=1;
#endif

#ifdef CHAR_MACRO
#undef CHAR_MACRO
#endif
#if defined (INTEL_CXML)
#define CHAR_MACRO(char_var) &char_var, one
#else
#define CHAR_MACRO(char_var) &char_var
#endif

#include "Teuchos_ConfigDefs.hpp"
#include "Teuchos_ScalarTraits.hpp"
#include "Teuchos_LAPACK_wrappers.hpp"

/*! \class Teuchos::LAPACK
    \brief The Templated LAPACK Wrapper Class.

    The Teuchos::LAPACK class is a wrapper that encapsulates LAPACK
    (Linear Algebra Package).  LAPACK provides portable, high-
    performance implementations of linear, eigen, SVD, etc solvers.

    The standard LAPACK interface is Fortran-specific.  Unfortunately, the
    interface between C++ and Fortran is not standard across all computer
    platforms.  The Teuchos::LAPACK class provides C++ wrappers for the LAPACK
    kernels in order to insulate the rest of Teuchos from the details of C++ to Fortran
    translation.  A Teuchos::LAPACK object is essentially nothing, but allows access to 
    the LAPACK wrapper functions.

    Teuchos::LAPACK is a serial interface only.  This is appropriate since the standard
    LAPACK are only specified for serial execution (or shared memory parallel).

    \note 
	<ol>
	    	<li>These templates are specialized to use the Fortran LAPACK routines for
		scalar types \c float and \c double.
    
		<li>If Teuchos is configured with \c --enable-teuchos-std::complex then these templates
		are specialized for scalar types \c std::complex<float> and \c std::complex<double> also.

		<li>A short description is given for each method.  For more detailed documentation, see the
		LAPACK website (\c http://www.netlib.org/lapack/ ).
	</ol>
*/

namespace Teuchos
{

  template<class T>
  struct UndefinedLAPACKRoutine
  {
    // This function should not compile if there is an attempt to instantiate!
    static inline T notDefined() { return T::LAPACK_routine_not_defined_for_this_type(); }
  };

  template<typename OrdinalType, typename ScalarType>
  class LAPACK
  {    
  public:

    typedef typename Teuchos::ScalarTraits<ScalarType>::magnitudeType MagnitudeType;

    //! @name Constructors/Destructors.
    //@{ 

    //! Default Constructor.
    inline LAPACK(void) {}

    //! Copy Constructor.
    inline LAPACK(const LAPACK<OrdinalType, ScalarType>& lapack) {}

    //! Destructor.
    inline virtual ~LAPACK(void) {}
    //@}

    //! @name Symmetric Positive Definite Linear System Routines.
    //@{ 

    //! Computes the \c L*D*L' factorization of a Hermitian/symmetric positive definite tridiagonal matrix \c A.
    void PTTRF(const OrdinalType n, ScalarType* d, ScalarType* e, OrdinalType* info) const;

    //! Solves a tridiagonal system \c A*X=B using the \L*D*L' factorization of \c A computed by PTTRF.
    void PTTRS(const OrdinalType n, const OrdinalType nrhs, const ScalarType* d, const ScalarType* e, ScalarType* B, const OrdinalType ldb, OrdinalType* info) const;

    //! Computes Cholesky factorization of a real symmetric positive definite matrix \c A.    
    void POTRF(const char UPLO, const OrdinalType n, ScalarType* A, const OrdinalType lda, OrdinalType* info) const;

    //! Solves a system of linear equations \c A*X=B, where \c A is a symmetric positive definite matrix factored by POTRF and the \c nrhs solutions are returned in \c B.
    void POTRS(const char UPLO, const OrdinalType n, const OrdinalType nrhs, const ScalarType* A, const OrdinalType lda, ScalarType* B, const OrdinalType ldb, OrdinalType* info) const;

    //! Computes the inverse of a real symmetric positive definite matrix \c A using the Cholesky factorization \c A from POTRF.
    void POTRI(const char UPLO, const OrdinalType n, ScalarType* A, const OrdinalType lda, OrdinalType* info) const;

    //! Estimates the reciprocal of the condition number (1-norm) of a real symmetric positive definite matrix \c A using the Cholesky factorization from POTRF.

    void POCON(const char UPLO, const OrdinalType n, const ScalarType* A, const OrdinalType lda, const ScalarType anorm, ScalarType* rcond, ScalarType* WORK, OrdinalType* IWORK, OrdinalType* info) const;

    //! Computes the solution to a real system of linear equations \c A*X=B, where \c A is a symmetric positive definite matrix and the \c nrhs solutions are returned in \c B. 
    void POSV(const char UPLO, const OrdinalType n, const OrdinalType nrhs, ScalarType* A, const OrdinalType lda, ScalarType* B, const OrdinalType ldb, OrdinalType* info) const;

    //! Computes row and column scalings intended to equilibrate a symmetric positive definite matrix \c A and reduce its condition number (w.r.t. 2-norm).
    void POEQU(const OrdinalType n, const ScalarType* A, const OrdinalType lda, ScalarType* S, ScalarType* scond, ScalarType* amax, OrdinalType* info) const;

    //! Improves the computed solution to a system of linear equations when the coefficient matrix is symmetric positive definite, and provides error bounds and backward error estimates for the solution.
    void PORFS(const char UPLO, const OrdinalType n, const OrdinalType nrhs, ScalarType* A, const OrdinalType lda, const ScalarType* AF, const OrdinalType ldaf, const ScalarType* B, const OrdinalType ldb, ScalarType* X, const OrdinalType ldx, ScalarType* FERR, ScalarType* BERR, ScalarType* WORK, OrdinalType* IWORK, OrdinalType* info) const;

    //! Uses the Cholesky factorization to compute the solution to a real system of linear equations \c A*X=B, where \c A is symmetric positive definite.  System can be equilibrated by POEQU and iteratively refined by PORFS, if requested.
    void POSVX(const char FACT, const char UPLO, const OrdinalType n, const OrdinalType nrhs, ScalarType* A, const OrdinalType lda, ScalarType* AF, const OrdinalType ldaf, char EQUED, ScalarType* S, ScalarType* B, const OrdinalType ldb, ScalarType* X, const OrdinalType ldx, ScalarType* rcond, ScalarType* FERR, ScalarType* BERR, ScalarType* WORK, OrdinalType* IWORK, OrdinalType* info) const;
    //@}

    //! @name General Linear System Routines.
    //@{ 

    //! Solves an over/underdetermined real \c m by \c n linear system \c A using QR or LQ factorization of A.
    void GELS(const char TRANS, const OrdinalType m, const OrdinalType n, const OrdinalType nrhs, ScalarType* A, const OrdinalType lda, ScalarType* B, const OrdinalType ldb, ScalarType* WORK, const OrdinalType lwork, OrdinalType* info) const;

    //! Computes a QR factorization of a general \c m by \c n matrix \c A.
    void GEQRF( const OrdinalType m, const OrdinalType n, ScalarType* A, const OrdinalType lda, ScalarType* TAU, ScalarType* WORK, const OrdinalType lwork, OrdinalType* info) const;

    //! Computes an LU factorization of a general \c m by \c n matrix \c A using partial pivoting with row interchanges.
    void GETRF(const OrdinalType m, const OrdinalType n, ScalarType* A, const OrdinalType lda, OrdinalType* IPIV, OrdinalType* info) const;

    //! Solves a system of linear equations \c A*X=B or \c A'*X=B with a general \c n by \c n matrix \c A using the LU factorization computed by GETRF.
    void GETRS(const char TRANS, const OrdinalType n, const OrdinalType nrhs, const ScalarType* A, const OrdinalType lda, const OrdinalType* IPIV, ScalarType* B, const OrdinalType ldb, OrdinalType* info) const;

    //! Computes an LU factorization of a \c n by \c n matrix tridiagonal matrix \c A using partial pivoting with row interchanges.
    void GTTRF(const OrdinalType n, ScalarType* dl, ScalarType* d, ScalarType* du, ScalarType* du2, OrdinalType* IPIV, OrdinalType* info) const;

    //! Solves a system of linear equations \c A*X=B or \c A'*X=B or \c A^H*X=B with a tridiagonal matrix \c A using the LU factorization computed by GTTRF.
    void GTTRS(const char TRANS, const OrdinalType n, const OrdinalType nrhs, const ScalarType* dl, const ScalarType* d, const ScalarType* du, const ScalarType* du2, const OrdinalType* IPIV, ScalarType* B, const OrdinalType ldb, OrdinalType* info) const;

    //! Computes the inverse of a matrix \c A using the LU factorization computed by GETRF.
    void GETRI(const OrdinalType n, ScalarType* A, const OrdinalType lda, const OrdinalType* IPIV, ScalarType* WORK, const OrdinalType lwork, OrdinalType* info) const;

    //! Estimates the reciprocal of the condition number of a general real matrix \c A, in either the 1-norm or the infinity-norm, using the LU factorization computed by GETRF.
    void GECON(const char NORM, const OrdinalType n, const ScalarType* A, const OrdinalType lda, const ScalarType anorm, ScalarType* rcond, ScalarType* WORK, OrdinalType* IWORK, OrdinalType* info) const;

    //! Computes the solution to a real system of linear equations \c A*X=B, where \c A is factored through GETRF and the \c nrhs solutions are computed through GETRS.
    void GESV(const OrdinalType n, const OrdinalType nrhs, ScalarType* A, const OrdinalType lda, OrdinalType* IPIV, ScalarType* B, const OrdinalType ldb, OrdinalType* info) const;

    //! Computes row and column scalings intended to equilibrate an \c m by \c n matrix \c A and reduce its condition number.
    void GEEQU(const OrdinalType m, const OrdinalType n, const ScalarType* A, const OrdinalType lda, ScalarType* R, ScalarType* C, ScalarType* rowcond, ScalarType* colcond, ScalarType* amax, OrdinalType* info) const;

    //! Improves the computed solution to a system of linear equations and provides error bounds and backward error estimates for the solution.  Use after GETRF/GETRS.
    void GERFS(const char TRANS, const OrdinalType n, const OrdinalType nrhs, const ScalarType* A, const OrdinalType lda, const ScalarType* AF, const OrdinalType ldaf, const OrdinalType* IPIV, const ScalarType* B, const OrdinalType ldb, ScalarType* X, const OrdinalType ldx, ScalarType* FERR, ScalarType* BERR, ScalarType* WORK, OrdinalType* IWORK, OrdinalType* info) const;

    //! Uses the LU factorization to compute the solution to a real system of linear equations \c A*X=B, returning error bounds on the solution and a condition estimate.
    void GESVX(const char FACT, const char TRANS, const OrdinalType n, const OrdinalType nrhs, ScalarType* A, const OrdinalType lda, ScalarType* AF, const OrdinalType ldaf, OrdinalType* IPIV, char EQUED, ScalarType* R, ScalarType* C, ScalarType* B, const OrdinalType ldb, ScalarType* X, const OrdinalType ldx, ScalarType* rcond, ScalarType* FERR, ScalarType* BERR, ScalarType* WORK, OrdinalType* IWORK, OrdinalType* info) const;

    /*! \brief Reduces a real symmetric matrix \c A to tridiagonal form by orthogonal similarity transformations.
        \note This method is not defined when the ScalarType is \c std::complex<float> or \c std::complex<double>.
    */
    void SYTRD(const char UPLO, const OrdinalType n, ScalarType* A, const OrdinalType lda, ScalarType* D, ScalarType* E, ScalarType* TAU, ScalarType* WORK, const OrdinalType lwork, OrdinalType* info) const;

    //! Reduces a real general matrix \c A to upper Hessenberg form by orthogonal similarity transformations.
    void GEHRD(const OrdinalType n, const OrdinalType ilo, const OrdinalType ihi, ScalarType* A, const OrdinalType lda, ScalarType* TAU, ScalarType* WORK, const OrdinalType lwork, OrdinalType* info) const;

    //! Solves a triangular linear system of the form \c A*X=B or \c A**T*X=B, where \c A is a triangular matrix.
    void TRTRS(const char UPLO, const char TRANS, const char DIAG, const OrdinalType n, const OrdinalType nrhs, const ScalarType* A, const OrdinalType lda, ScalarType* B, const OrdinalType ldb, OrdinalType* info) const; 
    
    //@}

    //! @name Symmetric Eigenproblem Routines
    //@{ 
    /*! \brief Computes the eigenvalues and, optionally, eigenvectors of a symmetric \c n by \c n matrix \c A in packed storage.
        \note This method is not defined when the ScalarType is \c std::complex<float> or \c std::complex<double>.
    */
    void SPEV(const char JOBZ, const char UPLO, const OrdinalType n, ScalarType* AP, ScalarType* W, ScalarType* Z, const OrdinalType ldz, ScalarType* WORK, OrdinalType* info) const;

    /*! \brief Computes all the eigenvalues and, optionally, eigenvectors of a symmetric \c n by \c n matrix A.
        \note This method is not defined when the ScalarType is \c std::complex<float> or \c std::complex<double>.
    */
    void SYEV(const char JOBZ, const char UPLO, const OrdinalType n, ScalarType* A, const OrdinalType lda, ScalarType* W, ScalarType* WORK, const OrdinalType lwork, OrdinalType* info) const;

    /*! \brief Computes all the eigenvalues and, optionally, eigenvectors of a symmetric \c n by \c n matrix pencil \c {A,B}, where \c A is symmetric and \c B is symmetric positive-definite.
        \note This method is not defined when the ScalarType is \c std::complex<float> or \c std::complex<double>.
    */
    void SYGV(const OrdinalType itype, const char JOBZ, const char UPLO, const OrdinalType n, ScalarType* A, const OrdinalType lda, ScalarType* B, const OrdinalType ldb, ScalarType* W, ScalarType* WORK, const OrdinalType lwork, OrdinalType* info) const;

    /*! \brief Computes all the eigenvalues and, optionally, eigenvectors of a Hermitian \c n by \c n matrix A.
        \note This method will call SYEV when ScalarType is \c float or \c double.
    */
    void HEEV(const char JOBZ, const char UPLO, const OrdinalType n, ScalarType* A, const OrdinalType lda, MagnitudeType* W, ScalarType* WORK, const OrdinalType lwork, MagnitudeType* RWORK, OrdinalType* info) const;

    /*! \brief Computes all the eigenvalues and, optionally, eigenvectors of a generalized Hermitian-definite \c n by \c n matrix pencil \c {A,B}, where \c A is Hermitian and \c B is Hermitian positive-definite.
        \note This method will call SYGV when ScalarType is \c float or \c double.
    */
    void HEGV(const OrdinalType itype, const char JOBZ, const char UPLO, const OrdinalType n, ScalarType* A, const OrdinalType lda, ScalarType* B, const OrdinalType ldb, MagnitudeType* W, ScalarType* WORK, const OrdinalType lwork, MagnitudeType *RWORK, OrdinalType* info) const;

    //! Computes the eigenvalues and, optionally, eigenvectors of a symmetric tridiagonal \c n by \c n matrix \c A using implicit QL/QR.  The eigenvectors can only be computed if \c A was reduced to tridiagonal form by SYTRD.
    void STEQR(const char COMPZ, const OrdinalType n, ScalarType* D, ScalarType* E, ScalarType* Z, const OrdinalType ldz, ScalarType* WORK, OrdinalType* info) const;
    //@}

    //! @name Non-Hermitian Eigenproblem Routines
    //@{ 
    //! Computes the eigenvalues of a real upper Hessenberg matrix \c H and, optionally, the matrices \c T and \c Z from the Schur decomposition, where T is an upper quasi-triangular matrix and Z contains the Schur vectors. 
    void HSEQR(const char JOB, const char COMPZ, const OrdinalType n, const OrdinalType ilo, const OrdinalType ihi, ScalarType* H, const OrdinalType ldh, ScalarType* WR, ScalarType* WI, ScalarType* Z, const OrdinalType ldz, ScalarType* WORK, const OrdinalType lwork, OrdinalType* info) const;
    
    /*! Computes for an \c n by \c n nonsymmetric matrix \c A, the eigenvalues, the Schur form \c T, and, optionally, the matrix of Schur vectors \c Z. When \c ScalarType is \c float or \c double, the real Schur form is computed.
       \note (This is the version used for \c float and \c double, where \c select requires two arguments to represent a std::complex eigenvalue.)
    */
    void GEES(const char JOBVS, const char SORT, OrdinalType (*ptr2func)(ScalarType*, ScalarType*), const OrdinalType n, ScalarType* A, const OrdinalType lda, OrdinalType* sdim, ScalarType* WR, ScalarType* WI, ScalarType* VS, const OrdinalType ldvs, ScalarType* WORK, const OrdinalType lwork, OrdinalType* BWORK, OrdinalType* info) const;    

    /*! Computes for an \c n by \c n nonsymmetric matrix \c A, the eigenvalues, the Schur form \c T, and, optionally, the matrix of Schur vectors \c Z. When \c ScalarType is \c float or \c double, the real Schur form is computed.
       \note (This is the version used for \c std::complex<float> and \c std::complex<double>, where \c select requires one arguments to represent a std::complex eigenvalue.)
    */
    void GEES(const char JOBVS, const char SORT, OrdinalType (*ptr2func)(ScalarType*), const OrdinalType n, ScalarType* A, const OrdinalType lda, OrdinalType* sdim, ScalarType* W, ScalarType* VS, const OrdinalType ldvs, ScalarType* WORK, const OrdinalType lwork, MagnitudeType* RWORK, OrdinalType* BWORK, OrdinalType* info) const;    

    /*! Computes for an \c n by \c n nonsymmetric matrix \c A, the eigenvalues, the Schur form \c T, and, optionally, the matrix of Schur vectors \c Z. When \c ScalarType is \c float or \c double, the real Schur form is computed.
       \note (This is the version used for any \c ScalarType, when the user doesn't want to enable the sorting functionality.)
    */
    void GEES(const char JOBVS, const OrdinalType n, ScalarType* A, const OrdinalType lda, OrdinalType* sdim, MagnitudeType* WR, MagnitudeType* WI, ScalarType* VS, const OrdinalType ldvs, ScalarType* WORK, const OrdinalType lwork, MagnitudeType* RWORK, OrdinalType* BWORK, OrdinalType* info) const;    

    //! Computes for an \c n by \c n real nonsymmetric matrix \c A, the eigenvalues and, optionally, the left and/or right eigenvectors.
    void GEEV(const char JOBVL, const char JOBVR, const OrdinalType n, ScalarType* A, const OrdinalType lda, ScalarType* WR, ScalarType* WI, ScalarType* VL, const OrdinalType ldvl, ScalarType* VR, const OrdinalType ldvr, ScalarType* WORK, const OrdinalType lwork, OrdinalType* info) const;

    //! Computes for a pair of \c n by \c n nonsymmetric matrices (\c A,\c B) the generalized eigenvalues, and optionally, the left and/or right generalized eigenvectors.
    void GGEVX(const char BALANC, const char JOBVL, const char JOBVR, const char SENSE, const OrdinalType n, ScalarType* A, const OrdinalType lda, ScalarType* B, const OrdinalType ldb, MagnitudeType* ALPHAR, MagnitudeType* ALPHAI, ScalarType* BETA, ScalarType* VL, const OrdinalType ldvl, ScalarType* VR, const OrdinalType ldvr, OrdinalType* ilo, OrdinalType* ihi, MagnitudeType* LSCALE, MagnitudeType* RSCALE, MagnitudeType* abnrm, MagnitudeType* bbnrm, MagnitudeType* RCONDE, MagnitudeType* RCONDV, ScalarType* WORK, const OrdinalType lwork, OrdinalType* IWORK, OrdinalType* BWORK, OrdinalType* info) const;
    //@}

    //! @name Orthogonal matrix routines
    //@{ 
    /*! \brief Overwrites the general real matrix \c m by \c n matrix \c C with the product of \c C and \c Q, which is the product of \c k elementary reflectors, as returned by GEQRF.
    \note This method is not defined when the ScalarType is \c std::complex<float> or \c std::complex<double>.
    */
    void ORMQR(const char SIDE, const char TRANS, const OrdinalType m, const OrdinalType n, const OrdinalType k, ScalarType* A, const OrdinalType lda, const ScalarType* TAU, ScalarType* C, const OrdinalType ldc, ScalarType* WORK, const OrdinalType lwork, OrdinalType* info) const;

    /*! \brief Generates an \c m by \c n matrix Q with orthonormal columns which is defined as the first \n columns of a product of \c k elementary reflectors of order \c m, as returned by GEQRF.
    \note This method is not defined when the ScalarType is \c std::complex<float> or \c std::complex<double>.
    */
    void ORGQR(const OrdinalType m, const OrdinalType n, const OrdinalType k, ScalarType* A, const OrdinalType lda, const ScalarType* TAU, ScalarType* WORK, const OrdinalType lwork, OrdinalType* info) const;

    /*! \brief Generates an \c m by \c n matrix Q with orthonormal columns which is defined as the first \n columns of a product of \c k elementary reflectors of order \c m, as returned by GEQRF.
    \note This method will call ORGQR when the ScalarType is \c float or \c double.
    */
    void UNGQR(const OrdinalType m, const OrdinalType n, const OrdinalType k, ScalarType* A, const OrdinalType lda, const ScalarType* TAU, ScalarType* WORK, const OrdinalType lwork, OrdinalType* info) const;

    /*! \brief Generates a real orthogonal matrix \c Q which is the product of \c ihi-ilo elementary reflectors of order \c n, as returned by GEHRD.  On return \c Q is stored in \c A.
    \note This method is not defined when the ScalarType is \c std::complex<float> or \c std::complex<double>.
    */
    void ORGHR(const OrdinalType n, const OrdinalType ilo, const OrdinalType ihi, ScalarType* A, const OrdinalType lda, const ScalarType* TAU, ScalarType* WORK, const OrdinalType lwork, OrdinalType* info) const;

    /*! \brief Overwrites the general real \c m by \c n matrix \c C with the product of \c C and \c Q, which is a product of \c ihi-ilo elementary reflectors, as returned by GEHRD.
    \note This method is not defined when the ScalarType is \c std::complex<float> or \c std::complex<double>. 
    */
    void ORMHR(const char SIDE, const char TRANS, const OrdinalType m, const OrdinalType n, const OrdinalType ilo, const OrdinalType ihi, const ScalarType* A, const OrdinalType lda, const ScalarType* TAU, ScalarType* C, const OrdinalType ldc, ScalarType* WORK, const OrdinalType lwork, OrdinalType* info) const;
    //@}

    //! @name Triangular Matrix Routines
    //@{ 

    /*! Computes some or all of the right and/or left eigenvectors of an upper triangular matrix \c T. If ScalarType is \c float or \c double, then the matrix is quasi-triangular and arugments \c RWORK is ignored.
       \note (This is the version used for \c float and \c double, where \c select requires two arguments to represent a std::complex eigenvalue.)
    */
    void TREVC(const char SIDE, const char HOWMNY, OrdinalType (*ptr2func)(ScalarType*, ScalarType*), const OrdinalType n, const ScalarType* T, const OrdinalType ldt, ScalarType* VL, const OrdinalType ldvl, ScalarType* VR, const OrdinalType ldvr, const OrdinalType mm, OrdinalType* m, ScalarType* WORK, OrdinalType* info) const;

    /*! Computes some or all of the right and/or left eigenvectors of an upper triangular matrix \c T. If ScalarType is \c float or \c double, then the matrix is quasi-triangular and arugments \c RWORK is ignored.
       \note (This is the version used for \c std::complex<float> and \c std::complex<double>, where \c select requires one arguments to represent a std::complex eigenvalue.)
    */
    void TREVC(const char SIDE, const char HOWMNY, OrdinalType (*ptr2func)(ScalarType*), const OrdinalType n, const ScalarType* T, const OrdinalType ldt, ScalarType* VL, const OrdinalType ldvl, ScalarType* VR, const OrdinalType ldvr, const OrdinalType mm, OrdinalType* m, ScalarType* WORK, MagnitudeType* RWORK, OrdinalType* info) const;

    /*! Computes some or all of the right and/or left eigenvectors of an upper triangular matrix \c T. If ScalarType is \c float or \c double, then the matrix is quasi-triangular and arugments \c RWORK is ignored.
       \note (This is the version used for any \c ScalarType, when the user doesn't want to enable the selecting functionality, with HOWMNY='A'.)
    */
    void TREVC(const char SIDE, const OrdinalType n, const ScalarType* T, const OrdinalType ldt, ScalarType* VL, const OrdinalType ldvl, ScalarType* VR, const OrdinalType ldvr, const OrdinalType mm, OrdinalType* m, ScalarType* WORK, MagnitudeType* RWORK, OrdinalType* info) const;

    /*! Reorders the Schur factorization of a matrix \c T via unitary similarity transformations so that the diagonal element of \c T with row index \c ifst is moved to row \c ilst. If \c ScalarType is \c float or \c double, then \c T should be in real Schur form and the operation affects the diagonal block referenced by \c ifst.
      \note This method will ignore the WORK std::vector when ScalarType is \c std::complex<float> or \c std::complex<double>.
    */
    void TREXC(const char COMPQ, const OrdinalType n, ScalarType* T, const OrdinalType ldt, ScalarType* Q, const OrdinalType ldq, OrdinalType ifst, OrdinalType ilst, ScalarType* WORK, OrdinalType* info) const;

    //@}

    //! @name Rotation/Reflection generators
    //@{ 

    //! Generates a plane rotation that zeros out the second component of the input std::vector.
    void LARTG( const ScalarType f, const ScalarType g, MagnitudeType* c, ScalarType* s, ScalarType* r ) const;

    //! Generates an elementary reflector of order \c n that zeros out the last \c n-1 components of the input std::vector.
    void LARFG( const OrdinalType n, ScalarType* alpha, ScalarType* x, const OrdinalType incx, ScalarType* tau ) const;

    //@}

    //! @name Random number generators
    //@{ 
    //! Returns a random number from a uniform or normal distribution.
    ScalarType LARND( const OrdinalType idist, OrdinalType* seed ) const;

    //! Returns a std::vector of random numbers from a chosen distribution.
    void LARNV( const OrdinalType idist, OrdinalType* seed, const OrdinalType n, ScalarType* v ) const;    
    //@}

    //! @name Machine Characteristics Routines.
    //@{ 
    /*! \brief Determines machine parameters for floating point characteristics.
        \note This method is not defined when the ScalarType is \c std::complex<float> or \c std::complex<double>. 
    */
    ScalarType LAMCH(const char CMACH) const;

    /*! \brief Chooses problem-dependent parameters for the local environment.
	\note This method should give parameters for good, but not optimal, performance on many currently 
	available computers.
    */
    OrdinalType ILAENV( const OrdinalType ispec, const std::string& NAME, const std::string& OPTS, const OrdinalType N1 = -1, const OrdinalType N2 = -1, const OrdinalType N3 = -1, const OrdinalType N4 = -1 ) const;
    //@}

    //! @name Miscellaneous Utilities.
    //@{ 
    /*! \brief Computes x^2 + y^2 safely, to avoid overflow.
        \note This method is not defined when the ScalarType is \c std::complex<float> or \c std::complex<double>. 
    */
    ScalarType LAPY2(const ScalarType x, const ScalarType y) const;
    //@}
  };

  // END GENERAL TEMPLATE DECLARATION //

  // BEGIN GENERAL TEMPLATE IMPLEMENTATION //


  template<typename OrdinalType, typename ScalarType>
  void LAPACK<OrdinalType, ScalarType>::PTTRF(const OrdinalType n, ScalarType* d, ScalarType* e, OrdinalType* info) const
  {
    UndefinedLAPACKRoutine<ScalarType>::notDefined();
  }
  
  template<typename OrdinalType, typename ScalarType>
  void LAPACK<OrdinalType, ScalarType>::PTTRS(const OrdinalType n, const OrdinalType nrhs, const ScalarType* d, const ScalarType* e, ScalarType* B, const OrdinalType ldb, OrdinalType* info) const
  {
    UndefinedLAPACKRoutine<ScalarType>::notDefined();
  }
  
  template<typename OrdinalType, typename ScalarType>
  void LAPACK<OrdinalType, ScalarType>::POTRF(const char UPLO, const OrdinalType n, ScalarType* A, const OrdinalType lda, OrdinalType* info) const
  {
    UndefinedLAPACKRoutine<ScalarType>::notDefined();
  }
  
  template<typename OrdinalType, typename ScalarType>
  void LAPACK<OrdinalType, ScalarType>::POTRS(const char UPLO, const OrdinalType n, const OrdinalType nrhs, const ScalarType* A, const OrdinalType lda, ScalarType* B, const OrdinalType ldb, OrdinalType* info) const
  {
    UndefinedLAPACKRoutine<ScalarType>::notDefined();
  }
  
  template<typename OrdinalType, typename ScalarType>
  void LAPACK<OrdinalType, ScalarType>::POTRI(const char UPLO, const OrdinalType n, ScalarType* A, const OrdinalType lda, OrdinalType* info) const
  {
    UndefinedLAPACKRoutine<ScalarType>::notDefined();
  }
  
  template<typename OrdinalType, typename ScalarType>
  void LAPACK<OrdinalType, ScalarType>::POCON(const char UPLO, const OrdinalType n, const ScalarType* A, const OrdinalType lda, const ScalarType anorm, ScalarType* rcond, ScalarType* WORK, OrdinalType* IWORK, OrdinalType* info) const
  {
    UndefinedLAPACKRoutine<ScalarType>::notDefined();
  }  

  template<typename OrdinalType, typename ScalarType>
  void LAPACK<OrdinalType, ScalarType>::POSV(const char UPLO, const OrdinalType n, const OrdinalType nrhs, ScalarType* A, const OrdinalType lda, ScalarType* B, const OrdinalType ldb, OrdinalType* info) const
  {
    UndefinedLAPACKRoutine<ScalarType>::notDefined();
  }
  
  template<typename OrdinalType, typename ScalarType>
  void LAPACK<OrdinalType, ScalarType>::POEQU(const OrdinalType n, const ScalarType* A, const OrdinalType lda, ScalarType* S, ScalarType* scond, ScalarType* amax, OrdinalType* info) const
  {
    UndefinedLAPACKRoutine<ScalarType>::notDefined();
  }
  
  template<typename OrdinalType, typename ScalarType>
  void LAPACK<OrdinalType, ScalarType>::PORFS(const char UPLO, const OrdinalType n, const OrdinalType nrhs, ScalarType* A, const OrdinalType lda, const ScalarType* AF, const OrdinalType ldaf, const ScalarType* B, const OrdinalType ldb, ScalarType* X, const OrdinalType ldx, ScalarType* FERR, ScalarType* BERR, ScalarType* WORK, OrdinalType* IWORK, OrdinalType* info) const
  {
    UndefinedLAPACKRoutine<ScalarType>::notDefined();
  }
  
  template<typename OrdinalType, typename ScalarType>
  void LAPACK<OrdinalType, ScalarType>::POSVX(const char FACT, const char UPLO, const OrdinalType n, const OrdinalType nrhs, ScalarType* A, const OrdinalType lda, ScalarType* AF, const OrdinalType ldaf, char EQUED, ScalarType* S, ScalarType* B, const OrdinalType ldb, ScalarType* X, const OrdinalType ldx, ScalarType* rcond, ScalarType* FERR, ScalarType* BERR, ScalarType* WORK, OrdinalType* IWORK, OrdinalType* info) const
  {
    UndefinedLAPACKRoutine<ScalarType>::notDefined();
  }

  template<typename OrdinalType, typename ScalarType>
  void LAPACK<OrdinalType,ScalarType>::GELS(const char TRANS, const OrdinalType m, const OrdinalType n, const OrdinalType nrhs, ScalarType* A, const OrdinalType lda, ScalarType* B, const OrdinalType ldb, ScalarType* WORK, const OrdinalType lwork, OrdinalType* info) const
  {
    UndefinedLAPACKRoutine<ScalarType>::notDefined();
  }

  template<typename OrdinalType, typename ScalarType>  
  void LAPACK<OrdinalType,ScalarType>::GEQRF( const OrdinalType m, const OrdinalType n, ScalarType* A, const OrdinalType lda, ScalarType* TAU, ScalarType* WORK, const OrdinalType lwork, OrdinalType* info) const
  {
    UndefinedLAPACKRoutine<ScalarType>::notDefined();
  }

  template<typename OrdinalType, typename ScalarType>
  void LAPACK<OrdinalType,ScalarType>::GETRF(const OrdinalType m, const OrdinalType n, ScalarType* A, const OrdinalType lda, OrdinalType* IPIV, OrdinalType* info) const
  {
    UndefinedLAPACKRoutine<ScalarType>::notDefined();
  }
  
  template<typename OrdinalType, typename ScalarType>
  void LAPACK<OrdinalType,ScalarType>::GETRS(const char TRANS, const OrdinalType n, const OrdinalType nrhs, const ScalarType* A, const OrdinalType lda, const OrdinalType* IPIV, ScalarType* B, const OrdinalType ldb, OrdinalType* info) const
  {
    UndefinedLAPACKRoutine<ScalarType>::notDefined();
  }

  template<typename OrdinalType, typename ScalarType>
  void LAPACK<OrdinalType,ScalarType>::GTTRF(const OrdinalType n, ScalarType* dl, ScalarType* d, ScalarType* du, ScalarType* du2, OrdinalType* IPIV, OrdinalType* info) const
  {
    UndefinedLAPACKRoutine<ScalarType>::notDefined();
  }

  template<typename OrdinalType, typename ScalarType>
  void LAPACK<OrdinalType,ScalarType>::GTTRS(const char TRANS, const OrdinalType n, const OrdinalType nrhs, const ScalarType* dl, const ScalarType* d, const ScalarType* du, const ScalarType* du2, const OrdinalType* IPIV, ScalarType* B, const OrdinalType ldb, OrdinalType* info) const
  {
    UndefinedLAPACKRoutine<ScalarType>::notDefined();
  }
  
  template<typename OrdinalType, typename ScalarType>
  void LAPACK<OrdinalType,ScalarType>::GETRI(const OrdinalType n, ScalarType* A, const OrdinalType lda, const OrdinalType* IPIV, ScalarType* WORK, const OrdinalType lwork, OrdinalType* info) const
  {
    UndefinedLAPACKRoutine<ScalarType>::notDefined();
  }
  
  template<typename OrdinalType, typename ScalarType>
  void LAPACK<OrdinalType,ScalarType>::GECON(const char NORM, const OrdinalType n, const ScalarType* A, const OrdinalType lda, const ScalarType anorm, ScalarType* rcond, ScalarType* WORK, OrdinalType* IWORK, OrdinalType* info) const
  {
    UndefinedLAPACKRoutine<ScalarType>::notDefined();
  }
  
  template<typename OrdinalType, typename ScalarType>
  void LAPACK<OrdinalType,ScalarType>::GESV(const OrdinalType n, const OrdinalType nrhs, ScalarType* A, const OrdinalType lda, OrdinalType* IPIV, ScalarType* B, const OrdinalType ldb, OrdinalType* info) const
  {
    UndefinedLAPACKRoutine<ScalarType>::notDefined();
  }
  
  template<typename OrdinalType, typename ScalarType>
  void LAPACK<OrdinalType,ScalarType>::GEEQU(const OrdinalType m, const OrdinalType n, const ScalarType* A, const OrdinalType lda, ScalarType* R, ScalarType* C, ScalarType* rowcond, ScalarType* colcond, ScalarType* amax, OrdinalType* info) const
  {
    UndefinedLAPACKRoutine<ScalarType>::notDefined();
  }
  
  template<typename OrdinalType, typename ScalarType>
  void LAPACK<OrdinalType,ScalarType>::GERFS(const char TRANS, const OrdinalType n, const OrdinalType nrhs, const ScalarType* A, const OrdinalType lda, const ScalarType* AF, const OrdinalType ldaf, const OrdinalType* IPIV, const ScalarType* B, const OrdinalType ldb, ScalarType* X, const OrdinalType ldx, ScalarType* FERR, ScalarType* BERR, ScalarType* WORK, OrdinalType* IWORK, OrdinalType* info) const
  {
    UndefinedLAPACKRoutine<ScalarType>::notDefined();
  }
  
  template<typename OrdinalType, typename ScalarType>
  void LAPACK<OrdinalType,ScalarType>::GESVX(const char FACT, const char TRANS, const OrdinalType n, const OrdinalType nrhs, ScalarType* A, const OrdinalType lda, ScalarType* AF, const OrdinalType ldaf, OrdinalType* IPIV, char EQUED, ScalarType* R, ScalarType* C, ScalarType* B, const OrdinalType ldb, ScalarType* X, const OrdinalType ldx, ScalarType* rcond, ScalarType* FERR, ScalarType* BERR, ScalarType* WORK, OrdinalType* IWORK, OrdinalType* info) const
  {
    UndefinedLAPACKRoutine<ScalarType>::notDefined();
  }
  
  template<typename OrdinalType, typename ScalarType>
  void LAPACK<OrdinalType,ScalarType>::SYTRD(const char UPLO, const OrdinalType n, ScalarType* A, const OrdinalType lda, ScalarType* D, ScalarType* E, ScalarType* TAU, ScalarType* WORK, const OrdinalType lwork, OrdinalType* info) const
  {
    UndefinedLAPACKRoutine<ScalarType>::notDefined();
  }

  template<typename OrdinalType, typename ScalarType>
  void LAPACK<OrdinalType,ScalarType>::GEHRD(const OrdinalType n, const OrdinalType ilo, const OrdinalType ihi, ScalarType* A, const OrdinalType lda, ScalarType* TAU, ScalarType* WORK, const OrdinalType lwork, OrdinalType* info) const
  {
    UndefinedLAPACKRoutine<ScalarType>::notDefined();
  }
  
  template<typename OrdinalType, typename ScalarType>
  void LAPACK<OrdinalType,ScalarType>::TRTRS(const char UPLO, const char TRANS, const char DIAG, const OrdinalType n, const OrdinalType nrhs, const ScalarType* A, const OrdinalType lda, ScalarType* B, const OrdinalType ldb, OrdinalType* info) const 
  {
    UndefinedLAPACKRoutine<ScalarType>::notDefined();
  }
  
  template<typename OrdinalType, typename ScalarType>
  void LAPACK<OrdinalType,ScalarType>::SPEV(const char JOBZ, const char UPLO, const OrdinalType n, ScalarType* AP, ScalarType* W, ScalarType* Z, const OrdinalType ldz, ScalarType* WORK, OrdinalType* info) const
  {
    UndefinedLAPACKRoutine<ScalarType>::notDefined();
  }

  template<typename OrdinalType, typename ScalarType>
  void LAPACK<OrdinalType,ScalarType>::SYEV(const char JOBZ, const char UPLO, const OrdinalType n, ScalarType* A, const OrdinalType lda, ScalarType* W, ScalarType* WORK, const OrdinalType lwork, OrdinalType* info) const
  {
    UndefinedLAPACKRoutine<ScalarType>::notDefined();
  }

  template<typename OrdinalType, typename ScalarType>
  void LAPACK<OrdinalType,ScalarType>::SYGV(const OrdinalType itype, const char JOBZ, const char UPLO, const OrdinalType n, ScalarType* A, const OrdinalType lda, ScalarType* B, const OrdinalType ldb, ScalarType* W, ScalarType* WORK, const OrdinalType lwork, OrdinalType* info) const
  {
    UndefinedLAPACKRoutine<ScalarType>::notDefined();
  }

  template<typename OrdinalType, typename ScalarType>
  void LAPACK<OrdinalType,ScalarType>::HEEV(const char JOBZ, const char UPLO, const OrdinalType n, ScalarType* A, const OrdinalType lda, MagnitudeType* W, ScalarType* WORK, const OrdinalType lwork, MagnitudeType* RWORK, OrdinalType* info) const
  {
    UndefinedLAPACKRoutine<ScalarType>::notDefined();
  }

  template<typename OrdinalType, typename ScalarType>
  void LAPACK<OrdinalType,ScalarType>::HEGV(const OrdinalType itype, const char JOBZ, const char UPLO, const OrdinalType n, ScalarType* A, const OrdinalType lda, ScalarType* B, const OrdinalType ldb, MagnitudeType* W, ScalarType* WORK, const OrdinalType lwork, MagnitudeType* RWORK, OrdinalType* info) const
  {
    UndefinedLAPACKRoutine<ScalarType>::notDefined();
  }

  template<typename OrdinalType, typename ScalarType>
  void LAPACK<OrdinalType,ScalarType>::STEQR(const char COMPZ, const OrdinalType n, ScalarType* D, ScalarType* E, ScalarType* Z, const OrdinalType ldz, ScalarType* WORK, OrdinalType* info) const
  {
    UndefinedLAPACKRoutine<ScalarType>::notDefined();
  }

  template<typename OrdinalType, typename ScalarType>
  void LAPACK<OrdinalType, ScalarType>::HSEQR(const char JOB, const char COMPZ, const OrdinalType n, const OrdinalType ilo, const OrdinalType ihi, ScalarType* H, const OrdinalType ldh, ScalarType* WR, ScalarType* WI, ScalarType* Z, const OrdinalType ldz, ScalarType* WORK, const OrdinalType lwork, OrdinalType* info) const
  {
    UndefinedLAPACKRoutine<ScalarType>::notDefined();
  }
  
  template<typename OrdinalType, typename ScalarType>
  void LAPACK<OrdinalType, ScalarType>::GEES(const char JOBVS, const char SORT, OrdinalType (*ptr2func)(ScalarType*, ScalarType*), const OrdinalType n, ScalarType* A, const OrdinalType lda, OrdinalType* sdim, ScalarType* WR, ScalarType* WI, ScalarType* VS, const OrdinalType ldvs, ScalarType* WORK, const OrdinalType lwork, OrdinalType* BWORK, OrdinalType* info) const    
  {
    UndefinedLAPACKRoutine<ScalarType>::notDefined();
  }

  template<typename OrdinalType, typename ScalarType>
  void LAPACK<OrdinalType, ScalarType>::GEES(const char JOBVS, const char SORT, OrdinalType (*ptr2func)(ScalarType*), const OrdinalType n, ScalarType* A, const OrdinalType lda, OrdinalType* sdim, ScalarType* W, ScalarType* VS, const OrdinalType ldvs, ScalarType* WORK, const OrdinalType lwork, MagnitudeType *RWORK, OrdinalType* BWORK, OrdinalType* info) const    
  {
    UndefinedLAPACKRoutine<ScalarType>::notDefined();
  }

  template<typename OrdinalType, typename ScalarType>
  void LAPACK<OrdinalType, ScalarType>::GEES(const char JOBVS, const OrdinalType n, ScalarType* A, const OrdinalType lda, OrdinalType* sdim, MagnitudeType* WR, MagnitudeType* WI, ScalarType* VS, const OrdinalType ldvs, ScalarType* WORK, const OrdinalType lwork, MagnitudeType *RWORK, OrdinalType* BWORK, OrdinalType* info) const    
  {
    UndefinedLAPACKRoutine<ScalarType>::notDefined();
  }

  template<typename OrdinalType, typename ScalarType>
  void LAPACK<OrdinalType, ScalarType>::GEEV(const char JOBVL, const char JOBVR, const OrdinalType n, ScalarType* A, const OrdinalType lda, ScalarType* WR, ScalarType* WI, ScalarType* VL, const OrdinalType ldvl, ScalarType* VR, const OrdinalType ldvr, ScalarType* WORK, const OrdinalType lwork, OrdinalType* info) const
  {
    UndefinedLAPACKRoutine<ScalarType>::notDefined();
  }

  template<typename OrdinalType, typename ScalarType>
  void LAPACK<OrdinalType, ScalarType>::GGEVX(const char BALANC, const char JOBVL, const char JOBVR, const char SENSE, const OrdinalType n, ScalarType* A, const OrdinalType lda, ScalarType* B, const OrdinalType ldb, MagnitudeType* ALPHAR, MagnitudeType* ALPHAI, ScalarType* BETA, ScalarType* VL, const OrdinalType ldvl, ScalarType* VR, const OrdinalType ldvr, OrdinalType* ilo, OrdinalType* ihi, MagnitudeType* LSCALE, MagnitudeType* RSCALE, MagnitudeType* abnrm, MagnitudeType* bbnrm, MagnitudeType* RCONDE, MagnitudeType* RCONDV, ScalarType* WORK, const OrdinalType lwork, OrdinalType* IWORK, OrdinalType* BWORK, OrdinalType* info) const
  {
    UndefinedLAPACKRoutine<ScalarType>::notDefined();
  }

  template<typename OrdinalType, typename ScalarType>
  void LAPACK<OrdinalType, ScalarType>::ORMQR(const char SIDE, const char TRANS, const OrdinalType m, const OrdinalType n, const OrdinalType k, ScalarType* A, const OrdinalType lda, const ScalarType* TAU, ScalarType* C, const OrdinalType ldc, ScalarType* WORK, const OrdinalType lwork, OrdinalType* info) const
  {
    UndefinedLAPACKRoutine<ScalarType>::notDefined();
  }

  template<typename OrdinalType, typename ScalarType>
  void LAPACK<OrdinalType, ScalarType>::ORGQR(const OrdinalType m, const OrdinalType n, const OrdinalType k, ScalarType* A, const OrdinalType lda, const ScalarType* TAU, ScalarType* WORK, const OrdinalType lwork, OrdinalType* info) const
  {
    UndefinedLAPACKRoutine<ScalarType>::notDefined();
  }
  
  template<typename OrdinalType, typename ScalarType>
  void LAPACK<OrdinalType, ScalarType>::UNGQR(const OrdinalType m, const OrdinalType n, const OrdinalType k, ScalarType* A, const OrdinalType lda, const ScalarType* TAU, ScalarType* WORK, const OrdinalType lwork, OrdinalType* info) const
  {
    UndefinedLAPACKRoutine<ScalarType>::notDefined();
  }
  
  template<typename OrdinalType, typename ScalarType>
  void LAPACK<OrdinalType, ScalarType>::ORGHR(const OrdinalType n, const OrdinalType ilo, const OrdinalType ihi, ScalarType* A, const OrdinalType lda, const ScalarType* TAU, ScalarType* WORK, const OrdinalType lwork, OrdinalType* info) const
  {
    UndefinedLAPACKRoutine<ScalarType>::notDefined();
  }
  
  template<typename OrdinalType, typename ScalarType>
  void LAPACK<OrdinalType, ScalarType>::ORMHR(const char SIDE, const char TRANS, const OrdinalType m, const OrdinalType n, const OrdinalType ilo, const OrdinalType ihi, const ScalarType* A, const OrdinalType lda, const ScalarType* TAU, ScalarType* C, const OrdinalType ldc, ScalarType* WORK, const OrdinalType lwork, OrdinalType* info) const
  {
    UndefinedLAPACKRoutine<ScalarType>::notDefined();
  }
  
  template<typename OrdinalType, typename ScalarType>
  void LAPACK<OrdinalType, ScalarType>::TREVC(const char SIDE, const char HOWMNY, OrdinalType (*ptr2func)(ScalarType*, ScalarType*), const OrdinalType n, const ScalarType* T, const OrdinalType ldt, ScalarType* VL, const OrdinalType ldvl, ScalarType* VR, const OrdinalType ldvr, const OrdinalType mm, OrdinalType* m, ScalarType* WORK, OrdinalType* info) const
  {
    UndefinedLAPACKRoutine<ScalarType>::notDefined();
  }
  
  template<typename OrdinalType, typename ScalarType>
  void LAPACK<OrdinalType, ScalarType>::TREVC(const char SIDE, const char HOWMNY, OrdinalType (*ptr2func)(ScalarType*), const OrdinalType n, const ScalarType* T, const OrdinalType ldt, ScalarType* VL, const OrdinalType ldvl, ScalarType* VR, const OrdinalType ldvr, const OrdinalType mm, OrdinalType* m, ScalarType* WORK, MagnitudeType* RWORK, OrdinalType* info) const
  {
    UndefinedLAPACKRoutine<ScalarType>::notDefined();
  }
  
  template<typename OrdinalType, typename ScalarType>
  void LAPACK<OrdinalType, ScalarType>::TREVC(const char SIDE, const OrdinalType n, const ScalarType* T, const OrdinalType ldt, ScalarType* VL, const OrdinalType ldvl, ScalarType* VR, const OrdinalType ldvr, const OrdinalType mm, OrdinalType* m, ScalarType* WORK, MagnitudeType* RWORK, OrdinalType* info) const
  {
    UndefinedLAPACKRoutine<ScalarType>::notDefined();
  }
  
  template<typename OrdinalType, typename ScalarType>
  void LAPACK<OrdinalType, ScalarType>::TREXC(const char COMPQ, const OrdinalType n, ScalarType* T, const OrdinalType ldt, ScalarType* Q, const OrdinalType ldq, OrdinalType ifst, OrdinalType ilst, ScalarType* WORK, OrdinalType* info) const
  {
    UndefinedLAPACKRoutine<ScalarType>::notDefined();
  }

  template<typename OrdinalType, typename ScalarType>
  ScalarType LAPACK<OrdinalType, ScalarType>::LAMCH(const char CMACH) const
  {
    return UndefinedLAPACKRoutine<ScalarType>::notDefined();
  }

  template<typename OrdinalType, typename ScalarType>
  OrdinalType LAPACK<OrdinalType, ScalarType>::ILAENV( const OrdinalType ispec, const std::string& NAME, const std::string& OPTS, const OrdinalType N1, const OrdinalType N2, const OrdinalType N3, const OrdinalType N4 ) const
  {
    return UndefinedLAPACKRoutine<ScalarType>::notDefined();
  }
 
  template<typename OrdinalType, typename ScalarType>
  ScalarType LAPACK<OrdinalType, ScalarType>::LAPY2(const ScalarType x, const ScalarType y) const
  {
    return UndefinedLAPACKRoutine<ScalarType>::notDefined();
  }

  template<typename OrdinalType, typename ScalarType>
  void LAPACK<OrdinalType, ScalarType>::LARTG( const ScalarType f, const ScalarType g, MagnitudeType* c, ScalarType* s, ScalarType* r ) const
  {
    return UndefinedLAPACKRoutine<ScalarType>::notDefined();
  }

  template<typename OrdinalType, typename ScalarType>
  void LAPACK<OrdinalType, ScalarType>::LARFG( const OrdinalType n, ScalarType* alpha, ScalarType* x, const OrdinalType incx, ScalarType* tau ) const
  {
    return UndefinedLAPACKRoutine<ScalarType>::notDefined();
  }

  template<typename OrdinalType, typename ScalarType>
  ScalarType LAPACK<OrdinalType, ScalarType>::LARND( const OrdinalType idist, OrdinalType* seed ) const
  {
    return UndefinedLAPACKRoutine<ScalarType>::notDefined();
  }

  template<typename OrdinalType, typename ScalarType>
  void LAPACK<OrdinalType, ScalarType>::LARNV( const OrdinalType idist, OrdinalType* seed, const OrdinalType n, ScalarType* v ) const    
  {
    return UndefinedLAPACKRoutine<ScalarType>::notDefined();
  }

  // END GENERAL TEMPLATE IMPLEMENTATION //

#ifndef DOXYGEN_SHOULD_SKIP_THIS

  // BEGIN FLOAT PARTIAL SPECIALIZATION DECLARATION //

#ifdef HAVE_TEUCHOS_BLASFLOAT

  template<typename OrdinalType>
  class LAPACK<OrdinalType, float>
  {    
  public:
    inline LAPACK(void) {}
    inline LAPACK(const LAPACK<OrdinalType, float>& lapack) {}
    inline virtual ~LAPACK(void) {}

    // Symmetric positive definite linear system routines
    void POTRF(const char UPLO, const OrdinalType n, float* A, const OrdinalType lda, OrdinalType* info) const;
    void POTRS(const char UPLO, const OrdinalType n, const OrdinalType nrhs, const float* A, const OrdinalType lda, float* B, const OrdinalType ldb, OrdinalType* info) const;
    void PTTRF(const OrdinalType n, float* d, float* e, OrdinalType* info) const;
    void PTTRS(const OrdinalType n, const OrdinalType nrhs, const float* d, const float* e, float* B, const OrdinalType ldb, OrdinalType* info) const;
    void POTRI(const char UPLO, const OrdinalType n, float* A, const OrdinalType lda, OrdinalType* info) const;
    void POCON(const char UPLO, const OrdinalType n, const float* A, const OrdinalType lda, const float anorm, float* rcond, float* WORK, OrdinalType* IWORK, OrdinalType* info) const;
    void POSV(const char UPLO, const OrdinalType n, const OrdinalType nrhs, float* A, const OrdinalType lda, float* B, const OrdinalType ldb, OrdinalType* info) const;
    void POEQU(const OrdinalType n, const float* A, const OrdinalType lda, float* S, float* scond, float* amax, OrdinalType* info) const;
    void PORFS(const char UPLO, const OrdinalType n, const OrdinalType nrhs, float* A, const OrdinalType lda, const float* AF, const OrdinalType ldaf, const float* B, const OrdinalType ldb, float* X, const OrdinalType ldx, float* FERR, float* BERR, float* WORK, OrdinalType* IWORK, OrdinalType* info) const;
    void POSVX(const char FACT, const char UPLO, const OrdinalType n, const OrdinalType nrhs, float* A, const OrdinalType lda, float* AF, const OrdinalType ldaf, char EQUED, float* S, float* B, const OrdinalType ldb, float* X, const OrdinalType ldx, float* rcond, float* FERR, float* BERR, float* WORK, OrdinalType* IWORK, OrdinalType* info) const; 

    // General Linear System Routines
    void GELS(const char TRANS, const OrdinalType m, const OrdinalType n, const OrdinalType nrhs, float* A, const OrdinalType lda, float* B, const OrdinalType ldb, float* WORK, const OrdinalType lwork, OrdinalType* info) const;
    void GEQRF( const OrdinalType m, const OrdinalType n, float* A, const OrdinalType lda, float* TAU, float* WORK, const OrdinalType lwork, OrdinalType* info) const;
    void GETRF(const OrdinalType m, const OrdinalType n, float* A, const OrdinalType lda, OrdinalType* IPIV, OrdinalType* info) const;
    void GETRS(const char TRANS, const OrdinalType n, const OrdinalType nrhs, const float* A, const OrdinalType lda, const OrdinalType* IPIV, float* B, const OrdinalType ldb, OrdinalType* info) const;
    void GTTRF(const OrdinalType n, float* dl, float* d, float* du, float* du2, OrdinalType* IPIV, OrdinalType* info) const;
    void GTTRS(const char TRANS, const OrdinalType n, const OrdinalType nrhs, const float* dl, const float* d, const float* du, const float* du2, const OrdinalType* IPIV, float* B, const OrdinalType ldb, OrdinalType* info) const;


    void GETRI(const OrdinalType n, float* A, const OrdinalType lda, const OrdinalType* IPIV, float* WORK, const OrdinalType lwork, OrdinalType* info) const;
    void GECON(const char NORM, const OrdinalType n, const float* A, const OrdinalType lda, const float anorm, float* rcond, float* WORK, OrdinalType* IWORK, OrdinalType* info) const;
    void GESV(const OrdinalType n, const OrdinalType nrhs, float* A, const OrdinalType lda, OrdinalType* IPIV, float* B, const OrdinalType ldb, OrdinalType* info) const;
    void GEEQU(const OrdinalType m, const OrdinalType n, const float* A, const OrdinalType lda, float* R, float* C, float* rowcond, float* colcond, float* amax, OrdinalType* info) const;
    void GERFS(const char TRANS, const OrdinalType n, const OrdinalType nrhs, const float* A, const OrdinalType lda, const float* AF, const OrdinalType ldaf, const OrdinalType* IPIV, const float* B, const OrdinalType ldb, float* X, const OrdinalType ldx, float* FERR, float* BERR, float* WORK, OrdinalType* IWORK, OrdinalType* info) const;
    void GESVX(const char FACT, const char TRANS, const OrdinalType n, const OrdinalType nrhs, float* A, const OrdinalType lda, float* AF, const OrdinalType ldaf, OrdinalType* IPIV, char EQUED, float* R, float* C, float* B, const OrdinalType ldb, float* X, const OrdinalType ldx, float* rcond, float* FERR, float* BERR, float* WORK, OrdinalType* IWORK, OrdinalType* info) const;
    void SYTRD(const char UPLO, const OrdinalType n, float* A, const OrdinalType lda, float* D, float* E, float* TAU, float* WORK, const OrdinalType lwork, OrdinalType* info) const;
    void GEHRD(const OrdinalType n, const OrdinalType ilo, const OrdinalType ihi, float* A, const OrdinalType lda, float* TAU, float* WORK, const OrdinalType lwork, OrdinalType* info) const;
    void TRTRS(const char UPLO, const char TRANS, const char DIAG, const OrdinalType n, const OrdinalType nrhs, const float* A, const OrdinalType lda, float* B, const OrdinalType ldb, OrdinalType* info) const; 

    // Symmetric eigenvalue routines.
    void SPEV(const char JOBZ, const char UPLO, const OrdinalType n, float* AP, float* W, float* Z, const OrdinalType ldz, float* WORK, OrdinalType* info) const;
    void SYEV(const char JOBZ, const char UPLO, const OrdinalType n, float* A, const OrdinalType lda, float* W, float* WORK, const OrdinalType lwork, OrdinalType* info) const;
    void SYGV(const OrdinalType itype, const char JOBZ, const char UPLO, const OrdinalType n, float* A, const OrdinalType lda, float* B, const OrdinalType ldb, float* W, float* WORK, const OrdinalType lwork, OrdinalType* info) const;
    void HEEV(const char JOBZ, const char UPLO, const OrdinalType n, float* A, const OrdinalType lda, float* W, float* WORK, const OrdinalType lwork, float* RWORK, OrdinalType* info) const;
    void HEGV(const OrdinalType itype, const char JOBZ, const char UPLO, const OrdinalType n, float* A, const OrdinalType lda, float* B, const OrdinalType ldb, float* W, float* WORK, const OrdinalType lwork, float *RWORK, OrdinalType* info) const;
    void STEQR(const char COMPZ, const OrdinalType n, float* D, float* E, float* Z, const OrdinalType ldz, float* WORK, OrdinalType* info) const;

    // Non-Hermitian eigenvalue routines.
    void HSEQR(const char JOB, const char COMPZ, const OrdinalType n, const OrdinalType ilo, const OrdinalType ihi, float* H, const OrdinalType ldh, float* WR, float* WI, float* Z, const OrdinalType ldz, float* WORK, const OrdinalType lwork, OrdinalType* info) const;
    void GEES(const char JOBVS, const char SORT, OrdinalType (*ptr2func)(float*, float*), const OrdinalType n, float* A, const OrdinalType lda, OrdinalType* sdim, float* WR, float* WI, float* VS, const OrdinalType ldvs, float* WORK, const OrdinalType lwork, OrdinalType* BWORK, OrdinalType* info) const;    
    void GEES(const char JOBVS, const OrdinalType n, float* A, const OrdinalType lda, OrdinalType* sdim, float* WR, float* WI, float* VS, const OrdinalType ldvs, float* WORK, const OrdinalType lwork, float* RWORK, OrdinalType* BWORK, OrdinalType* info) const;    
    void GEEV(const char JOBVL, const char JOBVR, const OrdinalType n, float* A, const OrdinalType lda, float* WR, float* WI, float* VL, const OrdinalType ldvl, float* VR, const OrdinalType ldvr, float* WORK, const OrdinalType lwork, OrdinalType* info) const;
    void GGEVX(const char BALANC, const char JOBVL, const char JOBVR, const char SENSE, const OrdinalType n, float* A, const OrdinalType lda, float* B, const OrdinalType ldb, float* ALPHAR, float* ALPHAI, float* BETA, float* VL, const OrdinalType ldvl, float* VR, const OrdinalType ldvr, OrdinalType* ilo, OrdinalType* ihi, float* LSCALE, float* RSCALE, float* abnrm, float* bbnrm, float* RCONDE, float* RCONDV, float* WORK, const OrdinalType lwork, OrdinalType* IWORK, OrdinalType* BWORK, OrdinalType* info) const;

    // Orthogonal matrix routines.
    void ORMQR(const char SIDE, const char TRANS, const OrdinalType m, const OrdinalType n, const OrdinalType k, float* A, const OrdinalType lda, const float* TAU, float* C, const OrdinalType ldc, float* WORK, const OrdinalType lwork, OrdinalType* info) const;
    void ORGQR(const OrdinalType m, const OrdinalType n, const OrdinalType k, float* A, const OrdinalType lda, const float* TAU, float* WORK, const OrdinalType lwork, OrdinalType* info) const;
    void UNGQR(const OrdinalType m, const OrdinalType n, const OrdinalType k, float* A, const OrdinalType lda, const float* TAU, float* WORK, const OrdinalType lwork, OrdinalType* info) const;
    void ORGHR(const OrdinalType n, const OrdinalType ilo, const OrdinalType ihi, float* A, const OrdinalType lda, const float* TAU, float* WORK, const OrdinalType lwork, OrdinalType* info) const;
    void ORMHR(const char SIDE, const char TRANS, const OrdinalType m, const OrdinalType n, const OrdinalType ilo, const OrdinalType ihi, const float* A, const OrdinalType lda, const float* TAU, float* C, const OrdinalType ldc, float* WORK, const OrdinalType lwork, OrdinalType* info) const;

    // Triangular matrix routines.
    void TREVC(const char SIDE, const char HOWMNY, OrdinalType (*ptr2func)(float*, float*), const OrdinalType n, const float* T, const OrdinalType ldt, float* VL, const OrdinalType ldvl, float* VR, const OrdinalType ldvr, const OrdinalType mm, OrdinalType* m, float* WORK, OrdinalType* info) const;
    void TREVC(const char SIDE, const OrdinalType n, const float* T, const OrdinalType ldt, float* VL, const OrdinalType ldvl, float* VR, const OrdinalType ldvr, const OrdinalType mm, OrdinalType* m, float* WORK, float *RWORK, OrdinalType* info) const;
    void TREXC(const char COMPQ, const OrdinalType n, float* T, const OrdinalType ldt, float* Q, const OrdinalType ldq, OrdinalType ifst, OrdinalType ilst, float* WORK, OrdinalType* info) const;

    // Rotation/reflection generators
    void LARTG( const float f, const float g, float* c, float* s, float* r ) const;
    void LARFG( const OrdinalType n, float* alpha, float* x, const OrdinalType incx, float* tau ) const;

    // Random number generators
    float LARND( const OrdinalType idist, OrdinalType* seed ) const;
    void LARNV( const OrdinalType idist, OrdinalType* seed, const OrdinalType n, float* v ) const;    

    // Machine characteristics.
    float LAMCH(const char CMACH) const;
    OrdinalType ILAENV( const OrdinalType ispec, const std::string& NAME, const std::string& OPTS, const OrdinalType N1 = -1, const OrdinalType N2 = -1, const OrdinalType N3 = -1, const OrdinalType N4 = -1 ) const;

    // Miscellaneous routines.
    float LAPY2(const float x, const float y) const;

  };

  // END FLOAT PARTIAL SPECIALIZATION DECLARATION //

  // BEGIN FLOAT PARTIAL SPECIALIZATION IMPLEMENTATION //

  template<typename OrdinalType>
  void LAPACK<OrdinalType, float>::PTTRF(const OrdinalType n, float* d, float* e, OrdinalType* info) const
  {
    SPTTRF_F77(&n,d,e,info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType, float>::PTTRS(const OrdinalType n, const OrdinalType nrhs, const float* d, const float* e, float* B, const OrdinalType ldb, OrdinalType* info) const
  {
    SPTTRS_F77(&n,&nrhs,d,e,B,&ldb,info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType, float>::POTRF(const char UPLO, const OrdinalType n, float* A, const OrdinalType lda, OrdinalType* info) const
  {
    SPOTRF_F77(CHAR_MACRO(UPLO), &n, A, &lda, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType, float>::POTRS(const char UPLO, const OrdinalType n, const OrdinalType nrhs, const float* A, const OrdinalType lda, float* B, const OrdinalType ldb, OrdinalType* info) const
  {
    SPOTRS_F77(CHAR_MACRO(UPLO), &n, &nrhs, A, &lda, B, &ldb, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType, float>::POTRI(const char UPLO, const OrdinalType n, float* A, const OrdinalType lda, OrdinalType* info) const
  {
    SPOTRI_F77(CHAR_MACRO(UPLO), &n, A, &lda, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType, float>::POCON(const char UPLO, const OrdinalType n, const float* A, const OrdinalType lda, const float anorm, float* rcond, float* WORK, OrdinalType* IWORK, OrdinalType* info) const
  {
    SPOCON_F77(CHAR_MACRO(UPLO), &n, A, &lda, &anorm, rcond, WORK, IWORK, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType, float>::POSV(const char UPLO, const OrdinalType n, const OrdinalType nrhs, float* A, const OrdinalType lda, float* B, const OrdinalType ldb, OrdinalType* info) const
  {
    SPOSV_F77(CHAR_MACRO(UPLO), &n, &nrhs, A, &lda, B, &ldb, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType, float>::POEQU(const OrdinalType n, const float* A, const OrdinalType lda, float* S, float* scond, float* amax, OrdinalType* info) const
  {
    SPOEQU_F77(&n, A, &lda, S, scond, amax, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType, float>::PORFS(const char UPLO, const OrdinalType n, const OrdinalType nrhs, float* A, const OrdinalType lda, const float* AF, const OrdinalType ldaf, const float* B, const OrdinalType ldb, float* X, const OrdinalType ldx, float* FERR, float* BERR, float* WORK, OrdinalType* IWORK, OrdinalType* info) const
  {
    SPORFS_F77(CHAR_MACRO(UPLO), &n, &nrhs, A, &lda, AF, &ldaf, B, &ldb, X, &ldx, FERR, BERR, WORK, IWORK, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType, float>::POSVX(const char FACT, const char UPLO, const OrdinalType n, const OrdinalType nrhs, float* A, const OrdinalType lda, float* AF, const OrdinalType ldaf, char EQUED, float* S, float* B, const OrdinalType ldb, float* X, const OrdinalType ldx, float* rcond, float* FERR, float* BERR, float* WORK, OrdinalType* IWORK, OrdinalType* info) const
  {
    SPOSVX_F77(CHAR_MACRO(FACT), CHAR_MACRO(UPLO), &n, &nrhs, A, &lda, AF, &ldaf, CHAR_MACRO(EQUED), S, B, &ldb, X, &ldx, rcond, FERR, BERR, WORK, IWORK, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType,float>::GELS(const char TRANS, const OrdinalType m, const OrdinalType n, const OrdinalType nrhs, float* A, const OrdinalType lda, float* B, const OrdinalType ldb, float* WORK, const OrdinalType lwork, OrdinalType* info) const
  {
    SGELS_F77(CHAR_MACRO(TRANS), &m, &n, &nrhs, A, &lda, B, &ldb, WORK, &lwork, info);
  }
  
  template<typename OrdinalType>  
  void LAPACK<OrdinalType,float>::GEQRF( const OrdinalType m, const OrdinalType n, float* A, const OrdinalType lda, float* TAU, float* WORK, const OrdinalType lwork, OrdinalType* info) const
  {
    SGEQRF_F77(&m, &n, A, &lda, TAU, WORK, &lwork, info);
  }

  template<typename OrdinalType>
  void LAPACK<OrdinalType,float>::GETRF(const OrdinalType m, const OrdinalType n, float* A, const OrdinalType lda, OrdinalType* IPIV, OrdinalType* info) const
  {
    SGETRF_F77(&m, &n, A, &lda, IPIV, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType,float>::GETRS(const char TRANS, const OrdinalType n, const OrdinalType nrhs, const float* A, const OrdinalType lda, const OrdinalType* IPIV, float* B, const OrdinalType ldb, OrdinalType* info) const
  {
    SGETRS_F77(CHAR_MACRO(TRANS), &n, &nrhs, A, &lda, IPIV, B, &ldb, info);
  }

  template<typename OrdinalType>
  void LAPACK<OrdinalType,float>::GTTRF(const OrdinalType n, float* dl, float* d, float* du, float* du2, OrdinalType* IPIV, OrdinalType* info) const
  {
    SGTTRF_F77(&n, dl, d, du, du2, IPIV, info);
  }

  template<typename OrdinalType>
  void LAPACK<OrdinalType,float>::GTTRS(const char TRANS, const OrdinalType n, const OrdinalType nrhs, const float* dl, const float* d, const float* du, const float* du2, const OrdinalType* IPIV, float* B, const OrdinalType ldb, OrdinalType* info) const
  {
    SGTTRS_F77(CHAR_MACRO(TRANS), &n, &nrhs, dl, d, du, du2, IPIV, B, &ldb, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType,float>::GETRI(const OrdinalType n, float* A, const OrdinalType lda, const OrdinalType* IPIV, float* WORK, const OrdinalType lwork, OrdinalType* info) const
  {
    SGETRI_F77(&n, A, &lda, IPIV, WORK, &lwork, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType,float>::GECON(const char NORM, const OrdinalType n, const float* A, const OrdinalType lda, const float anorm, float* rcond, float* WORK, OrdinalType* IWORK, OrdinalType* info) const
  {
    SGECON_F77(CHAR_MACRO(NORM), &n, A, &lda, &anorm, rcond, WORK, IWORK, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType,float>::GESV(const OrdinalType n, const OrdinalType nrhs, float* A, const OrdinalType lda, OrdinalType* IPIV, float* B, const OrdinalType ldb, OrdinalType* info) const
  {
    SGESV_F77(&n, &nrhs, A, &lda, IPIV, B, &ldb, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType,float>::GEEQU(const OrdinalType m, const OrdinalType n, const float* A, const OrdinalType lda, float* R, float* C, float* rowcond, float* colcond, float* amax, OrdinalType* info) const
  {
    SGEEQU_F77(&m, &n, A, &lda, R, C, rowcond, colcond, amax, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType,float>::GERFS(const char TRANS, const OrdinalType n, const OrdinalType nrhs, const float* A, const OrdinalType lda, const float* AF, const OrdinalType ldaf, const OrdinalType* IPIV, const float* B, const OrdinalType ldb, float* X, const OrdinalType ldx, float* FERR, float* BERR, float* WORK, OrdinalType* IWORK, OrdinalType* info) const
  {
    SGERFS_F77(CHAR_MACRO(TRANS), &n, &nrhs, A, &lda, AF, &ldaf, IPIV, B, &ldb, X, &ldx, FERR, BERR, WORK, IWORK, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType,float>::GESVX(const char FACT, const char TRANS, const OrdinalType n, const OrdinalType nrhs, float* A, const OrdinalType lda, float* AF, const OrdinalType ldaf, OrdinalType* IPIV, char EQUED, float* R, float* C, float* B, const OrdinalType ldb, float* X, const OrdinalType ldx, float* rcond, float* FERR, float* BERR, float* WORK, OrdinalType* IWORK, OrdinalType* info) const
  {
    SGESVX_F77(CHAR_MACRO(FACT), CHAR_MACRO(TRANS), &n, &nrhs, A, &lda, AF, &ldaf, IPIV, CHAR_MACRO(EQUED), R, C, B, &ldb, X, &ldx, rcond, FERR, BERR, WORK, IWORK, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType,float>::SYTRD(const char UPLO, const OrdinalType n, float* A, const OrdinalType lda, float* D, float* E, float* TAU, float* WORK, const OrdinalType lwork, OrdinalType* info) const
  {
    SSYTRD_F77(CHAR_MACRO(UPLO), &n, A, &lda, D, E, TAU, WORK, &lwork, info);
  }

  template<typename OrdinalType>
  void LAPACK<OrdinalType,float>::GEHRD(const OrdinalType n, const OrdinalType ilo, const OrdinalType ihi, float* A, const OrdinalType lda, float* TAU, float* WORK, const OrdinalType lwork, OrdinalType* info) const
  {
    SGEHRD_F77(&n, &ilo, &ihi, A, &lda, TAU, WORK, &lwork, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType,float>::TRTRS(const char UPLO, const char TRANS, const char DIAG, const OrdinalType n, const OrdinalType nrhs, const float* A, const OrdinalType lda, float* B, const OrdinalType ldb, OrdinalType* info) const 
  {
    STRTRS_F77(CHAR_MACRO(UPLO), CHAR_MACRO(TRANS), CHAR_MACRO(DIAG), &n, &nrhs, A, &lda, B, &ldb, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType,float>::SPEV(const char JOBZ, const char UPLO, const OrdinalType n, float* AP, float* W, float* Z, const OrdinalType ldz, float* WORK, OrdinalType* info) const
  {
    SSPEV_F77(CHAR_MACRO(JOBZ), CHAR_MACRO(UPLO), &n, AP, W, Z, &ldz, WORK, info);
  }

  template<typename OrdinalType>
  void LAPACK<OrdinalType,float>::SYEV(const char JOBZ, const char UPLO, const OrdinalType n, float* A, const OrdinalType lda, float* W, float* WORK, const OrdinalType lwork, OrdinalType* info) const
  {
    SSYEV_F77(CHAR_MACRO(JOBZ), CHAR_MACRO(UPLO), &n, A, &lda, W, WORK, &lwork, info);
  }

  template<typename OrdinalType>
  void LAPACK<OrdinalType,float>::SYGV(const OrdinalType itype, const char JOBZ, const char UPLO, const OrdinalType n, float* A, const OrdinalType lda, float* B, const OrdinalType ldb, float* W, float* WORK, const OrdinalType lwork, OrdinalType* info) const
  {
    SSYGV_F77(&itype, CHAR_MACRO(JOBZ), CHAR_MACRO(UPLO), &n, A, &lda, B, &ldb, W, WORK, &lwork, info);
  }

  template<typename OrdinalType>
  void LAPACK<OrdinalType,float>::HEEV(const char JOBZ, const char UPLO, const OrdinalType n, float* A, const OrdinalType lda, float* W, float* WORK, const OrdinalType lwork, float* RWORK, OrdinalType* info) const
  {
    SSYEV_F77(CHAR_MACRO(JOBZ), CHAR_MACRO(UPLO), &n, A, &lda, W, WORK, &lwork, info);
  }

  template<typename OrdinalType>
  void LAPACK<OrdinalType,float>::HEGV(const OrdinalType itype, const char JOBZ, const char UPLO, const OrdinalType n, float* A, const OrdinalType lda, float* B, const OrdinalType ldb, float* W, float* WORK, const OrdinalType lwork, float *RWORK, OrdinalType* info) const
  {
    SSYGV_F77(&itype, CHAR_MACRO(JOBZ), CHAR_MACRO(UPLO), &n, A, &lda, B, &ldb, W, WORK, &lwork, info);
  }

  template<typename OrdinalType>
  void LAPACK<OrdinalType,float>::STEQR(const char COMPZ, const OrdinalType n, float* D, float* E, float* Z, const OrdinalType ldz, float* WORK, OrdinalType* info) const
  {
    SSTEQR_F77(CHAR_MACRO(COMPZ), &n, D, E, Z, &ldz, WORK, info);
  }

  template<typename OrdinalType>
  void LAPACK<OrdinalType, float>::HSEQR(const char JOB, const char COMPZ, const OrdinalType n, const OrdinalType ilo, const OrdinalType ihi, float* H, const OrdinalType ldh, float* WR, float* WI, float* Z, const OrdinalType ldz, float* WORK, const OrdinalType lwork, OrdinalType* info) const
  {
    SHSEQR_F77(CHAR_MACRO(JOB), CHAR_MACRO(COMPZ), &n, &ilo, &ihi, H, &ldh, WR, WI, Z, &ldz, WORK, &lwork, info);
  }

  template<typename OrdinalType>
  void LAPACK<OrdinalType, float>::GEES(const char JOBVS, const char SORT, OrdinalType (*ptr2func)(float*, float*), const OrdinalType n, float* A, const OrdinalType lda, OrdinalType* sdim, float* WR, float* WI, float* VS, const OrdinalType ldvs, float* WORK, const OrdinalType lwork, OrdinalType* BWORK, OrdinalType* info) const    
  {
    SGEES_F77(CHAR_MACRO(JOBVS), CHAR_MACRO(SORT), ptr2func, &n, A, &lda, sdim, WR, WI, VS, &ldvs, WORK, &lwork, BWORK, info);
  }

  template<typename OrdinalType>
  void LAPACK<OrdinalType, float>::GEES(const char JOBVS, const OrdinalType n, float* A, const OrdinalType lda, OrdinalType* sdim, float* WR, float* WI, float* VS, const OrdinalType ldvs, float* WORK, const OrdinalType lwork, float* RWORK, OrdinalType* BWORK, OrdinalType* info) const    
  {
    OrdinalType (*nullfptr)(float*,float*) = NULL;
    const char sort = 'N';
    SGEES_F77(CHAR_MACRO(JOBVS), CHAR_MACRO(sort), nullfptr, &n, A, &lda, sdim, WR, WI, VS, &ldvs, WORK, &lwork, BWORK, info);
  }

  template<typename OrdinalType>
  void LAPACK<OrdinalType, float>::GEEV(const char JOBVL, const char JOBVR, const OrdinalType n, float* A, const OrdinalType lda, float* WR, float* WI, float* VL, const OrdinalType ldvl, float* VR, const OrdinalType ldvr, float* WORK, const OrdinalType lwork, OrdinalType* info) const
  {
    SGEEV_F77(CHAR_MACRO(JOBVL), CHAR_MACRO(JOBVR), &n, A, &lda, WR, WI, VL, &ldvl, VR, &ldvr, WORK, &lwork, info);
  }

  template<typename OrdinalType>
  void LAPACK<OrdinalType,float>::GGEVX(const char BALANC, const char JOBVL, const char JOBVR, const char SENSE, const OrdinalType n, float* A, const OrdinalType lda, float* B, const OrdinalType ldb, float* ALPHAR, float* ALPHAI, float* BETA, float* VL, const OrdinalType ldvl, float* VR, const OrdinalType ldvr, OrdinalType* ilo, OrdinalType* ihi, float* LSCALE, float* RSCALE, float* abnrm, float* bbnrm, float* RCONDE, float* RCONDV, float* WORK, const OrdinalType lwork, OrdinalType* IWORK, OrdinalType* BWORK, OrdinalType* info) const
  {
    SGGEVX_F77(CHAR_MACRO(BALANC), CHAR_MACRO(JOBVL), CHAR_MACRO(JOBVR), CHAR_MACRO(SENSE), &n, A, &lda, B, &ldb, ALPHAR, ALPHAI, BETA, VL, &ldvl, VR, &ldvr, ilo, ihi, LSCALE, RSCALE, abnrm, bbnrm, RCONDE, RCONDV, WORK, &lwork, IWORK, BWORK, info);
  }

  template<typename OrdinalType>
  void LAPACK<OrdinalType, float>::ORMQR(const char SIDE, const char TRANS, const OrdinalType m, const OrdinalType n, const OrdinalType k, float* A, const OrdinalType lda, const float* TAU, float* C, const OrdinalType ldc, float* WORK, const OrdinalType lwork, OrdinalType* info) const
  {
    SORMQR_F77(CHAR_MACRO(SIDE), CHAR_MACRO(TRANS), &m, &n, &k, A, &lda, TAU, C, &ldc, WORK, &lwork, info);
  }

  template<typename OrdinalType>
  void LAPACK<OrdinalType, float>::ORGQR(const OrdinalType m, const OrdinalType n, const OrdinalType k, float* A, const OrdinalType lda, const float* TAU, float* WORK, const OrdinalType lwork, OrdinalType* info) const
  {
    SORGQR_F77( &m, &n, &k, A, &lda, TAU, WORK, &lwork, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType, float>::UNGQR(const OrdinalType m, const OrdinalType n, const OrdinalType k, float* A, const OrdinalType lda, const float* TAU, float* WORK, const OrdinalType lwork, OrdinalType* info) const
  {
    SORGQR_F77( &m, &n, &k, A, &lda, TAU, WORK, &lwork, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType, float>::ORGHR(const OrdinalType n, const OrdinalType ilo, const OrdinalType ihi, float* A, const OrdinalType lda, const float* TAU, float* WORK, const OrdinalType lwork, OrdinalType* info) const
  {
    SORGHR_F77(&n, &ilo, &ihi, A, &lda, TAU, WORK, &lwork, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType, float>::ORMHR(const char SIDE, const char TRANS, const OrdinalType m, const OrdinalType n, const OrdinalType ilo, const OrdinalType ihi, const float* A, const OrdinalType lda, const float* TAU, float* C, const OrdinalType ldc, float* WORK, const OrdinalType lwork, OrdinalType* info) const
  {
    SORMHR_F77(CHAR_MACRO(SIDE), CHAR_MACRO(TRANS), &m, &n, &ilo, &ihi, A, &lda, TAU, C, &ldc, WORK, &lwork, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType, float>::TREVC(const char SIDE, const char HOWMNY, OrdinalType (*ptr2func)(float*,float*), const OrdinalType n, const float* T, const OrdinalType ldt, float* VL, const OrdinalType ldvl, float* VR, const OrdinalType ldvr, const OrdinalType mm, OrdinalType* m, float* WORK, OrdinalType* info) const
  {
    STREVC_F77(CHAR_MACRO(SIDE), CHAR_MACRO(HOWMNY), ptr2func, &n, T, &ldt, VL, &ldvl, VR, &ldvr, &mm, m, WORK, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType, float>::TREVC(const char SIDE, const OrdinalType n, const float* T, const OrdinalType ldt, float* VL, const OrdinalType ldvl, float* VR, const OrdinalType ldvr, const OrdinalType mm, OrdinalType* m, float* WORK, float* RWORK, OrdinalType* info) const
  {
    OrdinalType (*nullfptr)(float*,float*) = NULL;
    const char whch = 'A';
    STREVC_F77(CHAR_MACRO(SIDE), CHAR_MACRO(whch), nullfptr, &n, T, &ldt, VL, &ldvl, VR, &ldvr, &mm, m, WORK, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType, float>::TREXC(const char COMPQ, const OrdinalType n, float* T, const OrdinalType ldt, float* Q, const OrdinalType ldq, OrdinalType ifst, OrdinalType ilst, float* WORK, OrdinalType* info) const
  {
    STREXC_F77(CHAR_MACRO(COMPQ), &n, T, &ldt, Q, &ldq, &ifst, &ilst, WORK, info);
  }

  template<typename OrdinalType>
  void LAPACK<OrdinalType, float>::LARTG( const float f, const float g, float* c, float* s, float* r ) const
  {
    SLARTG_F77(&f, &g, c, s, r);
  }

  template<typename OrdinalType>
  void LAPACK<OrdinalType, float>::LARFG( const OrdinalType n, float* alpha, float* x, const OrdinalType incx, float* tau ) const
  {
    SLARFG_F77(&n, alpha, x, &incx, tau);
  }

  template<typename OrdinalType>
  float LAPACK<OrdinalType, float>::LARND( const OrdinalType idist, OrdinalType* seed ) const
  {
    return(SLARND_F77(&idist, seed));
  }

  template<typename OrdinalType>
  void LAPACK<OrdinalType, float>::LARNV( const OrdinalType idist, OrdinalType* seed, const OrdinalType n, float* v ) const
  {
    SLARNV_F77(&idist, seed, &n, v);
  }

  template<typename OrdinalType>
  float LAPACK<OrdinalType, float>::LAMCH(const char CMACH) const
  {
    return(SLAMCH_F77(CHAR_MACRO(CMACH)));
  }

  template<typename OrdinalType>
  OrdinalType LAPACK<OrdinalType, float>::ILAENV( const OrdinalType ispec, const std::string& NAME, const std::string& OPTS, const OrdinalType N1, const OrdinalType N2, const OrdinalType N3, const OrdinalType N4 ) const
  {
    unsigned int opts_length = OPTS.length();
    // if user queries a Hermitian routine, change it to a symmetric routine
    std::string temp_NAME = "s" + NAME;
    if (temp_NAME.substr(1,2) == "he") {
      temp_NAME.replace(1,2,"sy");
    }
    unsigned int name_length = temp_NAME.length();
#if defined (INTEL_CXML)
    return ILAENV_F77(&ispec, &temp_NAME[0], name_length, &OPTS[0], opts_length, &N1, &N2, &N3, &N4 );
#else
    return ILAENV_F77(&ispec, &temp_NAME[0], &OPTS[0], &N1, &N2, &N3, &N4, name_length, opts_length );
#endif
  }
 
  template<typename OrdinalType>
  float LAPACK<OrdinalType, float>::LAPY2(const float x, const float y) const
  {
    return SLAPY2_F77(&x, &y);
  }

  // END FLOAT PARTIAL SPECIALIZATION IMPLEMENTATION //

#endif // HAVE_TEUCHOS_BLASFLOAT

  // BEGIN DOUBLE PARTIAL SPECIALIZATION DECLARATION //

  template<typename OrdinalType>
  class LAPACK<OrdinalType, double>
  {    
  public:
    inline LAPACK(void) {}
    inline LAPACK(const LAPACK<OrdinalType, double>& lapack) {}
    inline virtual ~LAPACK(void) {}

    // Symmetric positive definite linear system routines
    void PTTRF(const OrdinalType n, double* d, double* e, OrdinalType* info) const;
    void PTTRS(const OrdinalType n, const OrdinalType nrhs, const double* d, const double* e, double* B, const OrdinalType ldb, OrdinalType* info) const;
    void POTRF(const char UPLO, const OrdinalType n, double* A, const OrdinalType lda, OrdinalType* info) const;
    void POTRS(const char UPLO, const OrdinalType n, const OrdinalType nrhs, const double* A, const OrdinalType lda, double* B, const OrdinalType ldb, OrdinalType* info) const;
    void POTRI(const char UPLO, const OrdinalType n, double* A, const OrdinalType lda, OrdinalType* info) const;
    void POCON(const char UPLO, const OrdinalType n, const double* A, const OrdinalType lda, const double anorm, double* rcond, double* WORK, OrdinalType* IWORK, OrdinalType* info) const;
    void POSV(const char UPLO, const OrdinalType n, const OrdinalType nrhs, double* A, const OrdinalType lda, double* B, const OrdinalType ldb, OrdinalType* info) const;
    void POEQU(const OrdinalType n, const double* A, const OrdinalType lda, double* S, double* scond, double* amax, OrdinalType* info) const;
    void PORFS(const char UPLO, const OrdinalType n, const OrdinalType nrhs, double* A, const OrdinalType lda, const double* AF, const OrdinalType ldaf, const double* B, const OrdinalType ldb, double* X, const OrdinalType ldx, double* FERR, double* BERR, double* WORK, OrdinalType* IWORK, OrdinalType* info) const;
    void POSVX(const char FACT, const char UPLO, const OrdinalType n, const OrdinalType nrhs, double* A, const OrdinalType lda, double* AF, const OrdinalType ldaf, char EQUED, double* S, double* B, const OrdinalType ldb, double* X, const OrdinalType ldx, double* rcond, double* FERR, double* BERR, double* WORK, OrdinalType* IWORK, OrdinalType* info) const; 

    // General linear system routines
    void GELS(const char TRANS, const OrdinalType m, const OrdinalType n, const OrdinalType nrhs, double* A, const OrdinalType lda, double* B, const OrdinalType ldb, double* WORK, const OrdinalType lwork, OrdinalType* info) const;
    void GEQRF( const OrdinalType m, const OrdinalType n, double* A, const OrdinalType lda, double* TAU, double* WORK, const OrdinalType lwork, OrdinalType* info) const;
    void GETRF(const OrdinalType m, const OrdinalType n, double* A, const OrdinalType lda, OrdinalType* IPIV, OrdinalType* info) const;
    void GETRS(const char TRANS, const OrdinalType n, const OrdinalType nrhs, const double* A, const OrdinalType lda, const OrdinalType* IPIV, double* B, const OrdinalType ldb, OrdinalType* info) const;
    void GTTRF(const OrdinalType n, double* dl, double* d, double* du, double* du2, OrdinalType* IPIV, OrdinalType* info) const;
    void GTTRS(const char TRANS, const OrdinalType n, const OrdinalType nrhs, const double* dl, const double* d, const double* du, const double* du2, const OrdinalType* IPIV, double* B, const OrdinalType ldb, OrdinalType* info) const;
    void GETRI(const OrdinalType n, double* A, const OrdinalType lda, const OrdinalType* IPIV, double* WORK, const OrdinalType lwork, OrdinalType* info) const;
    void GECON(const char NORM, const OrdinalType n, const double* A, const OrdinalType lda, const double anorm, double* rcond, double* WORK, OrdinalType* IWORK, OrdinalType* info) const;
    void GESV(const OrdinalType n, const OrdinalType nrhs, double* A, const OrdinalType lda, OrdinalType* IPIV, double* B, const OrdinalType ldb, OrdinalType* info) const;
    void GEEQU(const OrdinalType m, const OrdinalType n, const double* A, const OrdinalType lda, double* R, double* C, double* rowcond, double* colcond, double* amax, OrdinalType* info) const;
    void GERFS(const char TRANS, const OrdinalType n, const OrdinalType nrhs, const double* A, const OrdinalType lda, const double* AF, const OrdinalType ldaf, const OrdinalType* IPIV, const double* B, const OrdinalType ldb, double* X, const OrdinalType ldx, double* FERR, double* BERR, double* WORK, OrdinalType* IWORK, OrdinalType* info) const;
    void GESVX(const char FACT, const char TRANS, const OrdinalType n, const OrdinalType nrhs, double* A, const OrdinalType lda, double* AF, const OrdinalType ldaf, OrdinalType* IPIV, char EQUED, double* R, double* C, double* B, const OrdinalType ldb, double* X, const OrdinalType ldx, double* rcond, double* FERR, double* BERR, double* WORK, OrdinalType* IWORK, OrdinalType* info) const;
    void SYTRD(const char UPLO, const OrdinalType n, double* A, const OrdinalType lda, double* D, double* E, double* TAU, double* WORK, const OrdinalType lwork, OrdinalType* info) const;
    void GEHRD(const OrdinalType n, const OrdinalType ilo, const OrdinalType ihi, double* A, const OrdinalType lda, double* TAU, double* WORK, const OrdinalType lwork, OrdinalType* info) const;
    void TRTRS(const char UPLO, const char TRANS, const char DIAG, const OrdinalType n, const OrdinalType nrhs, const double* A, const OrdinalType lda, double* B, const OrdinalType ldb, OrdinalType* info) const; 

    // Symmetric eigenproblem routines.
    void SPEV(const char JOBZ, const char UPLO, const OrdinalType n, double* AP, double* W, double* Z, const OrdinalType ldz, double* WORK, OrdinalType* info) const;
    void SYEV(const char JOBZ, const char UPLO, const OrdinalType n, double* A, const OrdinalType lda, double* W, double* WORK, const OrdinalType lwork, OrdinalType* info) const;
    void SYGV(const OrdinalType itype, const char JOBZ, const char UPLO, const OrdinalType n, double* A, const OrdinalType lda, double* B, const OrdinalType ldb, double* W, double* WORK, const OrdinalType lwork, OrdinalType* info) const;
    void HEEV(const char JOBZ, const char UPLO, const OrdinalType n, double* A, const OrdinalType lda, double* W, double* WORK, const OrdinalType lwork, double* RWORK, OrdinalType* info) const;
    void HEGV(const OrdinalType itype, const char JOBZ, const char UPLO, const OrdinalType n, double* A, const OrdinalType lda, double* B, const OrdinalType ldb, double* W, double* WORK, const OrdinalType lwork, double *RWORK, OrdinalType* info) const;
    void STEQR(const char COMPZ, const OrdinalType n, double* D, double* E, double* Z, const OrdinalType ldz, double* WORK, OrdinalType* info) const;

    // Non-Hermitian eigenproblem routines.
    void HSEQR(const char JOB, const char COMPZ, const OrdinalType n, const OrdinalType ilo, const OrdinalType ihi, double* H, const OrdinalType ldh, double* WR, double* WI, double* Z, const OrdinalType ldz, double* WORK, const OrdinalType lwork, OrdinalType* info) const;
    void GEES(const char JOBVS, const char SORT, OrdinalType (*ptr2func)(double*, double*), const OrdinalType n, double* A, const OrdinalType lda, OrdinalType* sdim, double* WR, double* WI, double* VS, const OrdinalType ldvs, double* WORK, const OrdinalType lwork, OrdinalType* BWORK, OrdinalType* info) const;    
    void GEES(const char JOBVS, const OrdinalType n, double* A, const OrdinalType lda, OrdinalType* sdim, double* WR, double* WI, double* VS, const OrdinalType ldvs, double* WORK, const OrdinalType lwork, double* RWORK, OrdinalType* BWORK, OrdinalType* info) const;    
    void GEEV(const char JOBVL, const char JOBVR, const OrdinalType n, double* A, const OrdinalType lda, double* WR, double* WI, double* VL, const OrdinalType ldvl, double* VR, const OrdinalType ldvr, double* WORK, const OrdinalType lwork, OrdinalType* info) const;
    void GGEVX(const char BALANC, const char JOBVL, const char JOBVR, const char SENSE, const OrdinalType n, double* A, const OrdinalType lda, double* B, const OrdinalType ldb, double* ALPHAR, double* ALPHAI, double* BETA, double* VL, const OrdinalType ldvl, double* VR, const OrdinalType ldvr, OrdinalType* ilo, OrdinalType* ihi, double* LSCALE, double* RSCALE, double* abnrm, double* bbnrm, double* RCONDE, double* RCONDV, double* WORK, const OrdinalType lwork, OrdinalType* IWORK, OrdinalType* BWORK, OrdinalType* info) const;

    // Orthogonal matrix routines.
    void ORMQR(const char SIDE, const char TRANS, const OrdinalType m, const OrdinalType n, const OrdinalType k, double* A, const OrdinalType lda, const double* TAU, double* C, const OrdinalType ldc, double* WORK, const OrdinalType lwork, OrdinalType* info) const;
    void ORGQR(const OrdinalType m, const OrdinalType n, const OrdinalType k, double* A, const OrdinalType lda, const double* TAU, double* WORK, const OrdinalType lwork, OrdinalType* info) const;
    void UNGQR(const OrdinalType m, const OrdinalType n, const OrdinalType k, double* A, const OrdinalType lda, const double* TAU, double* WORK, const OrdinalType lwork, OrdinalType* info) const;
    void ORGHR(const OrdinalType n, const OrdinalType ilo, const OrdinalType ihi, double* A, const OrdinalType lda, const double* TAU, double* WORK, const OrdinalType lwork, OrdinalType* info) const;
    void ORMHR(const char SIDE, const char TRANS, const OrdinalType m, const OrdinalType n, const OrdinalType ilo, const OrdinalType ihi, const double* A, const OrdinalType lda, const double* TAU, double* C, const OrdinalType ldc, double* WORK, const OrdinalType lwork, OrdinalType* info) const;

    // Triangular matrix routines.
    void TREVC(const char SIDE, const char HOWMNY, OrdinalType (*ptr2func)(double*,double*), const OrdinalType n, const double* T, const OrdinalType ldt, double* VL, const OrdinalType ldvl, double* VR, const OrdinalType ldvr, const OrdinalType mm, OrdinalType* m, double* WORK, OrdinalType* info) const;
    void TREVC(const char SIDE, const OrdinalType n, const double* T, const OrdinalType ldt, double* VL, const OrdinalType ldvl, double* VR, const OrdinalType ldvr, const OrdinalType mm, OrdinalType* m, double* WORK, double* RWORK, OrdinalType* info) const;
    void TREXC(const char COMPQ, const OrdinalType n, double* T, const OrdinalType ldt, double* Q, const OrdinalType ldq, OrdinalType ifst, OrdinalType ilst, double* WORK, OrdinalType* info) const;

    // Rotation/reflection generators
    void LARTG( const double f, const double g, double* c, double* s, double* r ) const;
    void LARFG( const OrdinalType n, double* alpha, double* x, const OrdinalType incx, double* tau ) const;

    // Random number generators
    double LARND( const OrdinalType idist, OrdinalType* seed ) const;
    void LARNV( const OrdinalType idist, OrdinalType* seed, const OrdinalType n, double* v ) const;    

    // Machine characteristic routines.
    double LAMCH(const char CMACH) const;
    OrdinalType ILAENV( const OrdinalType ispec, const std::string& NAME, const std::string& OPTS, const OrdinalType N1 = -1, const OrdinalType N2 = -1, const OrdinalType N3 = -1, const OrdinalType N4 = -1 ) const;

    // Miscellaneous routines.
    double LAPY2(const double x, const double y) const;

  };

  // END DOUBLE PARTIAL SPECIALIZATION DECLARATION //

  // BEGIN DOUBLE PARTIAL SPECIALIZATION IMPLEMENTATION //


  template<typename OrdinalType>
  void LAPACK<OrdinalType, double>::PTTRF(const OrdinalType n, double* d, double* e, OrdinalType* info) const
  {
    DPTTRF_F77(&n,d,e,info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType, double>::PTTRS(const OrdinalType n, const OrdinalType nrhs, const double* d, const double* e, double* B, const OrdinalType ldb, OrdinalType* info) const
  {
    DPTTRS_F77(&n,&nrhs,d,e,B,&ldb,info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType, double>::POTRF(const char UPLO, const OrdinalType n, double* A, const OrdinalType lda, OrdinalType* info) const
  {
    DPOTRF_F77(CHAR_MACRO(UPLO), &n, A, &lda, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType, double>::POTRS(const char UPLO, const OrdinalType n, const OrdinalType nrhs, const double* A, const OrdinalType lda, double* B, const OrdinalType ldb, OrdinalType* info) const
  {
    DPOTRS_F77(CHAR_MACRO(UPLO), &n, &nrhs, A, &lda, B, &ldb, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType, double>::POTRI(const char UPLO, const OrdinalType n, double* A, const OrdinalType lda, OrdinalType* info) const
  {
    DPOTRI_F77(CHAR_MACRO(UPLO), &n, A, &lda, info);
  }
  
  template<typename OrdinalType>
    void LAPACK<OrdinalType, double>::POCON(const char UPLO, const OrdinalType n, const double* A, const OrdinalType lda, const double anorm, double* rcond, double* WORK, OrdinalType* IWORK, OrdinalType* info) const
  {
    DPOCON_F77(CHAR_MACRO(UPLO), &n, A, &lda, &anorm, rcond, WORK, IWORK, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType, double>::POSV(const char UPLO, const OrdinalType n, const OrdinalType nrhs, double* A, const OrdinalType lda, double* B, const OrdinalType ldb, OrdinalType* info) const
  {
    DPOSV_F77(CHAR_MACRO(UPLO), &n, &nrhs, A, &lda, B, &ldb, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType, double>::POEQU(const OrdinalType n, const double* A, const OrdinalType lda, double* S, double* scond, double* amax, OrdinalType* info) const
  {
    DPOEQU_F77(&n, A, &lda, S, scond, amax, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType, double>::PORFS(const char UPLO, const OrdinalType n, const OrdinalType nrhs, double* A, const OrdinalType lda, const double* AF, const OrdinalType ldaf, const double* B, const OrdinalType ldb, double* X, const OrdinalType ldx, double* FERR, double* BERR, double* WORK, OrdinalType* IWORK, OrdinalType* info) const
  {
    DPORFS_F77(CHAR_MACRO(UPLO), &n, &nrhs, A, &lda, AF, &ldaf, B, &ldb, X, &ldx, FERR, BERR, WORK, IWORK, info);
  }
  
  template<typename OrdinalType>
    void LAPACK<OrdinalType, double>::POSVX(const char FACT, const char UPLO, const OrdinalType n, const OrdinalType nrhs, double* A, const OrdinalType lda, double* AF, const OrdinalType ldaf, char EQUED, double* S, double* B, const OrdinalType ldb, double* X, const OrdinalType ldx, double* rcond, double* FERR, double* BERR, double* WORK, OrdinalType* IWORK, OrdinalType* info) const 
  {
    DPOSVX_F77(CHAR_MACRO(FACT), CHAR_MACRO(UPLO), &n, &nrhs, A, &lda, AF, &ldaf, CHAR_MACRO(EQUED), S, B, &ldb, X, &ldx, rcond, FERR, BERR, WORK, IWORK, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType,double>::GELS(const char TRANS, const OrdinalType m, const OrdinalType n, const OrdinalType nrhs, double* A, const OrdinalType lda, double* B, const OrdinalType ldb, double* WORK, const OrdinalType lwork, OrdinalType* info) const
  {
    DGELS_F77(CHAR_MACRO(TRANS), &m, &n, &nrhs, A, &lda, B, &ldb, WORK, &lwork, &info);
  }
  
  template<typename OrdinalType>  
  void LAPACK<OrdinalType,double>::GEQRF( const OrdinalType m, const OrdinalType n, double* A, const OrdinalType lda, double* TAU, double* WORK, const OrdinalType lwork, OrdinalType* info) const
  {
    DGEQRF_F77(&m, &n, A, &lda, TAU, WORK, &lwork, info);
  }

  template<typename OrdinalType>
  void LAPACK<OrdinalType,double>::GETRF(const OrdinalType m, const OrdinalType n, double* A, const OrdinalType lda, OrdinalType* IPIV, OrdinalType* info) const
  {
    DGETRF_F77(&m, &n, A, &lda, IPIV, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType,double>::GETRS(const char TRANS, const OrdinalType n, const OrdinalType nrhs, const double* A, const OrdinalType lda, const OrdinalType* IPIV, double* B, const OrdinalType ldb, OrdinalType* info) const
  {
    DGETRS_F77(CHAR_MACRO(TRANS), &n, &nrhs, A, &lda, IPIV, B, &ldb, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType,double>::GTTRF(const OrdinalType n, double* dl, double* d, double* du, double* du2, OrdinalType* IPIV, OrdinalType* info) const
  {
    DGTTRF_F77(&n, dl, d, du, du2, IPIV, info);
  }

  template<typename OrdinalType>
  void LAPACK<OrdinalType,double>::GTTRS(const char TRANS, const OrdinalType n, const OrdinalType nrhs, const double* dl, const double* d, const double* du, const double* du2, const OrdinalType* IPIV, double* B, const OrdinalType ldb, OrdinalType* info) const
  {
    DGTTRS_F77(CHAR_MACRO(TRANS), &n, &nrhs, dl, d, du, du2, IPIV, B, &ldb, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType,double>::GETRI(const OrdinalType n, double* A, const OrdinalType lda, const OrdinalType* IPIV, double* WORK, const OrdinalType lwork, OrdinalType* info) const
  {
    DGETRI_F77(&n, A, &lda, IPIV, WORK, &lwork, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType,double>::GECON(const char NORM, const OrdinalType n, const double* A, const OrdinalType lda, const double anorm, double* rcond, double* WORK, OrdinalType* IWORK, OrdinalType* info) const
  {
    DGECON_F77(CHAR_MACRO(NORM), &n, A, &lda, &anorm, rcond, WORK, IWORK, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType,double>::GESV(const OrdinalType n, const OrdinalType nrhs, double* A, const OrdinalType lda, OrdinalType* IPIV, double* B, const OrdinalType ldb, OrdinalType* info) const
  {
    DGESV_F77(&n, &nrhs, A, &lda, IPIV, B, &ldb, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType,double>::GEEQU(const OrdinalType m, const OrdinalType n, const double* A, const OrdinalType lda, double* R, double* C, double* rowcond, double* colcond, double* amax, OrdinalType* info) const
  {
    DGEEQU_F77(&m, &n, A, &lda, R, C, rowcond, colcond, amax, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType,double>::GERFS(const char TRANS, const OrdinalType n, const OrdinalType nrhs, const double* A, const OrdinalType lda, const double* AF, const OrdinalType ldaf, const OrdinalType* IPIV, const double* B, const OrdinalType ldb, double* X, const OrdinalType ldx, double* FERR, double* BERR, double* WORK, OrdinalType* IWORK, OrdinalType* info) const
  {
    DGERFS_F77(CHAR_MACRO(TRANS), &n, &nrhs, A, &lda, AF, &ldaf, IPIV, B, &ldb, X, &ldx, FERR, BERR, WORK, IWORK, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType,double>::GESVX(const char FACT, const char TRANS, const OrdinalType n, const OrdinalType nrhs, double* A, const OrdinalType lda, double* AF, const OrdinalType ldaf, OrdinalType* IPIV, char EQUED, double* R, double* C, double* B, const OrdinalType ldb, double* X, const OrdinalType ldx, double* rcond, double* FERR, double* BERR, double* WORK, OrdinalType* IWORK, OrdinalType* info) const
  {
    DGESVX_F77(CHAR_MACRO(FACT), CHAR_MACRO(TRANS), &n, &nrhs, A, &lda, AF, &ldaf, IPIV, CHAR_MACRO(EQUED), R, C, B, &ldb, X, &ldx, rcond, FERR, BERR, WORK, IWORK, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType,double>::SYTRD(const char UPLO, const OrdinalType n, double* A, const OrdinalType lda, double* D, double* E, double* TAU, double* WORK, const OrdinalType lwork, OrdinalType* info) const
  {
    DSYTRD_F77(CHAR_MACRO(UPLO), &n, A, &lda, D, E, TAU, WORK, &lwork, info);
  }

  template<typename OrdinalType>
  void LAPACK<OrdinalType, double>::GEHRD(const OrdinalType n, const OrdinalType ilo, const OrdinalType ihi, double* A, const OrdinalType lda, double* TAU, double* WORK, const OrdinalType lwork, OrdinalType* info) const
  {
    DGEHRD_F77(&n, &ilo, &ihi, A, &lda, TAU, WORK, &lwork, info);
  }

  template<typename OrdinalType>
  void LAPACK<OrdinalType,double>::TRTRS(const char UPLO, const char TRANS, const char DIAG, const OrdinalType n, const OrdinalType nrhs, const double* A, const OrdinalType lda, double* B, const OrdinalType ldb, OrdinalType* info) const 
  {
    DTRTRS_F77(CHAR_MACRO(UPLO), CHAR_MACRO(TRANS), CHAR_MACRO(DIAG), &n, &nrhs, A, &lda, B, &ldb, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType,double>::SPEV(const char JOBZ, const char UPLO, const OrdinalType n, double* AP, double* W, double* Z, const OrdinalType ldz, double* WORK, OrdinalType* info) const
  {
    DSPEV_F77(CHAR_MACRO(JOBZ), CHAR_MACRO(UPLO), &n, AP, W, Z, &ldz, WORK, info);
  }

  template<typename OrdinalType>
  void LAPACK<OrdinalType,double>::SYEV(const char JOBZ, const char UPLO, const OrdinalType n, double* A, const OrdinalType lda, double* W, double* WORK, const OrdinalType lwork, OrdinalType* info) const
  {
    DSYEV_F77(CHAR_MACRO(JOBZ), CHAR_MACRO(UPLO), &n, A, &lda, W, WORK, &lwork, info);
  }

  template<typename OrdinalType>
  void LAPACK<OrdinalType,double>::SYGV(const OrdinalType itype, const char JOBZ, const char UPLO, const OrdinalType n, double* A, const OrdinalType lda, double* B, const OrdinalType ldb, double* W, double* WORK, const OrdinalType lwork, OrdinalType* info) const
  {
    DSYGV_F77(&itype, CHAR_MACRO(JOBZ), CHAR_MACRO(UPLO), &n, A, &lda, B, &ldb, W, WORK, &lwork, info);
  }

  template<typename OrdinalType>
  void LAPACK<OrdinalType,double>::HEEV(const char JOBZ, const char UPLO, const OrdinalType n, double* A, const OrdinalType lda, double* W, double* WORK, const OrdinalType lwork, double* RWORK, OrdinalType* info) const
  {
    DSYEV_F77(CHAR_MACRO(JOBZ), CHAR_MACRO(UPLO), &n, A, &lda, W, WORK, &lwork, info);
  }

  template<typename OrdinalType>
  void LAPACK<OrdinalType,double>::HEGV(const OrdinalType itype, const char JOBZ, const char UPLO, const OrdinalType n, double* A, const OrdinalType lda, double* B, const OrdinalType ldb, double* W, double* WORK, const OrdinalType lwork, double *RWORK, OrdinalType* info) const
  {
    DSYGV_F77(&itype, CHAR_MACRO(JOBZ), CHAR_MACRO(UPLO), &n, A, &lda, B, &ldb, W, WORK, &lwork, info);
  }

  template<typename OrdinalType>
  void LAPACK<OrdinalType,double>::STEQR(const char COMPZ, const OrdinalType n, double* D, double* E, double* Z, const OrdinalType ldz, double* WORK, OrdinalType* info) const
  {
    DSTEQR_F77(CHAR_MACRO(COMPZ), &n, D, E, Z, &ldz, WORK, info);
  }

  template<typename OrdinalType>
  void LAPACK<OrdinalType, double>::HSEQR(const char JOB, const char COMPZ, const OrdinalType n, const OrdinalType ilo, const OrdinalType ihi, double* H, const OrdinalType ldh, double* WR, double* WI, double* Z, const OrdinalType ldz, double* WORK, const OrdinalType lwork, OrdinalType* info) const
  {
    DHSEQR_F77(CHAR_MACRO(JOB), CHAR_MACRO(COMPZ), &n, &ilo, &ihi, H, &ldh, WR, WI, Z, &ldz, WORK, &lwork, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType, double>::GEES(const char JOBVS, const char SORT, OrdinalType (*ptr2func)(double*, double*), const OrdinalType n, double* A, const OrdinalType lda, OrdinalType* sdim, double* WR, double* WI, double* VS, const OrdinalType ldvs, double* WORK, const OrdinalType lwork, OrdinalType* BWORK, OrdinalType* info) const    
  {
    DGEES_F77(CHAR_MACRO(JOBVS), CHAR_MACRO(SORT), ptr2func, &n, A, &lda, sdim, WR, WI, VS, &ldvs, WORK, &lwork, BWORK, info);
  }

  template<typename OrdinalType>
  void LAPACK<OrdinalType, double>::GEES(const char JOBVS, const OrdinalType n, double* A, const OrdinalType lda, OrdinalType* sdim, double* WR, double* WI, double* VS, const OrdinalType ldvs, double* WORK, const OrdinalType lwork, double* RWORK, OrdinalType* BWORK, OrdinalType* info) const    
  {
    OrdinalType (*nullfptr)(double*,double*) = NULL;
    const char sort = 'N';
    DGEES_F77(CHAR_MACRO(JOBVS), CHAR_MACRO(sort), nullfptr, &n, A, &lda, sdim, WR, WI, VS, &ldvs, WORK, &lwork, BWORK, info);
  }

  template<typename OrdinalType>
  void LAPACK<OrdinalType, double>::GEEV(const char JOBVL, const char JOBVR, const OrdinalType n, double* A, const OrdinalType lda, double* WR, double* WI, double* VL, const OrdinalType ldvl, double* VR, const OrdinalType ldvr, double* WORK, const OrdinalType lwork, OrdinalType* info) const
  {
    DGEEV_F77(CHAR_MACRO(JOBVL), CHAR_MACRO(JOBVR), &n, A, &lda, WR, WI, VL, &ldvl, VR, &ldvr, WORK, &lwork, info);
  }

  template<typename OrdinalType>
  void LAPACK<OrdinalType, double>::GGEVX(const char BALANC, const char JOBVL, const char JOBVR, const char SENSE, const OrdinalType n, double* A, const OrdinalType lda, double* B, const OrdinalType ldb, double* ALPHAR, double* ALPHAI, double* BETA, double* VL, const OrdinalType ldvl, double* VR, const OrdinalType ldvr, OrdinalType* ilo, OrdinalType* ihi, double* LSCALE, double* RSCALE, double* abnrm, double* bbnrm, double* RCONDE, double* RCONDV, double* WORK, const OrdinalType lwork, OrdinalType* IWORK, OrdinalType* BWORK, OrdinalType* info) const
  {
    DGGEVX_F77(CHAR_MACRO(BALANC), CHAR_MACRO(JOBVL), CHAR_MACRO(JOBVR), CHAR_MACRO(SENSE), &n, A, &lda, B, &ldb, ALPHAR, ALPHAI, BETA, VL, &ldvl, VR, &ldvr, ilo, ihi, LSCALE, RSCALE, abnrm, bbnrm, RCONDE, RCONDV, WORK, &lwork, IWORK, BWORK, info);
  }

  template<typename OrdinalType>
  void LAPACK<OrdinalType, double>::ORMQR(const char SIDE, const char TRANS, const OrdinalType m, const OrdinalType n, const OrdinalType k, double* A, const OrdinalType lda, const double* TAU, double* C, const OrdinalType ldc, double* WORK, const OrdinalType lwork, OrdinalType* info) const
  {
    DORMQR_F77(CHAR_MACRO(SIDE), CHAR_MACRO(TRANS), &m, &n, &k, A, &lda, TAU, C, &ldc, WORK, &lwork, info);
  }

  template<typename OrdinalType>
  void LAPACK<OrdinalType, double>::ORGQR(const OrdinalType m, const OrdinalType n, const OrdinalType k, double* A, const OrdinalType lda, const double* TAU, double* WORK, const OrdinalType lwork, OrdinalType* info) const
  {
    DORGQR_F77( &m, &n, &k, A, &lda, TAU, WORK, &lwork, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType, double>::UNGQR(const OrdinalType m, const OrdinalType n, const OrdinalType k, double* A, const OrdinalType lda, const double* TAU, double* WORK, const OrdinalType lwork, OrdinalType* info) const
  {
    DORGQR_F77( &m, &n, &k, A, &lda, TAU, WORK, &lwork, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType, double>::ORGHR(const OrdinalType n, const OrdinalType ilo, const OrdinalType ihi, double* A, const OrdinalType lda, const double* TAU, double* WORK, const OrdinalType lwork, OrdinalType* info) const
  {
    DORGHR_F77(&n, &ilo, &ihi, A, &lda, TAU, WORK, &lwork, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType, double>::ORMHR(const char SIDE, const char TRANS, const OrdinalType m, const OrdinalType n, const OrdinalType ilo, const OrdinalType ihi, const double* A, const OrdinalType lda, const double* TAU, double* C, const OrdinalType ldc, double* WORK, const OrdinalType lwork, OrdinalType* info) const
  {
    DORMHR_F77(CHAR_MACRO(SIDE), CHAR_MACRO(TRANS), &m, &n, &ilo, &ihi, A, &lda, TAU, C, &ldc, WORK, &lwork, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType, double>::TREVC(const char SIDE, const char HOWMNY, OrdinalType (*ptr2func)(double*,double*), const OrdinalType n, const double* T, const OrdinalType ldt, double* VL, const OrdinalType ldvl, double* VR, const OrdinalType ldvr, const OrdinalType mm, OrdinalType* m, double* WORK, OrdinalType* info) const
  {
    DTREVC_F77(CHAR_MACRO(SIDE), CHAR_MACRO(HOWMNY), ptr2func, &n, T, &ldt, VL, &ldvl, VR, &ldvr, &mm, m, WORK, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType, double>::TREVC(const char SIDE, const OrdinalType n, const double* T, const OrdinalType ldt, double* VL, const OrdinalType ldvl, double* VR, const OrdinalType ldvr, const OrdinalType mm, OrdinalType* m, double* WORK, double* RWORK, OrdinalType* info) const
  {
    OrdinalType (*nullfptr)(double*,double*) = NULL;
    const char whch = 'A';
    DTREVC_F77(CHAR_MACRO(SIDE), CHAR_MACRO(whch), nullfptr, &n, T, &ldt, VL, &ldvl, VR, &ldvr, &mm, m, WORK, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType, double>::TREXC(const char COMPQ, const OrdinalType n, double* T, const OrdinalType ldt, double* Q, const OrdinalType ldq, OrdinalType ifst, OrdinalType ilst, double* WORK, OrdinalType* info) const
  {
    DTREXC_F77(CHAR_MACRO(COMPQ), &n, T, &ldt, Q, &ldq, &ifst, &ilst, WORK, info);
  }

  template<typename OrdinalType>
  void LAPACK<OrdinalType, double>::LARTG( const double f, const double g, double* c, double* s, double* r ) const
  {
    DLARTG_F77(&f, &g, c, s, r);
  }

  template<typename OrdinalType>
  void LAPACK<OrdinalType, double>::LARFG( const OrdinalType n, double* alpha, double* x, const OrdinalType incx, double* tau ) const
  {
    DLARFG_F77(&n, alpha, x, &incx, tau);
  }

  template<typename OrdinalType>
  double LAPACK<OrdinalType, double>::LARND( const OrdinalType idist, OrdinalType* seed ) const
  {
    return(DLARND_F77(&idist, seed));
  }

  template<typename OrdinalType>
  void LAPACK<OrdinalType, double>::LARNV( const OrdinalType idist, OrdinalType* seed, const OrdinalType n, double* v ) const
  {
    DLARNV_F77(&idist, seed, &n, v);
  }

  template<typename OrdinalType>
  double LAPACK<OrdinalType, double>::LAMCH(const char CMACH) const
  {
    return(DLAMCH_F77(CHAR_MACRO(CMACH)));
  }

  template<typename OrdinalType>
  OrdinalType LAPACK<OrdinalType, double>::ILAENV( const OrdinalType ispec, const std::string& NAME, const std::string& OPTS, const OrdinalType N1, const OrdinalType N2, const OrdinalType N3, const OrdinalType N4 ) const
  {
    unsigned int opts_length = OPTS.length();
    // if user queries a Hermitian routine, change it to a symmetric routine
    std::string temp_NAME = "d" + NAME;
    if (temp_NAME.substr(1,2) == "he") {
      temp_NAME.replace(1,2,"sy");
    }
    unsigned int name_length = temp_NAME.length();
#if defined(INTEL_CXML)
    return ILAENV_F77(&ispec, &temp_NAME[0], name_length, &OPTS[0], opts_length, &N1, &N2, &N3, &N4 );
#else
# if defined(__INTEL_COMPILER) && defined(_WIN32)
    return 0;
# else
    return ILAENV_F77(&ispec, &temp_NAME[0], &OPTS[0], &N1, &N2, &N3, &N4, name_length, opts_length );
# endif
#endif
  }
 
  template<typename OrdinalType>
  double LAPACK<OrdinalType, double>::LAPY2(const double x, const double y) const
  {
    return DLAPY2_F77(&x, &y);
  }

  // END DOUBLE PARTIAL SPECIALIZATION IMPLEMENTATION //

#ifdef HAVE_TEUCHOS_COMPLEX

#ifdef HAVE_TEUCHOS_BLASFLOAT

  // BEGIN COMPLEX<FLOAT> PARTIAL SPECIALIZATION DECLARATION //

  template<typename OrdinalType>
  class LAPACK<OrdinalType, std::complex<float> >
  {    
  public:
    inline LAPACK(void) {}
    inline LAPACK(const LAPACK<OrdinalType, std::complex<float> >& lapack) {}
    inline virtual ~LAPACK(void) {}

    // Symmetric positive definite linear system routines
    void PTTRF(const OrdinalType n, std::complex<float>* d, std::complex<float>* e, OrdinalType* info) const;
    void PTTRS(const OrdinalType n, const OrdinalType nrhs, const std::complex<float>* d, const std::complex<float>* e, std::complex<float>* B, const OrdinalType ldb, OrdinalType* info) const;
    void POTRF(const char UPLO, const OrdinalType n, std::complex<float>* A, const OrdinalType lda, OrdinalType* info) const;
    void POTRS(const char UPLO, const OrdinalType n, const OrdinalType nrhs, const std::complex<float>* A, const OrdinalType lda, std::complex<float>* B, const OrdinalType ldb, OrdinalType* info) const;
    void POTRI(const char UPLO, const OrdinalType n, std::complex<float>* A, const OrdinalType lda, OrdinalType* info) const;
    void POCON(const char UPLO, const OrdinalType n, const std::complex<float>* A, const OrdinalType lda, const float anorm, float* rcond, std::complex<float>* WORK, float* rwork, OrdinalType* info) const;
    void POSV(const char UPLO, const OrdinalType n, const OrdinalType nrhs, std::complex<float>* A, const OrdinalType lda, std::complex<float>* B, const OrdinalType ldb, OrdinalType* info) const;
    void POEQU(const OrdinalType n, const std::complex<float>* A, const OrdinalType lda, float* S, float* scond, float* amax, OrdinalType* info) const;
    void PORFS(const char UPLO, const OrdinalType n, const OrdinalType nrhs, std::complex<float>* A, const OrdinalType lda, const std::complex<float>* AF, const OrdinalType ldaf, const std::complex<float>* B, const OrdinalType ldb, std::complex<float>* X, const OrdinalType ldx, float* FERR, float* BERR, std::complex<float>* WORK, float* RWORK, OrdinalType* info) const;
    void POSVX(const char FACT, const char UPLO, const OrdinalType n, const OrdinalType nrhs, std::complex<float>* A, const OrdinalType lda, std::complex<float>* AF, const OrdinalType ldaf, char EQUED, float* S, std::complex<float>* B, const OrdinalType ldb, std::complex<float>* X, const OrdinalType ldx, float* rcond, float* FERR, float* BERR, std::complex<float>* WORK, float* RWORK, OrdinalType* info) const; 

    // General Linear System Routines
    void GELS(const char TRANS, const OrdinalType m, const OrdinalType n, const OrdinalType nrhs, std::complex<float>* A, const OrdinalType lda, std::complex<float>* B, const OrdinalType ldb, std::complex<float>* WORK, const OrdinalType lwork, OrdinalType* info) const;
    void GEQRF( const OrdinalType m, const OrdinalType n, std::complex<float>* A, const OrdinalType lda, std::complex<float>* TAU, std::complex<float>* WORK, const OrdinalType lwork, OrdinalType* info) const;
    void UNGQR(const OrdinalType m, const OrdinalType n, const OrdinalType k, std::complex<float>* A, const OrdinalType lda, const std::complex<float>* TAU, std::complex<float>* WORK, const OrdinalType lwork, OrdinalType* info) const;
    void GETRF(const OrdinalType m, const OrdinalType n, std::complex<float>* A, const OrdinalType lda, OrdinalType* IPIV, OrdinalType* info) const;
    void GETRS(const char TRANS, const OrdinalType n, const OrdinalType nrhs, const std::complex<float>* A, const OrdinalType lda, const OrdinalType* IPIV, std::complex<float>* B, const OrdinalType ldb, OrdinalType* info) const;
    void GTTRF(const OrdinalType n, std::complex<float>* dl, std::complex<float>* d, std::complex<float>* du, std::complex<float>* du2, OrdinalType* IPIV, OrdinalType* info) const;
    void GTTRS(const char TRANS, const OrdinalType n, const OrdinalType nrhs, const std::complex<float>* dl, const std::complex<float>* d, const std::complex<float>* du, const std::complex<float>* du2, const OrdinalType* IPIV, std::complex<float>* B, const OrdinalType ldb, OrdinalType* info) const;
    void GETRI(const OrdinalType n, std::complex<float>* A, const OrdinalType lda, const OrdinalType* IPIV, std::complex<float>* WORK, const OrdinalType lwork, OrdinalType* info) const;
    void GECON(const char NORM, const OrdinalType n, const std::complex<float>* A, const OrdinalType lda, const float anorm, float* rcond, std::complex<float>* WORK, float* RWORK, OrdinalType* info) const;
    void GESV(const OrdinalType n, const OrdinalType nrhs, std::complex<float>* A, const OrdinalType lda, OrdinalType* IPIV, std::complex<float>* B, const OrdinalType ldb, OrdinalType* info) const;
    void GEEQU(const OrdinalType m, const OrdinalType n, const std::complex<float>* A, const OrdinalType lda, float* R, float* C, float* rowcond, float* colcond, float* amax, OrdinalType* info) const;
    void GERFS(const char TRANS, const OrdinalType n, const OrdinalType nrhs, const std::complex<float>* A, const OrdinalType lda, const std::complex<float>* AF, const OrdinalType ldaf, const OrdinalType* IPIV, const std::complex<float>* B, const OrdinalType ldb, std::complex<float>* X, const OrdinalType ldx, float* FERR, float* BERR, std::complex<float>* WORK, float* RWORK, OrdinalType* info) const;
    void GESVX(const char FACT, const char TRANS, const OrdinalType n, const OrdinalType nrhs, std::complex<float>* A, const OrdinalType lda, std::complex<float>* AF, const OrdinalType ldaf, OrdinalType* IPIV, char EQUED, float* R, float* C, std::complex<float>* B, const OrdinalType ldb, std::complex<float>* X, const OrdinalType ldx, float* rcond, float* FERR, float* BERR, std::complex<float>* WORK, float* RWORK, OrdinalType* info) const;
    void GEHRD(const OrdinalType n, const OrdinalType ilo, const OrdinalType ihi, std::complex<float>* A, const OrdinalType lda, std::complex<float>* TAU, std::complex<float>* WORK, const OrdinalType lwork, OrdinalType* info) const;
    void TRTRS(const char UPLO, const char TRANS, const char DIAG, const OrdinalType n, const OrdinalType nrhs, const std::complex<float>* A, const OrdinalType lda, std::complex<float>* B, const OrdinalType ldb, OrdinalType* info) const; 

    // Symmetric eigenvalue routines.
    void STEQR(const char COMPZ, const OrdinalType n, float* D, float* E, std::complex<float>* Z, const OrdinalType ldz, float* WORK, OrdinalType* info) const;
    void HEEV(const char JOBZ, const char UPLO, const OrdinalType n, std::complex<float>* A, const OrdinalType lda, float* W, std::complex<float>* WORK, const OrdinalType lwork, float* RWORK, OrdinalType* info) const;
    void HEGV(const OrdinalType itype, const char JOBZ, const char UPLO, const OrdinalType n, std::complex<float>* A, const OrdinalType lda, std::complex<float>* B, const OrdinalType ldb, float* W, std::complex<float>* WORK, const OrdinalType lwork, float *RWORK, OrdinalType* info) const;

    // Non-Hermitian eigenvalue routines.
    void HSEQR(const char JOB, const char COMPZ, const OrdinalType n, const OrdinalType ilo, const OrdinalType ihi, std::complex<float>* H, const OrdinalType ldh, std::complex<float>* W, std::complex<float>* Z, const OrdinalType ldz, std::complex<float>* WORK, const OrdinalType lwork, OrdinalType* info) const;
    void GEES(const char JOBVS, const char SORT, OrdinalType (*ptr2func)(std::complex<float>*), const OrdinalType n, std::complex<float>* A, const OrdinalType lda, OrdinalType* sdim, std::complex<float>* W, std::complex<float>* VS, const OrdinalType ldvs, std::complex<float>* WORK, const OrdinalType lwork, float* RWORK, OrdinalType* BWORK, OrdinalType* info) const;    
    void GEES(const char JOBVS, const OrdinalType n, std::complex<float>* A, const OrdinalType lda, OrdinalType* sdim, float* WR, float* WI, std::complex<float>* VS, const OrdinalType ldvs, std::complex<float>* WORK, const OrdinalType lwork, float* RWORK, OrdinalType* BWORK, OrdinalType* info) const;    
    void GEEV(const char JOBVL, const char JOBVR, const OrdinalType n, std::complex<float>* A, const OrdinalType lda, std::complex<float>* W, std::complex<float>* VL, const OrdinalType ldvl, std::complex<float>* VR, const OrdinalType ldvr, std::complex<float>* WORK, const OrdinalType lwork, float* RWORK, OrdinalType* info) const;
//    void GGEVX(const char BALANC, const char JOBVL, const char JOBVR, const char SENSE, const OrdinalType n, std::complex<float>* A, const OrdinalType lda, std::complex<float>* B, const OrdinalType ldb, float* ALPHAR, float* ALPHAI, std::complex<float>* BETA, std::complex<float>* VL, const OrdinalType ldvl, std::complex<float>* VR, const OrdinalType ldvr, OrdinalType* ilo, OrdinalType* ihi, float* LSCALE, float* RSCALE, float* abnrm, float* bbnrm, float* RCONDE, float* RCONDV, std::complex<float>* work, const OrdinalType lwork, OrdinalType* IWORK, OrdinalType* BWORK, OrdinalType* info) const;

    // Triangular matrix routines.
    void TREVC(const char SIDE, const char HOWMNY, OrdinalType (*ptr2func)(std::complex<float>*), const OrdinalType n, const std::complex<float>* T, const OrdinalType ldt, std::complex<float>* VL, const OrdinalType ldvl, std::complex<float>* VR, const OrdinalType ldvr, const OrdinalType mm, OrdinalType* m, std::complex<float>* WORK, float* RWORK, OrdinalType* info) const;
    void TREVC(const char SIDE, const OrdinalType n, const std::complex<float>* T, const OrdinalType ldt, std::complex<float>* VL, const OrdinalType ldvl, std::complex<float>* VR, const OrdinalType ldvr, const OrdinalType mm, OrdinalType* m, std::complex<float>* WORK, float* RWORK, OrdinalType* info) const;
    void TREXC(const char COMPQ, const OrdinalType n, std::complex<float>* T, const OrdinalType ldt, std::complex<float>* Q, const OrdinalType ldq, OrdinalType ifst, OrdinalType ilst, std::complex<float>* WORK, OrdinalType* info) const;

    // Rotation/reflection generators
    void LARTG( const std::complex<float> f, const std::complex<float> g, float* c, std::complex<float>* s, std::complex<float>* r ) const;
    void LARFG( const OrdinalType n, std::complex<float>* alpha, std::complex<float>* x, const OrdinalType incx, std::complex<float>* tau ) const;

    // Random number generators
    std::complex<float> LARND( const OrdinalType idist, OrdinalType* seed ) const;
    void LARNV( const OrdinalType idist, OrdinalType* seed, const OrdinalType n, std::complex<float>* v ) const;    

    // Machine characteristics
    OrdinalType ILAENV( const OrdinalType ispec, const std::string& NAME, const std::string& OPTS, const OrdinalType N1 = -1, const OrdinalType N2 = -1, const OrdinalType N3 = -1, const OrdinalType N4 = -1 ) const;

  };

  // END COMPLEX<FLOAT> PARTIAL SPECIALIZATION DECLARATION //

  // BEGIN COMPLEX<FLOAT> PARTIAL SPECIALIZATION IMPLEMENTATION //

  template<typename OrdinalType>
  void LAPACK<OrdinalType, std::complex<float> >::PTTRF(const OrdinalType n, std::complex<float>* d, std::complex<float>* e, OrdinalType* info) const
  {
    CPTTRF_F77(&n,d,e,info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType, std::complex<float> >::PTTRS(const OrdinalType n, const OrdinalType nrhs, const std::complex<float>* d, const std::complex<float>* e, std::complex<float>* B, const OrdinalType ldb, OrdinalType* info) const
  {
    CPTTRS_F77(&n,&nrhs,d,e,B,&ldb,info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType, std::complex<float> >::POTRF(const char UPLO, const OrdinalType n, std::complex<float>* A, const OrdinalType lda, OrdinalType* info) const
  {
    CPOTRF_F77(CHAR_MACRO(UPLO), &n, A, &lda, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType, std::complex<float> >::POTRS(const char UPLO, const OrdinalType n, const OrdinalType nrhs, const std::complex<float>* A, const OrdinalType lda, std::complex<float>* B, const OrdinalType ldb, OrdinalType* info) const
  {
    CPOTRS_F77(CHAR_MACRO(UPLO), &n, &nrhs, A, &lda, B, &ldb, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType, std::complex<float> >::POTRI(const char UPLO, const OrdinalType n, std::complex<float>* A, const OrdinalType lda, OrdinalType* info) const
  {
    CPOTRI_F77(CHAR_MACRO(UPLO), &n, A, &lda, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType, std::complex<float> >::POCON(const char UPLO, const OrdinalType n, const std::complex<float>* A, const OrdinalType lda, const float anorm, float* rcond, std::complex<float>* WORK, float* RWORK, OrdinalType* info) const
  {
    CPOCON_F77(CHAR_MACRO(UPLO), &n, A, &lda, &anorm, rcond, WORK, RWORK, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType, std::complex<float> >::POSV(const char UPLO, const OrdinalType n, const OrdinalType nrhs, std::complex<float>* A, const OrdinalType lda, std::complex<float>* B, const OrdinalType ldb, OrdinalType* info) const
  {
    CPOSV_F77(CHAR_MACRO(UPLO), &n, &nrhs, A, &lda, B, &ldb, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType, std::complex<float> >::POEQU(const OrdinalType n, const std::complex<float>* A, const OrdinalType lda, float* S, float* scond, float* amax, OrdinalType* info) const
  {
    CPOEQU_F77(&n, A, &lda, S, scond, amax, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType, std::complex<float> >::PORFS(const char UPLO, const OrdinalType n, const OrdinalType nrhs, std::complex<float>* A, const OrdinalType lda, const std::complex<float>* AF, const OrdinalType ldaf, const std::complex<float>* B, const OrdinalType ldb, std::complex<float>* X, const OrdinalType ldx, float* FERR, float* BERR, std::complex<float>* WORK, float* RWORK, OrdinalType* info) const
  {
    CPORFS_F77(CHAR_MACRO(UPLO), &n, &nrhs, A, &lda, AF, &ldaf, B, &ldb, X, &ldx, FERR, BERR, WORK, RWORK, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType, std::complex<float> >::POSVX(const char FACT, const char UPLO, const OrdinalType n, const OrdinalType nrhs, std::complex<float>* A, const OrdinalType lda, std::complex<float>* AF, const OrdinalType ldaf, char EQUED, float* S, std::complex<float>* B, const OrdinalType ldb, std::complex<float>* X, const OrdinalType ldx, float* rcond, float* FERR, float* BERR, std::complex<float>* WORK, float* RWORK, OrdinalType* info) const
  {
    CPOSVX_F77(CHAR_MACRO(FACT), CHAR_MACRO(UPLO), &n, &nrhs, A, &lda, AF, &ldaf, CHAR_MACRO(EQUED), S, B, &ldb, X, &ldx, rcond, FERR, BERR, WORK, RWORK, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType,std::complex<float> >::GELS(const char TRANS, const OrdinalType m, const OrdinalType n, const OrdinalType nrhs, std::complex<float>* A, const OrdinalType lda, std::complex<float>* B, const OrdinalType ldb, std::complex<float>* WORK, const OrdinalType lwork, OrdinalType* info) const
  {
    CGELS_F77(CHAR_MACRO(TRANS), &m, &n, &nrhs, A, &lda, B, &ldb, WORK, &lwork, info);
  }
  
  template<typename OrdinalType>  
  void LAPACK<OrdinalType,std::complex<float> >::GEQRF( const OrdinalType m, const OrdinalType n, std::complex<float>* A, const OrdinalType lda, std::complex<float>* TAU, std::complex<float>* WORK, const OrdinalType lwork, OrdinalType* info) const
  {
    CGEQRF_F77(&m, &n, A, &lda, TAU, WORK, &lwork, info);
  }

  template<typename OrdinalType>
  void LAPACK<OrdinalType,std::complex<float> >::UNGQR(const OrdinalType m, const OrdinalType n, const OrdinalType k, std::complex<float>* A, const OrdinalType lda, const std::complex<float>* TAU, std::complex<float>* WORK, const OrdinalType lwork, OrdinalType* info) const
  {
    CUNGQR_F77( &m, &n, &k, A, &lda, TAU, WORK, &lwork, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType,std::complex<float> >::GETRF(const OrdinalType m, const OrdinalType n, std::complex<float>* A, const OrdinalType lda, OrdinalType* IPIV, OrdinalType* info) const
  {
    CGETRF_F77(&m, &n, A, &lda, IPIV, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType,std::complex<float> >::GETRS(const char TRANS, const OrdinalType n, const OrdinalType nrhs, const std::complex<float>* A, const OrdinalType lda, const OrdinalType* IPIV, std::complex<float>* B , const OrdinalType ldb, OrdinalType* info) const
  {
    CGETRS_F77(CHAR_MACRO(TRANS), &n, &nrhs, A, &lda, IPIV, B, &ldb, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType,std::complex<float> >::GTTRF(const OrdinalType n, std::complex<float>* dl, std::complex<float>* d, std::complex<float>* du, std::complex<float>* du2, OrdinalType* IPIV, OrdinalType* info) const
  {
    CGTTRF_F77(&n, dl, d, du, du2, IPIV, info);
  }

  template<typename OrdinalType>
  void LAPACK<OrdinalType,std::complex<float> >::GTTRS(const char TRANS, const OrdinalType n, const OrdinalType nrhs, const std::complex<float>* dl, const std::complex<float>* d, const std::complex<float>* du, const std::complex<float>* du2, const OrdinalType* IPIV, std::complex<float>* B, const OrdinalType ldb, OrdinalType* info) const
  {
    CGTTRS_F77(CHAR_MACRO(TRANS), &n, &nrhs, dl, d, du, du2, IPIV, B, &ldb, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType,std::complex<float> >::GETRI(const OrdinalType n, std::complex<float>* A, const OrdinalType lda, const OrdinalType* IPIV, std::complex<float>* WORK, const OrdinalType lwork, OrdinalType* info) const
  {
    CGETRI_F77(&n, A, &lda, IPIV, WORK, &lwork, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType,std::complex<float> >::GECON(const char NORM, const OrdinalType n, const std::complex<float>* A, const OrdinalType lda, const float anorm, float* rcond, std::complex<float>* WORK, float* RWORK, OrdinalType* info) const
  {
    CGECON_F77(CHAR_MACRO(NORM), &n, A, &lda, &anorm, rcond, WORK, RWORK, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType,std::complex<float> >::GESV(const OrdinalType n, const OrdinalType nrhs, std::complex<float>* A, const OrdinalType lda, OrdinalType* IPIV, std::complex<float>* B, const OrdinalType ldb, OrdinalType* info) const
  {
    CGESV_F77(&n, &nrhs, A, &lda, IPIV, B, &ldb, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType,std::complex<float> >::GEEQU(const OrdinalType m, const OrdinalType n, const std::complex<float>* A, const OrdinalType lda, float* R, float* C, float* rowcond, float* colcond, float* amax, OrdinalType* info) const
  {
    CGEEQU_F77(&m, &n, A, &lda, R, C, rowcond, colcond, amax, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType,std::complex<float> >::GERFS(const char TRANS, const OrdinalType n, const OrdinalType nrhs, const std::complex<float>* A, const OrdinalType lda, const std::complex<float>* AF, const OrdinalType ldaf, const OrdinalType* IPIV, const std::complex<float>* B, const OrdinalType ldb, std::complex<float>* X, const OrdinalType ldx, float* FERR, float* BERR, std::complex<float>* WORK, float* RWORK, OrdinalType* info) const
  {
    CGERFS_F77(CHAR_MACRO(TRANS), &n, &nrhs, A, &lda, AF, &ldaf, IPIV, B, &ldb, X, &ldx, FERR, BERR, WORK, RWORK, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType,std::complex<float> >::GESVX(const char FACT, const char TRANS, const OrdinalType n, const OrdinalType nrhs, std::complex<float>* A, const OrdinalType lda, std::complex<float>* AF, const OrdinalType ldaf, OrdinalType* IPIV, char EQUED, float* R, float* C, std::complex<float>* B, const OrdinalType ldb, std::complex<float>* X, const OrdinalType ldx, float* rcond, float* FERR, float* BERR, std::complex<float>* WORK, float* RWORK, OrdinalType* info) const
  {
    CGESVX_F77(CHAR_MACRO(FACT), CHAR_MACRO(TRANS), &n, &nrhs, A, &lda, AF, &ldaf, IPIV, CHAR_MACRO(EQUED), R, C, B, &ldb, X, &ldx, rcond, FERR, BERR, WORK, RWORK, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType,std::complex<float> >::GEHRD(const OrdinalType n, const OrdinalType ilo, const OrdinalType ihi, std::complex<float>* A, const OrdinalType lda, std::complex<float>* TAU, std::complex<float>* WORK, const OrdinalType lwork, OrdinalType* info) const
  {
    CGEHRD_F77(&n, &ilo, &ihi, A, &lda, TAU, WORK, &lwork, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType,std::complex<float> >::TRTRS(const char UPLO, const char TRANS, const char DIAG, const OrdinalType n, const OrdinalType nrhs, const std::complex<float>* A, const OrdinalType lda, std::complex<float>* B, const OrdinalType ldb, OrdinalType* info) const 
  {
    CTRTRS_F77(CHAR_MACRO(UPLO), CHAR_MACRO(TRANS), CHAR_MACRO(DIAG), &n, &nrhs, A, &lda, B, &ldb, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType,std::complex<float> >::STEQR(const char COMPZ, const OrdinalType n, float* D, float* E, std::complex<float>* Z, const OrdinalType ldz, float* WORK, OrdinalType* info) const
  {
    CSTEQR_F77(CHAR_MACRO(COMPZ), &n, D, E, Z, &ldz, WORK, info);
  }

  template<typename OrdinalType>
  void LAPACK<OrdinalType,std::complex<float> >::HEEV(const char JOBZ, const char UPLO, const OrdinalType n, std::complex<float> * A, const OrdinalType lda, float * W, std::complex<float> * WORK, const OrdinalType lwork, float* RWORK, OrdinalType* info) const
  {
    CHEEV_F77(CHAR_MACRO(JOBZ), CHAR_MACRO(UPLO), &n, A, &lda, W, WORK, &lwork, RWORK, info);
  }

  template<typename OrdinalType>
  void LAPACK<OrdinalType,std::complex<float> >::HEGV(const OrdinalType itype, const char JOBZ, const char UPLO, const OrdinalType n, std::complex<float> * A, const OrdinalType lda, std::complex<float> * B, const OrdinalType ldb, float * W, std::complex<float> * WORK, const OrdinalType lwork, float *RWORK, OrdinalType* info) const
  {
    CHEGV_F77(&itype, CHAR_MACRO(JOBZ), CHAR_MACRO(UPLO), &n, A, &lda, B, &ldb, W, WORK, &lwork, RWORK, info);
  }

  template<typename OrdinalType>
  void LAPACK<OrdinalType, std::complex<float> >::HSEQR(const char JOB, const char COMPZ, const OrdinalType n, const OrdinalType ilo, const OrdinalType ihi, std::complex<float>* H, const OrdinalType ldh, std::complex<float>* W, std::complex<float>* Z, const OrdinalType ldz, std::complex<float>* WORK, const OrdinalType lwork, OrdinalType* info) const
  {
    CHSEQR_F77(CHAR_MACRO(JOB), CHAR_MACRO(COMPZ), &n, &ilo, &ihi, H, &ldh, W, Z, &ldz, WORK, &lwork, info);
  }

  template<typename OrdinalType>
  void LAPACK<OrdinalType, std::complex<float> >::GEES(const char JOBVS, const char SORT, OrdinalType (*ptr2func)(std::complex<float>*), const OrdinalType n, std::complex<float>* A, const OrdinalType lda, OrdinalType* sdim, std::complex<float>* W, std::complex<float>* VS, const OrdinalType ldvs, std::complex<float>* WORK, const OrdinalType lwork, float* RWORK, OrdinalType* BWORK, OrdinalType* info) const    
  {
    CGEES_F77(CHAR_MACRO(JOBVS), CHAR_MACRO(SORT), ptr2func, &n, A, &lda, sdim, W, VS, &ldvs, WORK, &lwork, RWORK, BWORK, info);
  }

  template<typename OrdinalType>
  void LAPACK<OrdinalType, std::complex<float> >::GEES(const char JOBVS, const OrdinalType n, std::complex<float>* A, const OrdinalType lda, OrdinalType* sdim, float* WR, float* WI, std::complex<float>* VS, const OrdinalType ldvs, std::complex<float>* WORK, const OrdinalType lwork, float* RWORK, OrdinalType* BWORK, OrdinalType* info) const    
  {
    OrdinalType (*nullfptr)(std::complex<float>*) = NULL;
    std::vector< std::complex<float> > W(n);
    const char sort = 'N';
    CGEES_F77(CHAR_MACRO(JOBVS), CHAR_MACRO(sort), nullfptr, &n, A, &lda, sdim, &W[0], VS, &ldvs, WORK, &lwork, RWORK, BWORK, info);
    for (int i=0; i<n; i++) {
      WR[i] = W[i].real();
      WI[i] = W[i].imag();
    }
  }

  template<typename OrdinalType>
  void LAPACK<OrdinalType, std::complex<float> >::GEEV(const char JOBVL, const char JOBVR, const OrdinalType n, std::complex<float>* A, const OrdinalType lda, std::complex<float>* W, std::complex<float>* VL, const OrdinalType ldvl, std::complex<float>* VR, const OrdinalType ldvr, std::complex<float>* WORK, const OrdinalType lwork, float* RWORK, OrdinalType* info) const
  {
    CGEEV_F77(CHAR_MACRO(JOBVL), CHAR_MACRO(JOBVR), &n, A, &lda, W, VL, &ldvl, VR, &ldvr, WORK, &lwork, RWORK, info);
  }

/*
  template<typename OrdinalType>
  void LAPACK<OrdinalType, std::complex<float> >::GGEVX(const char BALANC, const char JOBVL, const char JOBVR, const char SENSE, const OrdinalType n, std::complex<float>* A, const OrdinalType lda, std::complex<float>* B, const OrdinalType ldb, float* ALPHAR, float* ALPHAI, std::complex<float>* BETA, std::complex<float>* VL, const OrdinalType ldvl, std::complex<float>* VR, const OrdinalType ldvr, OrdinalType* ilo, OrdinalType* ihi, float* LSCALE, float* RSCALE, float* abnrm, float* bbnrm, float* RCONDE, float* RCONDV, std::complex<float>* WORK, const OrdinalType lwork, OrdinalType* IWORK, OrdinalType* BWORK, OrdinalType* info) const
  {
    std::vector< std::complex<float> > ALPHA(n);
    CGGEVX_F77(CHAR_MACRO(BALANC), CHAR_MACRO(JOBVL), CHAR_MACRO(JOBVR), CHAR_MACRO(SENSE), &n, A, &lda, B, &ldb, &ALPHA[0], BETA, VL, &ldvl, VR, &ldvr, ilo, ihi, LSCALE, RSCALE, abnrm, bbnrm, RCONDE, RCONDV, WORK, &lwork, IWORK, BWORK, info);
    for (int i=0; i<n; i++) {
      ALPHAR[i] = ALPHA[i].real();
      ALPHAI[i] = ALPHA[i].imag();
    }
  }
*/  

  template<typename OrdinalType>
  void LAPACK<OrdinalType, std::complex<float> >::TREVC(const char SIDE, const char HOWMNY, OrdinalType (*ptr2func)(std::complex<float>*), const OrdinalType n, const std::complex<float>* T, const OrdinalType ldt, std::complex<float>* VL, const OrdinalType ldvl, std::complex<float>* VR, const OrdinalType ldvr, const OrdinalType mm, OrdinalType* m, std::complex<float>* WORK, float* RWORK, OrdinalType* info) const
  {
    CTREVC_F77(CHAR_MACRO(SIDE), CHAR_MACRO(HOWMNY), ptr2func, &n, T, &ldt, VL, &ldvl, VR, &ldvr, &mm, m, WORK, RWORK, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType, std::complex<float> >::TREVC(const char SIDE, const OrdinalType n, const std::complex<float>* T, const OrdinalType ldt, std::complex<float>* VL, const OrdinalType ldvl, std::complex<float>* VR, const OrdinalType ldvr, const OrdinalType mm, OrdinalType* m, std::complex<float>* WORK, float* RWORK, OrdinalType* info) const
  {
    OrdinalType (*nullfptr)(std::complex<float>*) = NULL;
    const char whch = 'A';
    CTREVC_F77(CHAR_MACRO(SIDE), CHAR_MACRO(whch), nullfptr, &n, T, &ldt, VL, &ldvl, VR, &ldvr, &mm, m, WORK, RWORK, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType, std::complex<float> >::TREXC(const char COMPQ, const OrdinalType n, std::complex<float>* T, const OrdinalType ldt, std::complex<float>* Q, const OrdinalType ldq, OrdinalType ifst, OrdinalType ilst, std::complex<float>* WORK, OrdinalType* info) const
  {
    CTREXC_F77(CHAR_MACRO(COMPQ), &n, T, &ldt, Q, &ldq, &ifst, &ilst, info);
  }

  template<typename OrdinalType>
  void LAPACK<OrdinalType, std::complex<float> >::LARTG( const std::complex<float> f, const std::complex<float> g, float* c, std::complex<float>* s, std::complex<float>* r ) const
  {
    CLARTG_F77(&f, &g, c, s, r);
  }

  template<typename OrdinalType>
  void LAPACK<OrdinalType, std::complex<float> >::LARFG( const OrdinalType n, std::complex<float>* alpha, std::complex<float>* x, const OrdinalType incx, std::complex<float>* tau ) const
  {
    CLARFG_F77(&n, alpha, x, &incx, tau);
  }

  template<typename OrdinalType>
  std::complex<float> LAPACK<OrdinalType, std::complex<float> >::LARND( const OrdinalType idist, OrdinalType* seed ) const
  {
    return(CLARND_F77(&idist, seed));
  }

  template<typename OrdinalType>
  void LAPACK<OrdinalType, std::complex<float> >::LARNV( const OrdinalType idist, OrdinalType* seed, const OrdinalType n, std::complex<float>* v ) const
  {
    CLARNV_F77(&idist, seed, &n, v);
  }

  template<typename OrdinalType>
  OrdinalType LAPACK<OrdinalType, std::complex<float> >::ILAENV( const OrdinalType ispec, const std::string& NAME, const std::string& OPTS, const OrdinalType N1, const OrdinalType N2, const OrdinalType N3, const OrdinalType N4 ) const
  {
    unsigned int opts_length = OPTS.length();
    std::string temp_NAME = "c" + NAME;
    unsigned int name_length = temp_NAME.length();
#if defined (INTEL_CXML)
    return ILAENV_F77(&ispec, &temp_NAME[0], name_length, &OPTS[0], opts_length, &N1, &N2, &N3, &N4 );
#else
    return ILAENV_F77(&ispec, &temp_NAME[0], &OPTS[0], &N1, &N2, &N3, &N4, name_length, opts_length );
#endif
  }

  // END COMPLEX<FLOAT> PARTIAL SPECIALIZATION IMPLEMENTATION //

#endif // HAVE_TEUCHOS_BLASFLOAT

  // BEGIN COMPLEX<DOUBLE> PARTIAL SPECIALIZATION DECLARATION //

  template<typename OrdinalType>
  class LAPACK<OrdinalType, std::complex<double> >
  {    
  public:
    inline LAPACK(void) {}
    inline LAPACK(const LAPACK<OrdinalType, std::complex<double> >& lapack) {}
    inline virtual ~LAPACK(void) {}

    // Symmetric positive definite linear system routines
    void PTTRF(const OrdinalType n, std::complex<double>* d, std::complex<double>* e, OrdinalType* info) const;
    void PTTRS(const OrdinalType n, const OrdinalType nrhs, const std::complex<double>* d, const std::complex<double>* e, std::complex<double>* B, const OrdinalType ldb, OrdinalType* info) const;
    void POTRF(const char UPLO, const OrdinalType n, std::complex<double>* A, const OrdinalType lda, OrdinalType* info) const;
    void POTRS(const char UPLO, const OrdinalType n, const OrdinalType nrhs, const std::complex<double>* A, const OrdinalType lda, std::complex<double>* B, const OrdinalType ldb, OrdinalType* info) const;
    void POTRI(const char UPLO, const OrdinalType n, std::complex<double>* A, const OrdinalType lda, OrdinalType* info) const;
    void POCON(const char UPLO, const OrdinalType n, const std::complex<double>* A, const OrdinalType lda, const double anorm, double* rcond, std::complex<double>* WORK, double* RWORK, OrdinalType* info) const;
    void POSV(const char UPLO, const OrdinalType n, const OrdinalType nrhs, std::complex<double>* A, const OrdinalType lda, std::complex<double>* B, const OrdinalType ldb, OrdinalType* info) const;
    void POEQU(const OrdinalType n, const std::complex<double>* A, const OrdinalType lda, double* S, double* scond, double* amax, OrdinalType* info) const;
    void PORFS(const char UPLO, const OrdinalType n, const OrdinalType nrhs, std::complex<double>* A, const OrdinalType lda, const std::complex<double>* AF, const OrdinalType ldaf, const std::complex<double>* B, const OrdinalType ldb, std::complex<double>* X, const OrdinalType ldx, double* FERR, double* BERR, std::complex<double>* WORK, double* RWORK, OrdinalType* info) const;
    void POSVX(const char FACT, const char UPLO, const OrdinalType n, const OrdinalType nrhs, std::complex<double>* A, const OrdinalType lda, std::complex<double>* AF, const OrdinalType ldaf, char EQUED, double* S, std::complex<double>* B, const OrdinalType ldb, std::complex<double>* X, const OrdinalType ldx, double* rcond, double* FERR, double* BERR, std::complex<double>* WORK, double* RWORK, OrdinalType* info) const; 

    // General Linear System Routines
    void GELS(const char TRANS, const OrdinalType m, const OrdinalType n, const OrdinalType nrhs, std::complex<double>* A, const OrdinalType lda, std::complex<double>* B, const OrdinalType ldb, std::complex<double>* WORK, const OrdinalType lwork, OrdinalType* info) const;
    void GEQRF( const OrdinalType m, const OrdinalType n, std::complex<double>* A, const OrdinalType lda, std::complex<double>* TAU, std::complex<double>* WORK, const OrdinalType lwork, OrdinalType* info) const;
    void UNGQR(const OrdinalType m, const OrdinalType n, const OrdinalType k, std::complex<double>* A, const OrdinalType lda, const std::complex<double>* TAU, std::complex<double>* WORK, const OrdinalType lwork, OrdinalType* info) const;
    void GETRF(const OrdinalType m, const OrdinalType n, std::complex<double>* A, const OrdinalType lda, OrdinalType* IPIV, OrdinalType* info) const;
    void GETRS(const char TRANS, const OrdinalType n, const OrdinalType nrhs, const std::complex<double>* A, const OrdinalType lda, const OrdinalType* IPIV, std::complex<double>* B, const OrdinalType ldb, OrdinalType* info) const;
    void GTTRF(const OrdinalType n, std::complex<double>* dl, std::complex<double>* d, std::complex<double>* du, std::complex<double>* du2, OrdinalType* IPIV, OrdinalType* info) const;
    void GTTRS(const char TRANS, const OrdinalType n, const OrdinalType nrhs, const std::complex<double>* dl, const std::complex<double>* d, const std::complex<double>* du, const std::complex<double>* du2, const OrdinalType* IPIV, std::complex<double>* B, const OrdinalType ldb, OrdinalType* info) const;
    void GETRI(const OrdinalType n, std::complex<double>* A, const OrdinalType lda, const OrdinalType* IPIV, std::complex<double>* WORK, const OrdinalType lwork, OrdinalType* info) const;
    void GECON(const char NORM, const OrdinalType n, const std::complex<double>* A, const OrdinalType lda, const double anorm, double* rcond, std::complex<double>* WORK, double* RWORK, OrdinalType* info) const;
    void GESV(const OrdinalType n, const OrdinalType nrhs, std::complex<double>* A, const OrdinalType lda, OrdinalType* IPIV, std::complex<double>* B, const OrdinalType ldb, OrdinalType* info) const;
    void GEEQU(const OrdinalType m, const OrdinalType n, const std::complex<double>* A, const OrdinalType lda, double* R, double* C, double* rowcond, double* colcond, double* amax, OrdinalType* info) const;
    void GERFS(const char TRANS, const OrdinalType n, const OrdinalType nrhs, const std::complex<double>* A, const OrdinalType lda, const std::complex<double>* AF, const OrdinalType ldaf, const OrdinalType* IPIV, const std::complex<double>* B, const OrdinalType ldb, std::complex<double>* X, const OrdinalType ldx, double* FERR, double* BERR, std::complex<double>* WORK, double* RWORK, OrdinalType* info) const;
    void GESVX(const char FACT, const char TRANS, const OrdinalType n, const OrdinalType nrhs, std::complex<double>* A, const OrdinalType lda, std::complex<double>* AF, const OrdinalType ldaf, OrdinalType* IPIV, char EQUED, double* R, double* C, std::complex<double>* B, const OrdinalType ldb, std::complex<double>* X, const OrdinalType ldx, double* rcond, double* FERR, double* BERR, std::complex<double>* WORK, double* RWORK, OrdinalType* info) const;
    void GEHRD(const OrdinalType n, const OrdinalType ilo, const OrdinalType ihi, std::complex<double>* A, const OrdinalType lda, std::complex<double>* TAU, std::complex<double>* WORK, const OrdinalType lwork, OrdinalType* info) const;
    void TRTRS(const char UPLO, const char TRANS, const char DIAG, const OrdinalType n, const OrdinalType nrhs, const std::complex<double>* A, const OrdinalType lda, std::complex<double>* B, const OrdinalType ldb, OrdinalType* info) const; 

    // Symmetric eigenvalue routines.
    void STEQR(const char COMPZ, const OrdinalType n, double* D, double* E, std::complex<double>* Z, const OrdinalType ldz, double* WORK, OrdinalType* info) const;
    void HEEV(const char JOBZ, const char UPLO, const OrdinalType n, std::complex<double>* A, const OrdinalType lda, double* W, std::complex<double>* WORK, const OrdinalType lwork, double* RWORK, OrdinalType* info) const;
    void HEGV(const OrdinalType itype, const char JOBZ, const char UPLO, const OrdinalType n, std::complex<double>* A, const OrdinalType lda, std::complex<double>* B, const OrdinalType ldb, double* W, std::complex<double>* WORK, const OrdinalType lwork, double *RWORK, OrdinalType* info) const;

    // Non-hermitian eigenvalue routines.
    void HSEQR(const char JOB, const char COMPZ, const OrdinalType n, const OrdinalType ilo, const OrdinalType ihi, std::complex<double>* H, const OrdinalType ldh, std::complex<double>* W, std::complex<double>* Z, const OrdinalType ldz, std::complex<double>* WORK, const OrdinalType lwork, OrdinalType* info) const;
    void GEES(const char JOBVS, const char SORT, OrdinalType (*ptr2func)(std::complex<double>*), const OrdinalType n, std::complex<double>* A, const OrdinalType lda, OrdinalType* sdim, std::complex<double>* W, std::complex<double>* VS, const OrdinalType ldvs, std::complex<double>* WORK, const OrdinalType lwork, double* RWORK, OrdinalType* BWORK, OrdinalType* info) const;    
    void GEES(const char JOBVS, const OrdinalType n, std::complex<double>* A, const OrdinalType lda, OrdinalType* sdim, double* WR, double* WI, std::complex<double>* VS, const OrdinalType ldvs, std::complex<double>* WORK, const OrdinalType lwork, double* RWORK, OrdinalType* BWORK, OrdinalType* info) const;    
    void GEEV(const char JOBVL, const char JOBVR, const OrdinalType n, std::complex<double>* A, const OrdinalType lda, std::complex<double>* W, std::complex<double>* VL, const OrdinalType ldvl, std::complex<double>* VR, const OrdinalType ldvr, std::complex<double>* WORK, const OrdinalType lwork, double* RWORK, OrdinalType* info) const;
//    void GGEVX(const char BALANC, const char JOBVL, const char JOBVR, const char SENSE, const OrdinalType n, std::complex<double>* A, const OrdinalType lda, std::complex<double>* B, const OrdinalType ldb, double* ALPHAR, double* ALPHAI, std::complex<double>* BETA, std::complex<double>* VL, const OrdinalType ldvl, std::complex<double>* VR, const OrdinalType ldvr, OrdinalType* ilo, OrdinalType* ihi, double* LSCALE, double* RSCALE, double* abnrm, double* bbnrm, double* RCONDE, double* RCONDV, std::complex<double>* work, const OrdinalType lwork, OrdinalType* IWORK, OrdinalType* BWORK, OrdinalType* info) const;

    // Triangular matrix routines.
    void TREVC(const char SIDE, const char HOWMNY, OrdinalType (*ptr2func)(std::complex<double>*), const OrdinalType n, const std::complex<double>* T, const OrdinalType ldt, std::complex<double>* VL, const OrdinalType ldvl, std::complex<double>* VR, const OrdinalType ldvr, const OrdinalType mm, OrdinalType* m, std::complex<double>* WORK, double* RWORK, OrdinalType* info) const;
    void TREVC(const char SIDE, const OrdinalType n, const std::complex<double>* T, const OrdinalType ldt, std::complex<double>* VL, const OrdinalType ldvl, std::complex<double>* VR, const OrdinalType ldvr, const OrdinalType mm, OrdinalType* m, std::complex<double>* WORK, double* RWORK, OrdinalType* info) const;
    void TREXC(const char COMPQ, const OrdinalType n, std::complex<double>* T, const OrdinalType ldt, std::complex<double>* Q, const OrdinalType ldq, OrdinalType ifst, OrdinalType ilst, std::complex<double>* WORK, OrdinalType* info) const;

    // Rotation/reflection generators
    void LARTG( const std::complex<double> f, const std::complex<double> g, double* c, std::complex<double>* s, std::complex<double>* r ) const;
    void LARFG( const OrdinalType n, std::complex<double>* alpha, std::complex<double>* x, const OrdinalType incx, std::complex<double>* tau ) const;

    // Random number generators
    std::complex<double> LARND( const OrdinalType idist, OrdinalType* seed ) const;
    void LARNV( const OrdinalType idist, OrdinalType* seed, const OrdinalType n, std::complex<double>* v ) const;    

    // Machine characteristics
    OrdinalType ILAENV( const OrdinalType ispec, const std::string& NAME, const std::string& OPTS, const OrdinalType N1 = -1, const OrdinalType N2 = -1, const OrdinalType N3 = -1, const OrdinalType N4 = -1 ) const;

  };

  // END COMPLEX<DOUBLE> PARTIAL SPECIALIZATION DECLARATION //

  // BEGIN COMPLEX<DOUBLE> PARTIAL SPECIALIZATION IMPLEMENTATION //

  template<typename OrdinalType>
  void LAPACK<OrdinalType, std::complex<double> >::PTTRF(const OrdinalType n, std::complex<double>* d, std::complex<double>* e, OrdinalType* info) const
  {
    ZPTTRF_F77(&n,d,e,info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType, std::complex<double> >::PTTRS(const OrdinalType n, const OrdinalType nrhs, const std::complex<double>* d, const std::complex<double>* e, std::complex<double>* B, const OrdinalType ldb, OrdinalType* info) const
  {
    ZPTTRS_F77(&n,&nrhs,d,e,B,&ldb,info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType, std::complex<double> >::POTRF(const char UPLO, const OrdinalType n, std::complex<double>* A, const OrdinalType lda, OrdinalType* info) const
  {
    ZPOTRF_F77(CHAR_MACRO(UPLO), &n, A, &lda, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType, std::complex<double> >::POTRS(const char UPLO, const OrdinalType n, const OrdinalType nrhs, const std::complex<double>* A, const OrdinalType lda, std::complex<double>* B, const OrdinalType ldb, OrdinalType* info) const
  {
    ZPOTRS_F77(CHAR_MACRO(UPLO), &n, &nrhs, A, &lda, B, &ldb, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType, std::complex<double> >::POTRI(const char UPLO, const OrdinalType n, std::complex<double>* A, const OrdinalType lda, OrdinalType* info) const
  {
    ZPOTRI_F77(CHAR_MACRO(UPLO), &n, A, &lda, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType, std::complex<double> >::POCON(const char UPLO, const OrdinalType n, const std::complex<double>* A, const OrdinalType lda, const double anorm, double* rcond, std::complex<double>* WORK, double* RWORK, OrdinalType* info) const
  {
    ZPOCON_F77(CHAR_MACRO(UPLO), &n, A, &lda, &anorm, rcond, WORK, RWORK, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType, std::complex<double> >::POSV(const char UPLO, const OrdinalType n, const OrdinalType nrhs, std::complex<double>* A, const OrdinalType lda, std::complex<double>* B, const OrdinalType ldb, OrdinalType* info) const
  {
    ZPOSV_F77(CHAR_MACRO(UPLO), &n, &nrhs, A, &lda, B, &ldb, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType, std::complex<double> >::POEQU(const OrdinalType n, const std::complex<double>* A, const OrdinalType lda, double* S, double* scond, double* amax, OrdinalType* info) const
  {
    ZPOEQU_F77(&n, A, &lda, S, scond, amax, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType, std::complex<double> >::PORFS(const char UPLO, const OrdinalType n, const OrdinalType nrhs, std::complex<double>* A, const OrdinalType lda, const std::complex<double>* AF, const OrdinalType ldaf, const std::complex<double>* B, const OrdinalType ldb, std::complex<double>* X, const OrdinalType ldx, double* FERR, double* BERR, std::complex<double>* WORK, double* RWORK, OrdinalType* info) const
  {
    ZPORFS_F77(CHAR_MACRO(UPLO), &n, &nrhs, A, &lda, AF, &ldaf, B, &ldb, X, &ldx, FERR, BERR, WORK, RWORK, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType, std::complex<double> >::POSVX(const char FACT, const char UPLO, const OrdinalType n, const OrdinalType nrhs, std::complex<double>* A, const OrdinalType lda, std::complex<double>* AF, const OrdinalType ldaf, char EQUED, double* S, std::complex<double>* B, const OrdinalType ldb, std::complex<double>* X, const OrdinalType ldx, double* rcond, double* FERR, double* BERR, std::complex<double>* WORK, double* RWORK, OrdinalType* info) const
  {
    ZPOSVX_F77(CHAR_MACRO(FACT), CHAR_MACRO(UPLO), &n, &nrhs, A, &lda, AF, &ldaf, CHAR_MACRO(EQUED), S, B, &ldb, X, &ldx, rcond, FERR, BERR, WORK, RWORK, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType,std::complex<double> >::GELS(const char TRANS, const OrdinalType m, const OrdinalType n, const OrdinalType nrhs, std::complex<double>* A, const OrdinalType lda, std::complex<double>* B, const OrdinalType ldb, std::complex<double>* WORK, const OrdinalType lwork, OrdinalType* info) const
  {
    ZGELS_F77(CHAR_MACRO(TRANS), &m, &n, &nrhs, A, &lda, B, &ldb, WORK, &lwork, info);
  }
  
  template<typename OrdinalType>  
  void LAPACK<OrdinalType,std::complex<double> >::GEQRF( const OrdinalType m, const OrdinalType n, std::complex<double>* A, const OrdinalType lda, std::complex<double>* TAU, std::complex<double>* WORK, const OrdinalType lwork, OrdinalType* info) const
  {
    ZGEQRF_F77(&m, &n, A, &lda, TAU, WORK, &lwork, info);
  }

  template<typename OrdinalType>
  void LAPACK<OrdinalType,std::complex<double> >::UNGQR(const OrdinalType m, const OrdinalType n, const OrdinalType k, std::complex<double>* A, const OrdinalType lda, const std::complex<double>* TAU, std::complex<double>* WORK, const OrdinalType lwork, OrdinalType* info) const
  {
    ZUNGQR_F77( &m, &n, &k, A, &lda, TAU, WORK, &lwork, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType,std::complex<double> >::GETRF(const OrdinalType m, const OrdinalType n, std::complex<double>* A, const OrdinalType lda, OrdinalType* IPIV, OrdinalType* info) const
  {
    ZGETRF_F77(&m, &n, A, &lda, IPIV, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType,std::complex<double> >::GETRS(const char TRANS, const OrdinalType n, const OrdinalType nrhs, const std::complex<double>* A, const OrdinalType lda, const OrdinalType* IPIV, std::complex<double>* B, const OrdinalType ldb, OrdinalType* info) const
  {
    ZGETRS_F77(CHAR_MACRO(TRANS), &n, &nrhs, A, &lda, IPIV, B, &ldb, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType,std::complex<double> >::GTTRF(const OrdinalType n, std::complex<double>* dl, std::complex<double>* d, std::complex<double>* du, std::complex<double>* du2, OrdinalType* IPIV, OrdinalType* info) const
  {
    ZGTTRF_F77(&n, dl, d, du, du2, IPIV, info);
  }

  template<typename OrdinalType>
  void LAPACK<OrdinalType,std::complex<double> >::GTTRS(const char TRANS, const OrdinalType n, const OrdinalType nrhs, const std::complex<double>* dl, const std::complex<double>* d, const std::complex<double>* du, const std::complex<double>* du2, const OrdinalType* IPIV, std::complex<double>* B, const OrdinalType ldb, OrdinalType* info) const
  {
    ZGTTRS_F77(CHAR_MACRO(TRANS), &n, &nrhs, dl, d, du, du2, IPIV, B, &ldb, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType,std::complex<double> >::GETRI(const OrdinalType n, std::complex<double>* A, const OrdinalType lda, const OrdinalType* IPIV, std::complex<double>* WORK, const OrdinalType lwork, OrdinalType* info) const
  {
    ZGETRI_F77(&n, A, &lda, IPIV, WORK, &lwork, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType,std::complex<double> >::GECON(const char NORM, const OrdinalType n, const std::complex<double>* A, const OrdinalType lda, const double anorm, double* rcond, std::complex<double>* WORK, double* RWORK, OrdinalType* info) const
  {
    ZGECON_F77(CHAR_MACRO(NORM), &n, A, &lda, &anorm, rcond, WORK, RWORK, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType,std::complex<double> >::GESV(const OrdinalType n, const OrdinalType nrhs, std::complex<double>* A, const OrdinalType lda, OrdinalType* IPIV, std::complex<double>* B, const OrdinalType ldb, OrdinalType* info) const
  {
    ZGESV_F77(&n, &nrhs, A, &lda, IPIV, B, &ldb, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType,std::complex<double> >::GEEQU(const OrdinalType m, const OrdinalType n, const std::complex<double>* A, const OrdinalType lda, double* R, double* C, double* rowcond, double* colcond, double* amax, OrdinalType* info) const
  {
    ZGEEQU_F77(&m, &n, A, &lda, R, C, rowcond, colcond, amax, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType,std::complex<double> >::GERFS(const char TRANS, const OrdinalType n, const OrdinalType nrhs, const std::complex<double>* A, const OrdinalType lda, const std::complex<double>* AF, const OrdinalType ldaf, const OrdinalType* IPIV, const std::complex<double>* B, const OrdinalType ldb, std::complex<double>* X, const OrdinalType ldx, double* FERR, double* BERR, std::complex<double>* WORK, double* RWORK, OrdinalType* info) const
  {
    ZGERFS_F77(CHAR_MACRO(TRANS), &n, &nrhs, A, &lda, AF, &ldaf, IPIV, B, &ldb, X, &ldx, FERR, BERR, WORK, RWORK, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType,std::complex<double> >::GESVX(const char FACT, const char TRANS, const OrdinalType n, const OrdinalType nrhs, std::complex<double>* A, const OrdinalType lda, std::complex<double>* AF, const OrdinalType ldaf, OrdinalType* IPIV, char EQUED, double* R, double* C, std::complex<double>* B, const OrdinalType ldb, std::complex<double>* X, const OrdinalType ldx, double* rcond, double* FERR, double* BERR, std::complex<double>* WORK, double* RWORK, OrdinalType* info) const
  {
    ZGESVX_F77(CHAR_MACRO(FACT), CHAR_MACRO(TRANS), &n, &nrhs, A, &lda, AF, &ldaf, IPIV, CHAR_MACRO(EQUED), R, C, B, &ldb, X, &ldx, rcond, FERR, BERR, WORK, RWORK, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType,std::complex<double> >::GEHRD(const OrdinalType n, const OrdinalType ilo, const OrdinalType ihi, std::complex<double>* A, const OrdinalType lda, std::complex<double>* TAU, std::complex<double>* WORK, const OrdinalType lwork, OrdinalType* info) const
  {
    ZGEHRD_F77(&n, &ilo, &ihi, A, &lda, TAU, WORK, &lwork, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType,std::complex<double> >::TRTRS(const char UPLO, const char TRANS, const char DIAG, const OrdinalType n, const OrdinalType nrhs, const std::complex<double>* A, const OrdinalType lda, std::complex<double>* B, const OrdinalType ldb, OrdinalType* info) const 
  {
    ZTRTRS_F77(CHAR_MACRO(UPLO), CHAR_MACRO(TRANS), CHAR_MACRO(DIAG), &n, &nrhs, A, &lda, B, &ldb, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType,std::complex<double> >::STEQR(const char COMPZ, const OrdinalType n, double* D, double* E, std::complex<double>* Z, const OrdinalType ldz, double* WORK, OrdinalType* info) const
  {
    ZSTEQR_F77(CHAR_MACRO(COMPZ), &n, D, E, Z, &ldz, WORK, info);
  }

  template<typename OrdinalType>
  void LAPACK<OrdinalType,std::complex<double> >::HEEV(const char JOBZ, const char UPLO, const OrdinalType n, std::complex<double> * A, const OrdinalType lda, double * W, std::complex<double> * WORK, const OrdinalType lwork, double* RWORK, OrdinalType* info) const
  {
    ZHEEV_F77(CHAR_MACRO(JOBZ), CHAR_MACRO(UPLO), &n, A, &lda, W, WORK, &lwork, RWORK, info);
  }

  template<typename OrdinalType>
  void LAPACK<OrdinalType,std::complex<double> >::HEGV(const OrdinalType itype, const char JOBZ, const char UPLO, const OrdinalType n, std::complex<double> * A, const OrdinalType lda, std::complex<double> * B, const OrdinalType ldb, double * W, std::complex<double> * WORK, const OrdinalType lwork, double *RWORK, OrdinalType* info) const
  {
    ZHEGV_F77(&itype, CHAR_MACRO(JOBZ), CHAR_MACRO(UPLO), &n, A, &lda, B, &ldb, W, WORK, &lwork, RWORK, info);
  }

  template<typename OrdinalType>
  void LAPACK<OrdinalType, std::complex<double> >::HSEQR(const char JOB, const char COMPZ, const OrdinalType n, const OrdinalType ilo, const OrdinalType ihi, std::complex<double>* H, const OrdinalType ldh, std::complex<double>* W, std::complex<double>* Z, const OrdinalType ldz, std::complex<double>* WORK, const OrdinalType lwork, OrdinalType* info) const
  {
    ZHSEQR_F77(CHAR_MACRO(JOB), CHAR_MACRO(COMPZ), &n, &ilo, &ihi, H, &ldh, W, Z, &ldz, WORK, &lwork, info);
  }

  template<typename OrdinalType>
  void LAPACK<OrdinalType, std::complex<double> >::GEES(const char JOBVS, const char SORT, OrdinalType (*ptr2func)(std::complex<double>*), const OrdinalType n, std::complex<double>* A, const OrdinalType lda, OrdinalType* sdim, std::complex<double>* W, std::complex<double>* VS, const OrdinalType ldvs, std::complex<double>* WORK, const OrdinalType lwork, double* RWORK, OrdinalType* BWORK, OrdinalType* info) const    
  {
    ZGEES_F77(CHAR_MACRO(JOBVS), CHAR_MACRO(SORT), ptr2func, &n, A, &lda, sdim, W, VS, &ldvs, WORK, &lwork, RWORK, BWORK, info);
  }

  template<typename OrdinalType>
  void LAPACK<OrdinalType, std::complex<double> >::GEES(const char JOBVS, const OrdinalType n, std::complex<double>* A, const OrdinalType lda, OrdinalType* sdim, double* WR, double* WI, std::complex<double>* VS, const OrdinalType ldvs, std::complex<double>* WORK, const OrdinalType lwork, double* RWORK, OrdinalType* BWORK, OrdinalType* info) const    
  {
    OrdinalType (*nullfptr)(std::complex<double>*) = NULL;
    std::vector< std::complex<double> > W(n);
    const char sort = 'N';
    ZGEES_F77(CHAR_MACRO(JOBVS), CHAR_MACRO(sort), nullfptr, &n, A, &lda, sdim, &W[0], VS, &ldvs, WORK, &lwork, RWORK, BWORK, info);
    for (int i=0; i<n; i++) {
      WR[i] = W[i].real();
      WI[i] = W[i].imag();
    }
  }

  template<typename OrdinalType>
  void LAPACK<OrdinalType, std::complex<double> >::GEEV(const char JOBVL, const char JOBVR, const OrdinalType n, std::complex<double>* A, const OrdinalType lda, std::complex<double>* W, std::complex<double>* VL, const OrdinalType ldvl, std::complex<double>* VR, const OrdinalType ldvr, std::complex<double>* WORK, const OrdinalType lwork, double* RWORK, OrdinalType* info) const
  {
    ZGEEV_F77(CHAR_MACRO(JOBVL), CHAR_MACRO(JOBVR), &n, A, &lda, W, VL, &ldvl, VR, &ldvr, WORK, &lwork, RWORK, info);
  }

/*
  template<typename OrdinalType>
  void LAPACK<OrdinalType, std::complex<double> >::GGEVX(const char BALANC, const char JOBVL, const char JOBVR, const char SENSE, const OrdinalType n, std::complex<double>* A, const OrdinalType lda, std::complex<double>* B, const OrdinalType ldb, double* ALPHAR, double* ALPHAI, std::complex<double>* BETA, std::complex<double>* VL, const OrdinalType ldvl, std::complex<double>* VR, const OrdinalType ldvr, OrdinalType* ilo, OrdinalType* ihi, double* LSCALE, double* RSCALE, double* abnrm, double* bbnrm, double* RCONDE, double* RCONDV, std::complex<double>* WORK, const OrdinalType lwork, OrdinalType* IWORK, OrdinalType* BWORK, OrdinalType* info) const
  {
    std::vector< std::complex<double> > ALPHA(n);
    ZGGEVX_F77(CHAR_MACRO(BALANC), CHAR_MACRO(JOBVL), CHAR_MACRO(JOBVR), CHAR_MACRO(SENSE), &n, A, &lda, B, &ldb, &ALPHA[0], BETA, VL, &ldvl, VR, &ldvr, ilo, ihi, LSCALE, RSCALE, abnrm, bbnrm, RCONDE, RCONDV, WORK, &lwork, IWORK, BWORK, info);
    for (int i=0; i<n; i++) {
      ALPHAR[i] = ALPHA[i].real();
      ALPHAI[i] = ALPHA[i].imag();
    }
  }
*/

  template<typename OrdinalType>
  void LAPACK<OrdinalType, std::complex<double> >::TREVC(const char SIDE, const char HOWMNY, OrdinalType (*ptr2func)(std::complex<double>*), const OrdinalType n, const std::complex<double>* T, const OrdinalType ldt, std::complex<double>* VL, const OrdinalType ldvl, std::complex<double>* VR, const OrdinalType ldvr, const OrdinalType mm, OrdinalType* m, std::complex<double>* WORK, double* RWORK, OrdinalType* info) const
  {
    ZTREVC_F77(CHAR_MACRO(SIDE), CHAR_MACRO(HOWMNY), ptr2func, &n, T, &ldt, VL, &ldvl, VR, &ldvr, &mm, m, WORK, RWORK, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType, std::complex<double> >::TREVC(const char SIDE, const OrdinalType n, const std::complex<double>* T, const OrdinalType ldt, std::complex<double>* VL, const OrdinalType ldvl, std::complex<double>* VR, const OrdinalType ldvr, const OrdinalType mm, OrdinalType* m, std::complex<double>* WORK, double* RWORK, OrdinalType* info) const
  {
    OrdinalType (*nullfptr)(std::complex<double>*) = NULL;
    const char whch = 'A';
    ZTREVC_F77(CHAR_MACRO(SIDE), CHAR_MACRO(whch), nullfptr, &n, T, &ldt, VL, &ldvl, VR, &ldvr, &mm, m, WORK, RWORK, info);
  }
  
  template<typename OrdinalType>
  void LAPACK<OrdinalType, std::complex<double> >::TREXC(const char COMPQ, const OrdinalType n, std::complex<double>* T, const OrdinalType ldt, std::complex<double>* Q, const OrdinalType ldq, OrdinalType ifst, OrdinalType ilst, std::complex<double>* WORK, OrdinalType* info) const
  {
    ZTREXC_F77(CHAR_MACRO(COMPQ), &n, T, &ldt, Q, &ldq, &ifst, &ilst, info);
  }

  template<typename OrdinalType>
  void LAPACK<OrdinalType, std::complex<double> >::LARTG( const std::complex<double> f, const std::complex<double> g, double* c, std::complex<double>* s, std::complex<double>* r ) const
  {
    ZLARTG_F77(&f, &g, c, s, r);
  }

  template<typename OrdinalType>
  void LAPACK<OrdinalType, std::complex<double> >::LARFG( const OrdinalType n, std::complex<double>* alpha, std::complex<double>* x, const OrdinalType incx, std::complex<double>* tau ) const
  {
    ZLARFG_F77(&n, alpha, x, &incx, tau);
  }

  template<typename OrdinalType>
  std::complex<double> LAPACK<OrdinalType, std::complex<double> >::LARND( const OrdinalType idist, OrdinalType* seed ) const
  {
    return(ZLARND_F77(&idist, seed));
  }

  template<typename OrdinalType>
  void LAPACK<OrdinalType, std::complex<double> >::LARNV( const OrdinalType idist, OrdinalType* seed, const OrdinalType n, std::complex<double>* v ) const
  {
    ZLARNV_F77(&idist, seed, &n, v);
  }

  template<typename OrdinalType>
  OrdinalType LAPACK<OrdinalType, std::complex<double> >::ILAENV( const OrdinalType ispec, const std::string& NAME, const std::string& OPTS, const OrdinalType N1, const OrdinalType N2, const OrdinalType N3, const OrdinalType N4 ) const
  {
    unsigned int opts_length = OPTS.length();
    std::string temp_NAME = "z" + NAME;
    unsigned int name_length = temp_NAME.length();
#if defined (INTEL_CXML)
    return ILAENV_F77(&ispec, &temp_NAME[0], name_length, &OPTS[0], opts_length, &N1, &N2, &N3, &N4 );
#else
    return ILAENV_F77(&ispec, &temp_NAME[0], &OPTS[0], &N1, &N2, &N3, &N4, name_length, opts_length );
#endif
  }

  // END COMPLEX<DOUBLE> PARTIAL SPECIALIZATION IMPLEMENTATION //

#endif // HAVE_TEUCHOS_COMPLEX

#endif // DOXYGEN_SHOULD_SKIP_THIS

} // namespace Teuchos

#endif // _TEUCHOS_LAPACK_HPP_

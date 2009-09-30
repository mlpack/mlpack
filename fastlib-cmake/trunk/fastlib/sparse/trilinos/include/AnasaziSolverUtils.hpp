// @HEADER
// ***********************************************************************
//
//                 Anasazi: Block Eigensolvers Package
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

#ifndef ANASAZI_SOLVER_UTILS_HPP
#define ANASAZI_SOLVER_UTILS_HPP

/*!     \file AnasaziSolverUtils.hpp
        \brief Class which provides internal utilities for the Anasazi solvers.
*/

/*!    \class Anasazi::SolverUtils
       \brief Anasazi's templated, static class providing utilities for
       the solvers.

       This class provides concrete, templated implementations of utilities necessary
       for the solvers.  These utilities include
       sorting, orthogonalization, projecting/solving local eigensystems, and sanity
       checking.  These are internal utilties, so the user should not alter this class.

       \author Ulrich Hetmaniuk, Rich Lehoucq, and Heidi Thornquist
*/

#include "AnasaziConfigDefs.hpp"
#include "AnasaziMultiVecTraits.hpp"
#include "AnasaziOperatorTraits.hpp"
#include "Teuchos_ScalarTraits.hpp"

#include "AnasaziOutputManager.hpp"
#include "Teuchos_BLAS.hpp"
#include "Teuchos_LAPACK.hpp"
#include "Teuchos_SerialDenseMatrix.hpp"

namespace Anasazi {

  template<class ScalarType, class MV, class OP>
  class SolverUtils 
  {  
  public:
    typedef typename Teuchos::ScalarTraits<ScalarType>::magnitudeType MagnitudeType;
    typedef typename Teuchos::ScalarTraits<ScalarType>  SCT;
    
    //! @name Constructor/Destructor
    //@{ 

    //! Constructor.  
    SolverUtils();
    
    //! Destructor.
    virtual ~SolverUtils() {};
    
    //@}
    
    //! @name Sorting Methods
    //@{ 
    
    //! Permute the vectors in a multivector according to the permutation vector \c perm, and optionally the residual vector \c resids
    static void permuteVectors(const int n, const std::vector<int> &perm, MV &Q, std::vector< typename Teuchos::ScalarTraits<ScalarType>::magnitudeType >* resids = 0);

    //! Permute the columns of a Teuchos::SerialDenseMatrix according to the permutation vector \c perm
    static void permuteVectors(const std::vector<int> &perm, Teuchos::SerialDenseMatrix<int,ScalarType> &Q);

    //@} 

    //! @name Basis update methods
    //@{

    //! Apply a sequence of Householder reflectors (from \c GEQRF) to a multivector, using minimal workspace.
    /*! 
      @param k [in] the number of Householder reflectors composing the product
      @param V [in/out] the multivector to be modified, with \f$n\f$ columns
      @param H [in] a \f$n \times k\f$ matrix containing the encoded Householder vectors, as returned from \c GEQRF (see below)
      @param tau [in] the \f$n\f$ coefficients for the Householder reflects, as returned from \c GEQRF
      @param workMV [work] (optional) a multivector used for workspace. it need contain only a single vector; it if contains more, only the first vector will be modified.

      This routine applies a sequence of Householder reflectors, \f$H_1 H_2 \cdots H_k\f$, to a multivector \f$V\f$. The 
      reflectors are applied individually, as rank-one updates to the multivector. The benefit of this is that the only 
      required workspace is a one-column multivector. This workspace can be provided by the user. If it is not, it will
      be allocated locally on each call to applyHouse.

      Each \f$H_i\f$ (\f$i=1,\ldots,k \leq n\f$) has the form<br>
      \f$ H_i = I - \tau_i v_i v_i^T \f$ <br>
      where \f$\tau_i\f$ is a scalar and \f$v_i\f$ is a vector with
      \f$v_i(1:i-1) = 0\f$ and \f$e_i^T v_i = 1\f$; \f$v(i+1:n)\f$ is stored below <tt>H(i,i)</tt>
      and \f$\tau_i\f$ in <tt>tau[i-1]</tt>. (Note: zero-based indexing used for data structures \c H and \c tau, while one-based indexing used for mathematic object \f$v_i\f$).

      If the multivector is \f$m \times n\f$ and we apply \f$k\f$ Householder reflectors, the total cost of the method is
      \f$4mnk - 2m(k^2-k)\f$ flops. For \f$k=n\f$, this becomes \f$2mn^2\f$, the same as for a matrix-matrix multiplication by the accumulated Householder reflectors.
     */
    static void applyHouse(int k, MV &V, const Teuchos::SerialDenseMatrix<int,ScalarType> &H, const std::vector<ScalarType> &tau, Teuchos::RCP<MV> workMV = Teuchos::null);

    //@}

    //! @name Eigensolver Projection Methods
    //@{ 

    //! Routine for computing the first NEV generalized eigenpairs of the Hermitian pencil <tt>(KK, MM)</tt>
    /*!
      @param size [in] Dimension of the eigenproblem (KK, MM)
      @param KK [in] Hermitian "stiffness" matrix 
      @param MM [in] Hermitian positive-definite "mass" matrix
      @param EV [in] Dense matrix to store the nev eigenvectors 
      @param theta [in] Array to store the eigenvalues (Size = nev )
      @param nev [in/out] Number of the smallest eigenvalues requested (in) / computed (out)
      @param esType [in] Flag to select the algorithm
      <ul>
      <li> esType =  0  (default) Uses LAPACK routine (Cholesky factorization of MM)
                        with deflation of MM to get orthonormality of 
                        eigenvectors (\f$S^TMMS = I\f$)
      <li> esType =  1  Uses LAPACK routine (Cholesky factorization of MM)
                        (no check of orthonormality)
      <li> esType = 10  Uses LAPACK routine for simple eigenproblem on KK
                        (MM is not referenced in this case)
      </ul>

      \note The code accesses only the upper triangular part of KK and MM.
      \return Integer \c info on the status of the computation
      // Return the integer info on the status of the computation
      <ul>
      <li> info = 0 >> Success
      <li> info = - 20 >> Failure in LAPACK routine
      </ul>
    */
    static int directSolver(int size, const Teuchos::SerialDenseMatrix<int,ScalarType> &KK, 
                     Teuchos::RCP<const Teuchos::SerialDenseMatrix<int,ScalarType> > MM,
                     Teuchos::SerialDenseMatrix<int,ScalarType> &EV,
                     std::vector< typename Teuchos::ScalarTraits<ScalarType>::magnitudeType > &theta,
                     int &nev, int esType = 0);
    //@}

    //! @name Sanity Checking Methods
    //@{ 

    //! Return the maximum coefficient of the matrix \f$M * X - MX\f$ scaled by the maximum coefficient of \c MX.
    /*! \note When \c M is not specified, the identity is used.
     */
    static typename Teuchos::ScalarTraits<ScalarType>::magnitudeType errorEquality(const MV &X, const MV &MX, Teuchos::RCP<const OP> M = Teuchos::null);
    
    //@}
    
  private:

    //! @name Internal Typedefs
    //@{ 

    typedef MultiVecTraits<ScalarType,MV> MVT;
    typedef OperatorTraits<ScalarType,MV,OP> OPT;

    //@}
  };

  //-----------------------------------------------------------------------------
  // 
  //  CONSTRUCTOR
  //
  //-----------------------------------------------------------------------------  

  template<class ScalarType, class MV, class OP>
  SolverUtils<ScalarType, MV, OP>::SolverUtils() {}


  //-----------------------------------------------------------------------------
  // 
  //  SORTING METHODS
  //
  //-----------------------------------------------------------------------------
  
  //////////////////////////////////////////////////////////////////////////
  // permuteVectors for MV
  template<class ScalarType, class MV, class OP>
  void SolverUtils<ScalarType, MV, OP>::permuteVectors(
              const int n,
              const std::vector<int> &perm, 
              MV &Q, 
              std::vector< typename Teuchos::ScalarTraits<ScalarType>::magnitudeType >* resids)
  {
    // Permute the vectors according to the permutation vector \c perm, and
    // optionally the residual vector \c resids
    
    int i, j;
    std::vector<int> permcopy(perm), swapvec(n-1);
    std::vector<int> index(1);
    ScalarType one = Teuchos::ScalarTraits<ScalarType>::one();
    ScalarType zero = Teuchos::ScalarTraits<ScalarType>::zero();

    TEST_FOR_EXCEPTION(n > MVT::GetNumberVecs(Q), std::invalid_argument, "Anasazi::SolverUtils::permuteVectors(): argument n larger than width of input multivector.");

    // We want to recover the elementary permutations (individual swaps) 
    // from the permutation vector. Do this by constructing the inverse
    // of the permutation, by sorting them to {1,2,...,n}, and recording
    // the elementary permutations of the inverse.
    for (i=0; i<n-1; i++) {
      //
      // find i in the permcopy vector
      for (j=i; j<n; j++) {
        if (permcopy[j] == i) {
          // found it at index j
          break;
        }
        TEST_FOR_EXCEPTION(j == n-1, std::invalid_argument, "Anasazi::SolverUtils::permuteVectors(): permutation index invalid.");
      }
      //
      // Swap two scalars
      std::swap<int>( permcopy[j], permcopy[i] );

      swapvec[i] = j;
    }
      
    // now apply the elementary permutations of the inverse in reverse order
    for (i=n-2; i>=0; i--) {
      j = swapvec[i];
      //
      // Swap (i,j)
      //
      // Swap residuals (if they exist)
      if (resids) {
        std::swap<MagnitudeType>(  (*resids)[i], (*resids)[j] );
      }
      //
      // Swap corresponding vectors
      index[0] = j;
      Teuchos::RCP<MV> tmpQ = MVT::CloneCopy( Q, index );
      Teuchos::RCP<MV> tmpQj = MVT::CloneView( Q, index );
      index[0] = i;
      Teuchos::RCP<MV> tmpQi = MVT::CloneView( Q, index );
      MVT::MvAddMv( one, *tmpQi, zero, *tmpQi, *tmpQj );
      MVT::MvAddMv( one, *tmpQ, zero, *tmpQ, *tmpQi );
    }
  }


  //////////////////////////////////////////////////////////////////////////
  // permuteVectors for MV
  template<class ScalarType, class MV, class OP>
  void SolverUtils<ScalarType, MV, OP>::permuteVectors(
              const std::vector<int> &perm, 
              Teuchos::SerialDenseMatrix<int,ScalarType> &Q)
  {
    // Permute the vectors in Q according to the permutation vector \c perm, and
    // optionally the residual vector \c resids
    Teuchos::BLAS<int,ScalarType> blas;
    const int n = perm.size();
    const int m = Q.numRows();
    
    TEST_FOR_EXCEPTION(n != Q.numCols(), std::invalid_argument, "Anasazi::SolverUtils::permuteVectors(): size of permutation vector not equal to number of columns.");

    // Sort the primitive ritz vectors
    Teuchos::SerialDenseMatrix<int,ScalarType> copyQ( Q );
    for (int i=0; i<n; i++) {
      blas.COPY(m, copyQ[perm[i]], 1, Q[i], 1);
    }
  }


  //-----------------------------------------------------------------------------
  // 
  //  BASIS UPDATE METHODS
  //
  //-----------------------------------------------------------------------------

  // apply householder reflectors to multivector
  template<class ScalarType, class MV, class OP>
  void SolverUtils<ScalarType, MV, OP>::applyHouse(int k, MV &V, const Teuchos::SerialDenseMatrix<int,ScalarType> &H, const std::vector<ScalarType> &tau, Teuchos::RCP<MV> workMV) {

    const int n = MVT::GetNumberVecs(V);
    const ScalarType ONE = SCT::one();
    const ScalarType ZERO = SCT::zero();

    // early exit if V has zero-size or if k==0
    if (MVT::GetNumberVecs(V) == 0 || MVT::GetVecLength(V) == 0 || k == 0) {
      return;
    }

    if (workMV == Teuchos::null) {
      // user did not give us any workspace; allocate some
      workMV = MVT::Clone(V,1);
    }
    else if (MVT::GetNumberVecs(*workMV) > 1) {
      std::vector<int> first(1);
      first[0] = 0;
      workMV = MVT::CloneView(*workMV,first);
    }
    else {
      TEST_FOR_EXCEPTION(MVT::GetNumberVecs(*workMV) < 1,std::invalid_argument,"Anasazi::SolverUtils::applyHouse(): work multivector was empty.");
    }
    // Q = H_1 ... H_k is square, with as many rows as V has vectors
    // however, H need only have k columns, one each for the k reflectors.
    TEST_FOR_EXCEPTION( H.numCols() != k, std::invalid_argument,"Anasazi::SolverUtils::applyHouse(): H must have at least k columns.");
    TEST_FOR_EXCEPTION( (int)tau.size() != k, std::invalid_argument,"Anasazi::SolverUtils::applyHouse(): tau must have at least k entries.");
    TEST_FOR_EXCEPTION( H.numRows() != MVT::GetNumberVecs(V), std::invalid_argument,"Anasazi::SolverUtils::applyHouse(): Size of H,V are inconsistent.");

    // perform the loop
    // flops: Sum_{i=0:k-1} 4 m (n-i) == 4mnk - 2m(k^2- k)
    for (int i=0; i<k; i++) {
      // apply V H_i+1 = V - tau_i+1 (V v_i+1) v_i+1^T
      // because of the structure of v_i+1, this transform does not affect the first i columns of V
      std::vector<int> activeind(n-i);
      for (int j=0; j<n-i; j++) activeind[j] = j+i;
      Teuchos::RCP<MV> actV = MVT::CloneView(V,activeind);

      // note, below H_i, v_i and tau_i are mathematical objects which use 1-based indexing
      // while H, v and tau are data structures using 0-based indexing

      // get v_i+1: i-th column of H
      Teuchos::SerialDenseMatrix<int,ScalarType> v(Teuchos::Copy,H,n-i,1,i,i);
      // v_i+1(1:i) = 0: this isn't part of v
      // e_i+1^T v_i+1 = 1 = v(0)
      v(0,0) = ONE;

      // compute -tau_i V v_i
      // tau_i+1 is tau[i]
      // flops: 2 m n-i
      MVT::MvTimesMatAddMv(-tau[i],*actV,v,ZERO,*workMV);

      // perform V = V + workMV v_i^T
      // flops: 2 m n-i
      Teuchos::SerialDenseMatrix<int,ScalarType> vT(v,Teuchos::CONJ_TRANS);
      MVT::MvTimesMatAddMv(ONE,*workMV,vT,ONE,*actV);

      actV = Teuchos::null;
    }
  }

  
  //-----------------------------------------------------------------------------
  // 
  //  EIGENSOLVER PROJECTION METHODS
  //
  //-----------------------------------------------------------------------------
  
  template<class ScalarType, class MV, class OP>
  int SolverUtils<ScalarType, MV, OP>::directSolver(
      int size, 
      const Teuchos::SerialDenseMatrix<int,ScalarType> &KK, 
      Teuchos::RCP<const Teuchos::SerialDenseMatrix<int,ScalarType> > MM,
      Teuchos::SerialDenseMatrix<int,ScalarType> &EV,
      std::vector< typename Teuchos::ScalarTraits<ScalarType>::magnitudeType > &theta,
      int &nev, int esType)
  {
    // Routine for computing the first NEV generalized eigenpairs of the symmetric pencil (KK, MM)
    //
    // Parameter variables:
    //
    // size : Dimension of the eigenproblem (KK, MM)
    //
    // KK : Hermitian "stiffness" matrix 
    //
    // MM : Hermitian positive-definite "mass" matrix
    //
    // EV : Matrix to store the nev eigenvectors 
    //
    // theta : Array to store the eigenvalues (Size = nev )
    //
    // nev : Number of the smallest eigenvalues requested (input)
    //       Number of the smallest computed eigenvalues (output)
    //       Routine may compute and return more or less eigenvalues than requested.
    //
    // esType : Flag to select the algorithm
    //
    // esType =  0 (default) Uses LAPACK routine (Cholesky factorization of MM)
    //                       with deflation of MM to get orthonormality of 
    //                       eigenvectors (S^T MM S = I)
    //
    // esType =  1           Uses LAPACK routine (Cholesky factorization of MM)
    //                       (no check of orthonormality)
    //
    // esType = 10           Uses LAPACK routine for simple eigenproblem on KK
    //                       (MM is not referenced in this case)
    //
    // Note: The code accesses only the upper triangular part of KK and MM.
    //
    // Return the integer info on the status of the computation
    //
    // info = 0 >> Success
    //
    // info < 0 >> error in the info-th argument
    // info = - 20 >> Failure in LAPACK routine

    // Define local arrays

    // Create blas/lapack objects.
    Teuchos::LAPACK<int,ScalarType> lapack;
    Teuchos::BLAS<int,ScalarType> blas;

    int rank = 0;
    int info = 0;
    
    if (size < nev || size < 0) {
      return -1;
    }
    if (KK.numCols() < size || KK.numRows() < size) {
      return -2;
    }
    if ((esType == 0 || esType == 1)) {
      if (MM == Teuchos::null) {
        return -3;
      }
      else if (MM->numCols() < size || MM->numRows() < size) {
        return -3;
      }
    }
    if (EV.numCols() < size || EV.numRows() < size) {
      return -4;
    }
    if (theta.size() < (unsigned int) size) {
      return -5;
    }
    if (nev <= 0) {
      return -6;
    }

    // Query LAPACK for the "optimal" block size for HEGV
    std::string lapack_name = "hetrd";
    std::string lapack_opts = "u";
    int NB = lapack.ILAENV(1, lapack_name, lapack_opts, size, -1, -1, -1);
    int lwork = size*(NB+1);
    std::vector<ScalarType> work(lwork);
    std::vector<MagnitudeType> rwork(3*size-2);
    // tt contains the eigenvalues from HEGV, which are necessarily real, and
    // HEGV expects this vector to be real as well
    std::vector<MagnitudeType> tt( size );
    typedef typename std::vector<MagnitudeType>::iterator MTIter;

    MagnitudeType tol = SCT::magnitude(SCT::squareroot(SCT::eps()));
    // MagnitudeType tol = 1e-12;
    ScalarType zero = Teuchos::ScalarTraits<ScalarType>::zero();
    ScalarType one = Teuchos::ScalarTraits<ScalarType>::one();

    Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > KKcopy, MMcopy;
    Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > U;

    switch (esType) {
      default:
      case 0:
        //
        // Use LAPACK to compute the generalized eigenvectors
        //
        for (rank = size; rank > 0; --rank) {

          U = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>(rank,rank) );              
          //
          // Copy KK & MM
          //
          KKcopy = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>( Teuchos::Copy, KK, rank, rank ) );
          MMcopy = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>( Teuchos::Copy, *MM, rank, rank ) );
          //
          // Solve the generalized eigenproblem with LAPACK
          //
          info = 0;
          lapack.HEGV(1, 'V', 'U', rank, KKcopy->values(), KKcopy->stride(), 
              MMcopy->values(), MMcopy->stride(), &tt[0], &work[0], lwork,
              &rwork[0], &info);
          //
          // Treat error messages
          //
          if (info < 0) {
            std::cerr << std::endl;
            std::cerr << "Anasazi::SolverUtils::directSolver(): In HEGV, argument " << -info << "has an illegal value.\n";
            std::cerr << std::endl;
            return -20;
          }
          if (info > 0) {
            if (info > rank)
              rank = info - rank;
            continue;
          }
          //
          // Check the quality of eigenvectors ( using mass-orthonormality )
          //
          MMcopy = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>( Teuchos::Copy, *MM, rank, rank ) );
          for (int i = 0; i < rank; ++i) {
            for (int j = 0; j < i; ++j) {
              (*MMcopy)(i,j) = SCT::conjugate((*MM)(j,i));
            }
          }
          // U = 0*U + 1*MMcopy*KKcopy = MMcopy * KKcopy
          TEST_FOR_EXCEPTION( 
              U->multiply(Teuchos::NO_TRANS,Teuchos::NO_TRANS,one,*MMcopy,*KKcopy,zero) != 0,
              std::logic_error, "Anasazi::SolverUtils::directSolver() call to Teuchos::SerialDenseMatrix::multiply() returned an error.");
          // MMcopy = 0*MMcopy + 1*KKcopy^H*U = KKcopy^H * MMcopy * KKcopy
          TEST_FOR_EXCEPTION( 
              MMcopy->multiply(Teuchos::CONJ_TRANS,Teuchos::NO_TRANS,one,*KKcopy,*U,zero) != 0,
              std::logic_error, "Anasazi::SolverUtils::directSolver() call to Teuchos::SerialDenseMatrix::multiply() returned an error.");
          MagnitudeType maxNorm = SCT::magnitude(zero);
          MagnitudeType maxOrth = SCT::magnitude(zero);
          for (int i = 0; i < rank; ++i) {
            for (int j = i; j < rank; ++j) {
              if (j == i)
                maxNorm = SCT::magnitude((*MMcopy)(i,j) - one) > maxNorm
                  ? SCT::magnitude((*MMcopy)(i,j) - one) : maxNorm;            
              else 
                maxOrth = SCT::magnitude((*MMcopy)(i,j)) > maxOrth
                  ? SCT::magnitude((*MMcopy)(i,j)) : maxOrth;
            }
          }
          /*        if (verbose > 4) {
                    std::cout << " >> Local eigensolve >> Size: " << rank;
                    std::cout.precision(2);
                    std::cout.setf(std::ios::scientific, std::ios::floatfield);
                    std::cout << " Normalization error: " << maxNorm;
                    std::cout << " Orthogonality error: " << maxOrth;
                    std::cout << endl;
                    }*/
          if ((maxNorm <= tol) && (maxOrth <= tol)) {
            break;
          }
        } // for (rank = size; rank > 0; --rank)
        //
        // Copy the computed eigenvectors and eigenvalues
        // ( they may be less than the number requested because of deflation )
        //
        // std::cout << "directSolve    rank: " << rank << "\tsize: " << size << endl;
        nev = (rank < nev) ? rank : nev;
        EV.putScalar( zero );
        std::copy(tt.begin(),tt.begin()+nev,theta.begin());
        for (int i = 0; i < nev; ++i) {
          blas.COPY( rank, (*KKcopy)[i], 1, EV[i], 1 );
        }
        break;

      case 1:
        //
        // Use the Cholesky factorization of MM to compute the generalized eigenvectors
        //
        // Copy KK & MM
        //
        KKcopy = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>( Teuchos::Copy, KK, size, size ) );
        MMcopy = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>( Teuchos::Copy, *MM, size, size ) );
        //
        // Solve the generalized eigenproblem with LAPACK
        //
        info = 0;
        lapack.HEGV(1, 'V', 'U', size, KKcopy->values(), KKcopy->stride(), 
            MMcopy->values(), MMcopy->stride(), &tt[0], &work[0], lwork,
            &rwork[0], &info);
        //
        // Treat error messages
        //
        if (info < 0) {
          std::cerr << std::endl;
          std::cerr << "Anasazi::SolverUtils::directSolver(): In HEGV, argument " << -info << "has an illegal value.\n";
          std::cerr << std::endl;
          return -20;
        }
        if (info > 0) {
          if (info > size)
            nev = 0;
          else {
            std::cerr << std::endl;
            std::cerr << "Anasazi::SolverUtils::directSolver(): In HEGV, DPOTRF or DHEEV returned an error code (" << info << ").\n";
            std::cerr << std::endl;
            return -20; 
          }
        }
        //
        // Copy the eigenvectors and eigenvalues
        //
        std::copy(tt.begin(),tt.begin()+nev,theta.begin());
        for (int i = 0; i < nev; ++i) {
          blas.COPY( size, (*KKcopy)[i], 1, EV[i], 1 );
        }
        break;

      case 10:
        //
        // Simple eigenproblem
        //
        // Copy KK
        //
        KKcopy = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>( Teuchos::Copy, KK, size, size ) );
        //
        // Solve the generalized eigenproblem with LAPACK
        //
        lapack.HEEV('V', 'U', size, KKcopy->values(), KKcopy->stride(), &tt[0], &work[0], lwork, &rwork[0], &info);
        //
        // Treat error messages
        if (info != 0) {
          std::cerr << std::endl;
          if (info < 0) 
            std::cerr << "Anasazi::SolverUtils::directSolver(): In DHEEV, argument " << -info << " has an illegal value\n";
          else
            std::cerr << "Anasazi::SolverUtils::directSolver(): In DHEEV, the algorithm failed to converge (" << info << ").\n";
          std::cerr << std::endl;
          info = -20;
          break;
        }
        //
        // Copy the eigenvectors
        //
        std::copy(tt.begin(),tt.begin()+nev,theta.begin());
        for (int i = 0; i < nev; ++i) {
          blas.COPY( size, (*KKcopy)[i], 1, EV[i], 1 );
        }
        break;
    }

    return info;
  }


  //-----------------------------------------------------------------------------
  // 
  //  SANITY CHECKING METHODS
  //
  //-----------------------------------------------------------------------------

  template<class ScalarType, class MV, class OP>
  typename Teuchos::ScalarTraits<ScalarType>::magnitudeType 
  SolverUtils<ScalarType, MV, OP>::errorEquality(const MV &X, const MV &MX, Teuchos::RCP<const OP> M)
  {
    // Return the maximum coefficient of the matrix M * X - MX
    // scaled by the maximum coefficient of MX.
    // When M is not specified, the identity is used.
    
    MagnitudeType maxDiff = SCT::magnitude(SCT::zero());
    
    int xc = MVT::GetNumberVecs(X);
    int mxc = MVT::GetNumberVecs(MX);
    
    TEST_FOR_EXCEPTION(xc != mxc,std::invalid_argument,"Anasazi::SolverUtils::errorEquality(): input multivecs have different number of columns.");
    if (xc == 0) {
      return maxDiff;
    }
    
    MagnitudeType maxCoeffX = SCT::magnitude(SCT::zero());
    std::vector<MagnitudeType> tmp( xc );
    MVT::MvNorm(MX, &tmp);

    for (int i = 0; i < xc; ++i) {
      maxCoeffX = (tmp[i] > maxCoeffX) ? tmp[i] : maxCoeffX;
    }

    std::vector<int> index( 1 );
    Teuchos::RCP<MV> MtimesX; 
    if (M != Teuchos::null) {
      MtimesX = MVT::Clone( X, xc );
      OPT::Apply( *M, X, *MtimesX );
    }
    else {
      MtimesX = MVT::CloneCopy(X);
    }
    MVT::MvAddMv( -1.0, MX, 1.0, *MtimesX, *MtimesX );
    MVT::MvNorm( *MtimesX, &tmp );
   
    for (int i = 0; i < xc; ++i) {
      maxDiff = (tmp[i] > maxDiff) ? tmp[i] : maxDiff;
    }
    
    return (maxCoeffX == 0.0) ? maxDiff : maxDiff/maxCoeffX;
    
  }    
  
} // end namespace Anasazi

#endif // ANASAZI_SOLVER_UTILS_HPP


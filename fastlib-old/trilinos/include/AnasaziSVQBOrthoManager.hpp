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


/*! \file AnasaziSVQBOrthoManager.hpp
  \brief Orthogonalization manager based on the SVQB technique described in 
  "A Block Orthogonalization Procedure With Constant Synchronization Requirements", A. Stathapoulos and K. Wu
*/

#ifndef ANASAZI_SVQB_ORTHOMANAGER_HPP
#define ANASAZI_SVQB_ORTHOMANAGER_HPP

/*!   \class Anasazi::SVQBOrthoManager
      \brief An implementation of the Anasazi::MatOrthoManager that performs orthogonalization
      using the SVQB iterative orthogonalization technique described by Stathapoulos and Wu. This orthogonalization routine,
      while not returning the upper triangular factors of the popular Gram-Schmidt method, has a communication 
      cost (measured in number of communication calls) that is independent of the number of columns in the basis.
      
      \author Chris Baker, Ulrich Hetmaniuk, Rich Lehoucq, and Heidi Thornquist
*/

#include "AnasaziConfigDefs.hpp"
#include "AnasaziMultiVecTraits.hpp"
#include "AnasaziOperatorTraits.hpp"
#include "AnasaziMatOrthoManager.hpp"
#include "Teuchos_LAPACK.hpp"

namespace Anasazi {

  template<class ScalarType, class MV, class OP>
  class SVQBOrthoManager : public MatOrthoManager<ScalarType,MV,OP> {

  private:
    typedef typename Teuchos::ScalarTraits<ScalarType>::magnitudeType MagnitudeType;
    typedef Teuchos::ScalarTraits<ScalarType>  SCT;
    typedef Teuchos::ScalarTraits<MagnitudeType>  SCTM;
    typedef MultiVecTraits<ScalarType,MV>      MVT;
    typedef OperatorTraits<ScalarType,MV,OP>   OPT;
    std::string dbgstr;


  public:

    //! @name Constructor/Destructor
    //@{ 
    //! Constructor specifying re-orthogonalization tolerance.
    SVQBOrthoManager( Teuchos::RCP<const OP> Op = Teuchos::null, bool debug = false );


    //! Destructor
    ~SVQBOrthoManager() {};
    //@}


    //! @name Methods implementing Anasazi::MatOrthoManager
    //@{ 


    /*! \brief Given a list of mutually orthogonal and internally orthonormal bases \c Q, this method
     * projects a multivector \c X onto the space orthogonal to the individual <tt>Q[i]</tt>, 
     * optionally returning the coefficients of \c X for the individual <tt>Q[i]</tt>. All of this is done with respect
     * to the inner product innerProd().
     *
     * After calling this routine, \c X will be orthogonal to each of the <tt>Q[i]</tt>.
     *
     @param X [in/out] The multivector to be modified.<br>
       On output, the columns of \c X will be orthogonal to each <tt>Q[i]</tt>, satisfying
       \f[
       X_{out} = X_{in} - \sum_i Q[i] \langle Q[i], X_{in} \rangle
       \f]

     @param MX [in/out] The image of \c X under the inner product operator \c Op. 
       If \f$ MX != 0\f$: On input, this is expected to be consistent with \c Op \cdot X. On output, this is updated consistent with updates to \c X.
       If \f$ MX == 0\f$ or \f$ Op == 0\f$: \c MX is not referenced.

     @param C [out] The coefficients of \c X in the bases <tt>Q[i]</tt>. If <tt>C[i]</tt> is a non-null pointer 
       and <tt>C[i]</tt> matches the dimensions of \c X and <tt>Q[i]</tt>, then the coefficients computed during the orthogonalization
       routine will be stored in the matrix <tt>C[i]</tt>, similar to calling
       \code
          innerProd( Q[i], X, C[i] );
       \endcode
       If <tt>C[i]</tt> points to a Teuchos::SerialDenseMatrix with size
       inconsistent with \c X and \c <tt>Q[i]</tt>, then a std::invalid_argument
       exception will be thrown. Otherwise, if <tt>C.size() < i</tt> or
       <tt>C[i]</tt> is a null pointer, the caller will not have access to the
       computed coefficients.

     @param Q [in] A list of multivector bases specifying the subspaces to be orthogonalized against, satisfying 
     \f[
        \langle Q[i], Q[j] \rangle = I \quad\textrm{if}\quad i=j
     \f]
     and
     \f[
        \langle Q[i], Q[j] \rangle = 0 \quad\textrm{if}\quad i \neq j\ .
     \f]
    */
    void projectMat ( 
        MV &X, 
        Teuchos::RCP<MV> MX = Teuchos::null, 
        Teuchos::Array<Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > > C = Teuchos::tuple(Teuchos::null), 
        Teuchos::Array<Teuchos::RCP<const MV> > Q = Teuchos::tuple(Teuchos::null) ) const;


    /*! \brief This method takes a multivector \c X and attempts to compute an orthonormal basis for \f$colspan(X)\f$, with respect to innerProd().
     *
     * This method does not compute an upper triangular coefficient matrix \c B.
     *
     * This routine returns an integer \c rank stating the rank of the computed basis. If \c X does not have full rank and the normalize() routine does 
     * not attempt to augment the subspace, then \c rank may be smaller than the number of columns in \c X. In this case, only the first \c rank columns of 
     * output \c X and first \c rank rows of \c B will be valid.
     *  
     * The method attempts to find a basis with dimension equal to the number of columns in \c X. It does this by augmenting linearly dependent 
     * vectors in \c X with random directions. A finite number of these attempts will be made; therefore, it is possible that the dimension of the 
     * computed basis is less than the number of vectors in \c X.
     *
     @param X [in/out] The multivector to be modified.<br>
       On output, the first \c rank columns of \c X satisfy
       \f[
          \langle X[i], X[j] \rangle = \delta_{ij}\ .
       \f]
       Also, 
       \f[
          X_{in}(1:m,1:n) = X_{out}(1:m,1:rank) B(1:rank,1:n)
       \f]
       where \c m is the number of rows in \c X and \c n is the number of columns in \c X.

     @param MX [in/out] The image of \c X under the inner product operator \c Op. 
       If \f$ MX != 0\f$: On input, this is expected to be consistent with \c Op \cdot X. On output, this is updated consistent with updates to \c X.
       If \f$ MX == 0\f$ or \f$ Op == 0\f$: \c MX is not referenced.

     @param B [out] The coefficients of the original \c X with respect to the computed basis. If \c B is a non-null pointer and \c B matches the dimensions of \c B, then the
     coefficients computed during the orthogonalization routine will be stored in \c B, similar to calling 
       \code
          innerProd( Xout, Xin, B );
       \endcode
     If \c B points to a Teuchos::SerialDenseMatrix with size inconsistent with \c X, then a std::invalid_argument exception will be thrown. Otherwise, if \c B is null, the caller will not have
     access to the computed coefficients. This matrix is not necessarily triangular (as in a QR factorization); see the documentation of specific orthogonalization managers.<br>
     In general, \c B has no non-zero structure.

     @return Rank of the basis computed by this method, less than or equal to the number of columns in \c X. This specifies how many columns in the returned \c X and rows in the returned \c B are valid.
    */
    int normalizeMat ( 
        MV &X, 
        Teuchos::RCP<MV> MX = Teuchos::null, 
        Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > B = Teuchos::tuple(Teuchos::null) ) const;


    /*! \brief Given a set of bases <tt>Q[i]</tt> and a multivector \c X, this method computes an orthonormal basis for \f$colspan(X) - \sum_i colspan(Q[i])\f$.
     *
     *  This routine returns an integer \c rank stating the rank of the computed basis. If the subspace \f$colspan(X) - \sum_i colspan(Q[i])\f$ does not 
     *  have dimension as large as the number of columns of \c X and the orthogonalization manager doe not attempt to augment the subspace, then \c rank 
     *  may be smaller than the number of columns of \c X. In this case, only the first \c rank columns of output \c X and first \c rank rows of \c B will 
     *  be valid.
     *
     * The method attempts to find a basis with dimension the same as the number of columns in \c X. It does this by augmenting linearly dependent 
     * vectors with random directions. A finite number of these attempts will be made; therefore, it is possible that the dimension of the 
     * computed basis is less than the number of vectors in \c X.
     *
     @param X [in/out] The multivector to be modified.<br>
       On output, the first \c rank columns of \c X satisfy
       \f[
            \langle X[i], X[j] \rangle = \delta_{ij} \quad \textrm{and} \quad \langle X, Q[i] \rangle = 0\ .
       \f]
       Also, 
       \f[
          X_{in}(1:m,1:n) = X_{out}(1:m,1:rank) B(1:rank,1:n) + \sum_i Q[i] C[i]
       \f]
       where \c m is the number of rows in \c X and \c n is the number of columns in \c X.

     @param MX [in/out] The image of \c X under the inner product operator \c Op. 
       If \f$ MX != 0\f$: On input, this is expected to be consistent with \c Op \cdot X. On output, this is updated consistent with updates to \c X.
       If \f$ MX == 0\f$ or \f$ Op == 0\f$: \c MX is not referenced.

     @param C [out] The coefficients of \c X in the <tt>Q[i]</tt>. If <tt>C[i]</tt> is a non-null pointer 
       and <tt>C[i]</tt> matches the dimensions of \c X and <tt>Q[i]</tt>, then the coefficients computed during the orthogonalization
       routine will be stored in the matrix <tt>C[i]</tt>, similar to calling
       \code
          innerProd( Q[i], X, C[i] );
       \endcode
       If <tt>C[i]</tt> points to a Teuchos::SerialDenseMatrix with size
       inconsistent with \c X and \c <tt>Q[i]</tt>, then a std::invalid_argument
       exception will be thrown. Otherwise, if <tt>C.size() < i</tt> or
       <tt>C[i]</tt> is a null pointer, the caller will not have access to the
       computed coefficients.

     @param B [out] The coefficients of the original \c X with respect to the computed basis. If \c B is a non-null pointer and \c B matches the dimensions of \c B, then the
     coefficients computed during the orthogonalization routine will be stored in \c B, similar to calling 
       \code
          innerProd( Xout, Xin, B );
       \endcode
     If \c B points to a Teuchos::SerialDenseMatrix with size inconsistent with \c X, then a std::invalid_argument exception will be thrown. Otherwise, if \c B is null, the caller will not have
     access to the computed coefficients. This matrix is not necessarily triangular (as in a QR factorization); see the documentation of specific orthogonalization managers.<br>
     In general, \c B has no non-zero structure.

     @param Q [in] A list of multivector bases specifying the subspaces to be orthogonalized against, satisfying 
     \f[
        \langle Q[i], Q[j] \rangle = I \quad\textrm{if}\quad i=j
     \f]
     and
     \f[
        \langle Q[i], Q[j] \rangle = 0 \quad\textrm{if}\quad i \neq j\ .
     \f]

     @return Rank of the basis computed by this method, less than or equal to the number of columns in \c X. This specifies how many columns in the returned \c X and rows in the returned \c B are valid.

    */
    int projectAndNormalizeMat ( 
        MV &X, 
        Teuchos::RCP<MV> MX = Teuchos::null, 
        Teuchos::Array<Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > > C = Teuchos::tuple(Teuchos::null), 
        Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > B = Teuchos::null, 
        Teuchos::Array<Teuchos::RCP<const MV> > Q = Teuchos::tuple(Teuchos::null) ) const;

    //@}

    //! @name Error methods
    //@{ 

    /*! \brief This method computes the error in orthonormality of a multivector, measured
     * as the Frobenius norm of the difference <tt>innerProd(X,Y) - I</tt>.
     *  The method has the option of exploiting a caller-provided \c MX.
     */
    typename Teuchos::ScalarTraits<ScalarType>::magnitudeType 
    orthonormErrorMat(const MV &X, Teuchos::RCP<const MV> MX = Teuchos::null) const;

    /*! \brief This method computes the error in orthogonality of two multivectors, measured
     * as the Frobenius norm of <tt>innerProd(X,Y)</tt>.
     *  The method has the option of exploiting a caller-provided \c MX.
     */
    typename Teuchos::ScalarTraits<ScalarType>::magnitudeType 
    orthogErrorMat(const MV &X1, Teuchos::RCP<const MV> MX1, const MV &X2) const;

    //@}

  private:
    
    MagnitudeType eps_;
    bool debug_;
  
    // ! Routine to find an orthogonal/orthonormal basis for the 
    int findBasis( MV &X, Teuchos::RCP<MV> MX, 
                   Teuchos::Array<Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > > C, 
                   Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > B, 
                   Teuchos::Array<Teuchos::RCP<const MV> > Q,
                   bool normalize ) const;
  };


  //////////////////////////////////////////////////////////////////////////////////////////////////
  // Constructor
  template<class ScalarType, class MV, class OP>
  SVQBOrthoManager<ScalarType,MV,OP>::SVQBOrthoManager( Teuchos::RCP<const OP> Op, bool debug) 
    : MatOrthoManager<ScalarType,MV,OP>(Op), dbgstr("                    *** "), debug_(debug) {
    
    Teuchos::LAPACK<int,MagnitudeType> lapack;
    eps_ = lapack.LAMCH('E');
    if (debug_) {
      std::cout << "eps_ == " << eps_ << std::endl;
    }
  }


  //////////////////////////////////////////////////////////////////////////////////////////////////
  // Compute the distance from orthonormality
  template<class ScalarType, class MV, class OP>
  typename Teuchos::ScalarTraits<ScalarType>::magnitudeType 
  SVQBOrthoManager<ScalarType,MV,OP>::orthonormErrorMat(const MV &X, Teuchos::RCP<const MV> MX) const {
    const ScalarType ONE = SCT::one();
    int rank = MVT::GetNumberVecs(X);
    Teuchos::SerialDenseMatrix<int,ScalarType> xTx(rank,rank);
    innerProdMat(X,X,MX,xTx);
    for (int i=0; i<rank; i++) {
      xTx(i,i) -= ONE;
    }
    return xTx.normFrobenius();
  }

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // Compute the distance from orthogonality
  template<class ScalarType, class MV, class OP>
  typename Teuchos::ScalarTraits<ScalarType>::magnitudeType 
  SVQBOrthoManager<ScalarType,MV,OP>::orthogErrorMat(const MV &X1, Teuchos::RCP<const MV> MX1, const MV &X2) const {
    int r1 = MVT::GetNumberVecs(X1);
    int r2 = MVT::GetNumberVecs(X2);
    Teuchos::SerialDenseMatrix<int,ScalarType> xTx(r2,r1);
    innerProdMat(X2,X1,MX1,xTx);
    return xTx.normFrobenius();
  }


  //////////////////////////////////////////////////////////////////////////////////////////////////
  // Find an Op-orthonormal basis for span(X) - span(W)
  template<class ScalarType, class MV, class OP>
  int SVQBOrthoManager<ScalarType, MV, OP>::projectAndNormalizeMat(
                                    MV &X, Teuchos::RCP<MV> MX, 
                                    Teuchos::Array<Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > > C, 
                                    Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > B, 
                                    Teuchos::Array<Teuchos::RCP<const MV> > Q ) const {

    return findBasis(X,MX,C,B,Q,true);
  }



  //////////////////////////////////////////////////////////////////////////////////////////////////
  // Find an Op-orthonormal basis for span(X), with rank numvectors(X)
  template<class ScalarType, class MV, class OP>
  int SVQBOrthoManager<ScalarType, MV, OP>::normalizeMat(
                            MV &X, Teuchos::RCP<MV> MX, 
                            Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > B ) const {
    Teuchos::Array<Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > > C;
    Teuchos::Array<Teuchos::RCP<const MV> > Q;
    return findBasis(X,MX,C,B,Q,true);
  }



  //////////////////////////////////////////////////////////////////////////////////////////////////
  template<class ScalarType, class MV, class OP>
  void SVQBOrthoManager<ScalarType, MV, OP>::projectMat(
                                MV &X, Teuchos::RCP<MV> MX, 
                                Teuchos::Array<Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > > C, 
                                Teuchos::Array<Teuchos::RCP<const MV> > Q) const {
    findBasis(X,MX,C,Teuchos::null,Q,false);
  }


  //////////////////////////////////////////////////////////////////////////////////////////////////
  // Find an Op-orthonormal basis for span(X), with the option of extending the subspace so that 
  // the rank is numvectors(X)
  // 
  // Tracking the coefficients (C[i] and B) for this code is complicated by the fact that the loop
  // structure looks like
  // do
  //    project
  //    do
  //       ortho
  //    end
  // end
  // However, the recurrence for the coefficients is not complicated:
  // B = I
  // C = 0
  // do 
  //    project yields newC
  //    C = C + newC*B
  //    do 
  //       ortho yields newR
  //       B = newR*B
  //    end
  // end
  // This holds for each individual C[i] (which correspond to the list of bases we are orthogonalizing
  // against).
  //
  template<class ScalarType, class MV, class OP>
  int SVQBOrthoManager<ScalarType, MV, OP>::findBasis(
                MV &X, Teuchos::RCP<MV> MX, 
                Teuchos::Array<Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > > C, 
                Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > B,
                Teuchos::Array<Teuchos::RCP<const MV> > Q,
                bool normalize) const {

    const ScalarType ONE  = SCT::one();
    const MagnitudeType MONE = SCTM::one();
    const MagnitudeType ZERO = SCTM::zero();

    int numGS = 0,
        numSVQB = 0,
        numRand = 0;

    // get sizes of X,MX
    int xc = MVT::GetNumberVecs(X);
    int xr = MVT::GetVecLength( X );

    // get sizes of Q[i]
    int nq = Q.length();
    int qr = (nq == 0) ? 0 : MVT::GetVecLength(*Q[0]);
    int qsize = 0;
    std::vector<int> qcs(nq);
    for (int i=0; i<nq; i++) {
      qcs[i] = MVT::GetNumberVecs(*Q[i]);
      qsize += qcs[i];
    }

    if (normalize == true && qsize + xc > xr) {
      // not well-posed
      TEST_FOR_EXCEPTION( true, std::invalid_argument, 
                          "Anasazi::SVQBOrthoManager::findBasis(): Orthogonalization constraints not feasible" );
    }

    // try to short-circuit as early as possible
    if (normalize == false && (qsize == 0 || xc == 0)) {
      // nothing to do
      return 0;
    }
    else if (normalize == true && (xc == 0 || xr == 0)) {
      // normalize requires X not empty
      TEST_FOR_EXCEPTION( true, std::invalid_argument, 
                          "Anasazi::SVQBOrthoManager::findBasis(): X must be non-empty" );
    }

    // check that Q matches X
    TEST_FOR_EXCEPTION( qsize != 0 && qr != xr , std::invalid_argument, 
                        "Anasazi::SVQBOrthoManager::findBasis(): Size of X not consistant with size of Q" );

    /* If we don't have enough C, expanding it creates null references
     * If we have too many, resizing just throws away the later ones
     * If we have exactly as many as we have Q, this call has no effect
     */
    C.resize(nq);
    Teuchos::Array<Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > > newC(nq);
    // check the size of the C[i] against the Q[i] and consistency between Q[i]
    for (int i=0; i<nq; i++) {
      // check size of Q[i]
      TEST_FOR_EXCEPTION( MVT::GetVecLength( *Q[i] ) != qr, std::invalid_argument, 
                          "Anasazi::SVQBOrthoManager::findBasis(): Size of Q not mutually consistant" );
      TEST_FOR_EXCEPTION( qr < qcs[i], std::invalid_argument, 
                          "Anasazi::SVQBOrthoManager::findBasis(): Q has less rows than columns" );
      // check size of C[i]
      if ( C[i] == Teuchos::null ) {
        C[i] = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>(qcs[i],xc) );
      }
      else {
        TEST_FOR_EXCEPTION( C[i]->numRows() != qcs[i] || C[i]->numCols() != xc, std::invalid_argument, 
                            "Anasazi::SVQBOrthoManager::findBasis(): Size of Q not consistant with C" );
      }
      // clear C[i]
      C[i]->putScalar(ZERO);
      newC[i] = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>(*C[i]) );
    }


    ////////////////////////////////////////////////////////
    // Allocate necessary storage
    // C were allocated above
    // Allocate MX and B (if necessary)
    // Set B = I
    if (normalize == true) {
      if ( B == Teuchos::null ) {
        B = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>(xc,xc) );
      }
      TEST_FOR_EXCEPTION( B->numRows() != xc || B->numCols() != xc, std::invalid_argument, 
                          "Anasazi::SVQBOrthoManager::findBasis(): Size of B not consistant with X" );
      // set B to I
      B->putScalar(ZERO);
      for (int i=0; i<xc; i++) {
        (*B)(i,i) = MONE;
      }
    }
    /******************************************
     *  If _hasOp == false, DO NOT MODIFY MX  *
     ****************************************** 
     * if Op==null, MX == X (via pointer)
     * Otherwise, either the user passed in MX or we will allocate and compute it
     *
     * workX will be a multivector of the same size as X, used to perform X*S when normalizing
     */
    Teuchos::RCP<MV> workX;
    if (normalize) {
      workX = MVT::Clone(X,xc);
    }
    if (this->_hasOp) {
      if (MX == Teuchos::null) {
        // we need to allocate space for MX
        MX = MVT::Clone(X,xc);
        OPT::Apply(*(this->_Op),X,*MX);
        this->_OpCounter += MVT::GetNumberVecs(X);
      }
    }
    else {
      MX = Teuchos::rcp(&X,false);
    }
    std::vector<MagnitudeType> normX(xc), invnormX(xc);
    Teuchos::SerialDenseMatrix<int,ScalarType> XtMX(xc,xc), workU(1,1);
    Teuchos::LAPACK<int,ScalarType> lapack;
    /**********************************************************************
     * allocate storage for eigenvectors,eigenvalues of X^T Op X, and for
     * the work space needed to compute this xc-by-xc eigendecomposition
     **********************************************************************/
    std::vector<ScalarType> work;
    std::vector<MagnitudeType> lambda, lambdahi, rwork;
    if (normalize) {
      // get size of work from ILAENV
      int lwork = lapack.ILAENV(1,"hetrd","VU",xc,-1,-1,-1);
      // lwork >= (nb+1)*n for complex
      // lwork >= (nb+2)*n for real
      TEST_FOR_EXCEPTION( lwork < 0, OrthoError, 
                          "Anasazi::SVQBOrthoManager::findBasis(): Error code from ILAENV" );

      lwork = (lwork+2)*xc;
      work.resize(lwork);
      // size of rwork is max(1,3*xc-2)
      lwork = (3*xc-2 > 1) ? 3*xc - 2 : 1;
      rwork.resize(lwork);
      // size of lambda is xc
      lambda.resize(xc);
      lambdahi.resize(xc);
      workU.reshape(xc,xc);
    }

    // test sizes of X,MX
    int mxc = (this->_hasOp) ? MVT::GetNumberVecs( *MX ) : xc;
    int mxr = (this->_hasOp) ? MVT::GetVecLength( *MX )  : xr;
    TEST_FOR_EXCEPTION( xc != mxc || xr != mxr, std::invalid_argument, 
                        "Anasazi::SVQBOrthoManager::findBasis(): Size of X not consistant with MX" );

    // sentinel to continue the outer loop (perform another projection step)
    bool doGramSchmidt = true;          
    // variable for testing orthonorm/orthog 
    MagnitudeType tolerance = MONE/SCTM::squareroot(eps_);

    // outer loop
    while (doGramSchmidt) {

      ////////////////////////////////////////////////////////////////////////////////////
      // perform projection
      if (qsize > 0) {

        numGS++;

        // Compute the norms of the vectors
        normMat(X,MX,&normX);
        // normalize the vectors
        Teuchos::RCP<MV> Xi,MXi;
        std::vector<int> ind(1);
        for (int i=0; i<xc; i++) {
          invnormX[i] = (normX[i] == ZERO) ? ZERO : MONE/normX[i];
          ind[0] = i;
          Xi = MVT::CloneView(X,ind);
          MVT::MvAddMv(ZERO,*Xi,invnormX[i],*Xi,*Xi);
          Xi = Teuchos::null;
          if (this->_hasOp) {
            MXi = MVT::CloneView(*MX,ind);
            MVT::MvAddMv(ZERO,*MXi,invnormX[i],*MXi,*MXi);
            MXi = Teuchos::null;
          }
        }
        // check that vectors are normalized now
        if (debug_) {
          std::vector<MagnitudeType> nrm2(xc);
          std::cout << dbgstr << "max post-scale norm: (with/without MX) : ";
          MagnitudeType maxpsnw = ZERO, maxpsnwo = ZERO;
          normMat(X,MX,&nrm2);
          for (int i=0; i<xc; i++) {
            maxpsnw = (nrm2[i] > maxpsnw ? nrm2[i] : maxpsnw);
          }
          MatOrthoManager<ScalarType,MV,OP>::norm(X,&nrm2);
          for (int i=0; i<xc; i++) {
            maxpsnwo = (nrm2[i] > maxpsnwo ? nrm2[i] : maxpsnwo);
          }
          std::cout << "(" << maxpsnw << "," << maxpsnwo << ")" << std::endl;
        }
        // project the vectors onto the Qi
        for (int i=0; i<nq; i++) {
          innerProdMat(*Q[i],X,MX,*newC[i]);
        }
        // remove the components in Qi from X
        for (int i=0; i<nq; i++) {
          MVT::MvTimesMatAddMv(-ONE,*Q[i],*newC[i],ONE,X);
        }
        // un-scale the vectors 
        for (int i=0; i<xc; i++) {
          ind[0] = i;
          Xi = MVT::CloneView(X,ind);
          MVT::MvAddMv(ZERO,*Xi,normX[i],*Xi,*Xi);
          Xi = Teuchos::null;
        }
        // Recompute the vectors in MX
        if (this->_hasOp) {
          OPT::Apply(*(this->_Op),X,*MX);
          this->_OpCounter += MVT::GetNumberVecs(X);
        }

        //          
        // Compute largest column norm of 
        //            (  C[0]   )  
        //        C = (  ....   )
        //            ( C[nq-1] )
        MagnitudeType maxNorm = ZERO;
        for  (int j=0; j<xc; j++) {
          MagnitudeType sum = ZERO;
          for (int k=0; k<nq; k++) {
            for (int i=0; i<qcs[k]; i++) {
              sum += SCT::magnitude((*newC[k])(i,j))*SCT::magnitude((*newC[k])(i,j));
            }
          }
          maxNorm = (sum > maxNorm) ? sum : maxNorm;
        }
              
        // do we perform another GS?
        if (maxNorm < 0.36) {
          doGramSchmidt = false;
        }

        // unscale newC to reflect the scaling of X
        for (int k=0; k<nq; k++) {
          for (int j=0; j<xc; j++) {
            for (int i=0; i<qcs[k]; i++) {
              (*newC[k])(i,j) *= normX[j];
            }
          }
        }
        // accumulate into C
        if (normalize) {
          // we are normalizing
          int info;
          for (int i=0; i<nq; i++) {
            info = C[i]->multiply(Teuchos::NO_TRANS,Teuchos::NO_TRANS,ONE,*newC[i],*B,ONE);
            TEST_FOR_EXCEPTION(info != 0, std::logic_error, "Anasazi::SVQBOrthoManager::findBasis(): Input error to SerialDenseMatrix::multiply.");
          }
        }
        else {
          // not normalizing
          for (int i=0; i<nq; i++) {
            (*C[i]) += *newC[i];
          }
        }
      }
      else { // qsize == 0... don't perform projection
        // don't do any more outer loops; all we need is to call the normalize code below
        doGramSchmidt = false;
      }


      ////////////////////////////////////////////////////////////////////////////////////
      // perform normalization
      if (normalize) {

        MagnitudeType condT = tolerance;
        
        while (condT >= tolerance) {

          numSVQB++;

          // compute X^T Op X
          innerProdMat(X,X,MX,XtMX);

          // compute scaling matrix for XtMX: D^{.5} and D^{-.5} (D-half  and  D-half-inv)
          std::vector<MagnitudeType> Dh(xc), Dhi(xc);
          for (int i=0; i<xc; i++) {
            Dh[i]  = SCT::magnitude(SCT::squareroot(XtMX(i,i)));
            Dhi[i] = (Dh[i] == ZERO ? ZERO : MONE/Dh[i]);
          }
          // scale XtMX :   S = D^{-.5} * XtMX * D^{-.5}
          for (int i=0; i<xc; i++) {
            for (int j=0; j<xc; j++) {
              XtMX(i,j) *= Dhi[i]*Dhi[j];
            }
          }

          // compute the eigenvalue decomposition of S=U*Lambda*U^T (using upper part)
          int info;
          lapack.HEEV('V', 'U', xc, XtMX.values(), XtMX.stride(), &lambda[0], &work[0], work.size(), &rwork[0], &info);
          TEST_FOR_EXCEPTION( info != 0, OrthoError, 
                              "Anasazi::SVQBOrthoManager::findBasis(): Error code from HEEV" );
          if (debug_) {
            std::cout << dbgstr << "eigenvalues of XtMX: (";
            for (int i=0; i<xc-1; i++) {
              std::cout << lambda[i] << ",";
            }
            std::cout << lambda[xc-1] << ")" << std::endl;
          }

          // remember, HEEV orders the eigenvalues from smallest to largest
          // examine condition number of Lambda, compute Lambda^{-.5}
          MagnitudeType maxLambda = lambda[xc-1],
                        minLambda = lambda[0];
          int iZeroMax = -1;
          for (int i=0; i<xc; i++) {
            if (lambda[i] < 10*eps_*maxLambda) {      // finish: this was eps_*eps_*maxLambda
              iZeroMax = i;
              lambda[i]  = ZERO;
              lambdahi[i] = ZERO;
            }
            /*
            else if (lambda[i] < eps_*maxLambda) {
              lambda[i]  = SCTM::squareroot(eps_*maxLambda);
              lambdahi[i] = MONE/lambda[i];
            }
            */
            else {
              lambda[i] = SCTM::squareroot(lambda[i]);
              lambdahi[i] = MONE/lambda[i];
            }
          }

          // compute X * D^{-.5} * U * Lambda^{-.5} and new Op*X
          //
          // copy X into workX
          std::vector<int> ind(xc);
          for (int i=0; i<xc; i++) {ind[i] = i;}
          MVT::SetBlock(X,ind,*workX);
          //
          // compute D^{-.5}*U*Lambda^{-.5} into workU
          workU.assign(XtMX);
          for (int j=0; j<xc; j++) {
            for (int i=0; i<xc; i++) {
              workU(i,j) *= Dhi[i]*lambdahi[j];
            }
          }
          // compute workX * workU into X
          MVT::MvTimesMatAddMv(ONE,*workX,workU,ZERO,X);
          //
          // note, it seems important to apply Op exactly for large condition numbers.
          // for small condition numbers, we can update MX "implicitly"
          // this trick reduces the number of applications of Op
          if (this->_hasOp) {
            if (maxLambda >= tolerance * minLambda) {
              // explicit update of MX
              OPT::Apply(*(this->_Op),X,*MX);
              this->_OpCounter += MVT::GetNumberVecs(X);
            }
            else {
              // implicit update of MX
              // copy MX into workX
              MVT::SetBlock(*MX,ind,*workX);
              //
              // compute workX * workU into MX
              MVT::MvTimesMatAddMv(ONE,*workX,workU,ZERO,*MX);
            }
          }

          // accumulate new B into previous B
          // B = Lh * U^H * Dh * B
          for (int j=0; j<xc; j++) {
            for (int i=0; i<xc; i++) {
              workU(i,j) = Dh[i] * (*B)(i,j);
            }
          }
          info = B->multiply(Teuchos::CONJ_TRANS,Teuchos::NO_TRANS,ONE,XtMX,workU,ZERO);
          TEST_FOR_EXCEPTION(info != 0, std::logic_error, "Anasazi::SVQBOrthoManager::findBasis(): Input error to SerialDenseMatrix::multiply.");
          for (int j=0; j<xc ;j++) {
            for (int i=0; i<xc; i++) {
              (*B)(i,j) *= lambda[i];
            }
          }

          // check iZeroMax (rank indicator)
          if (iZeroMax >= 0) {
            if (debug_) {
              std::cout << dbgstr << "augmenting multivec with " << iZeroMax+1 << " random directions" << std::endl;
            }

            numRand++;
            // put random info in the first iZeroMax+1 vectors of X,MX
            std::vector<int> ind(iZeroMax+1);
            for (int i=0; i<iZeroMax+1; i++) {
              ind[i] = i;
            }
            Teuchos::RCP<MV> Xnull,MXnull;
            Xnull = MVT::CloneView(X,ind);
            MVT::MvRandom(*Xnull);
            if (this->_hasOp) {
              MXnull = MVT::CloneView(*MX,ind);
              OPT::Apply(*(this->_Op),*Xnull,*MXnull);
              this->_OpCounter += MVT::GetNumberVecs(*Xnull);
              MXnull = Teuchos::null;
            }
            Xnull = Teuchos::null;
            condT = tolerance;
            doGramSchmidt = true;
            break; // break from while(condT > tolerance)
          }

          condT = SCTM::magnitude(maxLambda / minLambda);
          if (debug_) {
            std::cout << dbgstr << "condT: " << condT << std::endl;
          }
          
        } // end while (condT >= tolerance)

        if ((doGramSchmidt == false) && (condT > SCTM::squareroot(tolerance))) {
          doGramSchmidt = true;
        }
      }
      // end if(normalize)
       
    } // end while (doGramSchmidt)

    if (debug_) {
      std::cout << dbgstr << "(numGS,numSVQB,numRand)                : " 
           << "(" << numGS 
           << "," << numSVQB 
           << "," << numRand 
           << ")" << std::endl;
    }

    return xc;
  }

} // namespace Anasazi

#endif // ANASAZI_SVQB_ORTHOMANAGER_HPP


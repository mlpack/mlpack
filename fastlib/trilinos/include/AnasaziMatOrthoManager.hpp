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

/*! \file AnasaziMatOrthoManager.hpp
  \brief  Templated virtual class for providing orthogonalization/orthonormalization methods with matrix-based 
          inner products.
*/

#ifndef ANASAZI_MATORTHOMANAGER_HPP
#define ANASAZI_MATORTHOMANAGER_HPP

/*! \class Anasazi::MatOrthoManager
  
  \brief Anasazi's templated virtual class for providing routines for orthogonalization and 
  orthonormalization of multivectors using matrix-based inner products.

  This class extends Anasazi::OrthoManager by providing extra calling arguments to orthogonalization
  routines, to reduce the cost of applying the inner product in cases where the user already
  has the image of the source multivector under the inner product matrix.

  A concrete implementation of this class is necessary. The user can create
  their own implementation if those supplied are not suitable for their needs.
  
  \author Chris Baker, Ulrich Hetmaniuk, Rich Lehoucq, and Heidi Thornquist
*/

#include "AnasaziConfigDefs.hpp"
#include "AnasaziTypes.hpp"
#include "AnasaziOrthoManager.hpp"
#include "AnasaziMultiVecTraits.hpp"
#include "AnasaziOperatorTraits.hpp"

namespace Anasazi {

  template <class ScalarType, class MV, class OP>
  class MatOrthoManager : public OrthoManager<ScalarType,MV> {
  public:
    //! @name Constructor/Destructor
    //@{ 
    //! Default constructor.
    MatOrthoManager(Teuchos::RCP<const OP> Op = Teuchos::null);

    //! Destructor.
    virtual ~MatOrthoManager() {};
    //@}

    //! @name Accessor routines
    //@{ 

    //! Set operator used for inner product.
    void setOp( Teuchos::RCP<const OP> Op );

    //! Get operator used for inner product.
    Teuchos::RCP<const OP> getOp() const;

    //! Retrieve operator counter.
    /*! This counter returns the number of applications of the operator specifying the inner 
     * product. When the operator is applied to a multivector, the counter is incremented by the
     * number of vectors in the multivector. If the operator is not specified, the counter is never 
     * incremented.
     */
    int getOpCounter() const;

    //! Reset the operator counter to zero.
    /*! See getOpCounter() for more details.
     */
    void resetOpCounter();

    //@}

    //! @name Matrix-based Orthogonality Methods 
    //@{ 

    /*! \brief Provides a matrix-based inner product.
     *
     * Provides the inner product 
     * \f[
     *    \langle x, y \rangle = x^H M y
     * \f]
     * Optionally allows the provision of \f$M y\f$. See OrthoManager::innerProd() for more details.
     *
     */
    void innerProdMat( const MV& X, const MV& Y, Teuchos::RCP<const MV> MY, 
                         Teuchos::SerialDenseMatrix<int,ScalarType>& Z ) const;

    /*! \brief Provides the norm induced by the matrix-based inner product.
     *
     *  Provides the norm:
     *  \f[
     *     \|x\|_M = \sqrt{x^T H y}
     *  \f]
     *  Optionally allows the provision of \f$M x\f$. See OrthoManager::norm() for more details.
     */
    void normMat(const MV& X, Teuchos::RCP<const MV> MX, 
              std::vector< typename Teuchos::ScalarTraits<ScalarType>::magnitudeType > *normvec ) const;

    /*! \brief Provides matrix-based projection method.
     *
     * This method optionally allows the provision of \f$M X\f$. See OrthoManager::project() for more details.
     */
    virtual void projectMat ( 
        MV &X, 
        Teuchos::RCP<MV> MX = Teuchos::null, 
        Teuchos::Array<Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > > C = Teuchos::tuple(Teuchos::null), 
        Teuchos::Array<Teuchos::RCP<const MV> > Q = Teuchos::tuple(Teuchos::null) ) const = 0;

    /*! \brief Provides matrix-based orthonormalization method.
     *
     * This method optionally allows the provision of \f$M X\f$. See orthoManager::normalize() for more details.
    */
    virtual int normalizeMat ( 
        MV &X, 
        Teuchos::RCP<MV> MX = Teuchos::null, 
        Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > B = Teuchos::null ) const = 0;


    /*! \brief Provides matrix-based projection/orthonormalization method.
     *
     * This method optionally allows the provision of \f$M X\f$. See orthoManager::projectAndNormalize() for more details.
    */
    virtual int projectAndNormalizeMat ( 
        MV &X, Teuchos::RCP<MV> MX = Teuchos::null, 
        Teuchos::Array<Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > > C = Teuchos::tuple(Teuchos::null), 
        Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > B = Teuchos::null, 
        Teuchos::Array<Teuchos::RCP<const MV> > Q = Teuchos::tuple(Teuchos::null) ) const = 0;

    /*! \brief This method computes the error in orthonormality of a multivector.
     *
     *  This method optionally allows optionally exploits a caller-provided \c MX.
     */
    virtual typename Teuchos::ScalarTraits<ScalarType>::magnitudeType 
    orthonormErrorMat(const MV &X, Teuchos::RCP<const MV> MX = Teuchos::null) const = 0;

    /*! \brief This method computes the error in orthogonality of two multivectors.
     *
     *  This method optionally allows optionally exploits a caller-provided \c MX.
     */
    virtual typename Teuchos::ScalarTraits<ScalarType>::magnitudeType 
    orthogErrorMat(const MV &X1, Teuchos::RCP<const MV> MX1, const MV &X2) const = 0;

    //@}

    //! @name Methods implementing Anasazi::OrthoManager
    //@{ 

    /*! \brief Implements the interface OrthoManager::innerProd(). 
     *
     * This method calls 
     * \code
     * innerProdMat(X,Teuchos::null,Y,Z);
     * \endcode
     */
    void innerProd( const MV& X, const MV& Y, Teuchos::SerialDenseMatrix<int,ScalarType>& Z ) const;

    /*! \brief Implements the interface OrthoManager::norm(). 
     *
     * This method calls 
     * \code
     * normMat(X,Teuchos::null,normvec);
     * \endcode
     */
    void norm( const MV& X, std::vector< typename Teuchos::ScalarTraits<ScalarType>::magnitudeType > *normvec ) const;
    
    /*! \brief Implements the interface OrthoManager::project(). 
     *
     * This method calls 
     * \code
     * projectMat(X,Teuchos::null,C,Z);
     * \endcode
     */
    void project ( MV &X, 
                   Teuchos::Array<Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > > C = Teuchos::tuple(Teuchos::null), 
                   Teuchos::Array<Teuchos::RCP<const MV> > Q = Teuchos::tuple(Teuchos::null)) const;

    /*! \brief Implements the interface OrthoManager::normalize(). 
     *
     * This method calls 
     * \code
     * normalizeMat(X,Teuchos::null,B);
     * \endcode
     */
    int normalize ( MV &X, Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > B = Teuchos::null) const;

    /*! \brief Implements the interface OrthoManager::projectAndNormalize(). 
     *
     * This method calls 
     * \code
     * projectAndNormalizeMat(X,Teuchos::null,C,B,Q);
     * \endcode
     */
    int projectAndNormalize ( MV &X, 
                              Teuchos::Array<Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > > C = Teuchos::tuple(Teuchos::null), 
                              Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > B = Teuchos::null, 
                              Teuchos::Array<Teuchos::RCP<const MV> > Q = Teuchos::tuple(Teuchos::null) ) const;

    /*! \brief Implements the interface OrthoManager::orthonormError(). 
     *
     * This method calls 
     * \code
     * orthonormErrorMat(X,Teuchos::null);
     * \endcode
     */
    typename Teuchos::ScalarTraits<ScalarType>::magnitudeType 
    orthonormError(const MV &X) const;

    /*! \brief Implements the interface OrthoManager::orthogError(). 
     *
     * This method calls 
     * \code
     * orthogErrorMat(X1,Teuchos::null,X2);
     * \endcode
     */
    typename Teuchos::ScalarTraits<ScalarType>::magnitudeType 
    orthogError(const MV &X1, const MV &X2) const;

    //@}

  protected:
    Teuchos::RCP<const OP> _Op;
    bool _hasOp;
    mutable int _OpCounter;

  };

  template <class ScalarType, class MV, class OP>
  MatOrthoManager<ScalarType,MV,OP>::MatOrthoManager(Teuchos::RCP<const OP> Op)
      : _Op(Op), _hasOp(Op!=Teuchos::null) {}

  template <class ScalarType, class MV, class OP>
  void MatOrthoManager<ScalarType,MV,OP>::setOp( Teuchos::RCP<const OP> Op ) 
  { 
    _Op = Op; 
    _hasOp = (_Op != Teuchos::null);
  }

  template <class ScalarType, class MV, class OP>
  Teuchos::RCP<const OP> MatOrthoManager<ScalarType,MV,OP>::getOp() const 
  { 
    return _Op; 
  } 

  template <class ScalarType, class MV, class OP>
  int MatOrthoManager<ScalarType,MV,OP>::getOpCounter() const 
  {
    return _OpCounter;
  }

  template <class ScalarType, class MV, class OP>
  void MatOrthoManager<ScalarType,MV,OP>::resetOpCounter() 
  {
    _OpCounter = 0;
  }

  template <class ScalarType, class MV, class OP>
  void MatOrthoManager<ScalarType,MV,OP>::innerProd( 
      const MV& X, const MV& Y, Teuchos::SerialDenseMatrix<int,ScalarType>& Z ) const 
  {
    typedef Teuchos::ScalarTraits<ScalarType> SCT;
    typedef MultiVecTraits<ScalarType,MV>     MVT;
    typedef OperatorTraits<ScalarType,MV,OP>  OPT;

    Teuchos::RCP<const MV> P,Q;
    Teuchos::RCP<MV> R;

    if (_hasOp) {
      // attempt to minimize the amount of work in applying 
      if ( MVT::GetNumberVecs(X) < MVT::GetNumberVecs(Y) ) {
        R = MVT::Clone(X,MVT::GetNumberVecs(X));
        OPT::Apply(*_Op,X,*R);
        _OpCounter += MVT::GetNumberVecs(X);
        P = R;
        Q = Teuchos::rcp( &Y, false );
      }
      else {
        P = Teuchos::rcp( &X, false );
        R = MVT::Clone(Y,MVT::GetNumberVecs(Y));
        OPT::Apply(*_Op,Y,*R);
        _OpCounter += MVT::GetNumberVecs(Y);
        Q = R;
      }
    }
    else {
      P = Teuchos::rcp( &X, false );
      Q = Teuchos::rcp( &Y, false );
    }

    MVT::MvTransMv(SCT::one(),*P,*Q,Z);
  }

  template <class ScalarType, class MV, class OP>
  void MatOrthoManager<ScalarType,MV,OP>::innerProdMat( 
      const MV& X, const MV& Y, Teuchos::RCP<const MV> MY, Teuchos::SerialDenseMatrix<int,ScalarType>& Z ) const 
  {
    typedef Teuchos::ScalarTraits<ScalarType> SCT;
    typedef MultiVecTraits<ScalarType,MV>     MVT;
    typedef OperatorTraits<ScalarType,MV,OP>  OPT;

    Teuchos::RCP<MV> P,Q;

    if ( MY == Teuchos::null ) {
      innerProd(X,Y,Z);
    }
    else if ( _hasOp ) {
      // the user has done the matrix vector for us
      MVT::MvTransMv(SCT::one(),X,*MY,Z);
    }
    else {
      // there is no matrix vector
      MVT::MvTransMv(SCT::one(),X,Y,Z);
    }
  }

  template <class ScalarType, class MV, class OP>
  void MatOrthoManager<ScalarType,MV,OP>::norm( 
      const MV& X, std::vector< typename Teuchos::ScalarTraits<ScalarType>::magnitudeType > *normvec ) const 
  {
    this->normMat(X,Teuchos::null,normvec);
  }

  template <class ScalarType, class MV, class OP>
  void MatOrthoManager<ScalarType,MV,OP>::normMat( 
      const MV& X, Teuchos::RCP<const MV> MX, 
      std::vector< typename Teuchos::ScalarTraits<ScalarType>::magnitudeType > *normvec ) const 
  {
    typedef Teuchos::ScalarTraits<ScalarType> SCT;
    typedef MultiVecTraits<ScalarType,MV>     MVT;
    typedef OperatorTraits<ScalarType,MV,OP>  OPT;

    if (!_hasOp) {
      MX = Teuchos::rcp(&X,false);
    }
    else if (MX == Teuchos::null) {
      Teuchos::RCP<MV> R = MVT::Clone(X,MVT::GetNumberVecs(X));
      OPT::Apply(*_Op,X,*R);
      _OpCounter += MVT::GetNumberVecs(X);
      MX = R;
    }

    Teuchos::SerialDenseMatrix<int,ScalarType> z(1,1);
    Teuchos::RCP<const MV> Xi, MXi;
    std::vector<int> ind(1);
    for (int i=0; i<MVT::GetNumberVecs(X); i++) {
      ind[0] = i;
      Xi = MVT::CloneView(X,ind);
      MXi = MVT::CloneView(*MX,ind);
      MVT::MvTransMv(SCT::one(),*Xi,*MXi,z);
      (*normvec)[i] = SCT::magnitude( SCT::squareroot( z(0,0) ) );
    }
  }

  template <class ScalarType, class MV, class OP>
  void MatOrthoManager<ScalarType,MV,OP>::project ( 
      MV &X, 
      Teuchos::Array<Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > > C, 
      Teuchos::Array<Teuchos::RCP<const MV> > Q) const 
  {
    this->projectMat(X,Teuchos::null,C,Q);
  }

  template <class ScalarType, class MV, class OP>
  int MatOrthoManager<ScalarType,MV,OP>::normalize ( 
      MV &X, Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > B ) const 
  {
    return this->normalizeMat(X,Teuchos::null,B);
  }

  template <class ScalarType, class MV, class OP>
  int MatOrthoManager<ScalarType,MV,OP>::projectAndNormalize ( 
      MV &X, 
      Teuchos::Array<Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > > C, 
      Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > B, 
      Teuchos::Array<Teuchos::RCP<const MV> > Q ) const 
  {
    return this->projectAndNormalizeMat(X,Teuchos::null,C,B,Q);
  }

  template <class ScalarType, class MV, class OP>
  typename Teuchos::ScalarTraits<ScalarType>::magnitudeType 
  MatOrthoManager<ScalarType,MV,OP>::orthonormError(const MV &X) const 
  {
    return this->orthonormErrorMat(X,Teuchos::null);
  }

  template <class ScalarType, class MV, class OP>
  typename Teuchos::ScalarTraits<ScalarType>::magnitudeType 
  MatOrthoManager<ScalarType,MV,OP>::orthogError(const MV &X1, const MV &X2) const 
  {
    return this->orthogErrorMat(X1,Teuchos::null,X2);
  }

} // end of Anasazi namespace


#endif

// end of file AnasaziMatOrthoManager.hpp

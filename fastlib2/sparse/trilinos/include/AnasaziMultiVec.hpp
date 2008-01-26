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

/*! \file AnasaziMultiVec.hpp
  \brief Templated virtual class for creating multi-vectors that can interface with the Anasazi::MultiVecTraits class
*/

#ifndef ANASAZI_MULTI_VEC_HPP
#define ANASAZI_MULTI_VEC_HPP

#include "AnasaziConfigDefs.hpp"
#include "AnasaziMultiVecTraits.hpp"

namespace Anasazi {


/*! 	\class MultiVec

	\brief Anasazi's templated virtual class for constructing a multi-vector that can interface with the 
	MultiVecTraits class used by the eigensolvers.

	A concrete implementation of this class is necessary.  The user can create
	their own implementation if those supplied are not suitable for their needs.

	\author Ulrich Hetmaniuk, Rich Lehoucq, and Heidi Thornquist
*/
template <class ScalarType>
class MultiVec {
public:

    //! @name Constructor/Destructor
	//@{ 
	//! Anasazi::MultiVec constructor.
	MultiVec() {};

	//! Anasazi::MultiVec destructor.
	virtual ~MultiVec () {};

	//@}
    //! @name Creation methods
	//@{ 

	/*! \brief Creates a new empty Anasazi::MultiVec containing \c numvecs columns.

	    \return Pointer to the new multivector	
	*/

	virtual MultiVec<ScalarType> * Clone ( const int numvecs ) const = 0;

	/*! \brief Creates a new Anasazi::MultiVec and copies contents of \c *this into
	    the new vector (deep copy).
	
	    \return Pointer to the new multivector	
	*/
	
	virtual MultiVec<ScalarType> * CloneCopy () const = 0;
	
	/*! \brief Creates a new Anasazi::MultiVec and copies the selected contents of \c *this 
	    into the new vector (deep copy).  The copied 
	    vectors from \c *this are indicated by the \c index.size() indices in \c index.

	    \return Pointer to the new multivector	
	*/

	virtual MultiVec<ScalarType> * CloneCopy ( const std::vector<int>& index ) const = 0;
	
	/*! \brief Creates a new Anasazi::MultiVec that shares the selected contents of \c *this.
	    The index of the \c numvecs vectors shallow copied from \c *this are indicated by the
	    indices given in \c index.

	    \return Pointer to the new multivector	
	*/

	virtual MultiVec<ScalarType> * CloneView ( const std::vector<int>& index ) = 0;
	//@}

  //! @name Attribute methods	
	//@{ 
	//! Obtain the vector length of *this.

	virtual int GetVecLength () const = 0;

	//! Obtain the number of vectors in *this.

	virtual int GetNumberVecs () const = 0;

	//@}
  //! @name Update methods
	//@{ 
	/*! \brief Update \c *this with \c alpha * \c A * \c B + \c beta * (\c *this).
	*/

	virtual void MvTimesMatAddMv ( ScalarType alpha, const MultiVec<ScalarType>& A, 
		const Teuchos::SerialDenseMatrix<int,ScalarType>& B, ScalarType beta ) = 0;

	/*! \brief Replace \c *this with \c alpha * \c A + \c beta * \c B.
	*/

	virtual void MvAddMv ( ScalarType alpha, const MultiVec<ScalarType>& A, ScalarType beta, const MultiVec<ScalarType>& B ) = 0;

	/*! \brief Compute a dense matrix \c B through the matrix-matrix multiply 
	   \c alpha * \c A^T * (\c *this).
	*/

	virtual void MvTransMv ( ScalarType alpha, const MultiVec<ScalarType>& A, Teuchos::SerialDenseMatrix<int,ScalarType>& B
#ifdef HAVE_ANASAZI_EXPERIMENTAL
				 , ConjType conj = Anasazi::CONJ
#endif				 
				 ) const = 0;
  
        /*! \brief Compute a vector \c b where the components are the individual dot-products, i.e.\c b[i] = \c A[i]^H*\c this[i] where \c A[i] is the i-th column of A.
   */
  
        virtual void MvDot ( const MultiVec<ScalarType>& A, std::vector<ScalarType>* b 
#ifdef HAVE_ANASAZI_EXPERIMENTAL
			     , ConjType conj = Anasazi::CONJ
#endif
			     ) const = 0;
  
	//@}
  //! @name Norm method
	//@{ 

	/*! \brief Compute the 2-norm of each individual vector of \c *this.  
	   Upon return, \c normvec[i] holds the 2-norm of the \c i-th vector of \c *this
	*/

	virtual void MvNorm ( std::vector<typename Teuchos::ScalarTraits<ScalarType>::magnitudeType>* normvec ) const = 0;

	//@}
  //! @name Initialization methods
	//@{ 
	/*! \brief Copy the vectors in \c A to a set of vectors in \c *this.  The \c 
  	    numvecs vectors in \c A are copied to a subset of vectors in \c *this
	    indicated by the indices given in \c index.
	*/

	virtual void SetBlock ( const MultiVec<ScalarType>& A, const std::vector<int>& index ) = 0;
	
	/*! \brief Scale each element of the vectors in \c *this with \c alpha.
	*/

	virtual void MvScale ( ScalarType alpha ) = 0;

	/*! \brief Scale each element of the \c i-th vector in \c *this with \c alpha[i].
	*/

        virtual void MvScale ( const std::vector<ScalarType>& alpha ) = 0;

	/*! \brief Fill the vectors in \c *this with random numbers.
	*/

	virtual void MvRandom () = 0;

	/*! \brief Replace each element of the vectors in \c *this with \c alpha.
	*/

	virtual void MvInit ( ScalarType alpha ) = 0;

	//@}
  //! @name Print method
	//@{ 
	/*! \brief Print \c *this multivector to the \c os output stream.
	*/
	virtual void MvPrint ( std::ostream& os ) const = 0;
	//@}

};


  ////////////////////////////////////////////////////////////////////
  //
  // Implementation of the Anasazi::MultiVecTraits for Anasazi::MultiVec.
  //
  ////////////////////////////////////////////////////////////////////

  /*! 
    \brief Template specialization of Anasazi::MultiVecTraits class using the Anasazi::MultiVec virtual
    base class.

    Any class that inherits from Anasazi::MultiVec will be accepted by the Anasazi templated solvers due to this
    interface to the Anasazi::MultiVecTraits class.
  */

  template<class ScalarType>
  class MultiVecTraits<ScalarType,MultiVec<ScalarType> >
  {
  public:

    //! @name Creation methods
    //@{ 

    /*! \brief Creates a new empty \c Anasazi::MultiVec containing \c numvecs columns.
      
    \return Reference-counted pointer to the new \c Anasazi::MultiVec.
    */
    static Teuchos::RCP<MultiVec<ScalarType> > Clone( const MultiVec<ScalarType>& mv, const int numvecs )
    { return Teuchos::rcp( const_cast<MultiVec<ScalarType>&>(mv).Clone(numvecs) ); }

    /*! \brief Creates a new \c Anasazi::MultiVec and copies contents of \c mv into the new vector (deep copy).
      
      \return Reference-counted pointer to the new \c Anasazi::MultiVec.
    */
    static Teuchos::RCP<MultiVec<ScalarType> > CloneCopy( const MultiVec<ScalarType>& mv )
    { return Teuchos::rcp( const_cast<MultiVec<ScalarType>&>(mv).CloneCopy() ); }

    /*! \brief Creates a new \c Anasazi::MultiVec and copies the selected contents of \c mv into the new vector (deep copy).  

      The copied vectors from \c mv are indicated by the \c index.size() indices in \c index.      
      \return Reference-counted pointer to the new \c Anasazi::MultiVec.
    */
    static Teuchos::RCP<MultiVec<ScalarType> > CloneCopy( const MultiVec<ScalarType>& mv, const std::vector<int>& index )
    { return Teuchos::rcp( const_cast<MultiVec<ScalarType>&>(mv).CloneCopy(index) ); }

    /*! \brief Creates a new \c Anasazi::MultiVec that shares the selected contents of \c mv (shallow copy).

    The index of the \c numvecs vectors shallow copied from \c mv are indicated by the indices given in \c index.
    \return Reference-counted pointer to the new \c Anasazi::MultiVec.
    */    
    static Teuchos::RCP<MultiVec<ScalarType> > CloneView( MultiVec<ScalarType>& mv, const std::vector<int>& index )
    { return Teuchos::rcp( mv.CloneView(index) ); }

    /*! \brief Creates a new const \c Anasazi::MultiVec that shares the selected contents of \c mv (shallow copy).

    The index of the \c numvecs vectors shallow copied from \c mv are indicated by the indices given in \c index.
    \return Reference-counted pointer to the new const \c Anasazi::MultiVec.
    */      
    static Teuchos::RCP<const MultiVec<ScalarType> > CloneView( const MultiVec<ScalarType>& mv, const std::vector<int>& index )
    { return Teuchos::rcp( const_cast<MultiVec<ScalarType>&>(mv).CloneView(index) ); }

    //@}

    //! @name Attribute methods
    //@{ 

    //! Obtain the vector length of \c mv.
    static int GetVecLength( const MultiVec<ScalarType>& mv )
    { return mv.GetVecLength(); }

    //! Obtain the number of vectors in \c mv
    static int GetNumberVecs( const MultiVec<ScalarType>& mv )
    { return mv.GetNumberVecs(); }

    //@}

    //! @name Update methods
    //@{ 

    /*! \brief Update \c mv with \f$ \alpha AB + \beta mv \f$.
     */
    static void MvTimesMatAddMv( ScalarType alpha, const MultiVec<ScalarType>& A, 
				 const Teuchos::SerialDenseMatrix<int,ScalarType>& B, 
				 ScalarType beta, MultiVec<ScalarType>& mv )
    { mv.MvTimesMatAddMv(alpha, A, B, beta); }

    /*! \brief Replace \c mv with \f$\alpha A + \beta B\f$.
     */
    static void MvAddMv( ScalarType alpha, const MultiVec<ScalarType>& A, ScalarType beta, const MultiVec<ScalarType>& B, MultiVec<ScalarType>& mv )
    { mv.MvAddMv(alpha, A, beta, B); }

    /*! \brief Compute a dense matrix \c B through the matrix-matrix multiply \f$ \alpha A^Tmv \f$.
     */
    static void MvTransMv( ScalarType alpha, const MultiVec<ScalarType>& A, const MultiVec<ScalarType>& mv, Teuchos::SerialDenseMatrix<int,ScalarType>& B
#ifdef HAVE_ANASAZI_EXPERIMENTAL
			   , ConjType conj = Anasazi::CONJ
#endif
			   )
    { mv.MvTransMv(alpha, A, B
#ifdef HAVE_ANASAZI_EXPERIMENTAL
		   , conj
#endif
		   ); }
    
    /*! \brief Compute a vector \c b where the components are the individual dot-products of the \c i-th columns of \c A and \c mv, i.e.\f$b[i] = A[i]^H mv[i]\f$.
     */
    static void MvDot( const MultiVec<ScalarType>& mv, const MultiVec<ScalarType>& A, std::vector<ScalarType>* b
#ifdef HAVE_ANASAZI_EXPERIMENTAL
		       , ConjType conj = Anasazi::CONJ
#endif
		       )
    { mv.MvDot( A, b
#ifdef HAVE_ANASAZI_EXPERIMENTAL
		, conj
#endif
		); }

    /*! \brief Scale each element of the vectors in \c *this with \c alpha.
     */    
    static void MvScale ( MultiVec<ScalarType>& mv, ScalarType alpha )
    { mv.MvScale( alpha ); }
    
    /*! \brief Scale each element of the \c i-th vector in \c *this with \c alpha[i].
     */
    static void MvScale ( MultiVec<ScalarType>& mv, const std::vector<ScalarType>& alpha )
    { mv.MvScale( alpha ); }
    
    //@}
    //! @name Norm method
    //@{ 

    /*! \brief Compute the 2-norm of each individual vector of \c mv.  
      Upon return, \c normvec[i] holds the value of \f$||mv_i||_2\f$, the \c i-th column of \c mv.
    */
    static void MvNorm( const MultiVec<ScalarType>& mv, std::vector<typename Teuchos::ScalarTraits<ScalarType>::magnitudeType>* normvec )
    { mv.MvNorm(normvec); }

    //@}
    //! @name Initialization methods
    //@{ 
    /*! \brief Copy the vectors in \c A to a set of vectors in \c mv indicated by the indices given in \c index.

    The \c numvecs vectors in \c A are copied to a subset of vectors in \c mv indicated by the indices given in \c index,
    i.e.<tt> mv[index[i]] = A[i]</tt>.
    */
    static void SetBlock( const MultiVec<ScalarType>& A, const std::vector<int>& index, MultiVec<ScalarType>& mv )
    { mv.SetBlock(A, index); }

    /*! \brief Replace the vectors in \c mv with random vectors.
     */
    static void MvRandom( MultiVec<ScalarType>& mv )
    { mv.MvRandom(); }

    /*! \brief Replace each element of the vectors in \c mv with \c alpha.
     */
    static void MvInit( MultiVec<ScalarType>& mv, ScalarType alpha = Teuchos::ScalarTraits<ScalarType>::zero() )
    { mv.MvInit(alpha); }

    //@}

    //! @name Print method
    //@{ 

    /*! \brief Print the \c mv multi-vector to the \c os output stream.
     */
    static void MvPrint( const MultiVec<ScalarType>& mv, std::ostream& os )
    { mv.MvPrint(os); }

    //@}
  };


} // namespace Anasazi

#endif

// end of file AnasaziMultiVec.hpp

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


#ifndef _TEUCHOS_SERIALDENSEVECTOR_HPP_
#define _TEUCHOS_SERIALDENSEVECTOR_HPP_

/*! \file Teuchos_SerialDenseVector.hpp
    \brief Templated serial dense std::vector class
*/

#include "Teuchos_ConfigDefs.hpp"
#include "Teuchos_Object.hpp" 
#include "Teuchos_SerialDenseMatrix.hpp"

/*! \class Teuchos::SerialDenseVector
    \brief This class creates and provides basic support for dense vectors of templated type as a specialization of Teuchos::SerialDenseMatrix.  Additional methods for the SerialDenseVector class, like mathematical methods, can be found documented in SerialDenseMatrix.
*/
namespace Teuchos {

  template<typename OrdinalType, typename ScalarType>
  class SerialDenseVector : public SerialDenseMatrix<OrdinalType,ScalarType> {
    
  public:
    //! @name Constructor/Destructor methods.
  //@{ 

    //! Default Constructor
    /*! Creates an empty std::vector of no length.  The Sizing methods should be used to size this matrix.  Values of this matrix should be set using the [] or the () operators.
    */
    SerialDenseVector();

    //! Shaped Constructor
    /*!
	\param length - Number of elements in this std::vector.

	Creates a shaped std::vector of length \c length.  All values are initialized to zero.  Values of this std::vector should be set using [] or the () operators.
    */
    SerialDenseVector(OrdinalType length);

    //! Shaped Constructor with Values
    /*!
	\param CV - Enumerated type set to Teuchos::Copy or Teuchos::View.
	\param values - Pointer to an array of ScalarType of the given \c length.
	\param length - Length of std::vector to be constructed.
    */
    SerialDenseVector(DataAccess CV, ScalarType* values, OrdinalType length);

    //! Copy Constructor
    SerialDenseVector(const SerialDenseVector<OrdinalType,ScalarType>& Source);

    //! Destructor
    virtual ~SerialDenseVector ();
  //@}

  //! @name Sizing methods.
  //@{ 

    //! Size method for changing the size of a SerialDenseVector, initializing entries to zero.
    /*!
	\param length - The length of the new std::vector.

	This allows the user to define the length of a SerialDenseVector at any point.
	This method can be called at any point after construction.  Any values previously in
	this object will be destroyed and the resized std::vector starts with all zero values.
    */
    int size(OrdinalType length) {return(SerialDenseMatrix<OrdinalType, ScalarType>::shape(length, 1));}

    //! Resizing method for changing the size of a SerialDenseVector, keeping the entries.
    /*!
	\param length - The length of the new std::vector.
	This allows the user to redefine the length of a SerialDenseVector at any point.
	This method can be called at any point after construction.  Any values previously in
	this object will be copied to the resized std::vector.
    */	
    int resize(OrdinalType length) {return(SerialDenseMatrix<OrdinalType,ScalarType>::reshape(length, 1));}
  //@}

  //! @name Comparison methods.
  //@{ 
    //! Equality of two matrices.
    /*! \return True if \e this std::vector and \c Operand are of the same length and have the same entries, else False will be returned.
    */
    bool operator == (const SerialDenseVector<OrdinalType, ScalarType> &Operand);

    //! Inequality of two matrices.
    /*! \return True if \e this std::vector and \c Operand are not of the same length or do not have the same entries, else False will be returned.
    */
    bool operator != (const SerialDenseVector<OrdinalType, ScalarType> &Operand);
  //@}

  //! @name Set methods.
  //@{ 

    //! Copies values from one std::vector to another.
    /*!
	The operator= copies the values from one existing SerialDenseVector to
	another.  If \c Source is a view (i.e. CV = Teuchos::View), then this
	method will return a view.  Otherwise, it will return a copy of \c Source.
	\e this will be resized if it is not large enough to copy \c Source into.
    */
    SerialDenseVector<OrdinalType,ScalarType>& operator = (const SerialDenseVector<OrdinalType,ScalarType>& Source);
  //@}

  //! @name Accessor methods.
  //@{ 
    //! Element access method (non-const).
    /*! Returns the ith element if x(i) is specified, the expression x[i] will return the same element.
	\return (*this)(index)
	\warning The validity of \c index will only be checked if Teuchos is configured with --enable-teuchos-abc.
    */
    ScalarType& operator () (OrdinalType index);
    
    //! Element access method (const).
    /*! Returns the ith element if x(i) is specified, the expression x[i] will return the same element.
	\return (*this)(index)
	\warning The validity of \c index will only be checked if Teuchos is configured with --enable-teuchos-abc.
    */
    const ScalarType& operator () (OrdinalType index) const;

    //! Element access method (non-const).
    /*! Returns the ith element if x[i] is specified, the expression x(i) will return the same element.
	\return (*this)[index]
	\warning The validity of \c index will only be checked if Teuchos is configured with --enable-teuchos-abc.
    */
    ScalarType& operator [] (OrdinalType index);

    //! Element access method (const).
    /*! Returns the ith element if x[i] is specified, the expression x(i) will return the same element.
    	\return (*this)[index]
	\warning The validity of \c index will only be checked if Teuchos is configured with --enable-teuchos-abc.
    */
    const ScalarType& operator [] (OrdinalType index) const;

  //@}

  //! @name Attribute methods.
  //@{ 
    //! Returns the length of this std::vector.
    OrdinalType length() const {return(this->numRows_);}
  //@}

  //! @name I/O methods.
  //@{ 
    //! Print method.  Define the behavior of the std::ostream << operator inherited from the Object class.
    virtual void print(std::ostream& os) const;
  //@}
};

  template<typename OrdinalType, typename ScalarType>
  SerialDenseVector<OrdinalType, ScalarType>::SerialDenseVector() : SerialDenseMatrix<OrdinalType,ScalarType>() {}

  template<typename OrdinalType, typename ScalarType>
  SerialDenseVector<OrdinalType, ScalarType>::SerialDenseVector( OrdinalType length ) : SerialDenseMatrix<OrdinalType,ScalarType>( length, 1 ) {}

  template<typename OrdinalType, typename ScalarType>
  SerialDenseVector<OrdinalType, ScalarType>::SerialDenseVector(DataAccess CV, ScalarType* values, OrdinalType length) : 
    SerialDenseMatrix<OrdinalType,ScalarType>( CV, values, length, length, 1 ) {}

  template<typename OrdinalType, typename ScalarType>
  SerialDenseVector<OrdinalType, ScalarType>::SerialDenseVector(const SerialDenseVector<OrdinalType, ScalarType> &Source) :
    SerialDenseMatrix<OrdinalType,ScalarType>( Source ) {}

  template<typename OrdinalType, typename ScalarType>
  SerialDenseVector<OrdinalType, ScalarType>::~SerialDenseVector() {}
  
  template<typename OrdinalType, typename ScalarType>
  SerialDenseVector<OrdinalType, ScalarType>& SerialDenseVector<OrdinalType,ScalarType>::operator = (const SerialDenseVector<OrdinalType, ScalarType>& Source) 
  {
    SerialDenseMatrix<OrdinalType,ScalarType>::operator=(Source); 
    return(*this);
  }

  template<typename OrdinalType, typename ScalarType>
  bool SerialDenseVector<OrdinalType, ScalarType>::operator == (const SerialDenseVector<OrdinalType, ScalarType> &Operand) 
  {
    bool result = 1;
    if(this->numRows_ != Operand.numRows_)
      {
	result = 0;
      }
    else
      {
	OrdinalType i;
	for(i = 0; i < this->numRows_; i++) {
	  if((*this)(i) != Operand(i))
	    {
	      return 0;
	    }
	}
      }
    return result;
  }

  template<typename OrdinalType, typename ScalarType>
  bool SerialDenseVector<OrdinalType, ScalarType>::operator != (const SerialDenseVector<OrdinalType, ScalarType> &Operand)
  {
    return !((*this)==Operand);
  }

  template<typename OrdinalType, typename ScalarType>
  void SerialDenseVector<OrdinalType, ScalarType>::print(std::ostream& os) const
  {
    os << std::endl;
    if(this->valuesCopied_)
      os << "Values_copied : yes" << std::endl;
    else
      os << "Values_copied : no" << std::endl;
      os << "Length : " << this->numRows_ << std::endl;
    if(this->numRows_ == 0) {
      os << "(std::vector is empty, no values to display)" << std::endl;
    } else {
      for(OrdinalType i = 0; i < this->numRows_; i++) {
	  os << (*this)(i) << " ";
      }
      os << std::endl;
    }
  }

  //----------------------------------------------------------------------------------------------------
  //   Accessor methods 
  //----------------------------------------------------------------------------------------------------

  template<typename OrdinalType, typename ScalarType>
  inline ScalarType& SerialDenseVector<OrdinalType, ScalarType>::operator () (OrdinalType index)
  {
#ifdef HAVE_TEUCHOS_ARRAY_BOUNDSCHECK
    this->checkIndex( index );
#endif
    return(this->values_[index]);
  }
  
  template<typename OrdinalType, typename ScalarType>
  inline const ScalarType& SerialDenseVector<OrdinalType, ScalarType>::operator () (OrdinalType index) const
  {
#ifdef HAVE_TEUCHOS_ARRAY_BOUNDSCHECK
    this->checkIndex( index );
#endif
    return(this->values_[index]);
  }
  
  template<typename OrdinalType, typename ScalarType>
  inline const ScalarType& SerialDenseVector<OrdinalType, ScalarType>::operator [] (OrdinalType index) const
  {
#ifdef HAVE_TEUCHOS_ARRAY_BOUNDSCHECK
    this->checkIndex( index );
#endif
    return(this->values_[index]);
  }
  
  template<typename OrdinalType, typename ScalarType>
  inline ScalarType& SerialDenseVector<OrdinalType, ScalarType>::operator [] (OrdinalType index)
  {
#ifdef HAVE_TEUCHOS_ARRAY_BOUNDSCHECK
    this->checkIndex( index );
#endif
    return(this->values_[index]);
  }

} // namespace Teuchos

#endif /* _TEUCHOS_SERIALDENSEVECTOR_HPP_ */

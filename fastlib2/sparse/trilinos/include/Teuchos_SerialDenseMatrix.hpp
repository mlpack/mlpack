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

// Kris
// 06.18.03 -- Removed comments/documentation; file too hard to edit otherwise. Will replace later.
//          -- Begin conversion from <ScalarType> template to <OrdinalType, ScalarType>
// 06.23.03 -- Finished conversion from <ScalarType> to <OrdinalType, ScalarType>
//          -- Tpetra_DenseMatrix.cpp is now obsolete
//          -- Added new constructor to allow construction of a submatrix
//          -- Altered copyMat to enable its use in new constructor
//          -- Commented out broken print() function
//          -- Fixed oneNorm() (uninitialized return variable was causing erroneous results)
// 06.24.03 -- Minor formatting changes
// 07.01.03 -- Added TempPrint() function to temporarily take the place of print() and operator<< while I figure out how to fix them
// 07.02.03 -- Added operator== and operator!= to make testing programs easier to write/read. Implementation of == isn't the most
//             efficient/robust, but it works. Will consider optimizing later.
//          -- Warning! Constructor DenseMatrix(DataAccess, const DenseMatrix<OrdinalType, ScalarType> &, int, int, int, int) (the
//             "submatrix grabber" constructor) does not work correctly when used with CV == View (always grabs submatrix from top
//             left corner).
// 07.07.03 -- Constructor bug detailed above (07.02) is now corrected (hopefully).
// 07.08.03 -- Move into Teuchos package/namespace

#ifndef _TEUCHOS_SERIALDENSEMATRIX_HPP_
#define _TEUCHOS_SERIALDENSEMATRIX_HPP_
/*! \file Teuchos_SerialDenseMatrix.hpp
    \brief Templated serial dense matrix class
*/

#include "Teuchos_CompObject.hpp"
#include "Teuchos_BLAS.hpp"
#include "Teuchos_ScalarTraits.hpp"
#include "Teuchos_DataAccess.hpp"
#include "Teuchos_ConfigDefs.hpp"
#include "Teuchos_TestForException.hpp"

/*! 	\class Teuchos::SerialDenseMatrix
	\brief This class creates and provides basic support for dense rectangular matrix of templated type.
*/
/** \example DenseMatrix/cxx_main.cpp
    This is an example of how to use the Teuchos::SerialDenseMatrix class.
*/


namespace Teuchos {  

  template<typename OrdinalType, typename ScalarType>
  class SerialDenseMatrix : public CompObject, public Object, public BLAS<OrdinalType, ScalarType>
  {
  public:

    //! @name Constructor/Destructor methods.
  //@{ 

    //! Default Constructor
    /*! Creates a empty matrix of no dimension.  The Shaping methods should be used to size this matrix.
	Values of this matrix should be set using the [], (), or = operators.	
    */
    SerialDenseMatrix();

    //! Shaped Constructor
    /*! 
	\param numRows - Number of rows in matrix.
	\param numCols - Number of columns in matrix.

	Creates a shaped matrix with \c numRows rows and \c numCols cols.  All values are initialized to 0.
	Values of this matrix should be set using the [] or the () operators.
    */
    SerialDenseMatrix(OrdinalType numRows, OrdinalType numCols);

    //! Shaped Constructor with Values
    /*!
	\param CV - Enumerated type set to Teuchos::Copy or Teuchos::View.
	\param values - Pointer to an array of ScalarType.  The first column starts at \c values,
		the second at \c values+stride, etc.
	\param stride - The stride between the columns of the matrix in memory.
	\param numRows - Number of rows in matrix.
	\param numCols - Number of columns in matrix.
    */
    SerialDenseMatrix(DataAccess CV, ScalarType* values, OrdinalType stride, OrdinalType numRows, OrdinalType numCols);

    //! Copy Constructor
    /*! \note A deep copy of the \c Source transposed can be obtained if \c trans=Teuchos::TRANS, \c else
      a non-transposed copy of \c Source is made.  There is no storage of the transpose state of the matrix
      within the SerialDenseMatrix class, so this information will not propogate to any operation performed
      on a matrix that has been copy constructed in transpose.
    */
    SerialDenseMatrix(const SerialDenseMatrix<OrdinalType, ScalarType> &Source, ETransp trans = Teuchos::NO_TRANS);

    //! Submatrix Copy Constructor
    /*! 
	\param CV - Enumerated type set to Teuchos::Copy or Teuchos::View.
	\param Source - Reference to another dense matrix from which values are to be copied.
	\param numRows - The number of rows in this matrix.
	\param numCols - The number of columns in this matrix.
	\param startRow - The row of \c Source from which the submatrix copy should start. 
	\param startCol - The column of \c Source from which the submatrix copy should start.

	Creates a shaped matrix with \c numRows rows and \c numCols columns, which is a submatrix of \c Source.
	If \c startRow and \c startCol are not given, then the submatrix is the leading submatrix of \c Source.
	Otherwise, the (1,1) entry in the copied matrix is the (\c startRow, \c startCol) entry of \c Source.
    */
    SerialDenseMatrix(DataAccess CV, const SerialDenseMatrix<OrdinalType, ScalarType> &Source, OrdinalType numRows, OrdinalType numCols, OrdinalType startRow=0, OrdinalType startCol=0);

    //! Destructor
    virtual ~SerialDenseMatrix();
  //@}

  //! @name Shaping methods.
  //@{ 
    //! Shape method for changing the size of a SerialDenseMatrix, initializing entries to zero.
    /*!
	\param numRows - The number of rows in this matrix.
	\param numCols - The number of columns in this matrix.

	This method allows the user to define the dimensions of a SerialDenseMatrix at any point.  This method
	can be called at any point after construction.  Any values previously in this object will be destroyed
	and the resized matrix starts of with all zero values.

	\return Integer error code, set to 0 if successful.
    */
    int shape(OrdinalType numRows, OrdinalType numCols);

    //! Same as <tt>shape()</tt> except leaves uninitialized.
    int shapeUninitialized(OrdinalType numRows, OrdinalType numCols);

    //! Reshaping method for changing the size of a SerialDenseMatrix, keeping the entries.
    /*!
	\param numRows - The number of rows in this matrix.
	\param numCols - The number of columns in this matrix.

	This method allows the user to redefine the dimensions of a SerialDenseMatrix at any point.  This method
	can be called at any point after construction.  Any values previously in this object will be copied into
	the reshaped matrix.

	\return Integer error code, set 0 if successful.
    */
    int reshape(OrdinalType numRows, OrdinalType numCols);


  //@}

  //! @name Set methods.
  //@{ 

    //! Copies values from one matrix to another.
    /*!
	The operator= copies the values from one existing SerialDenseMatrix to another. 
        If \c Source is a view (i.e. CV = Teuchos::View), then this method will
        return a view.  Otherwise, it will return a copy of \c Source.  \e this object
        will be resized if it is not large enough to copy \c Source into.
    */	
    SerialDenseMatrix<OrdinalType, ScalarType>& operator= (const SerialDenseMatrix<OrdinalType, ScalarType>& Source);

    //! Copies values from one matrix to another.
    /*!
	The operator= copies the values from one existing SerialDenseMatrix to another 
        if the dimension of both matrices are the same.  If not, \e this matrix
	will be returned unchanged.
    */	
    SerialDenseMatrix<OrdinalType, ScalarType>& assign (const SerialDenseMatrix<OrdinalType, ScalarType>& Source);

    //! Set all values in the matrix to a constant value.
    /*!
	\param value - Value to use; zero if none specified.
	\return Integer error code, set to 0 if successful.
    */
    int putScalar( const ScalarType value = Teuchos::ScalarTraits<ScalarType>::zero() );

    //! Set all values in the matrix to be random numbers.
    int random();

  //@}

  //! @name Accessor methods.
  //@{ 

    //! Element access method (non-const).
    /*! Returns the element in the ith row and jth column if A(i,j) is specified, the
	expression A[j][i] will return the same element.

  	\return Element from the specified \c rowIndex row and \c colIndex column.
	\warning The validity of \c rowIndex and \c colIndex will only be checked if Teuchos is
	configured with --enable-teuchos-abc.
    */
    ScalarType& operator () (OrdinalType rowIndex, OrdinalType colIndex);

    //! Element access method (const).
    /*! Returns the element in the ith row and jth column if A(i,j) is specified, the expression
	A[j][i] will return the same element.

  	\return Element from the specified \c rowIndex row and \c colIndex column.
	\warning The validity of \c rowIndex and \c colIndex will only be checked if Teuchos is
	configured with --enable-teuchos-abc.
    */
    const ScalarType& operator () (OrdinalType rowIndex, OrdinalType colIndex) const;

    //! Column access method (non-const).
    /*! Returns the pointer to the ScalarType array at the jth column if A[j] is specified, the expression
	A[j][i] will return the same element as A(i,j).

	\return Pointer to the ScalarType array at the \c colIndex column ( \c values_+colIndex*stride_ ).
	\warning The validity of \c colIndex will only be checked if Teuchos is	configured with 
	--enable-teuchos-abc.
    */
    ScalarType* operator [] (OrdinalType colIndex);

    //! Column access method (const).
    /*! Returns the pointer to the ScalarType array at the jth column if A[j] is specified, the expression
	A[j][i] will return the same element as A(i,j).

	\return Pointer to the ScalarType array at the \c colIndex column ( \c values_+colIndex*stride_ ).
	\warning The validity of \c colIndex will only be checked if Teuchos is	configured with 
	--enable-teuchos-abc.
    */
    const ScalarType* operator [] (OrdinalType colIndex) const;

    //! Data array access method.
    /*! \return Pointer to the ScalarType data array contained in the object. */
    ScalarType* values() const { return(values_); }

  //@}

  //! @name Mathematical methods.
  //@{ 

    //! Add another matrix to \e this matrix.
    /*! Add \c Source to \e this if the dimension of both matrices are the same.  If not, \e this matrix
	will be returned unchanged.
    */
    SerialDenseMatrix<OrdinalType, ScalarType>& operator+= (const SerialDenseMatrix<OrdinalType, ScalarType>& Source);

    //! Subtract another matrix from \e this matrix.
    /*! Subtract \c Source from \e this if the dimension of both matrices are the same.  If not, \e this matrix
	will be returned unchanged.
    */
    SerialDenseMatrix<OrdinalType, ScalarType>& operator-= (const SerialDenseMatrix<OrdinalType, ScalarType>& Source);

    //! Scale \c this matrix by \c alpha; \c *this = \c alpha*\c *this.
    /*!
	\param alpha Scalar to multiply \e this by.
	\return Integer error code, set to 0 if successful.
    */
    int scale ( const ScalarType alpha );

    //! Point-wise scale \c this matrix by \c A; i.e. *this(i,j) *= A(i,j)
    /*! The values of \c *this matrix will be point-wise scaled by the values in A.
	If A and \c this matrix are not the same dimension \c this will be returned unchanged.

	\param B Teuchos::SerialDenseMatrix used to perform element-wise scaling of \e this.
	\return Integer error code, set to 0 if successful.
    */
    int scale ( const SerialDenseMatrix<OrdinalType, ScalarType>& A );

    //! Multiply \c A * \c B and add them to \e this; \e this = \c beta * \e this + \c alpha*A*B.
    /*!
	\param transa - Use the transpose of \c A if transa = Teuchos::TRANS, else don't use the 
	transpose if transa = Teuchos::NOTRANS.
	\param transb - Use the transpose of \c B if transb = Teuchos::TRANS, else don't use the
	transpose if transb = Teuchos::NOTRANS.
	\param alpha - The scaling factor for \c A * \c B.
	\param A - SerialDenseMatrix
	\param B - SerialDenseMatrix
	\param beta - The scaling factor for \e this.

	If the matrices \c A and \c B are not of the right dimension, consistent with \e this, then \e this
	matrix will not be altered and -1 will be returned.
	\return Integer error code, set to 0 if successful.
    */
    int multiply (ETransp transa, ETransp transb, ScalarType alpha, const SerialDenseMatrix<OrdinalType, ScalarType> &A, const SerialDenseMatrix<OrdinalType, ScalarType> &B, ScalarType beta);
  //@}

  //! @name Comparison methods.
  //@{ 

    //! Equality of two matrices.
    /*! \return True if \e this matrix and \c Operand are of the same shape (rows and columns) and have
	the same entries, else False will be returned.
    */
    bool operator== (const SerialDenseMatrix<OrdinalType, ScalarType> &Operand);

    //! Inequality of two matrices.
    /*! \return True if \e this matrix and \c Operand of not of the same shape (rows and columns) or don't
	have the same entries, else False will be returned.
    */
    bool operator!= (const SerialDenseMatrix<OrdinalType, ScalarType> &Operand);

  //@}

  //! @name Attribute methods.
  //@{ 

    //! Returns the row dimension of this matrix.
    OrdinalType numRows() const { return(numRows_); }

    //! Returns the column dimension of this matrix.
    OrdinalType numCols() const { return(numCols_); }

    //! Returns the stride between the columns of this matrix in memory.
    OrdinalType stride() const { return(stride_); }
  //@}

  //! @name Norm methods.
  //@{ 

    //! Returns the 1-norm of the matrix.
    typename ScalarTraits<ScalarType>::magnitudeType normOne() const;

    //! Returns the Infinity-norm of the matrix.
    typename ScalarTraits<ScalarType>::magnitudeType normInf() const;

    //! Returns the Frobenius-norm of the matrix.
    typename ScalarTraits<ScalarType>::magnitudeType normFrobenius() const; 
  //@}

  //! @name I/O methods.
  //@{ 
    //! Print method.  Defines the behavior of the std::ostream << operator inherited from the Object class.
    virtual void print(std::ostream& os) const;

  //@}
  protected:
    void copyMat(ScalarType* inputMatrix, OrdinalType strideInput, OrdinalType numRows, OrdinalType numCols, ScalarType* outputMatrix, OrdinalType strideOutput, OrdinalType startRow, OrdinalType startCol, ScalarType alpha = ScalarTraits<ScalarType>::zero() );
    void deleteArrays();
    void checkIndex( OrdinalType rowIndex, OrdinalType colIndex = 0 ) const;
    OrdinalType numRows_;
    OrdinalType numCols_;
    OrdinalType stride_;
    bool valuesCopied_;
    ScalarType* values_;

  }; // class Teuchos_SerialDenseMatrix

  //----------------------------------------------------------------------------------------------------
  //  Constructors and Destructor
  //----------------------------------------------------------------------------------------------------

  template<typename OrdinalType, typename ScalarType>
  SerialDenseMatrix<OrdinalType, ScalarType>::SerialDenseMatrix() : CompObject(), numRows_(0), numCols_(0), stride_(0), valuesCopied_(false), values_(0) {}
  
  template<typename OrdinalType, typename ScalarType>
  SerialDenseMatrix<OrdinalType, ScalarType>::SerialDenseMatrix( OrdinalType numRows, OrdinalType numCols ) : CompObject(), numRows_(numRows), numCols_(numCols), stride_(numRows)
  {
    values_ = new ScalarType[stride_*numCols_];
    putScalar();
    valuesCopied_ = true;
  }

  template<typename OrdinalType, typename ScalarType>
  SerialDenseMatrix<OrdinalType, ScalarType>::SerialDenseMatrix(DataAccess CV, ScalarType* values, OrdinalType stride, OrdinalType numRows, OrdinalType numCols) : CompObject(), numRows_(numRows), numCols_(numCols), stride_(stride), valuesCopied_(false), values_(values)
  {
    if(CV == Copy)
      {
	stride_ = numRows_;
	values_ = new ScalarType[stride_*numCols_];
	copyMat(values, stride, numRows_, numCols_, values_, stride_, 0, 0, false);
	valuesCopied_ = true;
      }
  }
  
  template<typename OrdinalType, typename ScalarType>
  SerialDenseMatrix<OrdinalType, ScalarType>::SerialDenseMatrix(const SerialDenseMatrix<OrdinalType, ScalarType> &Source, ETransp trans) : CompObject(), numRows_(0), numCols_(0), stride_(0), valuesCopied_(true), values_(0)
  {
    if ( trans == Teuchos::NO_TRANS ) 
      {
	numRows_ = Source.numRows_;
	numCols_ = Source.numCols_;
	stride_ = numRows_;
	values_ = new ScalarType[stride_*numCols_];
	copyMat(Source.values_, Source.stride_, numRows_, numCols_, values_, stride_, 0, 0, false);
      } 
    else if ( trans == Teuchos::CONJ_TRANS && ScalarTraits<ScalarType>::isComplex ) 
      {	  
	numRows_ = Source.numCols_;
	numCols_ = Source.numRows_;
	stride_ = numRows_;
	values_ = new ScalarType[stride_*numCols_];
	for (OrdinalType j=0; j<numCols_; j++) {
	  for (OrdinalType i=0; i<numRows_; i++) {
	    values_[j*stride_ + i] = Teuchos::ScalarTraits<ScalarType>::conjugate(Source.values_[i*Source.stride_ + j]);
	  }
	}
      } 
    else 
      {
	numRows_ = Source.numCols_;
	numCols_ = Source.numRows_;
	stride_ = numRows_;
	values_ = new ScalarType[stride_*numCols_];
	for (OrdinalType j=0; j<numCols_; j++) {
	  for (OrdinalType i=0; i<numRows_; i++) {
	    values_[j*stride_ + i] = Source.values_[i*Source.stride_ + j];
	  }
	}
      }
  }
  
  template<typename OrdinalType, typename ScalarType>
  SerialDenseMatrix<OrdinalType, ScalarType>::SerialDenseMatrix(DataAccess CV, const SerialDenseMatrix<OrdinalType, ScalarType> &Source, OrdinalType numRows, OrdinalType numCols, OrdinalType startRow, OrdinalType startCol) : CompObject(), numRows_(numRows), numCols_(numCols), stride_(Source.stride_), valuesCopied_(false), values_(Source.values_)
  {
    if(CV == Copy)
      {
	stride_ = numRows;
	values_ = new ScalarType[stride_ * numCols];
	copyMat(Source.values_, Source.stride_, numRows, numCols, values_, stride_, startRow, startCol, false);
	valuesCopied_ = true;
      }
    else // CV == View
      {
	values_ = values_ + (stride_ * startCol) + startRow;
      }
  }
    
  template<typename OrdinalType, typename ScalarType>
  SerialDenseMatrix<OrdinalType, ScalarType>::~SerialDenseMatrix()
  {
    deleteArrays();
  }
  
  //----------------------------------------------------------------------------------------------------
  //  Shape methods 
  //----------------------------------------------------------------------------------------------------

  template<typename OrdinalType, typename ScalarType>
  int SerialDenseMatrix<OrdinalType, ScalarType>::shape(OrdinalType numRows, OrdinalType numCols)
  {
    deleteArrays(); // Get rid of anything that might be already allocated
    numRows_ = numRows;
    numCols_ = numCols;
    stride_ = numRows_;
    values_ = new ScalarType[stride_*numCols_];
    putScalar();
    valuesCopied_ = true;
    return(0);
  }

  template<typename OrdinalType, typename ScalarType>
  int SerialDenseMatrix<OrdinalType, ScalarType>::shapeUninitialized(OrdinalType numRows, OrdinalType numCols)
  {
    deleteArrays(); // Get rid of anything that might be already allocated
    numRows_ = numRows;
    numCols_ = numCols;
    stride_ = numRows_;
    values_ = new ScalarType[stride_*numCols_];
    valuesCopied_ = true;
    return(0);
  }
  
  template<typename OrdinalType, typename ScalarType>
  int SerialDenseMatrix<OrdinalType, ScalarType>::reshape(OrdinalType numRows, OrdinalType numCols)
  {
    // Allocate space for new matrix
    ScalarType* values_tmp = new ScalarType[numRows * numCols];
    ScalarType zero = ScalarTraits<ScalarType>::zero();
    for(OrdinalType k = 0; k < numRows * numCols; k++)
      {
	values_tmp[k] = zero;
      }
    OrdinalType numRows_tmp = TEUCHOS_MIN(numRows_, numRows);
    OrdinalType numCols_tmp = TEUCHOS_MIN(numCols_, numCols);
    if(values_ != 0)
      {
	copyMat(values_, stride_, numRows_tmp, numCols_tmp, values_tmp, numRows, 0, 0, false); // Copy principal submatrix of A to new A
      }
    deleteArrays(); // Get rid of anything that might be already allocated
    numRows_ = numRows;
    numCols_ = numCols;
    stride_ = numRows_;
    values_ = values_tmp; // Set pointer to new A
    valuesCopied_ = true;
    return(0);
  }
   
  //----------------------------------------------------------------------------------------------------
  //   Set methods 
  //----------------------------------------------------------------------------------------------------

  template<typename OrdinalType, typename ScalarType>
  int SerialDenseMatrix<OrdinalType, ScalarType>::putScalar( const ScalarType value )
  {
    // Set each value of the dense matrix to "value".
    for(OrdinalType j = 0; j < numCols_; j++) 
      {
	for(OrdinalType i = 0; i < numRows_; i++) 
	  {
	    values_[i + j*stride_] = value;
	  }
      }
    return 0;
  }    
    
  template<typename OrdinalType, typename ScalarType>
  int SerialDenseMatrix<OrdinalType, ScalarType>::random()
  {
    // Set each value of the dense matrix to a random value.
    for(OrdinalType j = 0; j < numCols_; j++) 
      {
	for(OrdinalType i = 0; i < numRows_; i++) 
	  {
	    values_[i + j*stride_] = ScalarTraits<ScalarType>::random();
	  }
      }
    return 0;
  }

  template<typename OrdinalType, typename ScalarType>
  SerialDenseMatrix<OrdinalType,ScalarType>& SerialDenseMatrix<OrdinalType, ScalarType>::operator= (const SerialDenseMatrix<OrdinalType,ScalarType>& Source) {
    if(this == &Source)
      return(*this); // Special case of source same as target
    if((!valuesCopied_) && (!Source.valuesCopied_) && (values_ == Source.values_))
      return(*this); // Special case of both are views to same data.

    // If the source is a view then we will return a view, else we will return a copy.
    if (!Source.valuesCopied_) {
      if(valuesCopied_)       { 
	// Clean up stored data if this was previously a copy.
	deleteArrays();
      }
      numRows_ = Source.numRows_; 
      numCols_ = Source.numCols_;
      stride_ = Source.stride_;
      values_ = Source.values_;
    }
    else {
      // If we were a view, we will now be a copy.
      if(!valuesCopied_) {
	numRows_ = Source.numRows_;
	numCols_ = Source.numCols_;
	stride_ = Source.numRows_;
	const OrdinalType newsize = stride_ * numCols_;
	if(newsize > 0) {
	  values_ = new ScalarType[newsize];
	  valuesCopied_ = true;
	}
	else {
	  values_ = 0;
	}
      }
      // If we were a copy, we will stay a copy.
      else {
	if((Source.numRows_ <= stride_) && (Source.numCols_ == numCols_)) { // we don't need to reallocate
	  numRows_ = Source.numRows_;
	  numCols_ = Source.numCols_;
	}
	else { // we need to allocate more space (or less space)
	  deleteArrays();
	  numRows_ = Source.numRows_;
	  numCols_ = Source.numCols_;
	  stride_ = Source.numRows_;
	  const OrdinalType newsize = stride_ * numCols_;
	  if(newsize > 0) {
	    values_ = new ScalarType[newsize];
	    valuesCopied_ = true;
	  }
	}
      }
      copyMat(Source.values_, Source.stride_, numRows_, numCols_, values_, stride_, 0, 0, false);
    } 
    return(*this);
  }

  template<typename OrdinalType, typename ScalarType>
  SerialDenseMatrix<OrdinalType, ScalarType>& SerialDenseMatrix<OrdinalType, ScalarType>::operator+= (const SerialDenseMatrix<OrdinalType,ScalarType>& Source )
  {
    // Check for compatible dimensions
    if ((numRows_ != Source.numRows_) || (numCols_ != Source.numCols_))
      {
	TEUCHOS_CHK_REF(*this); // Return *this without altering it.
      }
    copyMat(Source.values_, Source.stride_, numRows_, numCols_, values_, stride_, 0, 0, 1.0);
    return(*this);
  }

  template<typename OrdinalType, typename ScalarType>
  SerialDenseMatrix<OrdinalType, ScalarType>& SerialDenseMatrix<OrdinalType, ScalarType>::operator-= (const SerialDenseMatrix<OrdinalType,ScalarType>& Source )
  {
    // Check for compatible dimensions
    if ((numRows_ != Source.numRows_) || (numCols_ != Source.numCols_))
      {
	TEUCHOS_CHK_REF(*this); // Return *this without altering it.
      }
    copyMat(Source.values_, Source.stride_, numRows_, numCols_, values_, stride_, 0, 0, -1.0);
    return(*this);
  }

  template<typename OrdinalType, typename ScalarType>
  SerialDenseMatrix<OrdinalType,ScalarType>& SerialDenseMatrix<OrdinalType, ScalarType>::assign (const SerialDenseMatrix<OrdinalType,ScalarType>& Source) {
    if(this == &Source)
      return(*this); // Special case of source same as target
    if((!valuesCopied_) && (!Source.valuesCopied_) && (values_ == Source.values_))
      return(*this); // Special case of both are views to same data.

    // Check for compatible dimensions
    if ((numRows_ != Source.numRows_) || (numCols_ != Source.numCols_))
      {
	TEUCHOS_CHK_REF(*this); // Return *this without altering it.
      }
    copyMat(Source.values_, Source.stride_, numRows_, numCols_, values_, stride_, 0, 0 );
    return(*this);
  }

  //----------------------------------------------------------------------------------------------------
  //   Accessor methods 
  //----------------------------------------------------------------------------------------------------

  template<typename OrdinalType, typename ScalarType>
  inline ScalarType& SerialDenseMatrix<OrdinalType, ScalarType>::operator () (OrdinalType rowIndex, OrdinalType colIndex)
  {
#ifdef HAVE_TEUCHOS_ARRAY_BOUNDSCHECK
    checkIndex( rowIndex, colIndex );
#endif
    return(values_[colIndex * stride_ + rowIndex]);
  }
  
  template<typename OrdinalType, typename ScalarType>
  inline const ScalarType& SerialDenseMatrix<OrdinalType, ScalarType>::operator () (OrdinalType rowIndex, OrdinalType colIndex) const
  {
#ifdef HAVE_TEUCHOS_ARRAY_BOUNDSCHECK
    checkIndex( rowIndex, colIndex );
#endif
    return(values_[colIndex * stride_ + rowIndex]);
  }
  
  template<typename OrdinalType, typename ScalarType>
  inline const ScalarType* SerialDenseMatrix<OrdinalType, ScalarType>::operator [] (OrdinalType colIndex) const
  {
#ifdef HAVE_TEUCHOS_ARRAY_BOUNDSCHECK
    checkIndex( 0, colIndex );
#endif
    return(values_ + colIndex * stride_);
  }
  
  template<typename OrdinalType, typename ScalarType>
  inline ScalarType* SerialDenseMatrix<OrdinalType, ScalarType>::operator [] (OrdinalType colIndex)
  {
#ifdef HAVE_TEUCHOS_ARRAY_BOUNDSCHECK
    checkIndex( 0, colIndex );
#endif
  return(values_ + colIndex * stride_);
  }

  //----------------------------------------------------------------------------------------------------
  //   Norm methods 
  //----------------------------------------------------------------------------------------------------
  
  template<typename OrdinalType, typename ScalarType>
  typename ScalarTraits<ScalarType>::magnitudeType SerialDenseMatrix<OrdinalType, ScalarType>::normOne() const
  {
    OrdinalType i, j;
    typename ScalarTraits<ScalarType>::magnitudeType anorm = ScalarTraits<ScalarType>::magnitude(ScalarTraits<ScalarType>::zero());
    typename ScalarTraits<ScalarType>::magnitudeType absSum = ScalarTraits<ScalarType>::magnitude(ScalarTraits<ScalarType>::zero());
    ScalarType* ptr;
    for(j = 0; j < numCols_; j++)
      {
	ScalarType sum = 0;
	ptr = values_ + j * stride_;
	for(i = 0; i < numRows_; i++)
	  {
	    sum += ScalarTraits<ScalarType>::magnitude(*ptr++);
	  }
	absSum = ScalarTraits<ScalarType>::magnitude(sum);
	if(absSum > anorm)
	  {
	    anorm = absSum;
	  }
      }
    updateFlops(numRows_ * numCols_);
    return(anorm);
  }
  
  template<typename OrdinalType, typename ScalarType>
  typename ScalarTraits<ScalarType>::magnitudeType SerialDenseMatrix<OrdinalType, ScalarType>::normInf() const
  {
    OrdinalType i, j;
    typename ScalarTraits<ScalarType>::magnitudeType sum, anorm = ScalarTraits<ScalarType>::magnitude(ScalarTraits<ScalarType>::zero());
    
    for (i = 0; i < numRows_; i++) {
      sum = ScalarTraits<ScalarType>::magnitude(ScalarTraits<ScalarType>::zero());
      for (j=0; j< numCols_; j++) {
	sum += ScalarTraits<ScalarType>::magnitude(*(values_+i+j*stride_));
      }
      anorm = TEUCHOS_MAX( anorm, sum );
    }
    updateFlops(numRows_ * numCols_);
    return(anorm);
  }
  
  template<typename OrdinalType, typename ScalarType>
  typename ScalarTraits<ScalarType>::magnitudeType SerialDenseMatrix<OrdinalType, ScalarType>::normFrobenius() const
  {
    OrdinalType i, j;
    typename ScalarTraits<ScalarType>::magnitudeType anorm = ScalarTraits<ScalarType>::magnitude(ScalarTraits<ScalarType>::zero());
    for (j = 0; j < numCols_; j++) {
      for (i = 0; i < numRows_; i++) {
	anorm += ScalarTraits<ScalarType>::magnitude(values_[i+j*stride_]*values_[i+j*stride_]);
      }
    }
    anorm = ScalarTraits<ScalarType>::magnitude(ScalarTraits<ScalarType>::squareroot(anorm));
    updateFlops(numRows_ * numCols_);
    return(anorm);
  }
  
  //----------------------------------------------------------------------------------------------------
  //   Comparison methods 
  //----------------------------------------------------------------------------------------------------
  
  template<typename OrdinalType, typename ScalarType>
  bool SerialDenseMatrix<OrdinalType, ScalarType>::operator== (const SerialDenseMatrix<OrdinalType, ScalarType> &Operand)
  {
    bool result = 1;
    if((numRows_ != Operand.numRows_) || (numCols_ != Operand.numCols_))
      {
	result = 0;
      }
    else
      {
	OrdinalType i, j;
	for(i = 0; i < numRows_; i++)
	  {
	    for(j = 0; j < numCols_; j++)
	      {
		if((*this)(i, j) != Operand(i, j))
		  {
		    return 0;
		  }
	      }
	  }
      }
    return result;
  }
  
  template<typename OrdinalType, typename ScalarType>
  bool SerialDenseMatrix<OrdinalType, ScalarType>::operator!= (const SerialDenseMatrix<OrdinalType, ScalarType> &Operand)
  {
    return !((*this) == Operand);
  }
  
  //----------------------------------------------------------------------------------------------------
  //   Multiplication method 
  //----------------------------------------------------------------------------------------------------  

  template<typename OrdinalType, typename ScalarType>
  int SerialDenseMatrix<OrdinalType, ScalarType>::scale( const ScalarType alpha )
  {
    OrdinalType i, j;
    ScalarType* ptr;
    
    for (j=0; j<numCols_; j++) {
      ptr = values_ + j*stride_;
      for (i=0; i<numRows_; i++) { *ptr = alpha * (*ptr); ptr++; }
    }
    updateFlops( numRows_*numCols_ );
    return(0);
  }

  template<typename OrdinalType, typename ScalarType>
  int SerialDenseMatrix<OrdinalType, ScalarType>::scale( const SerialDenseMatrix<OrdinalType,ScalarType>& A )
  {
    OrdinalType i, j;
    ScalarType* ptr;
    
    // Check for compatible dimensions
    if ((numRows_ != A.numRows_) || (numCols_ != A.numCols_))
      {
        TEUCHOS_CHK_ERR(-1); // Return error
      }    
    for (j=0; j<numCols_; j++) {
      ptr = values_ + j*stride_;
      for (i=0; i<numRows_; i++) { *ptr = A(i,j) * (*ptr); ptr++; }
    }
    updateFlops( numRows_*numCols_ );
    return(0);
  }

  template<typename OrdinalType, typename ScalarType>
  int  SerialDenseMatrix<OrdinalType, ScalarType>::multiply(ETransp transa, ETransp transb, ScalarType alpha, const SerialDenseMatrix<OrdinalType, ScalarType> &A, const SerialDenseMatrix<OrdinalType, ScalarType> &B, ScalarType beta)
  {
    // Check for compatible dimensions
    OrdinalType A_nrows = (ETranspChar[transa]!='N') ? A.numCols() : A.numRows();
    OrdinalType A_ncols = (ETranspChar[transa]!='N') ? A.numRows() : A.numCols();
    OrdinalType B_nrows = (ETranspChar[transb]!='N') ? B.numCols() : B.numRows();
    OrdinalType B_ncols = (ETranspChar[transb]!='N') ? B.numRows() : B.numCols();
    if ((numRows_ != A_nrows) || (A_ncols != B_nrows) || (numCols_ != B_ncols))
      {
        TEUCHOS_CHK_ERR(-1); // Return error
      }
    // Call GEMM function
    this->GEMM(transa, transb, numRows_, numCols_, A_ncols, alpha, A.values(), A.stride(), B.values(), B.stride(), beta, values_, stride_);
    double nflops = 2 * numRows_;
    nflops *= numCols_;
    nflops *= A_ncols;
    updateFlops(nflops);
    return(0);
  }
  
  
  template<typename OrdinalType, typename ScalarType>
  void SerialDenseMatrix<OrdinalType, ScalarType>::print(std::ostream& os) const
  {
    os << std::endl;
    if(valuesCopied_)
      os << "Values_copied : yes" << std::endl;
    else
      os << "Values_copied : no" << std::endl;
      os << "Rows : " << numRows_ << std::endl;
      os << "Columns : " << numCols_ << std::endl;
      os << "LDA : " << stride_ << std::endl;
    if(numRows_ == 0 || numCols_ == 0) {
      os << "(matrix is empty, no values to display)" << std::endl;
    } else {
      for(OrdinalType i = 0; i < numRows_; i++) {
	for(OrdinalType j = 0; j < numCols_; j++){
	  os << (*this)(i,j) << " ";
	}
	os << std::endl;
      }
    }
  }

  //----------------------------------------------------------------------------------------------------
  //   Protected methods 
  //----------------------------------------------------------------------------------------------------  

  template<typename OrdinalType, typename ScalarType>
  inline void SerialDenseMatrix<OrdinalType, ScalarType>::checkIndex( OrdinalType rowIndex, OrdinalType colIndex ) const {
    TEST_FOR_EXCEPTION(rowIndex < 0 || rowIndex >= numRows_, std::out_of_range,
                       "SerialDenseMatrix<T>::checkIndex: "
                       "Row index " << rowIndex << " out of range [0, "<< numRows_ << ")");
    TEST_FOR_EXCEPTION(colIndex < 0 || colIndex >= numCols_, std::out_of_range,
                       "SerialDenseMatrix<T>::checkIndex: "
                       "Col index " << colIndex << " out of range [0, "<< numCols_ << ")");
  }

  template<typename OrdinalType, typename ScalarType>
  void SerialDenseMatrix<OrdinalType, ScalarType>::deleteArrays(void)
  {
    if (valuesCopied_)
      {
	delete [] values_;
	values_ = 0;
	valuesCopied_ = false;
      }
  }
  
  template<typename OrdinalType, typename ScalarType>
  void SerialDenseMatrix<OrdinalType, ScalarType>::copyMat(ScalarType* inputMatrix, OrdinalType strideInput, OrdinalType numRows, OrdinalType numCols, ScalarType* outputMatrix, OrdinalType strideOutput, OrdinalType startRow, OrdinalType startCol, ScalarType alpha)
  {
    OrdinalType i, j;
    ScalarType* ptr1 = 0;
    ScalarType* ptr2 = 0;
    for(j = 0; j < numCols; j++) {
	ptr1 = outputMatrix + (j * strideOutput);
	ptr2 = inputMatrix + (j + startCol) * strideInput + startRow;
	if (alpha != Teuchos::ScalarTraits<ScalarType>::zero() ) {
	  for(i = 0; i < numRows; i++)
	    {
	      *ptr1++ += alpha*(*ptr2++);
	    }
	} else {
	  for(i = 0; i < numRows; i++)
	    {
	      *ptr1++ = *ptr2++;
	    }
	}
    }
  }
  
} // namespace Teuchos


#endif /* _TEUCHOS_SERIALDENSEMATRIX_HPP_ */

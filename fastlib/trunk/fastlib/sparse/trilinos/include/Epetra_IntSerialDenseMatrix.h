
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

#ifndef EPETRA_INTSERIALDENSEMATRIX_H
#define EPETRA_INTSERIALDENSEMATRIX_H

#include "Epetra_Object.h" 

//! Epetra_IntSerialDenseMatrix: A class for constructing and using general dense integer matrices.

/*! The Epetra_IntSerialDenseMatrix class enables the construction and use of integer-valued, general
    dense matrices. 

The Epetra_IntSerialDenseMatrix class is intended to provide very basic support for dense rectangular matrices.


<b>Constructing Epetra_IntSerialDenseMatrix Objects</b>

There are four Epetra_IntSerialDenseMatrix constructors.  The first constructs a zero-sized object which should be made
to appropriate length using the Shape() or Reshape() functions and then filled with the [] or () operators. 
The second constructs an object sized to the dimensions specified, which should be filled with the [] or () operators.
The third is a constructor that accepts user
data as a 2D array, and the fourth is a copy constructor. The third constructor has
two data access modes (specified by the Epetra_DataAccess argument):
<ol>
  <li> Copy mode - Allocates memory and makes a copy of the user-provided data. In this case, the
       user data is not needed after construction.
  <li> View mode - Creates a "view" of the user data. In this case, the
       user data is required to remain intact for the life of the object.
</ol>

\warning View mode is \e extremely dangerous from a data hiding perspective.
Therefore, we strongly encourage users to develop code using Copy mode first and 
only use the View mode in a secondary optimization phase.

Epetra_IntSerialDenseMatrix constructors will throw an exception if an error occurrs.  
These exceptions will alway be negative integer values as follows:
<ol>
  <li> -1  Invalid dimension specified.
  <li> -2  Shape returned non-zero.
  <li> -3  Null pointer specified for user's data.
  <li> -99 Internal Epetra_IntSerialDenseMatrix error.  Contact developer.
</ol>

Other Epetra_IntSerialDenseMatrix functions that do not return an integer error code
(such as operators () and [] ) will throw an exception if an error occurrs. 
These exceptions will be integer values as follows:
<ol>
  <li> -1  Invalid row specified.
  <li> -2  Invalid column specified.
	<li> -5  Invalid assignment (type mismatch).
  <li> -99 Internal Epetra_IntSerialDenseMatrix error.  Contact developer.
</ol>


b<b>Extracting Data from Epetra_IntSerialDenseMatrix Objects</b>

Once a Epetra_IntSerialDenseMatrix is constructed, it is possible to view the data via access functions.

\warning Use of these access functions cam be \e extremely dangerous from a data hiding perspective.


<b>Vector and Utility Functions</b>

Once a Epetra_IntSerialDenseMatrix is constructed, several mathematical functions can be applied to
the object.  Specifically:
<ul>
  <li> Multiplication.
  <li> Norms.
</ul>


*/


//=========================================================================
class Epetra_IntSerialDenseMatrix : public Epetra_Object {

  public:
  
    //! @name Constructor/Destructor Methods
  //@{ 
  //! Default constructor; defines a zero size object.
  /*!
    Epetra_IntSerialDenseMatrix objects defined by the default constructor should be sized with the 
    Shape() or Reshape functions.  
    Values should be defined by using the [] or () operators.
   */
  Epetra_IntSerialDenseMatrix();
  
  //! Shaped constructor; defines a variable-sized object
  /*!
    \param In 
           NumRows - Number of rows in object.
    \param In 
           NumCols - Number of columns in object.

    Epetra_SerialDenseMatrix objects defined by the shaped constructor are already shaped to the
		dimensions given as a parameters. All values are initialized to 0. Calling this constructor 
		is equivalent to using the default constructor, and then calling the Shape function on it.
    Values should be defined by using the [] or () operators.
   */
  Epetra_IntSerialDenseMatrix(int NumRows, int NumCols);

  //! Set object values from two-dimensional array.
  /*!
    \param In 
           Epetra_DataAccess - Enumerated type set to Copy or View.
    \param In
           A - Pointer to an array of integer numbers.  The first vector starts at A.
	   The second vector starts at A+LDA, the third at A+2*LDA, and so on.
    \param In
           LDA - The "Leading Dimension", or stride between vectors in memory.
    \param In 
           NumRows - Number of rows in object.
    \param In 
           NumCols - Number of columns in object.

	   See Detailed Description section for further discussion.
  */
  Epetra_IntSerialDenseMatrix(Epetra_DataAccess CV, int* A, int LDA, int NumRows, int NumCols);
  
  //! Epetra_IntSerialDenseMatrix copy constructor.
	/*!
		This matrix will take on the data access mode of the Source matrix.
	*/
  Epetra_IntSerialDenseMatrix(const Epetra_IntSerialDenseMatrix& Source);

  //! Epetra_IntSerialDenseMatrix destructor.  
  virtual ~Epetra_IntSerialDenseMatrix ();
  //@}

  //! @name Shaping/sizing Methods
  //@{ 
  //! Set dimensions of a Epetra_IntSerialDenseMatrix object; init values to zero.
  /*!
    \param In 
           NumRows - Number of rows in object.
    \param In 
           NumCols - Number of columns in object.

	   Allows user to define the dimensions of a Epetra_IntSerialDenseMatrix at any point. This function can
	   be called at any point after construction.  Any values that were previously in this object are
	   destroyed and the resized matrix starts off with all zero values.

    \return Integer error code, set to 0 if successful.
  */
  int Shape(int NumRows, int NumCols);
  
  //! Reshape a Epetra_IntSerialDenseMatrix object.
  /*!
    \param In 
           NumRows - Number of rows in object.
    \param In 
           NumCols - Number of columns in object.

	   Allows user to define the dimensions of a Epetra_IntSerialDenseMatrix at any point. This function can
	   be called at any point after construction.  Any values that were previously in this object are
	   copied into the new shape.  If the new shape is smaller than the original, the upper left portion
	   of the original matrix (the principal submatrix) is copied to the new matrix.

    \return Integer error code, set to 0 if successful.
  */
  int Reshape(int NumRows, int NumCols);
  //@}
  
  //! @name Data Accessor methods
  //@{ 

  //! Computes the 1-Norm of the \e this matrix.
  /*!
    \return Integer error code, set to 0 if successful.
  */
  virtual int OneNorm();

  //! Computes the Infinity-Norm of the \e this matrix.
  virtual int InfNorm();

  //! Copy from one matrix to another.
  /*!
    The operator= allows one to copy the values from one existing IntSerialDenseMatrix to another.
		The left hand side matrix will take on the data access mode of the right hand side matrix. 

    \return Values of the left hand side matrix are modified by the values of the right hand side matrix.
  */
    Epetra_IntSerialDenseMatrix& operator = (const Epetra_IntSerialDenseMatrix& Source);

    //! Comparison operator.
    /*! operator== compares two Epetra_IntSerialDenseMatrix objects, returns false if sizes are different,
      or if any coefficients differ.
    */
    bool operator==(const Epetra_IntSerialDenseMatrix& rhs) const;

    //! Inequality operator
    /*! operator!= simply returns the negation of operator==.
     */
    bool operator!=(const Epetra_IntSerialDenseMatrix& rhs) const
    { return !(*this == rhs); }

  //! Element access function.
  /*!
    The parentheses operator returns the element in the ith row and jth column if A(i,j) is
    specified, the expression A[j][i] (note that i and j are reversed) will return the same element.
    Thus, A(i,j) = A[j][i] for all valid i and j.

    \return Element from the specified row and column.

		\warning No bounds checking is done unless Epetra is compiled with HAVE_EPETRA_ARRAY_BOUNDS_CHECK.
  */
    int& operator () (int RowIndex, int ColIndex);

  //! Element access function.
  /*!
    The parentheses operator returns the element in the ith row and jth column if A(i,j) is
    specified, the expression A[j][i] (note that i and j are reversed) will return the same element.
    Thus, A(i,j) = A[j][i] for all valid i and j.

    \return Element from the specified row and column.

		\warning No bounds checking is done unless Epetra is compiled with HAVE_EPETRA_ARRAY_BOUNDS_CHECK.
  */
    const int& operator () (int RowIndex, int ColIndex) const;

  //! Column access function.
  /*!
    The parentheses operator returns the element in the ith row and jth column if A(i,j) is
    specified, the expression A[j][i] (note that i and j are reversed) will return the same element.
    Thus, A(i,j) = A[j][i] for all valid i and j.

    \return Pointer to address of specified column.

    \warning No bounds checking can be done for the index i in the expression A[j][i].
		\warning No bounds checking is done unless Epetra is compiled with HAVE_EPETRA_ARRAY_BOUNDS_CHECK.
  */
    int* operator [] (int ColIndex);

  //! Column access function.
  /*!
    The parentheses operator returns the element in the ith row and jth column if A(i,j) is
    specified, the expression A[j][i] (note that i and j are reversed) will return the same element.
    Thus, A(i,j) = A[j][i] for all valid i and j.

    \return Pointer to address of specified column.

    \warning No bounds checking can be done for the index i in the expression A[j][i].
		\warning No bounds checking is done unless Epetra is compiled with HAVE_EPETRA_ARRAY_BOUNDS_CHECK.
  */
    const int* operator [] (int ColIndex) const;

  //! Set matrix values to random numbers.
  /*! 
		IntSerialDenseMatrix uses the random number generator provided by Epetra_Util.
		The matrix values will be set to random values on the interval (0, 2^31 - 1).

    \return Integer error code, set to 0 if successful.
  */
  int Random();
    
  //! Returns row dimension of system.
  int M() const {return(M_);};

  //! Returns column dimension of system.
  int N() const {return(N_);};

  //! Returns const pointer to the \e this matrix.
  const int* A() const {return(A_);};

  //! Returns pointer to the \e this matrix.
  int* A() {return(A_);};

  //! Returns the leading dimension of the \e this matrix.
  int LDA() const {return(LDA_);};

	//! Returns the data access mode of the \e this matrix.
	Epetra_DataAccess CV() const {return(CV_);};
  //@}
  
  //! @name I/O methods
  //@{ 
  //! Print service methods; defines behavior of ostream << operator.
  virtual void Print(ostream& os) const;
  //@}

  //! @name Expert-only unsupported methods
  //@{ 

  //! Reset an existing IntSerialDenseMatrix to point to another Matrix.
	/*! Allows an existing IntSerialDenseMatrix to become a View of another
		matrix's data, regardless of the DataAccess mode of the Source matrix.
		It is assumed that the Source matrix is an independent matrix, and 
		no checking is done to verify this.

		This is used by Epetra_CrsGraph in the OptimizeStorage method. It is used so that
		an existing (Copy) matrix can be converted to a View. This frees up
		memory that CrsGraph no longer needs.
		
		@param Source The IntSerialDenseMatrix this will become a view of.
		
		\return Integer error code, set to 0 if successful, and set to -1 
		if a type mismatch occured.
		
		\warning This method is extremely dangerous and should only be used by experts.
	*/
	
	int MakeViewOf(const Epetra_IntSerialDenseMatrix& Source);
	//@}

 protected:

	void CopyMat(int* Source, int Source_LDA, int NumRows, int NumCols, int* Target, int Target_LDA);
  void CleanupData();

	Epetra_DataAccess CV_;
	bool A_Copied_;
  int M_;
  int N_;
  int LDA_;
  int* A_;

};

// inlined definitions of op() and op[]
//=========================================================================
inline int& Epetra_IntSerialDenseMatrix::operator () (int RowIndex, int ColIndex) {
#ifdef HAVE_EPETRA_ARRAY_BOUNDS_CHECK
  if(RowIndex >= M_ || RowIndex < 0) 
		throw ReportError("Row index = " + toString(RowIndex) + 
											" Out of Range 0 - " + toString(M_-1),-1);
  if(ColIndex >= N_ || ColIndex < 0) 
		throw ReportError("Column index = " + toString(ColIndex) + 
											" Out of Range 0 - " + toString(N_-1),-2);
#endif
  return(A_[ColIndex*LDA_ + RowIndex]);
}
//=========================================================================
inline const int& Epetra_IntSerialDenseMatrix::operator () (int RowIndex, int ColIndex) const {
#ifdef HAVE_EPETRA_ARRAY_BOUNDS_CHECK
  if(RowIndex >= M_ || RowIndex < 0) 
		throw ReportError("Row index = " + toString(RowIndex) + 
											" Out of Range 0 - " + toString(M_-1),-1);
  if(ColIndex >= N_ || ColIndex < 0) 
		throw ReportError("Column index = " + toString(ColIndex) + 
											" Out of Range 0 - " + toString(N_-1),-2);
#endif
	return(A_[ColIndex * LDA_ + RowIndex]);
}
//=========================================================================
inline int* Epetra_IntSerialDenseMatrix::operator [] (int ColIndex) {
#ifdef HAVE_EPETRA_ARRAY_BOUNDS_CHECK
  if(ColIndex >= N_ || ColIndex < 0) 
		throw ReportError("Column index = " + toString(ColIndex) + 
											" Out of Range 0 - " + toString(N_-1),-2);
#endif
  return(A_+ ColIndex * LDA_);
}
//=========================================================================
inline const int* Epetra_IntSerialDenseMatrix::operator [] (int ColIndex) const {
#ifdef HAVE_EPETRA_ARRAY_BOUNDS_CHECK
  if(ColIndex >= N_ || ColIndex < 0) 
		throw ReportError("Column index = " + toString(ColIndex) + 
											" Out of Range 0 - " + toString(N_-1),-2);
#endif
  return(A_ + ColIndex * LDA_);
}
//=========================================================================

#endif /* EPETRA_INTSERIALDENSEMATRIX_H */

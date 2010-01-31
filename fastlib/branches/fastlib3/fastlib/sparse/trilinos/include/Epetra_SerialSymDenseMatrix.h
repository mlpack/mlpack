
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

#ifndef EPETRA_SERIALSYMDENSEMATRIX_H
#define EPETRA_SERIALSYMDENSEMATRIX_H
#include "Epetra_SerialDenseMatrix.h"


//! Epetra_SerialSymDenseMatrix: A class for constructing and using symmetric positive definite dense matrices.

/*! The Epetra_SerialSymDenseMatrix class enables the construction and use of 
    real-valued, symmetric positive definite, 
    double-precision dense matrices.  It is built on the Epetra_SerialDenseMatrix class which 
    in turn is built on the 
    BLAS via the Epetra_BLAS class. 

The Epetra_SerialSymDenseMatrix class is intended to provide full-featured support for solving 
linear and eigen system
problems for symmetric positive definite matrices.  It is written on top of BLAS and LAPACK 
and thus has excellent
performance and numerical capabilities.  Using this class, one can either perform simple 
factorizations and solves or
apply all the tricks available in LAPACK to get the best possible solution for very 
ill-conditioned problems.

<b>Epetra_SerialSymDenseMatrix vs. Epetra_LAPACK</b>

The Epetra_LAPACK class provides access to most of the same functionality as 
Epetra_SerialSymDenseMatrix.
The primary difference is that Epetra_LAPACK is a "thin" layer on top of 
LAPACK and Epetra_SerialSymDenseMatrix
attempts to provide easy access to the more sophisticated aspects of 
solving dense linear and eigensystems.
<ul>
<li> When you should use Epetra_LAPACK:  If you are simply looking for a 
     convenient wrapper around the Fortran LAPACK
     routines and you have a well-conditioned problem, you should probably use Epetra_LAPACK directly.
<li> When you should use Epetra_SerialSymDenseMatrix: If you want to (or potentially want to) 
     solve ill-conditioned 
     problems or want to work with a more object-oriented interface, you should 
     probably use Epetra_SerialSymDenseMatrix.
     
</ul>

<b>Constructing Epetra_SerialSymDenseMatrix Objects</b>

There are three Epetra_DenseMatrix constructors.  The first constructs a zero-sized object 
which should be made
to appropriate length using the Shape() or Reshape() functions and then filled with 
the [] or () operators. 
The second is a constructor that accepts user
data as a 2D array, the third is a copy constructor. The second constructor has
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

<b>Extracting Data from Epetra_SerialSymDenseMatrix Objects</b>

Once a Epetra_SerialSymDenseMatrix is constructed, it is possible to view the data via access functions.

\warning Use of these access functions cam be \e extremely dangerous from a data hiding perspective.


<b>Vector and Utility Functions</b>

Once a Epetra_SerialSymDenseMatrix is constructed, several mathematical functions can be applied to
the object.  Specifically:
<ul>
  <li> Multiplication.
  <li> Norms.
</ul>

<b>Counting floating point operations </b>
The Epetra_SerialSymDenseMatrix class has Epetra_CompObject as a base class.  Thus, floating 
point operations 
are counted and accumulated in the Epetra_Flop object (if any) that was set using the SetFlopCounter()
method in the Epetra_CompObject base class.

*/

//=========================================================================
class Epetra_SerialSymDenseMatrix : public Epetra_SerialDenseMatrix {

 public:
   //! @name Constructor/Destructor Methods
  //@{ 
  //! Default constructor; defines a zero size object.
  /*!
    Epetra_SerialSymDenseMatrix objects defined by the default constructor 
    should be sized with the Shape() 
    or Reshape() functions.  
    Values should be defined by using the [] or ()operators.

    Note: By default the active part of the matrix is assumed to be in the lower triangle.
    To set the upper part as active, call SetUpper(). 
    See Detailed Description section for further discussion.
   */
  Epetra_SerialSymDenseMatrix(void);
  //! Set object values from two-dimensional array.
  /*!
    \param In 
           Epetra_DataAccess - Enumerated type set to Copy or View.
    \param In
           A - Pointer to an array of double precision numbers.  The first vector starts at A.
	   The second vector starts at A+LDA, the third at A+2*LDA, and so on.
    \param In
           LDA - The "Leading Dimension", or stride between vectors in memory.
    \param In 
           NumRowsCols - Number of rows and columns in object.

	   Note: By default the active part of the matrix is assumed to be in the lower triangle.
	   To set the upper part as active, call SetUpper(). 
	   See Detailed Description section for further discussion.
  */
  Epetra_SerialSymDenseMatrix(Epetra_DataAccess CV, double *A, int LDA, int NumRowsCols);
  
  //! Epetra_SerialSymDenseMatrix copy constructor.
  
  Epetra_SerialSymDenseMatrix(const Epetra_SerialSymDenseMatrix& Source);
  
  
  //! Epetra_SerialSymDenseMatrix destructor.  
  virtual ~Epetra_SerialSymDenseMatrix ();
  //@}

  //! @name Set Methods
  //@{ 

  //let the compiler know we intend to overload the base-class Shape function,
  //rather than hide it.  
  using Epetra_SerialDenseMatrix::Shape;

  //! Set dimensions of a Epetra_SerialSymDenseMatrix object; init values to zero.
  /*!
    \param In 
           NumRowsCols - Number of rows and columns in object.

	   Allows user to define the dimensions of a Epetra_DenseMatrix at any point. This function can
	   be called at any point after construction.  Any values that were previously in this object are
	   destroyed and the resized matrix starts off with all zero values.

    \return Integer error code, set to 0 if successful.
  */
  int Shape(int NumRowsCols) {return(Epetra_SerialDenseMatrix::Shape(NumRowsCols,NumRowsCols));};

  //let the compiler know we intend to overload the base-class Reshape function,
  //rather than hide it.
  
  using Epetra_SerialDenseMatrix::Reshape;

  //! Reshape a Epetra_SerialSymDenseMatrix object.
  /*!
    \param In 
           NumRowsCols - Number of rows and columns in object.

	   Allows user to define the dimensions of a Epetra_SerialSymDenseMatrix at any point. This function can
	   be called at any point after construction.  Any values that were previously in this object are
	   copied into the new shape.  If the new shape is smaller than the original, the upper left portion
	   of the original matrix (the principal submatrix) is copied to the new matrix.

    \return Integer error code, set to 0 if successful.
  */
  int Reshape(int NumRowsCols) {return(Epetra_SerialDenseMatrix::Reshape(NumRowsCols,NumRowsCols));};


  //! Specify that the lower triangle of the \e this matrix should be used.
  void SetLower() {Upper_ = false; UPLO_ = 'L';};

  //! Specify that the upper triangle of the \e this matrix should be used.
  void SetUpper() {Upper_ = true; UPLO_ = 'U';};
  //@}

  //! @name Query methods
  //@{ 

  //! Returns true if upper triangle of \e this matrix has and will be used.
  bool Upper() const {return(Upper_);};

  //! Returns character value of UPLO used by LAPACK routines.
  char UPLO() const {return(UPLO_);};
  //@}

  //! @name Mathematical Methods
  //@{ 

  //! Inplace scalar-matrix product A = \e a A.
  /*! Scale a matrix, entry-by-entry using the value ScalarA.  This method is sensitive to 
      the UPLO() parameter.


  \param ScalarA (In) Scalar to multiply with A.

   \return Integer error code, set to 0 if successful.
	 
  */
  int  Scale ( double ScalarA );


  //! Computes the 1-Norm of the \e this matrix.
  /*!
    \return Integer error code, set to 0 if successful.
  */
  double NormOne() const;

  //! Computes the Infinity-Norm of the \e this matrix.
  double NormInf() const;

  //@}

  void CopyUPLOMat(bool Upper, double * A, int LDA, int NumRows);

  //! @name Deprecated methods (will be removed in later versions of this class)
  //@{ 

  //! Computes the 1-Norm of the \e this matrix (identical to NormOne() method).
  /*!
    \return Integer error code, set to 0 if successful.
  */
  double OneNorm() const {return(Epetra_SerialSymDenseMatrix::NormOne());};

  //! Computes the Infinity-Norm of the \e this matrix (identical to NormInf() method).
  double InfNorm() const {return(Epetra_SerialSymDenseMatrix::NormInf());};
  //@}

 private:

  bool Upper_;

  char UPLO_;


};

#endif /* EPETRA_SERIALSYMDENSEMATRIX_H */

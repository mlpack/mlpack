
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

#ifndef EPETRA_SERIALDENSEVECTOR_H
#define EPETRA_SERIALDENSEVECTOR_H

#include "Epetra_Object.h" 
#include "Epetra_SerialDenseMatrix.h"

//! Epetra_SerialDenseVector: A class for constructing and using dense vectors.

/*! The Epetra_SerialDenseVector class enables the construction and use of real-valued, 
    double-precision dense vectors.  It is built on the BLAS and LAPACK and derives from the Epetra_SerialDenseMatrix class.

The Epetra_SerialDenseVector class is intended to provide convenient vector notation but derives all signficant 
functionality from Epetra_SerialDenseMatrix.

<b>Constructing Epetra_SerialDenseVector Objects</b>

There are four Epetra_SerialDenseVector constructors.  The first constructs a zero-length object which should be made
to appropriate length using the Size() or Resize() functions and then filled with the [] or () operators. 
The second constructs an object sized to the dimension specified, which should be filled with the [] or () operators.
The third is a constructor that accepts user
data as a 1D array, and the fourth is a copy constructor. The third constructor has
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

<b>Extracting Data from Epetra_SerialDenseVector Objects</b>

Once a Epetra_SerialDenseVector is constructed, it is possible to view the data via access functions.

\warning Use of these access functions cam be \e extremely dangerous from a data hiding perspective.


The final useful function is Flops().  Each Epetra_SerialDenseVector object keep track of the number
of \e serial floating point operations performed using the specified object as the \e this argument
to the function.  The Flops() function returns this number as a double precision number.  Using this 
information, in conjunction with the Epetra_Time class, one can get accurate parallel performance
numbers.


*/


//=========================================================================
class Epetra_SerialDenseVector : public Epetra_SerialDenseMatrix{

  public:
  
    //! @name Constructors/destructors
  //@{ 
  //! Default constructor; defines a zero size object.
  /*!
    Epetra_SerialDenseVector objects defined by the default constructor should be sized with the 
    Size() or Resize functions.  
    Values should be defined by using the [] or () operators.
   */
  Epetra_SerialDenseVector();
  
  //! Sized constructor; defines a variable-sized object
  /*!
    \param In 
           Length - Length of vector.

    Epetra_SerialDenseVector objects defined by the sized constructor are already sized to the
		dimension given as a parameter. All values are initialized to 0. Calling this constructor 
		is equivalent to using the default constructor, and then calling the Size function on it.
    Values should be defined by using the [] or () operators.
   */
  Epetra_SerialDenseVector(int Length);

  //! Set object values from one-dimensional array.
  /*!
    \param In 
           Epetra_DataAccess - Enumerated type set to Copy or View.
    \param In
           Values - Pointer to an array of double precision numbers containing the values.
    \param In 
           Length - Length of vector.

	   See Detailed Description section for further discussion.
  */
  Epetra_SerialDenseVector(Epetra_DataAccess CV, double* Values, int Length);
  
  //! Epetra_SerialDenseVector copy constructor.
  
  Epetra_SerialDenseVector(const Epetra_SerialDenseVector& Source);
  

  //! Epetra_SerialDenseVector destructor.  
  virtual ~Epetra_SerialDenseVector ();
  //@}

  //! @name Post-construction modification routines
  //@{ 

  //! Set length of a Epetra_SerialDenseVector object; init values to zero.
  /*!
    \param In 
           Length - Length of vector object.

	   Allows user to define the dimension of a Epetra_SerialDenseVector. This function can
	   be called at any point after construction.  Any values that were previously in this object are
	   destroyed and the resized vector starts off with all zero values.

    \return Integer error code, set to 0 if successful.
  */
  int Size(int Length) {return(Epetra_SerialDenseMatrix::Shape(Length, 1));};
  
  //! Resize a Epetra_SerialDenseVector object.
  /*!
    \param In 
           Length - Length of vector object.

	   Allows user to define the dimension of a Epetra_SerialDenseVector. This function can
	   be called at any point after construction.  Any values that were previously in this object are
	   copied into the new size.  If the new shape is smaller than the original, the first Length values
	   are copied to the new vector.

    \return Integer error code, set to 0 if successful.
  */
  int Resize(int Length) {return(Epetra_SerialDenseMatrix::Reshape(Length, 1));};

  //@}

  //! @name Element access methods
  //@{ 
  //! Value copy from one vector to another.
  /*!
    The operator= allows one to copy the values from one existing SerialDenseVector to another, as
    long as there is enough room in the target to hold the source.

    \return Values of the left hand side vector are modified by the values of the right hand side vector.
  */
    Epetra_SerialDenseVector& operator = (const Epetra_SerialDenseVector& Source);

    //let the compiler know we intend to overload the base-class function
    //operator() rather than hide it.
    using Epetra_SerialDenseMatrix::operator();

  //! Element access function.
  /*!
    Returns the specified element of the vector.  Bounds checking is enforced.
    \return Specified element in vector.

    \warning No bounds checking is done unless Epetra is compiled with HAVE_EPETRA_ARRAY_BOUNDS_CHECK.
  */
    double& operator () (int Index);

  //! Element access function.
  /*!
    Returns the specified element of the vector.  Bounds checking is enforced.
    \return Specified element in vector.

    \warning No bounds checking is done unless Epetra is compiled with HAVE_EPETRA_ARRAY_BOUNDS_CHECK.
  */
    const double& operator () (int Index) const;

  //! Element access function.
  /*!
    Returns the specified element of the vector.
    \return Specified element in vector.

    \warning No bounds checking is done unless Epetra is compiled with HAVE_EPETRA_ARRAY_BOUNDS_CHECK.
  */
    double& operator [] (int Index);

  //! Column access function.
  /*!
    Returns the specified element of the vector.
    \return Specified element in vector.

    \warning No bounds checking is done unless Epetra is compiled with HAVE_EPETRA_ARRAY_BOUNDS_CHECK.
  */
    const double& operator [] (int Index) const;

  //@}

  //! @name Mathematical methods
  //@{ 
  //! Set vector values to random numbers.
  /*! 
		SerialDenseVector uses the random number generator provided by Epetra_Util.
		The vector values will be set to random values on the interval (-1.0, 1.0).

    \return Integer error code, set to 0 if successful.
  */
  int Random();
    
  //! Compute 1-norm of each vector in multi-vector.
  /*!
    \param x (In) Input vector x.

    \return Dot-product of the \e this vector and x.
  */
  double Dot(const Epetra_SerialDenseVector & x) const;

  //! Compute 1-norm of each vector in multi-vector.
  /*!
    \return 1-norm of the vector.
  */
  double Norm1() const;

  //! Compute 2-norm of each vector in multi-vector.
  /*!
    \param Out
    \return 2-norm of the vector.
  */
  double Norm2() const;

  //! Compute Inf-norm of each vector in multi-vector.
  /*!
    \return Infinity-norm of the vector.
  */
  double NormInf() const;

  //@}

  //! @name Attribute access methods
  //@{ 
  //! Returns length of vector.
  int Length() const {return(M_);};

  //! Returns pointer to the values in vector.
  double* Values() const {return(A_);};

	//! Returns the data access mode of the \e this vector.
	Epetra_DataAccess CV() const {return(CV_);};

  //@}

  //! @name I/O methods
  //@{ 
  //! Print service methods; defines behavior of ostream << operator.
  virtual void Print(ostream& os) const;
  //@}
};

// inlined definitions of op() and op[]
//=========================================================================
inline double& Epetra_SerialDenseVector::operator() (int Index)  {
#ifdef HAVE_EPETRA_ARRAY_BOUNDS_CHECK
  if (Index >= M_ || Index < 0) 
		throw ReportError("Index = " +toString(Index) + " Out of Range 0 - " + toString(M_-1), -1);
#endif
  return(A_[Index]);
}
//=========================================================================
inline const double& Epetra_SerialDenseVector::operator() (int Index) const  {
#ifdef HAVE_EPETRA_ARRAY_BOUNDS_CHECK
  if (Index >= M_ || Index < 0) 
		throw ReportError("Index = " +toString(Index) + " Out of Range 0 - " + toString(M_-1), -1);
#endif
   return(A_[Index]);
}
//=========================================================================
inline double& Epetra_SerialDenseVector::operator [] (int Index)  {
#ifdef HAVE_EPETRA_ARRAY_BOUNDS_CHECK
  if (Index >= M_ || Index < 0) 
		throw ReportError("Index = " +toString(Index) + " Out of Range 0 - " + toString(M_-1), -1);
#endif
   return(A_[Index]);
}
//=========================================================================
inline const double& Epetra_SerialDenseVector::operator [] (int Index) const  {
#ifdef HAVE_EPETRA_ARRAY_BOUNDS_CHECK
  if (Index >= M_ || Index < 0) 
		throw ReportError("Index = " +toString(Index) + " Out of Range 0 - " + toString(M_-1), -1);
#endif
   return(A_[Index]);
}
//=========================================================================

#endif /* EPETRA_SERIALDENSEVECTOR_H */

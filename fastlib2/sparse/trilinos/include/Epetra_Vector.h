
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

#ifndef EPETRA_VECTOR_H
#define EPETRA_VECTOR_H

#include "Epetra_MultiVector.h"
class Epetra_Map;

//! Epetra_Vector: A class for constructing and using dense vectors on a parallel computer.

/*! The Epetra_Vector class enables the construction and use of real-valued, 
    double-precision dense vectors in a distributed memory environment.  The distribution of the dense
    vector is determined in part by a Epetra_Comm object and a Epetra_Map (or Epetra_LocalMap
    or Epetra_BlockMap).

    This class is derived from the Epetra_MultiVector class.  As such, it has full access
    to all of the functionality provided in the Epetra_MultiVector class.

<b> Distributed Global vs. Replicated Local</b>
<ul>
  <li> Distributed Global Vectors - In most instances, a multi-vector will be partitioned
       across multiple memory images associated with multiple processors.  In this case, there is 
       a unique copy of each element and elements are spread across all processors specified by 
       the Epetra_Comm communicator.
  <li> Replicated Local Vectors - Some algorithms use vectors that are too small to
       be distributed across all processors.  Replicated local vectors handle
       these types of situation.
</ul>

<b>Constructing Epetra_Vectors</b>

There are four Epetra_Vector constructors.  The first is a basic constructor that allocates
space and sets all values to zero, the second is a 
copy constructor. The third and fourth constructors work with user data.  These constructors have
two data access modes:
<ol>
  <li> Copy mode - Allocates memory and makes a copy of the user-provided data. In this case, the
       user data is not needed after construction.
  <li> View mode - Creates a "view" of the user data. In this case, the
       user data is required to remain intact for the life of the vector.
</ol>

\warning View mode is \e extremely dangerous from a data hiding perspective.
Therefore, we strongly encourage users to develop code using Copy mode first and 
only use the View mode in a secondary optimization phase.

All Epetra_Vector constructors require a map argument that describes the layout of elements
on the parallel machine.  Specifically, 
\c map is a Epetra_Map, Epetra_LocalMap or Epetra_BlockMap object describing the desired
memory layout for the vector.

There are four different Epetra_Vector constructors:
<ul>
  <li> Basic - All values are zero.
  <li> Copy - Copy an existing vector.
  <li> Copy from or make view of user double array.
  <li> Copy or make view of a vector from a Epetra_MultiVector object.
</ul>

<b>Extracting Data from Epetra_Vectors</b>

Once a Epetra_Vector is constructed, it is possible to extract a copy of the values or create
a view of them.

\warning ExtractView functions are \e extremely dangerous from a data hiding perspective.
For both ExtractView fuctions, there is a corresponding ExtractCopy function.  We
strongly encourage users to develop code using ExtractCopy functions first and 
only use the ExtractView functions in a secondary optimization phase.

There are two Extract functions:
<ul>
  <li> ExtractCopy - Copy values into a user-provided array.
  <li> ExtractView - Set user-provided array to point to Epetra_Vector data.
</ul>

<b>Vector and Utility Functions</b>

Once a Epetra_Vector is constructed, a variety of mathematical functions can be applied to
the vector.  Specifically:
<ul>
  <li> Dot Products.
  <li> Vector Updates.
  <li> \e p Norms.
  <li> Weighted Norms.
  <li> Minimum, Maximum and Average Values.
</ul>

The final useful function is Flops().  Each Epetra_Vector object keep track of the number
of \e serial floating point operations performed using the specified object as the \e this argument
to the function.  The Flops() function returns this number as a double precision number.  Using this 
information, in conjunction with the Epetra_Time class, one can get accurate parallel performance
numbers.

\warning A Epetra_Map, Epetra_LocalMap or Epetra_BlockMap object is required for all 
  Epetra_Vector constructors.

*/

//=========================================================================
class Epetra_Vector : public Epetra_MultiVector {

  public:

    //! @name Constructors/destructors
  //@{ 
  //! Basic Epetra_Vector constuctor.
  /*! Creates a Epetra_Vector object and fills with zero values.  

    \param In 
           Map - A Epetra_LocalMap, Epetra_Map or Epetra_BlockMap.
    \param In 
           zeroOut - If <tt>true</tt> then the allocated memory will be zeroed
                     out initialy.  If <tt>false</tt> then this memory will not
                     be touched which can be significantly faster.

	   \warning Note that, because Epetra_LocalMap
	   derives from Epetra_Map and Epetra_Map derives from Epetra_BlockMap, this constructor works
	   for all three types of Epetra map classes.

    \return Pointer to a Epetra_Vector.

  */
  Epetra_Vector(const Epetra_BlockMap& Map, bool zeroOut = true);

  //! Epetra_Vector copy constructor.
  
  Epetra_Vector(const Epetra_Vector& Source);
  
  //! Set vector values from user array.
  /*!
    \param In 
           Epetra_DataAccess - Enumerated type set to Copy or View.
    \param In 
           Map - A Epetra_LocalMap, Epetra_Map or Epetra_BlockMap.
    \param In
           V - Pointer to an array of double precision numbers..

    \return Integer error code, set to 0 if successful.

	   See Detailed Description section for further discussion.
  */
  Epetra_Vector(Epetra_DataAccess CV, const Epetra_BlockMap& Map, double *V);

  //! Set vector values from a vector in an existing Epetra_MultiVector.
  /*!
    \param In 
           Epetra_DataAccess - Enumerated type set to Copy or View.
    \param In 
           Map - A Epetra_LocalMap, Epetra_Map or Epetra_BlockMap.
    \param In
           Source - An existing fully constructed Epetra_MultiVector.
    \param In
           Index - Index of vector to access.  

    \return Integer error code, set to 0 if successful.

	   See Detailed Description section for further discussion.
  */
  Epetra_Vector(Epetra_DataAccess CV, const Epetra_MultiVector& Source, int Index);

  //! Epetra_Vector destructor.  
    virtual ~Epetra_Vector ();
  //@}
  
  //! @name Post-construction modification routines
  //@{ 

  //! Replace values in a vector with a given indexed list of values, indices are in global index space.
  /*!
     Replace the Indices[i] entry in the \e this object with Values[i], for i=0; i<NumEntries.  The indices
     are in global index space.

    \param In
           NumEntries - Number of vector entries to modify.
    \param In
           Values - Values which will replace existing values in vector, of length NumEntries.
    \param In
           Indices - Indices in global index space corresponding to Values.

    \return Integer error code, set to 0 if successful, set to 1 if one or more indices are not associated with calling processor.
  */
  int ReplaceGlobalValues(int NumEntries, double * Values, int * Indices);

  //! Replace values in a vector with a given indexed list of values, indices are in local index space.
  /*!
     Replace the Indices[i] entry in the \e this object with Values[i], for i=0; i<NumEntries.  The indices
     are in local index space.

    \param In
           NumEntries - Number of vector entries to modify.
    \param In
           Values - Values which will replace existing values in vector, of length NumEntries.
    \param In
           Indices - Indices in local index space corresponding to Values.

    \return Integer error code, set to 0 if successful, set to 1 if one or more indices are not associated with calling processor.
  */
  int ReplaceMyValues(int NumEntries, double * Values, int * Indices);

  //! Sum values into a vector with a given indexed list of values, indices are in global index space.
  /*!
     Sum Values[i] into the Indices[i] entry in the \e this object, for i=0; i<NumEntries.  The indices
     are in global index space.

    \param In
           NumEntries - Number of vector entries to modify.
    \param In
           Values - Values which will replace existing values in vector, of length NumEntries.
    \param In
           Indices - Indices in global index space corresponding to Values.

    \return Integer error code, set to 0 if successful, set to 1 if one or more indices are not associated with calling processor.
  */
  int SumIntoGlobalValues(int NumEntries, double * Values, int * Indices);

  //! Sum values into a vector with a given indexed list of values, indices are in local index space.
  /*!
     Sum Values[i] into the Indices[i] entry in the \e this object, for i=0; i<NumEntries.  The indices
     are in local index space.

    \param In
           NumEntries - Number of vector entries to modify.
    \param In
           Values - Values which will replace existing values in vector, of length NumEntries.
    \param In
           Indices - Indices in local index space corresponding to Values.

    \return Integer error code, set to 0 if successful, set to 1 if one or more indices are not associated with calling processor.
  */
  int SumIntoMyValues(int NumEntries, double * Values, int * Indices);

  // Blockmap Versions

  //! Replace values in a vector with a given indexed list of values at the specified BlockOffset, indices are in global index space.
  /*!
     Replace the Indices[i] entry in the \e this object with Values[i], for i=0; i<NumEntries.  The indices
     are in global index space.  This method is intended for vector that are defined using block maps.  In this situation, 
     an index value is associated with one or more vector entries, depending on the element size of the given index.
     The BlockOffset argument indicates which vector entry to modify as an offset from the first vector entry associated with
     the given index.  The offset is used for each entry in the input list.

    \param In
           NumEntries - Number of vector entries to modify.
    \param In
           BlockOffset - Offset from the first vector entry associated with each of the given indices.
    \param In
           Values - Values which will replace existing values in vector, of length NumEntries.
    \param In
           Indices - Indices in global index space corresponding to Values.

    \return Integer error code, set to 0 if successful, set to 1 if one or more indices are not associated with calling processor.
  */
  int ReplaceGlobalValues(int NumEntries, int BlockOffset, double * Values, int * Indices);

  //! Replace values in a vector with a given indexed list of values at the specified BlockOffset, indices are in local index space.
  /*!
     Replace the (Indices[i], BlockOffset) entry in the \e this object with Values[i], for i=0; i<NumEntries.  The indices
     are in local index space.  This method is intended for vector that are defined using block maps.  In this situation, 
     an index value is associated with one or more vector entries, depending on the element size of the given index.
     The BlockOffset argument indicates which vector entry to modify as an offset from the first vector entry associated with
     the given index.  The offset is used for each entry in the input list.

    \param In
           NumEntries - Number of vector entries to modify.
    \param In
           BlockOffset - Offset from the first vector entry associated with each of the given indices.
    \param In
           Values - Values which will replace existing values in vector, of length NumEntries.
    \param In
           Indices - Indices in local index space corresponding to Values.

    \return Integer error code, set to 0 if successful, set to 1 if one or more indices are not associated with calling processor.
  */
  int ReplaceMyValues(int NumEntries, int BlockOffset, double * Values, int * Indices);

  //! Sum values into a vector with a given indexed list of values at the specified BlockOffset, indices are in global index space.
  /*!
     Sum Values[i] into the Indices[i] entry in the \e this object, for i=0; i<NumEntries.  The indices
     are in global index space.  This method is intended for vector that are defined using block maps.  In this situation, 
     an index value is associated with one or more vector entries, depending on the element size of the given index.
     The BlockOffset argument indicates which vector entry to modify as an offset from the first vector entry associated with
     the given index.  The offset is used for each entry in the input list.

    \param In
           NumEntries - Number of vector entries to modify.
    \param In
           BlockOffset - Offset from the first vector entry associated with each of the given indices.
    \param In
           Values - Values which will replace existing values in vector, of length NumEntries.
    \param In
           Indices - Indices in global index space corresponding to Values.

    \return Integer error code, set to 0 if successful, set to 1 if one or more indices are not associated with calling processor.
  */
  int SumIntoGlobalValues(int NumEntries, int BlockOffset, double * Values, int * Indices);

  //! Sum values into a vector with a given indexed list of values at the specified BlockOffset, indices are in local index space.
  /*!
     Sum Values[i] into the Indices[i] entry in the \e this object, for i=0; i<NumEntries.  The indices
     are in local index space.  This method is intended for vector that are defined using block maps.  In this situation, 
     an index value is associated with one or more vector entries, depending on the element size of the given index.
     The BlockOffset argument indicates which vector entry to modify as an offset from the first vector entry associated with
     the given index.  The offset is used for each entry in the input list.

    \param In
           NumEntries - Number of vector entries to modify.
    \param In
           BlockOffset - Offset from the first vector entry associated with each of the given indices.
    \param In
           Values - Values which will replace existing values in vector, of length NumEntries.
    \param In
           Indices - Indices in local index space corresponding to Values.

    \return Integer error code, set to 0 if successful, set to 1 if one or more indices are not associated with calling processor.
  */
  int SumIntoMyValues(int NumEntries, int BlockOffset, double * Values, int * Indices);
  //@}

  //! @name Extraction methods
  //@{ 

  //Let the compiler know we intend to overload the base-class ExtractCopy
  //function, rather than hide it.
  using Epetra_MultiVector::ExtractCopy;

  //! Put vector values into user-provided array.
  /*!
    \param Out
           V - Pointer to memory space that will contain the vector values.  

    \return Integer error code, set to 0 if successful.
  */
  int ExtractCopy(double *V) const;
  
  //Let the compiler know we intend to overload the base-class ExtractView
  //function, rather than hide it.
  using Epetra_MultiVector::ExtractView;

  //! Set user-provided address of V.
  /*!
    \param Out
           V - Address of a pointer to that will be set to point to the values of the vector.  

    \return Integer error code, set to 0 if successful.
  */
  int ExtractView(double **V) const;
  //@}

  //! @name Overloaded operators
  //@{ 

  //! Element access function.
  /*!
    \return V[Index].
  */
    double& operator [] (int index)
    {
#ifdef HAVE_EPETRA_ARRAY_BOUNDS_CHECK
      EPETRA_TEST_FOR_EXCEPTION(
        !( 0 <= index && index < this->MyLength() ), -99,
        "Epetra_Vector::operator[](int): "
        "The index = " << index << " does not fall in the range"
        "[0,"<<this->MyLength()<<")"
        );
#endif
      return Values_[index];
    }
  //! Element access function.
  /*!
    \return V[Index].
  */
    const double& operator [] (int index) const
    {
#ifdef HAVE_EPETRA_ARRAY_BOUNDS_CHECK
      EPETRA_TEST_FOR_EXCEPTION(
        !( 0 <= index && index < this->MyLength() ), -99,
        "Epetra_Vector::operator[](int) const: "
        "The index = " << index << " does not fall in the range"
        "[0,"<<this->MyLength()<<")"
        );
#endif
      return Values_[index];
    }
    //@}
    
    //! @name Expert-only unsupported methods
  //@{ 

  //Let the compiler know we intend to overload the base-class ResetView
  //function, rather than hide it.
  using Epetra_MultiVector::ResetView;

  //! Reset the view of an existing vector to point to new user data.
	/*! Allows the (very) light-weight replacement of multivector values for an
		  existing vector that was constructed using an Epetra_DataAccess mode of View.
			No checking is performed to see if the values passed in contain valid 
			data.  It is assumed that the user has verified the integrity of data before calling
			this method. This method is useful for situations where a vector is needed
			for use with an Epetra operator or matrix and the user is not passing in a multivector,
			or the multivector is being passed in with another map that is not exactly compatible
			with the operator, but has the correct number of entries.

			This method is used by AztecOO and Ifpack in the matvec and solve methods to improve
			performance and reduce repeated calls to constructors and destructors.

			@param Values Vector data.

			\return Integer error code, set to 0 if successful, -1 if the multivector was not created as a View.

			\warning This method is extremely dangerous and should only be used by experts.
	*/

	int ResetView(double * Values) {EPETRA_CHK_ERR(Epetra_MultiVector::ResetView(&Values)); return(0);};
	//@}
 private:

    int ChangeValues(int NumEntries, int BlockOffset, double * Values, int * Indices, bool IndicesGlobal, bool SumInto);

};

#endif /* EPETRA_VECTOR_H */

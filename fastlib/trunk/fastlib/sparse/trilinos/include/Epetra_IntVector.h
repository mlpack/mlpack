
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

#ifndef EPETRA_INTVECTOR_H
#define EPETRA_INTVECTOR_H

#include "Epetra_DistObject.h"
#include "Epetra_BlockMap.h"
#include "Epetra_Distributor.h"
class Epetra_Map;

//! Epetra_IntVector: A class for constructing and using dense integer vectors on a parallel computer.

/*! The Epetra_IntVector class enables the construction and use of integer
     dense vectors in a distributed memory environment.  The distribution of the dense
    vector is determined in part by a Epetra_Comm object and a Epetra_Map (or Epetra_LocalMap
    or Epetra_BlockMap).


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

<b>Constructing Epetra_IntVectors</b>

There are four Epetra_IntVector constructors.  The first is a basic constructor that allocates
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

All Epetra_IntVector constructors require a map argument that describes the layout of elements
on the parallel machine.  Specifically, 
\c map is a Epetra_Map, Epetra_LocalMap or Epetra_BlockMap object describing the desired
memory layout for the vector.

There are four different Epetra_IntVector constructors:
<ul>
  <li> Basic - All values are zero.
  <li> Copy - Copy an existing vector.
  <li> Copy from or make view of user int array.
</ul>

<b>Extracting Data from Epetra_IntVectors</b>

Once a Epetra_IntVector is constructed, it is possible to extract a copy of the values or create
a view of them.

\warning ExtractView functions are \e extremely dangerous from a data hiding perspective.
For both ExtractView fuctions, there is a corresponding ExtractCopy function.  We
strongly encourage users to develop code using ExtractCopy functions first and 
only use the ExtractView functions in a secondary optimization phase.

There are two Extract functions:
<ul>
  <li> ExtractCopy - Copy values into a user-provided array.
  <li> ExtractView - Set user-provided array to point to Epetra_IntVector data.
</ul>


\warning A Epetra_Map, Epetra_LocalMap or Epetra_BlockMap object is required for all 
  Epetra_IntVector constructors.

*/

//=========================================================================
class Epetra_IntVector : public Epetra_DistObject {

  public:

    //! @name Constructors/destructors
  //@{ 
  //! Basic Epetra_IntVector constuctor.
  /*! Creates a Epetra_IntVector object and, by default, fills with zero values.  

    \param In 
           Map - A Epetra_LocalMap, Epetra_Map or Epetra_BlockMap.

	   \warning Note that, because Epetra_LocalMap
	   derives from Epetra_Map and Epetra_Map derives from Epetra_BlockMap, this constructor works
	   for all three types of Epetra map classes.
  \param In
  zeroOut - If <tt>true</tt> then the allocated memory will be zeroed
            out initialy.  If <tt>false</tt> then this memory will not
            be touched which can be significantly faster.

    \return Pointer to a Epetra_IntVector.

  */
  Epetra_IntVector(const Epetra_BlockMap& Map, bool zeroOut = true);

  //! Epetra_IntVector copy constructor.
  
  Epetra_IntVector(const Epetra_IntVector& Source);
  
  //! Set vector values from user array.
  /*!
    \param In 
           Epetra_DataAccess - Enumerated type set to Copy or View.
    \param In 
           Map - A Epetra_LocalMap, Epetra_Map or Epetra_BlockMap.
    \param In
           V - Pointer to an array of integer numbers..

    \return Integer error code, set to 0 if successful.

	   See Detailed Description section for further discussion.
  */
  Epetra_IntVector(Epetra_DataAccess CV, const Epetra_BlockMap& Map, int *V);

  //! Epetra_IntVector destructor.  
  virtual ~Epetra_IntVector ();
  //@}
  

  //! @name Post-construction modification methods
  //@{ 
  //! Set all elements of the vector to Value
  int PutValue(int Value);
  //@}
  

  //! @name Extraction methods
  //@{ 


  //! Put vector values into user-provided array.
  /*!
    \param Out
           V - Pointer to memory space that will contain the vector values.  

    \return Integer error code, set to 0 if successful.
  */
  int ExtractCopy(int *V) const;
  
  //! Set user-provided address of V.
  /*!
    \param Out
           V - Address of a pointer to that will be set to point to the values of the vector.  

    \return Integer error code, set to 0 if successful.
  */
  int ExtractView(int **V) const;
  //@}

  //! @name Mathematical methods
  //@{ 
  //! Find maximum value
  /*!
    \return Maximum value across all processors.
  */
  int MaxValue();
 
  //! Find minimum value
  /*!
    \return Minimum value across all processors.
  */
  int MinValue();

  //@}
  
  //! @name Overloaded operators
  //@{ 

  //! = Operator.
  /*!
    \param In
           A - Epetra_IntVector to copy.

    \return Epetra_IntVector.
  */
  Epetra_IntVector& operator = (const Epetra_IntVector& Source);
  
  //! Element access function.
  /*!
    \return V[Index].
  */
    int& operator [] (int index) { return Values_[index]; }
  //! Element access function.
  /*!
    \return V[Index].
  */
    const int& operator [] (int index) const { return Values_[index]; }
    //@}
    
    //! @name Attribute access functions
  //@{ 

  //! Returns a pointer to an array containing the values of this vector.
  int * Values() const {return(Values_);};

  //! Returns the local vector length on the calling processor of vectors in the multi-vector.
  int MyLength() const {return(Map().NumMyPoints());};

  //! Returns the global vector length of vectors in the multi-vector.
  int GlobalLength() const {return(Map().NumGlobalPoints());};
  //@}

  //! @name I/O methods
  //@{ 

  //! Print method
  virtual void Print(ostream & os) const;
    //@}
 private:

  int AllocateForCopy();
  int DoCopy(int * V);
  int AllocateForView();
  int DoView(int * V);

   // Routines to implement Epetra_DistObject virtual methods
  int CheckSizes(const Epetra_SrcDistObject& A);

  int CopyAndPermute(const Epetra_SrcDistObject & Source,
                     int NumSameIDs, 
                     int NumPermuteIDs,
                     int * PermuteToLIDs,
                     int * PermuteFromLIDs,
                     const Epetra_OffsetIndex * Indexor);

  int PackAndPrepare(const Epetra_SrcDistObject & Source,
                     int NumExportIDs,
                     int * ExportLIDs,
                     int & LenExports,
                     char * & Exports,
                     int & SizeOfPacket,
                     int * Sizes,
                     bool& VarSizes,
                     Epetra_Distributor & Distor);
  
  int UnpackAndCombine(const Epetra_SrcDistObject & Source,
                       int NumImportIDs,
                       int * ImportLIDs, 
                       int LenImports, 
                       char * Imports,
                       int & SizeOfPacket, 
                       Epetra_Distributor & Distor,
                       Epetra_CombineMode CombineMode,
                       const Epetra_OffsetIndex * Indexor);

  int * Values_;
  bool UserAllocated_;
  bool Allocated_;
};

#endif /* EPETRA_INTVECTOR_H */

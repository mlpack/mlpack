
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

#ifndef EPETRA_LOCALMAP_H
#define EPETRA_LOCALMAP_H

//! Epetra_LocalMap: A class for replicating vectors and matrices across multiple processors.

/*! Small matrix and vector objects are often replicated on distributed memory
  parallel machines. The Epetra_LocalMap class allows construction of these replicated
  local objects and keeps information that describes 
  this distribution.  

  Epetra_LocalMap allows the storage and retrieval of the following information.  
  Once a Epetra_Map is constructed any of the following attributes can 
  be obtained
  by calling a query function that has the name as the attribute, e.g. to get the
  value of NumGlobalPoints, you can call a function NumGlobalElements().
  For attributes that
  are lists, the query functions return the list values in a user allocated array.  


  <ul>
  <li> NumMyElements - The number of elements owned by the calling processor.
  <li> IndexBase - The base integer value for indexed array references.  Typically this is 0
       for C/C++ and 1 for Fortran, but it can be set to any integer value.
  <li> Comm - The Epetra_Comm communicator.  This communicator can in turn be queried for
       processor rank and size information.
  </ul>

  The Epetra_LocalMap class is actually a derived class of Epetra_Map.  Epetra_Map is in turn derived
  from Epetra_BlockMap.  As such,  Epetra_LocalMap has full access to all the functions in these other
  map classes.

  In particular, the following function allows a boolean test:    

  <ul>
  <li> DistributedGlobal() - Returns false for a Epetra_LocalMap object.
  </ul>

  \warning A Epetra_Comm object is required for all Epetra_LocalMap constructors.

  \internal In the current implementation, Epetra_Map is the base class for Epetra_LocalMap.

*/
#include "Epetra_Map.h"

class Epetra_LocalMap : public Epetra_Map {
    
  public:
  //! Epetra_LocalMap constructor for a user-defined replicate distribution of elements.
  /*! Creates a map that puts NumMyElements on the calling processor. Each processor should
      pass in the same value for NumMyElements.

    \param In
            NumMyElements - Number of elements owned by the calling processor.
    
    \param In
            IndexBase - Minimum index value used for arrays that use this map.  Typically 0 for
	    C/C++ and 1 for Fortran.
	    
    \param In
            Comm - Epetra_Comm communicator containing information on the number of
	    processors.

    \return Pointer to a Epetra_Map object.

  */ 
	Epetra_LocalMap(int NumMyElements, int IndexBase, const Epetra_Comm& Comm);
	
  //! Epetra_LocalMap copy constructor.
  
	Epetra_LocalMap(const Epetra_LocalMap& map);
  
  //! Epetra_LocalMap destructor.
	
	virtual ~Epetra_LocalMap();
	
	//! Assignment Operator
	Epetra_LocalMap & operator=(const Epetra_LocalMap & map);
	
 private:
	
	int CheckInput();
	
};
#endif /* EPETRA_LOCALMAP_H */

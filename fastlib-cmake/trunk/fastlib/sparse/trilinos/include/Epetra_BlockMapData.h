
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

#ifndef EPETRA_BLOCKMAPDATA_H
#define EPETRA_BLOCKMAPDATA_H

#include "Epetra_Data.h"
#include "Epetra_IntSerialDenseVector.h"

class Epetra_Comm;
class Epetra_Directory;
class Epetra_HashTable;

//! Epetra_BlockMapData:  The Epetra BlockMap Data Class.
/*! The Epetra_BlockMapData class is an implementation detail of Epetra_BlockMap.
    It is reference-counted, and can be shared by multiple Epetra_BlockMap instances. 
    It derives from Epetra_Data, and inherits reference-counting from it.
*/

class Epetra_BlockMapData : public Epetra_Data {
  friend class Epetra_BlockMap;

 private:

  //! @name Constructor/Destructor Methods
  //@{ 

  //! Epetra_BlockMapData Default Constructor.
  Epetra_BlockMapData(int NumGlobalElements, int ElementSize, int IndexBase, const Epetra_Comm & Comm);

  //! Epetra_BlockMapData Destructor.
  ~Epetra_BlockMapData();

  //@}

  const Epetra_Comm * Comm_;

  Epetra_Directory* Directory_;

  Epetra_IntSerialDenseVector LID_;
  Epetra_IntSerialDenseVector MyGlobalElements_;
  Epetra_IntSerialDenseVector FirstPointInElementList_;
  Epetra_IntSerialDenseVector ElementSizeList_;
  Epetra_IntSerialDenseVector PointToElementList_;
  
  int NumGlobalElements_;
  int NumMyElements_;
  int IndexBase_;
  int ElementSize_;
  int MinMyElementSize_;
  int MaxMyElementSize_;
  int MinElementSize_;
  int MaxElementSize_;
  int MinAllGID_;
  int MaxAllGID_;
  int MinMyGID_;
  int MaxMyGID_;
  int MinLID_;
  int MaxLID_;
  int NumGlobalPoints_;
  int NumMyPoints_;
  
  bool ConstantElementSize_;
  bool LinearMap_;
  bool DistributedGlobal_;
  bool OneToOne_;

  int LastContiguousGID_;
  int LastContiguousGIDLoc_;
  Epetra_HashTable * LIDHash_;

	// these are intentionally declared but not defined. See Epetra Developer's Guide for details.
  Epetra_BlockMapData(const Epetra_BlockMapData & BlockMapData);
	Epetra_BlockMapData& operator=(const Epetra_BlockMapData & BlockMapData);

};
#endif /* EPETRA_BLOCKMAPDATA_H */


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

#ifndef EPETRA_CRSGRAPHDATA_H
#define EPETRA_CRSGRAPHDATA_H

#include "Epetra_Data.h"
#include "Epetra_DataAccess.h"
#include "Epetra_BlockMap.h"
#include "Epetra_IntSerialDenseVector.h"
class Epetra_Import;
class Epetra_Export;

//! Epetra_CrsGraphData:  The Epetra CrsGraph Data Class.
/*! The Epetra_CrsGraphData class is an implementation detail of Epetra_CrsGraph.
    It is reference-counted, and can be shared by multiple Epetra_CrsGraph instances. 
    It derives from Epetra_Data, and inherits reference-counting from it.
*/

class Epetra_CrsGraphData : public Epetra_Data {
  friend class Epetra_CrsGraph;

 private:

  //! @name Constructor/Destructor Methods
  //@{ 

  //! Epetra_CrsGraphData Default Constructor.
  Epetra_CrsGraphData(Epetra_DataAccess CV, const Epetra_BlockMap& RowMap, bool StaticProfile);

  //! Epetra_CrsGraphData Constructor (user provided ColMap).
  Epetra_CrsGraphData(Epetra_DataAccess CV, const Epetra_BlockMap& RowMap, const Epetra_BlockMap& ColMap, bool StaticProfile);

	//! Epetra_CrsGraphData copy constructor (not defined).
  Epetra_CrsGraphData(const Epetra_CrsGraphData& CrsGraphData);

  //! Epetra_CrsGraphData Destructor.
  ~Epetra_CrsGraphData();

  //@}

	//! Outputs state of almost all data members. (primarily used for testing purposes).
	/*! Output level: Uses same scheme as chmod. 4-bit = BlockMaps, 2-bit = Indices, 1-bit = Everything else.
		Default paramenter sets it to 3, which is everything but the BlockMaps. Commonly used options:
		1 = Everything except the BlockMaps & Indices_
		2 = Just Indices_
		3 = Everything except the BlockMaps
	*/
  void Print(ostream& os, int level = 3) const;

  //! Epetra_CrsGraphData assignment operator (not defined)
  Epetra_CrsGraphData& operator=(const Epetra_CrsGraphData& CrsGraphData);
  
  //! @name Helper methods called in CrsGraph. Mainly memory allocations and deallocations.
  //@{ 
  
  //! called by FillComplete (and TransformToLocal)
  int MakeImportExport();
  
  //! called by PackAndPrepare
  int ReAllocateAndCast(char*& UserPtr, int& Length, const int IntPacketSizeTimesNumTrans);
  
  //@}
  
  // Defined by CrsGraph::FillComplete and related
  Epetra_BlockMap RowMap_;
  Epetra_BlockMap ColMap_;
  Epetra_BlockMap DomainMap_;
  Epetra_BlockMap RangeMap_;
  
  const Epetra_Import* Importer_;
  const Epetra_Export* Exporter_;

  bool HaveColMap_;
  bool Filled_;
  bool Allocated_;
  bool Sorted_;
  bool StorageOptimized_;
  bool NoRedundancies_;
  bool IndicesAreGlobal_;
  bool IndicesAreLocal_;
  bool IndicesAreContiguous_;
  bool LowerTriangular_;
  bool UpperTriangular_;
  bool NoDiagonal_;
  bool GlobalConstantsComputed_;
  bool StaticProfile_;

  int IndexBase_;

  int NumGlobalEntries_;
  int NumGlobalBlockRows_;
  int NumGlobalBlockCols_;
  int NumGlobalBlockDiagonals_;
  int NumMyEntries_;
  int NumMyBlockRows_;
  int NumMyBlockCols_;
  int NumMyBlockDiagonals_;
  
  int MaxRowDim_;
  int MaxColDim_;
  int GlobalMaxRowDim_;
  int GlobalMaxColDim_;
  int MaxNumNonzeros_;
  int GlobalMaxNumNonzeros_;
  
  int NumGlobalNonzeros_;
  int NumGlobalRows_;
  int NumGlobalCols_;
  int NumGlobalDiagonals_;
  int NumMyNonzeros_;
  int NumMyRows_;
  int NumMyCols_;
  int NumMyDiagonals_;

  int MaxNumIndices_;
  int GlobalMaxNumIndices_;
  
  int** Indices_;
  Epetra_IntSerialDenseVector NumAllocatedIndicesPerRow_;
  Epetra_IntSerialDenseVector NumIndicesPerRow_;
  Epetra_IntSerialDenseVector IndexOffset_;
  Epetra_IntSerialDenseVector All_Indices_;
  Epetra_DataAccess CV_;
  
};

#endif /* EPETRA_CRSGRAPHDATA_H */

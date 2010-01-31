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

#ifndef EPETRA_FECRSGRAPH_H
#define EPETRA_FECRSGRAPH_H

#include "Epetra_Map.h"
#include "Epetra_CrsGraph.h"

/**
  Epetra Finite-Element CrsGraph. This class provides the ability to insert
  indices into a matrix-graph, where the indices represent dense submatrices
  such as element-stiffnesses that might arise from a finite-element
  application.

  In a parallel setting, indices may be submitted on the local processor
  for rows that do not reside in the local portion of the row-map. After
  all indices have been submitted, the GlobalAssemble method gathers all
  non-local graph rows to the appropriate 'owning' processors (an owning
  processor is a processor which has the row in its row-map).
 */
class Epetra_FECrsGraph : public Epetra_CrsGraph {
  public:

  /** Constructor */
  Epetra_FECrsGraph(Epetra_DataAccess CV,
		    const Epetra_BlockMap& RowMap,
		    int* NumIndicesPerRow,
		    bool ignoreNonLocalEntries=false);

  /** Constructor */
  Epetra_FECrsGraph(Epetra_DataAccess CV,
		    const Epetra_BlockMap& RowMap,
		    int NumIndicesPerRow,
		    bool ignoreNonLocalEntries=false);

  /** Constructor */
  Epetra_FECrsGraph(Epetra_DataAccess CV,
		    const Epetra_BlockMap& RowMap, 
		    const Epetra_BlockMap& ColMap,
		    int* NumIndicesPerRow,
		    bool ignoreNonLocalEntries=false);

  /** Constructor */
  Epetra_FECrsGraph(Epetra_DataAccess CV,
		    const Epetra_BlockMap& RowMap, 
		    const Epetra_BlockMap& ColMap,
		    int NumIndicesPerRow,
		    bool ignoreNonLocalEntries=false);

  /** Constructor */
  Epetra_FECrsGraph(const Epetra_FECrsGraph& Graph);

  /** Destructor */
  virtual ~Epetra_FECrsGraph();

  //Let the compiler know we intend to overload the base-class function
  //InsertGlobalIndices rather than hide it.
  using Epetra_CrsGraph::InsertGlobalIndices;

  /** Insert a rectangular, dense 'submatrix' of entries (matrix nonzero
      positions) into the graph.

    @param numRows Number of rows in the submatrix.
    @param rows List of row-numbers for the submatrix.
    @param numCols Number of columns in the submatrix.
    @param cols List of column-indices that will be used for each row in
        the 'rows' list.
  */
  int InsertGlobalIndices(int numRows, const int* rows,
			  int numCols, const int* cols);

   /** Gather any overlapping/shared data into the non-overlapping partitioning
      defined by the Map that was passed to this matrix at construction time.
      Data imported from other processors is stored on the owning processor
      with a "sumInto" or accumulate operation.
      This is a collective method -- every processor must enter it before any
      will complete it.

      ***NOTE***: When GlobalAssemble() calls FillComplete(), it passes the
      arguments 'DomainMap()' and 'RangeMap()', which are the map attributes
      held by the base-class CrsMatrix and its graph. If a rectangular matrix
      is being assembled, the domain-map and range-map must be specified by
      calling the other overloading of this method. Otherwise, GlobalAssemble()
      has no way of knowing what these maps should really be.


      @param callFillComplete option argument, defaults to true.
        Determines whether GlobalAssemble() internally calls the
        FillComplete() method on this matrix.

      @return error-code 0 if successful, non-zero if some error occurs
   */
  int GlobalAssemble(bool callFillComplete=true);

  /** Gather any overlapping/shared data into the non-overlapping partitioning
      defined by the Map that was passed to this matrix at construction time.
      Data imported from other processors is stored on the owning processor
      with a "sumInto" or accumulate operation.
      This is a collective method -- every processor must enter it before any
      will complete it.

      ***NOTE***: When GlobalAssemble() (the other overloading of this method)
      calls FillComplete(), it passes the arguments 'DomainMap()' and
      'RangeMap()', which are the map attributes already held by the base-class
      CrsMatrix and its graph. If a rectangular matrix is being assembled, the
      domain-map and range-map must be specified. Otherwise, GlobalAssemble()
      has no way of knowing what these maps should really be.


      @param domain_map user-supplied domain map for this matrix

      @param range_map user-supplied range map for this matrix

      @param callFillComplete option argument, defaults to true.
        Determines whether GlobalAssemble() internally calls the
        FillComplete() method on this matrix.

      @return error-code 0 if successful, non-zero if some error occurs
   */
  int GlobalAssemble(const Epetra_Map& domain_map,
                     const Epetra_Map& range_map,
                     bool callFillComplete=true);

 private:
  void DeleteMemory();
  int InsertNonlocalRow(int row, int offset);
  int InputNonlocalIndices(int row,
			   int numCols,
			   const int* cols);
  int InputNonlocalIndex(int rowoffset,
			 int col);

  int myFirstRow_;
  int myNumRows_;
  bool ignoreNonLocalEntries_;

  int numNonlocalRows_;
  int* nonlocalRows_;
  int* nonlocalRowLengths_;
  int* nonlocalRowAllocLengths_;
  int** nonlocalCols_;

  Epetra_FECrsGraph & operator=(const Epetra_FECrsGraph& Graph);


};//class Epetra_FECrsGraph

#endif

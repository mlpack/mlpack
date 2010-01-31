
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

#ifndef EPETRA_CRSGRAPH_H
#define EPETRA_CRSGRAPH_H

#include "Epetra_DistObject.h" 
#include "Epetra_CrsGraphData.h"
class Epetra_BlockMap;
class Epetra_Util;
class Epetra_Time;
class Epetra_Import;
class Epetra_Export;
class Epetra_Distributor;
class Epetra_RowMatrix;

//! Epetra_CrsGraph: A class for constructing and using sparse compressed row graphs.

/*! Epetra_CrsGraph enables the piecewise construction and use of sparse matrix graphs (the integer structure without
    values) where entries are intended for row access.  

    Epetra_CrsGraph is an attribute of all Epetra row-based matrix classes, defining their nonzero structure and also
    holding their Epetra_Map attributes.

<b>Constructing Epetra_CrsGraph objects</b>

Constructing Epetra_CrsGraph objects is a multi-step process.  The basic steps are as follows:
<ol>
  <li> Create Epetra_CrsGraph instance, including some initial storage,  via constructor. In
       addition to the copy constructor, Epetra_CrsGraph has four different constructors.  All four of these
          constructors have 
	  an argument, StaticProfile, which by default is set to false.  If it is set to true, then the 
	  profile (the number of indices per row as defined by NumIndicesPerRow) will be rigidly enforced.  
	  Although this takes away flexibility, it allows a single array to be allocated for all indices.
	  This decreases memory fragmentation and improves performance across many operations. A more detailed
	  discussion of the StaticProfile option is found below.
    <ol>
     <li> User-provided row map, variable nonzero profile: This constructor is used to define the 
          row distribution of the graph and specify a varying number of nonzero entries per row.
	  It is best to use this constructor when the user will be inserting entries using global index 
	  values and wants every column index to be included in the graph.  Note that in this case, the
	  column map will be built for the user when FillComplete() is called.  This constructor is also
	  appropriate for when there is a large variation in the number of indices per row.  If this is not
	  the case, the next constructor may be more convenient to use.
     <li> User-provided row map, fixed nonzero profile: This constructor is used to define the 
          row distribution of the graph and specify a fixed number of nonzero entries per row.
	  It is best to use this constructor when the user will be inserting entries using global index 
	  values and wants every column index to be included in the graph.  Note that in this case, the
	  column map will be built for the user when FillComplete() is called.  This constructor is also
	  appropriate for when there is little or no variation in the number of indices per row.
     <li> User-provided row map, user-provided column map and variable nonzero profile: 
          This constructor is used to define the 
          row \e and \e column distribution of the graph, and specify a varying number of nonzero entries per row.
	  It is best to use this constructor when the user will be inserting entries and already knows which columns
	  of the matrix should be included on each processor.  Note that in this case, the
	  column map will \e not be built for the user when FillComplete() is called.  Also, if the user attempts to 
	  insert a column index whose GID is not part of the column map on that process, the index will be
	  discarded.  This property can be used to "filter out" column entries that should be ignored.
	  This constructor is also
	  appropriate for when there is a large variation in the number of indices per row.  If this is not
	  the case, the next constructor may be more convenient to use.
     <li> User-provided row map, user-provided column map and fixed nonzero profile: 
          This constructor is used to define the 
          row \e and \e column distribution of the graph, and specify a fixed number of nonzero entries per row.
	  It is best to use this constructor when the user will be inserting entries and already knows which columns
	  of the matrix should be included on each processor.  Note that in this case, the
	  column map will \e not be built for the user when FillComplete() is called.  Also, if the user attempts to 
	  insert a column index whose GID is not part of the column map on that process, the index will be
	  discarded.  This constructor is also
	  appropriate for when there is little or no variation in the number of indices per row.
    </ol>
  <li> Enter row and column entry information via calls to the InsertGlobalIndices method.
  <li> Complete construction via FillComplete call, which performs the following tasks:
    <ol>
     <li>Transforms indices to local index space (after this, IndicesAreLocal()==true)
     <li>Sorts column-indices within each row
     <li>Compresses out any redundant indices within rows
     <li>Computes global data such as num-nonzeros, maximum row-lengths, etc.
    </ol>
  <li> (Optional) Optimize the graph storage via a call to OptimizeStorage.
</ol>

<b> Performance Enhancement Issues </b>

The Epetra_CrsGraph class attempts to address four basic types of situations, depending on the user's primary concern:

<ol>
 <li> Simple, flexible construction over minimal memory use or control of column indices:  In this case the user wants to provide only a row distribution
      of the graph and insert indices without worrying about memory allocation performance.  This type of user is best
      served by the constructor that requires only a row map, and a fixed number of indices per row.  In fact, setting NumIndicesPerRow=0
      is probably the best option.
 <li> Stronger control over memory allocation performance and use over flexibility and simplicity:  In this case the user explicitly set
      StaticProfile to true and will provide values, either a single global int or an array of int's, for NumIndicesPerRow, such that 
      the actual number of indices submitted to the graph will not exceed the estimates.  Because we know that NumIndicesPerRow will not 
      be exceeded, we can pre-allocate all of the storage for the graph as a single array.  This is typically much more efficient.
 <li> Explicit control over column indices:  In this case the user prescribes the column map.  Given the column map, any index that is
      submitted for entry into the graph will be included \e only if they are present in the list of GIDs for the column map on the
      processor that submits the index.  This feature allows the user to define a filter such that only certain columns will be kept.  The 
      user also prescribes the local ordering via this technique, since the ordering of GIDs in the column map imposes the local
      ordering.
 <li> Construction using local indices only:  In some situations, users may want to build a graph using local index values only.  In this
      case, the user must explicitly assign GIDs.  This is done by prescribing the column map, in the same way as the previous situation.
</ol>

Notes:
<ul>
<li>In all but the most advanced uses, users will typically \e not specify the column map.  In other words, graph entries will be submitted using 
GIDs not LIDs and all entries that are submitted are intended to be inserted into the graph.

<li>If a user is not particularly worried about performance, or really needs the flexibility associated with the first situation, then there 
is no need to explicitly manage the NumIndicesPerRow values or set StaticProfile to true.  In this case, it is best to set NumIndicesPerRow to
zero.

<li> Users who are concerned about performance should carefully manage NumIndicesPerRow and set StaticProfile to true.  This will give the best
performance and use the least amount of memory.

<li> A compromise approach would be to not set StaticProfile to true, giving the user flexibility, but then calling OptimizeStorage() once FillComplete()
has been called. This approach requires additional temporary memory because the graph will be copied into an efficient data structure and the old
memory deleted.  However, once the copy has been made, the resulting data structure is as efficient as when StaticProfile is used.
 </ul>

<b>Epetra_Map attributes</b>

Epetra_CrsGraph objects have four Epetra_Map attributes.

The Epetra_Map attributes can be obtained via these accessor methods:
<ul>
 <li>RowMap() Describes the numbering and distribution of the rows of the graph. The row-map exists and is valid
 for the entire life of the graph, having been passed in as a constructor argument. The set of graph rows is defined
 by the row-map and may not be changed. Rows may not be inserted or deleted by the user. The only change that may be
 made is that the user can replace the row-map with a compatible row-map (which is the same except for re-numbering)
 by calling the ReplaceRowMap() method.
 <li>ColMap() Describes the set of column-indices that appear in the rows in each processor's portion of the graph.
 Unless provided by the user at construction time, a valid column-map doesn't exist until FillComplete() is called.
 <li>RangeMap() Describes the range of the matrix operator. e.g., for a matrix-vector product operation, the result
   vector's map must be compatible with the range-map of the matrix operator. The range-map is usually the same as
   the row-map. The range-map is set equal to the row-map at graph creation time, but may be specified by the user
   when FillComplete() is called.
 <li>DomainMap() Describes the domain of the matrix operator. The domain-map can be specified by the user when
 FillComplete() is called. Until then, it is set equal to the row-map.
</ul>

It is important to note that while the row-map and the range-map are often the same, the column-map and the domain-map
are almost never the same. The set of entries in a distributed column-map almost always form overlapping sets, with
entries being associated with more than one processor. A domain-map, on the other hand, must be a 1-to-1 map, with
entries being associated with only a single processor.

<b>Global versus Local indices</b>

After creation and before FillComplete() has been called, the column-indices of the graph are in
the global space as received from the user. One of the tasks performed by FillComplete() is to
transform the indices to a local index space. The query methods IndicesAreGlobal() and IndicesAreLocal()
return true or false depending on whether this transformation has been performed or not.

Note the behavior of several graph methods:
<ul>
 <li>InsertGlobalIndices() returns an error if IndicesAreLocal()==true or StorageOptimized()==true
 <li>InsertMyIndices() returns an error if IndicesAreGlobal()==true or StorageOptimized()==true
 <li>RemoveGlobalIndices() returns an error if IndicesAreLocal()==true or if graph was constructed in View mode
 <li>RemoveMyIndices() returns an error if IndicesAreGlobal()==true or if graph was constructed in View mode
 <li>ExtractGlobalRowCopy() works regardless of state of indices
 <li>ExtractMyRowCopy() returns an error if IndicesAreGlobal()==true
 <li>ExtractGlobalRowView() returns an error if IndicesAreLocal()==true
 <li>ExtractMyRowView() returns an error if IndicesAreGlobal()==true
</ul>

Note that even after a graph is constructed, it is possible to add or remove entries.  However, 
FillComplete must then be called again to restore the graph to a consistent state.

*/    

class Epetra_CrsGraph: public Epetra_DistObject {
      
 public:

   //! @name Constructors/Destructor
  //@{ 
  //! Epetra_CrsGraph constuctor with variable number of indices per row.
  /*! Creates a Epetra_CrsGraph object and allocates storage.  
    
    \param CV - (In) A Epetra_DataAccess enumerated type set to Copy or View.
    \param RowMap - (In) An Epetra_BlockMap (or Epetra_Map or Epetra_LocalMap) listing the rows that this 
	   processor will contribute to.In
    \param NumIndicesPerRow - (In) An integer array of length NumMyRows
	   such that NumIndicesPerRow[i] indicates the (approximate if StaticProfile=false) number of entries in the ith row.
    \param StaticProfile - (In) Optional argument that indicates whether or not NumIndicesPerRow should be interpreted as an exact
           count of nonzeros, or should be used as an approximation.  By default this value is false, allowing the profile to be determined 
           dynamically.  If the user sets it to true, then the memory allocation for the Epetra_CrsGraph object will be done in one large
	   block, saving on memory fragmentation and generally improving the performance of matrix multiplication and solve kernels.
  */
  Epetra_CrsGraph(Epetra_DataAccess CV, const Epetra_BlockMap& RowMap, const int* NumIndicesPerRow, bool StaticProfile = false);
  
  //! Epetra_CrsGraph constuctor with fixed number of indices per row.
  /*! Creates a Epetra_CrsGraph object and allocates storage.  
    
    \param CV - (In) A Epetra_DataAccess enumerated type set to Copy or View.
    \param RowMap - (In) An Epetra_BlockMap (or Epetra_Map or Epetra_LocalMap) listing the rows that this 
	   processor will contribute to.
    \param NumIndicesPerRow - (In) An integer that indicates the (approximate if StaticProfile=false) number of entries in the each row.
	   Note that it is possible to use 0 for this value and let fill occur during the insertion phase.
    \param StaticProfile - (In) Optional argument that indicates whether or not NumIndicesPerRow should be interpreted as an exact
           count of nonzeros, or should be used as an approximation.  By default this value is false, allowing the profile to be determined 
           dynamically.  If the user sets it to true, then the memory allocation for the Epetra_CrsGraph object will be done in one large
	   block, saving on memory fragmentation and generally improving the performance of matrix multiplication and solve kernels.
	   
  */
  Epetra_CrsGraph(Epetra_DataAccess CV, const Epetra_BlockMap& RowMap, int NumIndicesPerRow, bool StaticProfile = false);
  
  //! Epetra_CrsGraph constuctor with variable number of indices per row.
  /*! Creates a Epetra_CrsGraph object and allocates storage.  
    
    \param CV - (In) A Epetra_DataAccess enumerated type set to Copy or View.
    \param RowMap - (In) An Epetra_BlockMap (or Epetra_Map or Epetra_LocalMap) listing the rows that this 
	   processor will contribute to.
    \param ColMap - (In) An Epetra_BlockMap (or Epetra_Map or Epetra_LocalMap) listing the columns that this 
	   processor will contribute to.
    \param NumIndicesPerRow - (In) An integer array of length NumMyRows
	   such that NumIndicesPerRow[i] indicates the (approximate if StaticProfile=false) number of entries in the ith row.
    \param StaticProfile - (In) Optional argument that indicates whether or not NumIndicesPerRow should be interpreted as an exact
           count of nonzeros, or should be used as an approximation.  By default this value is false, allowing the profile to be determined 
           dynamically.  If the user sets it to true, then the memory allocation for the Epetra_CrsGraph object will be done in one large
	   block, saving on memory fragmentation and generally improving the performance of matrix multiplication and solve kernels.
  */
  Epetra_CrsGraph(Epetra_DataAccess CV, const Epetra_BlockMap& RowMap, 
		  const Epetra_BlockMap& ColMap, const int* NumIndicesPerRow, bool StaticProfile = false);
  
  //! Epetra_CrsGraph constuctor with fixed number of indices per row.
  /*! Creates a Epetra_CrsGraph object and allocates storage.  
    
    \param CV - (In) A Epetra_DataAccess enumerated type set to Copy or View.
    \param RowMap - (In) An Epetra_BlockMap (or Epetra_Map or Epetra_LocalMap) listing the rows that this 
	   processor will contribute to.
    \param ColMap - (In) An Epetra_BlockMap (or Epetra_Map or Epetra_LocalMap) listing the columns that this 
	   processor will contribute to.
    \param In
           NumIndicesPerRow - An integer that indicates the (approximate if StaticProfile=false) number of entries in the each row.
	   Note that it is possible to use 0 for this value and let fill occur during the insertion phase.
    \param StaticProfile - (In) Optional argument that indicates whether or not NumIndicesPerRow should be interpreted as an exact
           count of nonzeros, or should be used as an approximation.  By default this value is false, allowing the profile to be determined 
           dynamically.  If the user sets it to true, then the memory allocation for the Epetra_CrsGraph object will be done in one large
	   block, saving on memory fragmentation and generally improving the performance of matrix multiplication and solve kernels.
	   
  */
  Epetra_CrsGraph(Epetra_DataAccess CV, const Epetra_BlockMap& RowMap, 
		  const Epetra_BlockMap& ColMap, int NumIndicesPerRow, bool StaticProfile = false);
  
  //! Copy constructor. 
	/*! This will create a Level 1 deep copy. This Graph will share ownership
		  of the CrsGraphData object with the right hand side Graph.
	*/
  Epetra_CrsGraph(const Epetra_CrsGraph& Graph);

  //! Epetra_CrsGraph Destructor
  virtual ~Epetra_CrsGraph();
  //@}
  
  //! @name Insertion/Removal methods
  //@{ 
  //! Enter a list of elements in a specified global row of the graph.
  /*!
    \param Row - (In) Global row number of indices.
    \param NumIndices - (In) Number of Indices.
    \param Indices - (In) Global column indices to insert.

    \return Integer error code, set to 0 if successful. If the insertion requires
     that additional memory be allocated for the row, a positive error code of 1
     is returned. If the graph is a 'View'
     mode graph, then a positive warning code of 2 will be returned if the 
     specified row already exists. Returns 1 if underlying graph data is shared
     by multiple graph instances.

    \pre IndicesAreGlobal()==true, StorageOptimized()==false
  */
  int InsertGlobalIndices(int GlobalRow, int NumIndices, int* Indices);
  
  //! Remove a list of elements from a specified global row of the graph.
  /*!
    \param Row - (In) Global row number of indices.
    \param NumIndices - (In) Number of Indices.
    \param Indices - (In) Global column indices to remove.
	   
    \return Integer error code, set to 0 if successful. Returns 1 if data is shared.

    \pre IndicesAreGlobal()==true, StorageOptimized()==false
  */
  int RemoveGlobalIndices(int GlobalRow, int NumIndices, int* Indices);
  
  //! Remove all indices from a specified global row of the graph.
  /*!
    \param Row - (In) Global row number of indices.

    \return Integer error code, set to 0 if successful. Returns 1 if data is shared.

    \pre IndicesAreGlobal()==true, StorageOptimized()==false
  */
  int RemoveGlobalIndices(int Row);

  
  //! Enter a list of elements in a specified local row of the graph.
  /*!
    \param Row - (In) Local row number of indices.
    \param NumIndices - (In) Number of Indices.
    \param Indices - (In) Local column indices to insert.

    \return Integer error code, set to 0 if successful. If the insertion requires
     that additional memory be allocated for the row, a positive error code of 1
     is returned. If one or more of the indices is ignored (due to not being
     contained in the column-map), then a positive warning code of 2 is returned.
     If the graph is a 'View' mode graph, then a positive warning code of 3 will
     be returned if the specified row already exists. Returns 1 if underlying
     graph data is shared by multiple graph instances.

    \pre IndicesAreLocal()==true, StorageOptimized()==false
  */
  int InsertMyIndices(int LocalRow, int NumIndices, int* Indices);
  
  //! Remove a list of elements from a specified local row of the graph.
  /*!
    \param Row - (In) Local row number of indices.
    \param NumIndices - (In) Number of Indices.
    \param Indices - (In) Local column indices to remove.
	   
    \return Integer error code, set to 0 if successful. Returns 1 if data is shared.

    \pre IndicesAreLocal()==true, StorageOptimized()==false
  */
  int RemoveMyIndices(int LocalRow, int NumIndices, int* Indices);
  
  //! Remove all indices from a specified local row of the graph.
  /*!
    \param Row - (In) Local row number of indices.

    \return Integer error code, set to 0 if successful. Returns 1 if data is shared.

    \pre IndicesAreLocal()==true, StorageOptimized()==false
  */
  int RemoveMyIndices(int Row);
  //@}

  //! @name Transformation methods
  //@{ 
  
  //! Tranform to local index space.  Perform other operations to allow optimal matrix operations.
  /*! This overloading of the FillComplete method assumes that the domain-map and range-map both equal
    the row-map, and simply calls FillComplete(RowMap(), RowMap()).
    \return Integer error code, set to 0 if successful. Returns 1 if data is shared (i.e., if the underlying graph-data
    object has a reference-count greater than 1).

    \post IndicesAreLocal()==true, Filled()==true
  */
  int FillComplete();

  //! Transform to local index space using specified Domain/Range maps.  Perform other operations to allow optimal matrix operations.
  /*! Performs this sequence of operations:
    <ol>
     <li>Transform indices to local index space
     <li>Sort column-indices within each row
     <li>Compress out any redundant indices within rows
     <li>Compute global data such as num-nonzeros, maximum row-lengths, etc.
    </ol>
    \return Integer error code, set to 0 if successful. Returns 1 if data is shared (i.e., if the underlying graph-data
    object has a reference-count greater than 1).

    \post IndicesAreLocal()==true, Filled()==true
  */
  int FillComplete(const Epetra_BlockMap& DomainMap, const Epetra_BlockMap& RangeMap);

  //! Make consecutive row index sections contiguous, minimize internal storage used for constructing graph.
  /*! After construction and during initialization (when indices are being added via InsertGlobalIndices() etc.), the column-
    indices for each row are held in a separate piece of allocated memory. This method moves the column-indices for all rows
    into one large contiguous array and eliminates internal storage that is not needed after graph construction. Calling this
    method can have a significant impact on memory costs and machine performance.

    If this object was constructed in View mode then this method can't make non-contiguous indices contiguous and will
    return a warning code of 1 if the viewed data isn't already contiguous.
    \return Integer error code, set to 0 if successful.

    \pre Filled()==true.
    \pre If CV=View when the graph was constructed, then this method will be effective \only if the indices of the graph were already contiguous.  In this case, the indices are left untouched and internal storage for the graph is minimized.

    \post StorageOptimized()==true, if successful
  */
  int OptimizeStorage();

  //@}

  //! @name Extraction methods
  //@{ 

  //! Extract a list of elements in a specified global row of the graph. Put into storage allocated by calling routine.
  /*!
    \param Row - (In) Global row number to get indices.
    \param LenOfIndices - (In) Length of Indices array.
    \param NumIndices - (Out) Number of Indices.
    \param Indices - (Out) Global column indices corresponding to values.

    \return Integer error code, set to 0 if successful.
  */
  int ExtractGlobalRowCopy(int GlobalRow, int LenOfIndices, int& NumIndices, int* Indices) const;

  //! Extract a list of elements in a specified local row of the graph. Put into storage allocated by calling routine.
  /*!
    \param Row - (In) Local row number to get indices.
    \param LenOfIndices - (In) Length of Indices array.
    \param NumIndices - (Out) Number of Indices.
    \param Indices - (Out) Local column indices corresponding to values.

    \return Integer error code, set to 0 if successful.

    \pre IndicesAreLocal()==true
  */
  int ExtractMyRowCopy(int LocalRow, int LenOfIndices, int& NumIndices, int* Indices) const;

  //! Get a view of the elements in a specified global row of the graph.
  /*!
    This function requires that the graph not be completed (FillComplete() was \e not called).
    \param Row - (In) Local row number to get indices.
    \param NumIndices - (Out) Number of Indices.
    \param Indices - (Out) Column indices corresponding to values.

    \return Integer error code, set to 0 if successful. Returns -1 if invalid row.  Returns -2 if graph is completed.

    \pre IndicesAreLocal()==false
  */
  int ExtractGlobalRowView(int GlobalRow, int& NumIndices, int*& Indices) const;

  //! Get a view of the elements in a specified local row of the graph.
  /*!
    This function requires that the graph be completed FillComplete() was called).
    \param Row - (In) Local row number to get indices.
    \param NumIndices - (Out) Number of Indices.
    \param Indices - (Out) Column indices corresponding to values.

    \return Integer error code, set to 0 if successful. Returns -1 if invalid row.  Returns -2 if graph is not completed.

    \pre IndicesAreLocal()==true
  */
  int ExtractMyRowView(int LocalRow, int& NumIndices, int*& Indices) const;
  //@}

  //! @name Graph Properties Query Methods
  //@{ 
  //! If FillComplete() has been called, this query returns true, otherwise it returns false.
	bool Filled() const {return(CrsGraphData_->Filled_);}
	
	//! If OptimizeStorage() has been called, this query returns true, otherwise it returns false.
	bool StorageOptimized() const {return(CrsGraphData_->StorageOptimized_);}
	
	//! If column indices are in global range, this query returns true, otherwise it returns false.
	bool IndicesAreGlobal() const {return(CrsGraphData_->IndicesAreGlobal_);}
	
	//! If column indices are in local range, this query returns true, otherwise it returns false.
	bool IndicesAreLocal() const {return(CrsGraphData_->IndicesAreLocal_);}
	
	//! If graph is lower triangular in local index space, this query returns true, otherwise it returns false.
	/*!
	  \pre Filled()==true
	*/
	bool LowerTriangular() const {return(CrsGraphData_->LowerTriangular_);}
	
	//! If graph is upper triangular in local index space, this query returns true, otherwise it returns false.
	/*!
	  \pre Filled()==true
	*/
	bool UpperTriangular() const {return(CrsGraphData_->UpperTriangular_);}
	
	//! If graph has no diagonal entries in global index space, this query returns true, otherwise it returns false.
	/*!
	  \pre Filled()==true
	*/
	bool NoDiagonal() const {return(CrsGraphData_->NoDiagonal_);}
	
	//! Returns true of GID is owned by the calling processor, otherwise it returns false.
	bool MyGlobalRow(int GID) const {return(RowMap().MyGID(GID));}

	//! Returns true if we have a well-defined ColMap, and returns false otherwise.
	/*! \pre We have a well-defined ColMap if a) a ColMap was passed in at construction, 
		or b) the MakeColMap function has been called. (Calling either of the FillComplete functions
		will result in MakeColMap being called.) 
	*/
	bool HaveColMap() const {return(CrsGraphData_->HaveColMap_);}
  //@}
  
  //! @name Atribute access functions
  //@{ 
    
	//! Returns the number of matrix rows on this processor.
	int NumMyRows() const {return(CrsGraphData_->NumMyRows_);}
	
	//! Returns the number of matrix rows in global matrix.
	int NumGlobalRows() const {return(CrsGraphData_->NumGlobalRows_);}
	
	//! Returns the number of entries in the set of column-indices that appear on this processor.
	/*! The set of column-indices that appear on this processor is the union of column-indices that
	  appear in all local rows. The size of this set isn't available until FillComplete() has been called.
	  \pre Filled()==true
	*/
	int NumMyCols() const {return(CrsGraphData_->NumMyCols_);}
	
	//! Returns the number of matrix columns in global matrix.
	/*!
	  \pre Filled()==true
	*/
	int NumGlobalCols() const {return(CrsGraphData_->NumGlobalCols_);}
	
	//! Returns the number of indices in the global graph.
	/*! Note that if the graph's maps are defined such that some nonzeros
            appear on more than one processor, then those nonzeros will be
            counted more than once. If the user wishes to assemble a graph from
            overlapping data, they can use Epetra_FECrsGraph.
	  \pre Filled()==true
	*/
	int NumGlobalNonzeros() const {return(CrsGraphData_->NumGlobalNonzeros_);}
	
	//! Returns the number of diagonal entries in the global graph, based on global row/column index comparisons.
	/*!
	  \pre Filled()==true
	*/
	int NumGlobalDiagonals() const {return(CrsGraphData_->NumGlobalDiagonals_);}
	
	//! Returns the number of diagonal entries in the local graph, based on global row/column index comparisons.
	/*!
	  \pre Filled()==true
	*/
	int NumMyDiagonals() const {return(CrsGraphData_->NumMyDiagonals_);}
	
	//! Returns the number of block matrix rows on this processor.
	int NumMyBlockRows() const {return(CrsGraphData_->NumMyBlockRows_);}
	
	//! Returns the number of Block matrix rows in global matrix.
	int NumGlobalBlockRows() const {return(CrsGraphData_->NumGlobalBlockRows_);}
	
	//! Returns the number of Block matrix columns on this processor.
	/*!
	  \pre Filled()==true
	*/
	int NumMyBlockCols() const {return(CrsGraphData_->NumMyBlockCols_);}
	
	//! Returns the number of Block matrix columns in global matrix.
	/*!
	  \pre Filled()==true
	*/
	int NumGlobalBlockCols() const {return(CrsGraphData_->NumGlobalBlockCols_);}
	
	//! Returns the number of Block diagonal entries in the local graph, based on global row/column index comparisons.
	/*!
	  \pre Filled()==true
	*/
	int NumMyBlockDiagonals() const {return(CrsGraphData_->NumMyBlockDiagonals_);}
	
	//! Returns the number of Block diagonal entries in the global graph, based on global row/column index comparisons.
	/*!
	  \pre Filled()==true
	*/
	int NumGlobalBlockDiagonals() const {return(CrsGraphData_->NumGlobalBlockDiagonals_);}
	
	//! Returns the number of entries in the global graph.
	/*!
	  \pre Filled()==true
	*/
	int NumGlobalEntries() const {return(CrsGraphData_->NumGlobalEntries_);}
	
	//! Returns the number of entries on this processor.
	/*!
	  \pre Filled()==true
	*/
	int NumMyEntries() const {return(CrsGraphData_->NumMyEntries_);}
	//! Returns the max row dimension of block entries on the processor.
	/*!
	  \pre Filled()==true
	*/
	int MaxRowDim() const {return(CrsGraphData_->MaxRowDim_);}
	
	//! Returns the max row dimension of block entries across all processors.
	/*!
	  \pre Filled()==true
	*/
	int GlobalMaxRowDim() const {return(CrsGraphData_->GlobalMaxRowDim_);}
	
	//! Returns the max column dimension of block entries on the processor.
	/*!
	  \pre Filled()==true
	*/
	int MaxColDim() const {return(CrsGraphData_->MaxColDim_);}
	
	//! Returns the max column dimension of block entries across all processors.
	/*!
	  \pre Filled()==true
	*/
	int GlobalMaxColDim() const {return(CrsGraphData_->GlobalMaxColDim_);}
	
	//! Returns the number of indices in the local graph.
	/*!
	  \pre Filled()==true
	*/
	int NumMyNonzeros() const {return(CrsGraphData_->NumMyNonzeros_);}
	
	//! Returns the current number of nonzero entries in specified global row on this processor.
	int NumGlobalIndices(int Row) const;
	
	//! Returns the allocated number of nonzero entries in specified global row on this processor.
	int NumAllocatedGlobalIndices(int Row) const;
	
	//! Returns the maximum number of nonzero entries across all rows on this processor.
	/*!
	  \pre Filled()==true
	*/
	int MaxNumIndices() const {return(CrsGraphData_->MaxNumIndices_);}
	
	//! Returns the maximun number of nonzero entries across all rows across all processors.
	/*!
	  \pre Filled()==true
	*/
	int GlobalMaxNumIndices() const {return(CrsGraphData_->GlobalMaxNumIndices_);}
	
	//! Returns the maximum number of nonzero points across all rows on this processor.
	/*! For each entry in the graph, let i = the GRID of the entry and j = the CGID of the entry.  Then
	    the entry size is the product of the rowmap elementsize of i and the colmap elementsize of i.
	    Let ki = sum of all entry sizes for the entries in the ith row. 
	    For example,
            if the ith block row had 5 block entries and the element size of each entry was 4-by-4, ki would be 80.
	    Then this function returns the max over all ki for all row on this processor.

	    \pre Filled()==true
	*/
	int MaxNumNonzeros() const {return(CrsGraphData_->MaxNumNonzeros_);}
	
	//! Returns the maximun number of nonzero points across all rows across all processors.
	/*! This function returns the max over all processor of MaxNumNonzeros().

	    \pre Filled()==true
	 */
	int GlobalMaxNumNonzeros() const {return(CrsGraphData_->GlobalMaxNumNonzeros_);}
	
	//! Returns the current number of nonzero entries in specified local row on this processor.
	int NumMyIndices(int Row) const {if (Row<0 || Row >= NumMyRows()) return(0);
	  if (StorageOptimized()) return(CrsGraphData_->IndexOffset_[Row+1] - CrsGraphData_->IndexOffset_[Row]); 
	  else return(CrsGraphData_->NumIndicesPerRow_[Row]);}
	
	//! Returns the allocated number of nonzero entries in specified local row on this processor.
	int NumAllocatedMyIndices(int Row) const {if (Row<0 || Row >= NumMyRows()) return(0);
	  if (StorageOptimized()) return(CrsGraphData_->IndexOffset_[Row+1] - CrsGraphData_->IndexOffset_[Row]); 
	  else return(CrsGraphData_->NumAllocatedIndicesPerRow_[Row]);}
	
	//! Returns the index base for row and column indices for this graph.
	int IndexBase() const {return(CrsGraphData_->IndexBase_);}
	
	//! Returns the RowMap associated with this graph.
	const Epetra_BlockMap& RowMap() const {return(Epetra_DistObject::Map());}
    
	/** Replaces the current RowMap with the user-specified map object, but only
	    if currentmap->PointSameAs(newmap) is true. This is a collective function.
	    Returns 0 if map is replaced, -1 if not.

	    \pre RowMap().PointSameAs(newmap)==true
	*/
	int ReplaceRowMap(const Epetra_BlockMap& newmap);

	/** Replaces the current ColMap with the user-specified map object, but only
	    if currentmap->PointSameAs(newmap) is true. This is a collective function.
	    Returns 0 if map is replaced, -1 if not.

	    \pre ColMap().PointSameAs(newmap)==true
	*/
	int ReplaceColMap(const Epetra_BlockMap& newmap);

	//! Returns the Column Map associated with this graph.
	/*!
	  \pre HaveColMap()==true
	 */
	const Epetra_BlockMap& ColMap() const {return(CrsGraphData_->ColMap_);}
	
	//! Returns the DomainMap associated with this graph.
	/*!
	  \pre Filled()==true
	 */
	const Epetra_BlockMap& DomainMap() const {return(CrsGraphData_->DomainMap_);}
	
	//! Returns the RangeMap associated with this graph.
	/*!
	  \pre Filled()==true
	 */
	const Epetra_BlockMap& RangeMap() const {return(CrsGraphData_->RangeMap_);}
	
	//! Returns the Importer associated with this graph.
	const Epetra_Import* Importer() const {return(CrsGraphData_->Importer_);}
	
	//! Returns the Exporter associated with this graph.
	const Epetra_Export* Exporter() const {return(CrsGraphData_->Exporter_);}
	
	//! Returns a pointer to the Epetra_Comm communicator associated with this graph.
	const Epetra_Comm& Comm() const {return(Epetra_DistObject::Comm());}
  //@}
  
  //! @name Local/Global ID methods
  //@{ 

	//! Returns the local row index for given global row index, returns -1 if no local row for this global row.
	int LRID(int GRID) const {return(RowMap().LID(GRID));}
	
	//! Returns the global row index for give local row index, returns IndexBase-1 if we don't have this local row.
	int GRID(int LRID) const {return(RowMap().GID(LRID));}
	
	//! Returns the local column index for given global column index, returns -1 if no local column for this global column.
	/*!
	  \pre HaveColMap()==true (If HaveColMap()==false, returns -1)
	 */
	int LCID(int GCID) const
	  {
	    return( CrsGraphData_->HaveColMap_ ? ColMap().LID(GCID) : -1 );
	  }
	
	//! Returns the global column index for give local column index, returns IndexBase-1 if we don't have this local column.
	/*!
	  \pre HaveColMap()==true (If HaveColMap()==false, returns -1)
	 */
	int GCID(int LCID) const
	  {
	    return( CrsGraphData_->HaveColMap_ ? ColMap().GID(LCID) : -1 );
	  }
	
	//! Returns true if the GRID passed in belongs to the calling processor in this map, otherwise returns false.
	bool MyGRID(int GRID) const {return(LRID(GRID) != -1);}
	
	//! Returns true if the LRID passed in belongs to the calling processor in this map, otherwise returns false.
	bool MyLRID(int LRID) const {return(GRID(LRID) != IndexBase() - 1);}
	
	//! Returns true if the GCID passed in belongs to the calling processor in this map, otherwise returns false.
	/*!
	  \pre HaveColMap()==true (If HaveColMap()==false, returns -1)
	 */
	bool MyGCID(int GCID) const {return(LCID(GCID) != -1);}
	
	//! Returns true if the LRID passed in belongs to the calling processor in this map, otherwise returns false.
	/*!
	  \pre HaveColMap()==true (If HaveColMap()==false, returns -1)
	 */
	bool MyLCID(int LCID) const {return(GCID(LCID) != IndexBase() - 1);}
  //@}
  
  //! @name Inlined Operator Methods
  //@{ 
	
	//! Inlined bracket operator for fast access to data. (Const and Non-const versions)
	/*! No error checking and dangerous for optimization purposes.
    \param Loc (In) - Local row.
	  
    \return reference to pointer to locally indexed Loc row in matrix.
  */

	inline int*  operator[]( int Loc ) { 
	  if (StorageOptimized()){ return(CrsGraphData_->All_Indices_.Values() + CrsGraphData_->IndexOffset_[Loc]);}
	  else return(CrsGraphData_->Indices_[Loc]); }

	inline int* operator[]( int Loc ) const { 
	  if (StorageOptimized()) { return(CrsGraphData_->All_Indices_.Values() +CrsGraphData_->IndexOffset_[Loc]);}
	  else return(CrsGraphData_->Indices_[Loc]); }

  //@}

	//! Assignment operator
	/*! This will do a Level 1 deep copy. It will share ownership of the CrsGraphData
		  with the right hand side Graph.
	*/
	Epetra_CrsGraph& operator = (const Epetra_CrsGraph& Source);

  //! @name I/O Methods
  //@{ 

  //! Print method
  virtual void Print(ostream& os) const;

	void PrintGraphData(ostream& os) const {CrsGraphData_->Print(os);}
	void PrintGraphData(ostream& os, int level) const {CrsGraphData_->Print(os, level);}
  //@}

  //! @name Deprecated methods:  These methods still work, but will be removed in a future version
  //@{ 

	//! Use ColMap() instead. 
	const Epetra_BlockMap& ImportMap() const {return(CrsGraphData_->ColMap_);}
	
	//! Use FillComplete() instead.
	int TransformToLocal();
	
	//! Use FillComplete(const Epetra_BlockMap& DomainMap, const Epetra_BlockMap& RangeMap) instead.
	int TransformToLocal(const Epetra_BlockMap* DomainMap, const Epetra_BlockMap* RangeMap);
	
  //@}

  //! @name Expert Users and Developers Only
  //@{ 

	//! Returns the reference count of CrsGraphData.
	/*! (Intended for testing purposes.) */
	int ReferenceCount() const {return(CrsGraphData_->ReferenceCount());}

	//! Returns a pointer to the CrsGraphData instance this CrsGraph uses. 
	/*! (Intended for developer use only for testing purposes.) */
	const Epetra_CrsGraphData* DataPtr() const {return(CrsGraphData_);}

  //@}	

	// functions listed in protected are the ones used by CrsMatrix and VbrMatrix.
	// functions listed in private are the ones that are really private.
	// (just pretend CrsMatrix and VbrMatrix derive from CrsGraph to understand the distinction.)
	friend class Epetra_CrsMatrix;
	friend class Epetra_VbrMatrix;
	friend class Epetra_FECrsGraph;
	friend class Epetra_FECrsMatrix;
	friend class Epetra_FEVbrMatrix;
	friend class Epetra_OffsetIndex;

 protected:
	int *All_Indices() const {
	  if (!StorageOptimized()) throw ReportError("This method: int *All_Indices() cannot be called when StorageOptimized()==false", -1);
	  else return(CrsGraphData_->All_Indices_.Values());}
	int *IndexOffset() const {
	  if (!StorageOptimized()) throw ReportError("This method: int *IndexOffset()  cannot be called when StorageOptimized()==false", -1);
	  else return(CrsGraphData_->IndexOffset_.Values());}
	int* NumIndicesPerRow() const {
	  if (StorageOptimized()) throw ReportError("This method: int* NumIndicesPerRow() cannot be called when StorageOptimized()==true", -1);
	  else return(CrsGraphData_->NumIndicesPerRow_.Values());}
	int* NumAllocatedIndicesPerRow() const {
	  if (StorageOptimized()) throw ReportError("This method: int* NumAllocatedIndicesPerRow() cannot be called when StorageOptimized()==true", -1);
	  else return(CrsGraphData_->NumAllocatedIndicesPerRow_.Values());}
	int** Indices() const {
	  if (StorageOptimized()) throw ReportError("This method: int** Indices() cannot be called when StorageOptimized()==true", -1);
	  else return(CrsGraphData_->Indices_);}
	int* Indices(int LocalRow) const {
	  if (StorageOptimized()) return(CrsGraphData_->All_Indices_.Values()+CrsGraphData_->IndexOffset_[LocalRow]);
	  else return(CrsGraphData_->Indices_[LocalRow]);}
	// If column indices are stored in one long array (via a call to OptimizeStorage), 
	// IndicesAreContiguous returns true, otherwise it returns false.
	bool IndicesAreContiguous() const {return(CrsGraphData_->IndicesAreContiguous_);}
	bool StaticProfile() const {return(CrsGraphData_->StaticProfile_);}
	bool GlobalConstantsComputed() const;
	bool FindGlobalIndexLoc(int LocalRow, int Index, int Start, int& Loc) const;
	bool FindGlobalIndexLoc(int NumIndices, const int* Indices, int Index, int Start, int& Loc) const;
	bool FindMyIndexLoc(int LocalRow, int Index, int Start, int& Loc) const;
	bool FindMyIndexLoc(int NumIndices, const int* Indices, int Index, int Start, int& Loc) const;
	int InsertIndices(int Row, int NumIndices, int* Indices);
	int MakeIndicesLocal(const Epetra_BlockMap& DomainMap, const Epetra_BlockMap& RangeMap);
	void SetIndicesAreLocal(bool Flag) {CrsGraphData_->IndicesAreLocal_ = Flag;}
	void SetIndicesAreGlobal(bool Flag) {CrsGraphData_->IndicesAreGlobal_ = Flag;}
	void SetSorted(bool Flag) {CrsGraphData_->Sorted_ = Flag;}

  //! Sort column indices, row-by-row, in ascending order.
  /*!
    \return Integer error code, set to 0 if successful. Returns 1 if data is shared.
  */
  int SortIndices();

  //! If SortIndices() has been called, this query returns true, otherwise it returns false.
  bool Sorted() const {return(CrsGraphData_->Sorted_);}
	
  //! Removes any redundant column indices in the rows of the graph.
  /*!
    \return Integer error code, set to 0 if successful. Returns 1 if data is shared.
  */
  int RemoveRedundantIndices();

  //! If RemoveRedundantIndices() has been called, this query returns true, otherwise it returns false.
  bool NoRedundancies() const {return(CrsGraphData_->NoRedundancies_);}

 private:
	void SetGlobalConstantsComputed(bool Flag) {CrsGraphData_->GlobalConstantsComputed_ = Flag;}
	void SetIndicesAreContiguous(bool Flag) {CrsGraphData_->IndicesAreContiguous_ = Flag;}
	void SetNoRedundancies(bool Flag) {CrsGraphData_->NoRedundancies_ = Flag;}
	void ComputeIndexState();
	int MakeColMap(const Epetra_BlockMap& DomainMap, const Epetra_BlockMap& RangeMap);
	int Allocate(const int* NumIndicesPerRow, int Inc, bool StaticProfile);
	//int ReAllocate();
	int ComputeGlobalConstants();
	void SetFilled(bool Flag) {CrsGraphData_->Filled_ = Flag;}
	bool Allocated() const {return(CrsGraphData_->Allocated_);}
	void SetAllocated(bool Flag) {CrsGraphData_->Allocated_ = Flag;}
	
	int CheckSizes(const Epetra_SrcDistObject& A);

	int CopyAndPermute(const Epetra_SrcDistObject& Source,
                           int NumSameIDs, 
                           int NumPermuteIDs,
                           int* PermuteToLIDs,
                           int* PermuteFromLIDs,
                           const Epetra_OffsetIndex * Indexor);
	int CopyAndPermuteRowMatrix(const Epetra_RowMatrix& A,
                                    int NumSameIDs, 
                                    int NumPermuteIDs,
                                    int* PermuteToLIDs,
                                    int* PermuteFromLIDs,
                                    const Epetra_OffsetIndex * Indexor);
	int CopyAndPermuteCrsGraph(const Epetra_CrsGraph& A,
                                   int NumSameIDs, 
                                   int NumPermuteIDs,
                                   int* PermuteToLIDs,
                                   int* PermuteFromLIDs,
                                   const Epetra_OffsetIndex * Indexor);
  
        int PackAndPrepare(const Epetra_SrcDistObject& Source,
                           int NumExportIDs,
                           int* ExportLIDs,
                           int& LenExports,
                           char*& Exports,
                           int& SizeOfPacket,
                           int * Sizes,
                           bool & VarSizes,
                           Epetra_Distributor& Distor);
        int PackAndPrepareCrsGraph(const Epetra_CrsGraph& A,
                                   int NumExportIDs,
                                   int* ExportLIDs,
                                   int& LenExports,
                                   char*& Exports,
                                   int& SizeOfPacket,
                                   int* Sizes,
                                   bool& VarSizes,
                                   Epetra_Distributor& Distor);
        int PackAndPrepareRowMatrix(const Epetra_RowMatrix& A,
                                    int NumExportIDs,
                                    int* ExportLIDs,
                                    int& LenExports,
                                    char*& Exports,
                                    int& SizeOfPacket,
                                    int* Sizes,
                                    bool& VarSizes,
                                    Epetra_Distributor& Distor);

        int UnpackAndCombine(const Epetra_SrcDistObject& Source,
                             int NumImportIDs,
                             int* ImportLIDs, 
                             int LenImports, 
                             char* Imports,
                             int& SizeOfPacket, 
                             Epetra_Distributor& Distor,
                             Epetra_CombineMode CombineMode,
                             const Epetra_OffsetIndex * Indexor);

	void CleanupData();

	Epetra_CrsGraphData* CrsGraphData_;	

};
#endif /* EPETRA_CRSGRAPH_H */

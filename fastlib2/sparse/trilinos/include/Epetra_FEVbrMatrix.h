
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

#ifndef EPETRA_FEVBRMATRIX_H
#define EPETRA_FEVBRMATRIX_H

#include <Epetra_VbrMatrix.h>
#include <Epetra_SerialDenseMatrix.h>

/** Epetra Finite-Element VbrMatrix. This class provides the ability to
    input finite-element style sub-matrix data, including sub-matrices with
    non-local rows (which could correspond to shared finite-element nodes for
    example). This class inherits Epetra_VbrMatrix, and so all Epetra_VbrMatrix
    functionality is also available.
*/    

class Epetra_FEVbrMatrix: public Epetra_VbrMatrix {
 public:

   //! @name Constructors/Destructor
  //@{ 
  //! Epetra_FEVbrMatrix constuctor with variable number of indices per row.
  /*! Creates a Epetra_FEVbrMatrix object and allocates storage.  
    
    \param In
           CV - A Epetra_DataAccess enumerated type set to Copy or View.
    \param In 
           RowMap - A Epetra_BlockMap listing the block rows that this processor
	   will contribute to.
    \param In
           NumBlockEntriesPerRow - An integer array of length NumRows
	   such that NumBlockEntriesPerRow[i] indicates the (approximate)
	   number of Block entries in the ith row.
  */
  Epetra_FEVbrMatrix(Epetra_DataAccess CV,
		     const Epetra_BlockMap& RowMap,
		     int *NumBlockEntriesPerRow,
		     bool ignoreNonLocalEntries=false);
  
  //! Epetra_FEVbrMatrix constuctor with fixed number of indices per row.
  /*! Creates a Epetra_FEVbrMatrix object and allocates storage.  
    
    \param In
           CV - A Epetra_DataAccess enumerated type set to Copy or View.
    \param In 
           RowMap - An Epetra_BlockMap listing the block rows that this
	   processor will contribute to.
    \param In
           NumBlockEntriesPerRow - An integer that indicates the (approximate)
	   number of Block entries in the each Block row.
	   Note that it is possible to use 0 for this value and let fill occur
	   during the insertion phase.
  */
  Epetra_FEVbrMatrix(Epetra_DataAccess CV,
		     const Epetra_BlockMap& RowMap,
		     int NumBlockEntriesPerRow,
		     bool ignoreNonLocalEntries=false);

  //! Epetra_FEVbrMatrix constuctor with variable number of indices per row.
  /*! Creates a Epetra_FEVbrMatrix object and allocates storage.  
    
    \param In
           CV - A Epetra_DataAccess enumerated type set to Copy or View.
    \param In 
           RowMap - A Epetra_BlockMap listing the block rows that this processor
	   will contribute to.
    \param In 
           ColMap - A Epetra_BlockMap listing the block columns to be contained
	   on this processor.
    \param In
           NumBlockEntriesPerRow - An integer array of length NumRows
	   such that NumBlockEntriesPerRow[i] indicates the (approximate)
	   number of Block entries in the ith row.
  */
  Epetra_FEVbrMatrix(Epetra_DataAccess CV,
		     const Epetra_BlockMap& RowMap,
		     const Epetra_BlockMap& ColMap,
		     int *NumBlockEntriesPerRow,
		     bool ignoreNonLocalEntries=false);
  
  //! Epetra_FEVbrMatrix constuctor with fixed number of indices per row.
  /*! Creates a Epetra_FEVbrMatrix object and allocates storage.  
    
    \param In
           CV - A Epetra_DataAccess enumerated type set to Copy or View.
    \param In 
           RowMap - An Epetra_BlockMap listing the block rows that this
	   processor will contribute to.
    \param In 
           ColMap - An Epetra_BlockMap listing the block columns to be contained
	   on this processor.
    \param In
           NumBlockEntriesPerRow - An integer that indicates the (approximate)
	   number of Block entries in the each Block row.
	   Note that it is possible to use 0 for this value and let fill occur
	   during the insertion phase.
  */
  Epetra_FEVbrMatrix(Epetra_DataAccess CV,
		     const Epetra_BlockMap& RowMap,
		     const Epetra_BlockMap& ColMap,
		     int NumBlockEntriesPerRow,
		     bool ignoreNonLocalEntries=false);

  /** Constructor with pre-constructed Graph.
   */
  Epetra_FEVbrMatrix(Epetra_DataAccess CV,
		     const Epetra_CrsGraph& Graph,
		     bool ignoreNonLocalEntries=false);

  /** Copy Constructor. */
  Epetra_FEVbrMatrix(const Epetra_FEVbrMatrix& src);

  //! Epetra_VbrMatrix Destructor
  virtual ~Epetra_FEVbrMatrix();
  //@}
  
  //! @name Insertion/Replace/SumInto methods
  //@{ 

  Epetra_FEVbrMatrix& operator=(const Epetra_FEVbrMatrix& src);

  //! Initialize all values in graph of the matrix with constant value.
  /*!
    \param In
           ScalarConstant - Value to use.

    \return Integer error code, set to 0 if successful.
  */
    int PutScalar(double ScalarConstant);

  //! Initiate insertion of a list of elements in a given global row of the matrix, values are inserted via SubmitEntry().
  /*!
    \param In
           BlockRow - Block Row number (in global coordinates) to put elements.
    \param In
           NumBlockEntries - Number of entries.
    \param In
           Indices - Global column indices corresponding to values.

    \return Integer error code, set to 0 if successful.
  */
    int BeginInsertGlobalValues(int BlockRow, int NumBlockEntries, int * BlockIndices);

  //! Initiate replacement of current values with this list of entries for a given global row of the matrix, values are replaced via SubmitEntry()
  /*!
    \param In
           Row - Block Row number (in global coordinates) to put elements.
    \param In
           NumBlockEntries - Number of entries.
    \param In
           Indices - Global column indices corresponding to values.

    \return Integer error code, set to 0 if successful.
  */
    int BeginReplaceGlobalValues(int BlockRow, int NumBlockEntries, int *BlockIndices);

  //! Initiate summing into current values with this list of entries for a given global row of the matrix, values are replaced via SubmitEntry()
  /*!
    \param In
           Row - Block Row number (in global coordinates) to put elements.
    \param In
           NumBlockEntries - Number of entries.
    \param In
           Indices - Global column indices corresponding to values.

    \return Integer error code, set to 0 if successful.
  */
    int BeginSumIntoGlobalValues(int BlockRow, int NumBlockEntries, int *BlockIndices);

    //Let the compiler know we intend to overload the base-class function
    //SubmitBlockEntry rather than hide it.
    using Epetra_VbrMatrix::SubmitBlockEntry;

    //! Submit a block entry to the indicated block row and column specified in the Begin routine.
    /* Submit a block entry that will recorded in the block row that was initiated by one of the
       Begin routines listed above.  Once a one of the following routines: BeginInsertGlobalValues(),
       BeginInsertMyValues(), BeginReplaceGlobalValues(), BeginReplaceMyValues(), BeginSumIntoGlobalValues(),
       BeginSumIntoMyValues(), you \e must call SubmitBlockEntry() NumBlockEntries times to register the values 
       corresponding to the block indices passed in to the Begin routine.  If the Epetra_VbrMatrix constuctor
       was called in Copy mode, the values will be copied.  However, no copying will be done until the EndSubmitEntries()
       function is call to complete submission of the current block row.  If the constructor was called in View mode, all
       block entries passed via SubmitBlockEntry() will not be copied, but a pointer will be set to point to the argument Values
       that was passed in by the user.

       For performance reasons, SubmitBlockEntry() does minimal processing of data.  Any processing that can be
       delayed is performed in EndSubmitEntries().

    \param In
           Values - The starting address of the values.
    \param In
           LDA - The stride between successive columns of Values.
    \param In
           NumRows - The number of rows passed in.
    \param In
           NumCols - The number of columns passed in.

    \return Integer error code, set to 0 if successful.
    */
    int SubmitBlockEntry(double *Values, int LDA, int NumRows, int NumCols);

    //! Completes processing of all data passed in for the current block row.
    /*! This function completes the processing of all block entries submitted via SubmitBlockEntry().  
        It also checks to make sure that SubmitBlockEntry was called the correct number of times as
	specified by the Begin routine that initiated the entry process.
    */

    int EndSubmitEntries();

    int GlobalAssemble(bool callFillComplete=true);

 private:
    int SetupForNonlocalSubmits(int BlockRow,
				int NumBlockEntries,
				int * BlockIndices, 
				bool IndicesAreLocal,
				Epetra_CombineMode SubmitMode);

    int InputNonlocalBlockEntry(double *Values, int LDA,
				int NumRows, int NumCols);

    int InsertNonlocalRow(int row, int offset, int numCols);

    void destroyNonlocalData();

    bool ignoreNonLocalEntries_;

    int numNonlocalBlockRows_;
    int* nonlocalBlockRows_;
    int* nonlocalBlockRowLengths_;
    int* nonlocalBlockRowAllocLengths_;
    int** nonlocalBlockCols_;

    //Triple-pointers are gross, but we need one here. We want a 2-D table of
    //pointer-to-matrix objects. If we only use a double-pointer, it would be
    //too hard to change the lengths of the rows of the table.

    Epetra_SerialDenseMatrix*** nonlocalCoefs_;

    //Following the approach Mike uses in Epetra_VbrMatrix, we need some state
    //variables to keep track of block-entry submits.
    int curRowOffset_;
    int curColOffset_;
    int curNumCols_;
    int* curCols_;
    int curMode_;
};

#endif /* EPETRA_FEVBRMATRIX_H */

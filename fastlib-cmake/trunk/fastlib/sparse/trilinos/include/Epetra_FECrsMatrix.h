
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

#ifndef EPETRA_FECRSMATRIX_H
#define EPETRA_FECRSMATRIX_H

#include <Epetra_CrsMatrix.h>
class Epetra_Map;
class Epetra_IntSerialDenseVector;
class Epetra_SerialDenseMatrix;

/** Epetra Finite-Element CrsMatrix. This class provides the ability to
    input finite-element style sub-matrix data, including sub-matrices with
    non-local rows (which could correspond to shared finite-element nodes for
    example). This class inherits Epetra_CrsMatrix, and so all Epetra_CrsMatrix
    functionality is also available.

    It is intended that this class will be used as follows:
    <ul>
    <li> Construct with either a map or graph that describes a (non-overlapping)
    data distribution.
    <li> Input data, including non-local data, using the methods
    InsertGlobalValues(), SumIntoGlobalValues() and/or ReplaceGlobalValues().
    <li> Call the method GlobalAssemble(), which gathers all non-local data
    onto the owning processors as determined by the map provided at
    construction. Users should note that the GlobalAssemble() method has an
    optional argument which determines whether GlobalAssemble() in turn calls
    FillComplete() after the data-exchange has occurred. If not explicitly
    supplied, this argument defaults to true.
    ***NOTE***: When GlobalAssemble() calls FillComplete(), it passes the
    arguments 'DomainMap()' and 'RangeMap()', which are the map attributes
    held by the base-class CrsMatrix and its graph. If a rectangular matrix
    is being assembled, the correct domain-map and range-map must be passed to
    GlobalAssemble (there are two overloadings of this method) -- otherwise, it
    has no way of knowing what these maps should really be.
    </ul>

    Sub-matrix data, which is assumed to be a rectangular 'table' of
    coefficients accompanied by 'scatter-indices', can be provided in three
    forms:
    <ul>
    <li>Fortran-style packed 1-D array.
    <li>C-style double-pointer, or list-of-rows.
    <li>Epetra_SerialDenseMatrix object.
    </ul>
    In all cases, a "format" parameter specifies whether the data is laid out
    in row-major or column-major order (i.e., whether coefficients for a row
    lie contiguously or whether coefficients for a column lie contiguously).
    See the documentation for the methods SumIntoGlobalValues() and
    ReplaceGlobalValues().

    Important notes:
    <ol>
    <li> Since Epetra_FECrsMatrix inherits Epetra_CrsMatrix, the semantics of
    the Insert/SumInto/Replace methods are the same as they are on
    Epetra_CrsMatrix, which is:
    <ul>
    <li>InsertGlobalValues() inserts values into the matrix only if the graph
    has not yet been finalized (FillComplete() has not yet been called). For
    non-local values, the call to InsertGlobalValues() may succeed but the
    GlobalAssemble() method may then fail because the non-local data is not
    actually inserted in the underlying matrix until GlobalAssemble() is called.
    <li>SumIntoGlobalValues() and ReplaceGlobalValues() only work for values
    that already exist in the matrix. In other words, these methods can not be
    used to put new values into the matrix.
    </ul>
    </ol>
*/
class Epetra_FECrsMatrix : public Epetra_CrsMatrix {
  public:
  /** Constructor. */
   Epetra_FECrsMatrix(Epetra_DataAccess CV,
		      const Epetra_Map& RowMap,
		      int* NumEntriesPerRow,
		      bool ignoreNonLocalEntries=false);

   /** Constructor. */
   Epetra_FECrsMatrix(Epetra_DataAccess CV,
		      const Epetra_Map& RowMap,
		      int NumEntriesPerRow,
		      bool ignoreNonLocalEntries=false);

  /** Constructor. */
   Epetra_FECrsMatrix(Epetra_DataAccess CV,
		      const Epetra_Map& RowMap,
		      const Epetra_Map& ColMap,
		      int* NumEntriesPerRow,
		      bool ignoreNonLocalEntries=false);

   /** Constructor. */
   Epetra_FECrsMatrix(Epetra_DataAccess CV,
		      const Epetra_Map& RowMap,
		      const Epetra_Map& ColMap,
		      int NumEntriesPerRow,
		      bool ignoreNonLocalEntries=false);

   /** Constructor. */
   Epetra_FECrsMatrix(Epetra_DataAccess CV,
		      const Epetra_CrsGraph& Graph,
		      bool ignoreNonLocalEntries=false);

   /** Copy Constructor. */
   Epetra_FECrsMatrix(const Epetra_FECrsMatrix& src);

   /** Destructor. */
   virtual ~Epetra_FECrsMatrix();

   /** Assignment operator */
   Epetra_FECrsMatrix& operator=(const Epetra_FECrsMatrix& src);

   enum { ROW_MAJOR = 0, COLUMN_MAJOR = 3 };

   //Let the compiler know we intend to overload the following base-class
   //functions, rather than hide them.
   using Epetra_CrsMatrix::SumIntoGlobalValues;
   using Epetra_CrsMatrix::InsertGlobalValues;
   using Epetra_CrsMatrix::ReplaceGlobalValues;

   /** Sum a Fortran-style table (single-dimensional packed-list) of
       coefficients into the matrix, adding them to any coefficients that
       may already exist at the specified row/column locations.

       @param numIndices Number of rows (and columns) in the sub-matrix.
       @param indices List of scatter-indices (rows and columns) for the
       sub-matrix.
       @param values List, length numIndices*numIndices. Square sub-matrix of
       coefficients, packed in a 1-D array. Data is packed either contiguously
       by row or by column, specified by the final parameter 'format'.
       @param format Specifies whether the data in 'values' is packed in
       column-major or row-major order. Valid values are
       Epetra_FECrsMatrix::ROW_MAJOR or Epetra_FECrsMatrix::COLUMN_MAJOR. This
       is an optional parameter, default value is COLUMN_MAJOR.
   */
   int SumIntoGlobalValues(int numIndices, const int* indices,
                           const double* values,
                           int format=Epetra_FECrsMatrix::COLUMN_MAJOR);

   /** Sum a Fortran-style table (single-dimensional packed-list) of
       coefficients into the matrix, adding them to any coefficients that
       may already exist at the specified row/column locations.

       @param numRows Number of rows in the sub-matrix.
       @param rows List of row-numbers (scatter-indices) for the sub-matrix.
       @param numCols Number of columns in the sub-matrix.
       @param cols List of column-numbers (scatter-indices) for the sub-matrix.
       @param values List, length numRows*numCols. Rectangular sub-matrix of
       coefficients, packed in a 1-D array. Data is packed either contiguously
       by row or by column, specified by the final parameter 'format'.
       @param format Specifies whether the data in 'values' is packed in
       column-major or row-major order. Valid values are
       Epetra_FECrsMatrix::ROW_MAJOR or Epetra_FECrsMatrix::COLUMN_MAJOR. This
       is an optional parameter, default value is COLUMN_MAJOR.
   */
   int SumIntoGlobalValues(int numRows, const int* rows,
                           int numCols, const int* cols,
                           const double* values,
                           int format=Epetra_FECrsMatrix::COLUMN_MAJOR);

   /** Sum C-style table (double-pointer, or list of lists) of coefficients
       into the matrix, adding them to any coefficients that
       may already exist at the specified row/column locations.

       @param numIndices Number of rows (and columns) in the sub-matrix.
       @param indices List of scatter-indices (rows and columns) for the
       sub-matrix.
       @param values Square sub-matrix of coefficients, provided in a 2-D
       array, or double-pointer.
       @param format Specifies whether the data in 'values' is packed in
       column-major or row-major order. Valid values are
       Epetra_FECrsMatrix::ROW_MAJOR or Epetra_FECrsMatrix::COLUMN_MAJOR. This
       is an optional parameter, default value is ROW_MAJOR.
   */
   int SumIntoGlobalValues(int numIndices, const int* indices,
                           const double* const* values,
                           int format=Epetra_FECrsMatrix::ROW_MAJOR);

   /** Sum C-style table (double-pointer, or list of lists) of coefficients
       into the matrix, adding them to any coefficients that
       may already exist at the specified row/column locations.

       @param numRows Number of rows in the sub-matrix.
       @param rows List of row-numbers (scatter-indices) for the sub-matrix.
       @param numCols Number of columns in the sub-matrix.
       @param cols List of column-numbers (scatter-indices) for the sub-matrix.
       @param values Rectangular sub-matrix of coefficients, provided in a 2-D
       array, or double-pointer.
       @param format Specifies whether the data in 'values' is packed in
       column-major or row-major order. Valid values are
       Epetra_FECrsMatrix::ROW_MAJOR or Epetra_FECrsMatrix::COLUMN_MAJOR. This
       is an optional parameter, default value is ROW_MAJOR.
   */
   int SumIntoGlobalValues(int numRows, const int* rows,
	                   int numCols, const int* cols,
                           const double* const* values,
                           int format=Epetra_FECrsMatrix::ROW_MAJOR);

   /** Insert a Fortran-style table (single-dimensional packed-list) of
       coefficients into the matrix.

       @param numIndices Number of rows (and columns) in the sub-matrix.
       @param indices List of scatter-indices (rows and columns) for the
       sub-matrix.
       @param values List, length numIndices*numIndices. Square sub-matrix of
       coefficients, packed in a 1-D array. Data is packed either contiguously
       by row or by column, specified by the final parameter 'format'.
       @param format Specifies whether the data in 'values' is packed in
       column-major or row-major order. Valid values are
       Epetra_FECrsMatrix::ROW_MAJOR or Epetra_FECrsMatrix::COLUMN_MAJOR. This
       is an optional parameter, default value is COLUMN_MAJOR.
   */
   int InsertGlobalValues(int numIndices, const int* indices,
                           const double* values,
                           int format=Epetra_FECrsMatrix::COLUMN_MAJOR);

   /** Insert a Fortran-style table (single-dimensional packed-list) of
       coefficients into the matrix.

       @param numRows Number of rows in the sub-matrix.
       @param rows List of row-numbers (scatter-indices) for the sub-matrix.
       @param numCols Number of columns in the sub-matrix.
       @param cols List of column-numbers (scatter-indices) for the sub-matrix.
       @param values List, length numRows*numCols. Rectangular sub-matrix of
       coefficients, packed in a 1-D array. Data is packed either contiguously
       by row or by column, specified by the final parameter 'format'.
       @param format Specifies whether the data in 'values' is packed in
       column-major or row-major order. Valid values are
       Epetra_FECrsMatrix::ROW_MAJOR or Epetra_FECrsMatrix::COLUMN_MAJOR. This
       is an optional parameter, default value is COLUMN_MAJOR.
   */
   int InsertGlobalValues(int numRows, const int* rows,
                           int numCols, const int* cols,
                           const double* values,
                           int format=Epetra_FECrsMatrix::COLUMN_MAJOR);

   /** Insert a C-style table (double-pointer, or list of lists) of coefficients
       into the matrix.

       @param numIndices Number of rows (and columns) in the sub-matrix.
       @param indices List of scatter-indices (rows and columns) for the
       sub-matrix.
       @param values Square sub-matrix of coefficients, provided in a 2-D
       array, or double-pointer.
       @param format Specifies whether the data in 'values' is packed in
       column-major or row-major order. Valid values are
       Epetra_FECrsMatrix::ROW_MAJOR or Epetra_FECrsMatrix::COLUMN_MAJOR. This
       is an optional parameter, default value is ROW_MAJOR.
   */
   int InsertGlobalValues(int numIndices, const int* indices,
                           const double* const* values,
                           int format=Epetra_FECrsMatrix::ROW_MAJOR);

   /** Insert a C-style table (double-pointer, or list of lists) of coefficients
       into the matrix.

       @param numRows Number of rows in the sub-matrix.
       @param rows List of row-numbers (scatter-indices) for the sub-matrix.
       @param numCols Number of columns in the sub-matrix.
       @param cols List of column-numbers (scatter-indices) for the sub-matrix.
       @param values Rectangular sub-matrix of coefficients, provided in a 2-D
       array, or double-pointer.
       @param format Specifies whether the data in 'values' is packed in
       column-major or row-major order. Valid values are
       Epetra_FECrsMatrix::ROW_MAJOR or Epetra_FECrsMatrix::COLUMN_MAJOR. This
       is an optional parameter, default value is ROW_MAJOR.
   */
   int InsertGlobalValues(int numRows, const int* rows,
	                   int numCols, const int* cols,
                           const double* const* values,
                           int format=Epetra_FECrsMatrix::ROW_MAJOR);

   /** Copy a Fortran-style table (single-dimensional packed-list) of
       coefficients into the matrix, replacing any coefficients that
       may already exist at the specified row/column locations.

       @param numIndices Number of rows (and columns) in the sub-matrix.
       @param indices List of scatter-indices (rows and columns) for the
       sub-matrix.
       @param values List, length numIndices*numIndices. Square sub-matrix of
       coefficients, packed in a 1-D array. Data is packed either contiguously
       by row or by column, specified by the final parameter 'format'.
       @param format Specifies whether the data in 'values' is packed in
       column-major or row-major order. Valid values are
       Epetra_FECrsMatrix::ROW_MAJOR or Epetra_FECrsMatrix::COLUMN_MAJOR. This
       is an optional parameter, default value is COLUMN_MAJOR.
   */
   int ReplaceGlobalValues(int numIndices, const int* indices,
                           const double* values,
                           int format=Epetra_FECrsMatrix::COLUMN_MAJOR);

   /** Copy Fortran-style table (single-dimensional packed-list) of coefficients
       into the matrix, replacing any coefficients that
       may already exist at the specified row/column locations.

       @param numRows Number of rows in the sub-matrix.
       @param rows List of row-numbers (scatter-indices) for the sub-matrix.
       @param numCols Number of columns in the sub-matrix.
       @param cols List, of column-numbers 
       (scatter-indices) for the sub-matrix.
       @param values List, length numRows*numCols. Rectangular sub-matrix of
       coefficients, packed in a 1-D array. Data is packed either contiguously
       by row or by column, specified by the final parameter 'format'.
       @param format Specifies whether the data in 'values' is packed in
       column-major or row-major order. Valid values are
       Epetra_FECrsMatrix::ROW_MAJOR or Epetra_FECrsMatrix::COLUMN_MAJOR. This
       is an optional parameter, default value is COLUMN_MAJOR.
   */
   int ReplaceGlobalValues(int numRows, const int* rows,
                           int numCols, const int* cols,
                           const double* values,
                           int format=Epetra_FECrsMatrix::COLUMN_MAJOR);

   /** Copy C-style table (double-pointer, or list of lists) of coefficients
       into the matrix, replacing any coefficients that
       may already exist at the specified row/column locations.

       @param numIndices Number of rows (and columns) in the sub-matrix.
       @param indices List of scatter-indices (rows and columns) for the
       sub-matrix.
       @param values Square sub-matrix of coefficients, provided in a 2-D
       array, or double-pointer.
       @param format Specifies whether the data in 'values' is packed in
       column-major or row-major order. Valid values are
       Epetra_FECrsMatrix::ROW_MAJOR or Epetra_FECrsMatrix::COLUMN_MAJOR. This
       is an optional parameter, default value is ROW_MAJOR.
   */
   int ReplaceGlobalValues(int numIndices, const int* indices,
                           const double* const* values,
                           int format=Epetra_FECrsMatrix::ROW_MAJOR);

   /** Copy C-style table (double-pointer, or list of lists) of coefficients
       into the matrix, replacing any coefficients that
       may already exist at the specified row/column locations.

       @param numRows Number of rows in the sub-matrix.
       @param rows List of row-numbers (scatter-indices) for the sub-matrix.
       @param numCols Number of columns in the sub-matrix.
       @param cols List of column-numbers (scatter-indices) for the sub-matrix.
       @param values Rectangular sub-matrix of coefficients, provided in a 2-D
       array, or double-pointer.
       @param format Specifies whether the data in 'values' is packed in
       column-major or row-major order. Valid values are
       Epetra_FECrsMatrix::ROW_MAJOR or Epetra_FECrsMatrix::COLUMN_MAJOR. This
       is an optional parameter, default value is ROW_MAJOR.
   */
   int ReplaceGlobalValues(int numRows, const int* rows,
                           int numCols, const int* cols,
                           const double* const* values,
                           int format=Epetra_FECrsMatrix::ROW_MAJOR);

   /** Sum a square structurally-symmetric sub-matrix into the global matrix.
       For non-square sub-matrices, see the other overloading of this method.

       @param indices List of scatter-indices. indices.Length() must be the same
       as values.M() and values.N().

       @param values Sub-matrix of coefficients. Must be square.

       @param format Optional format specifier, defaults to COLUMN_MAJOR.
   */
   int SumIntoGlobalValues(const Epetra_IntSerialDenseVector& indices,
			   const Epetra_SerialDenseMatrix& values,
			   int format=Epetra_FECrsMatrix::COLUMN_MAJOR);

   /** Sum a general sub-matrix into the global matrix.
       For square structurally-symmetric sub-matrices, see the other
       overloading of this method.

       @param rows List of row-indices. rows.Length() must be the same
       as values.M().

       @param cols List of column-indices. cols.Length() must be the same
       as values.N().

       @param values Sub-matrix of coefficients.

       @param format Optional format specifier, defaults to COLUMN_MAJOR.
   */
   int SumIntoGlobalValues(const Epetra_IntSerialDenseVector& rows,
			   const Epetra_IntSerialDenseVector& cols,
			   const Epetra_SerialDenseMatrix& values,
			   int format=Epetra_FECrsMatrix::COLUMN_MAJOR);

   /** Insert a square structurally-symmetric sub-matrix into the global matrix.
       For non-square sub-matrices, see the other overloading of this method.

       @param indices List of scatter-indices. indices.Length() must be the same
       as values.M() and values.N().

       @param values Sub-matrix of coefficients. Must be square.

       @param format Optional format specifier, defaults to COLUMN_MAJOR.
   */
   int InsertGlobalValues(const Epetra_IntSerialDenseVector& indices,
			   const Epetra_SerialDenseMatrix& values,
			   int format=Epetra_FECrsMatrix::COLUMN_MAJOR);

   /** Insert a general sub-matrix into the global matrix.
       For square structurally-symmetric sub-matrices, see the other
       overloading of this method.

       @param rows List of row-indices. rows.Length() must be the same
       as values.M().

       @param cols List of column-indices. cols.Length() must be the same
       as values.N().

       @param values Sub-matrix of coefficients.

       @param format Optional format specifier, defaults to COLUMN_MAJOR.
   */
   int InsertGlobalValues(const Epetra_IntSerialDenseVector& rows,
			   const Epetra_IntSerialDenseVector& cols,
			   const Epetra_SerialDenseMatrix& values,
			   int format=Epetra_FECrsMatrix::COLUMN_MAJOR);

   /** Use a square structurally-symmetric sub-matrix to replace existing
       values in the global matrix.
       For non-square sub-matrices, see the other overloading of this method.

       @param indices List of scatter-indices. indices.Length() must be the same
       as values.M() and values.N().

       @param values Sub-matrix of coefficients. Must be square.

       @param format Optional format specifier, defaults to COLUMN_MAJOR.
   */
   int ReplaceGlobalValues(const Epetra_IntSerialDenseVector& indices,
			   const Epetra_SerialDenseMatrix& values,
			   int format=Epetra_FECrsMatrix::COLUMN_MAJOR);

   /** Use a general sub-matrix to replace existing values.
       For square structurally-symmetric sub-matrices, see the other
       overloading of this method.

       @param rows List of row-indices. rows.Length() must be the same
       as values.M().

       @param cols List of column-indices. cols.Length() must be the same
       as values.N().

       @param values Sub-matrix of coefficients.

       @param format Optional format specifier, defaults to COLUMN_MAJOR.
   */
   int ReplaceGlobalValues(const Epetra_IntSerialDenseVector& rows,
			   const Epetra_IntSerialDenseVector& cols,
			   const Epetra_SerialDenseMatrix& values,
			   int format=Epetra_FECrsMatrix::COLUMN_MAJOR);

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

   /** Set whether or not non-local data values should be ignored. By default,
       non-local data values are NOT ignored.
    */
   void setIgnoreNonLocalEntries(bool flag) {
     ignoreNonLocalEntries_ = flag;
   }

  private:
   void DeleteMemory();

   enum {SUMINTO = 0, REPLACE = 1, INSERT = 2};

   int InputGlobalValues(int numRows, const int* rows,
                         int numCols, const int* cols,
                         const double* const* values,
                         int format,
                         int mode);

   int InputGlobalValues(int numRows, const int* rows,
                         int numCols, const int* cols,
                         const double* values,
                         int format,
                         int mode);

   int InputNonlocalGlobalValues(int row,
				 int numCols, const int* cols,
				 const double* values,
				 int mode);

   int InsertNonlocalRow(int row, int offset);

   int InputNonlocalValue(int rowoffset,
			  int col, double value,
			  int mode);

   int myFirstRow_;
   int myNumRows_;

   bool ignoreNonLocalEntries_;

   int numNonlocalRows_;
   int* nonlocalRows_;
   int* nonlocalRowLengths_;
   int* nonlocalRowAllocLengths_;
   int** nonlocalCols_;
   double** nonlocalCoefs_;

   double* workData_;
   int workDataLength_;
};//class Epetra_FECrsMatrix

#endif /* EPETRA_FECRSMATRIX_H */

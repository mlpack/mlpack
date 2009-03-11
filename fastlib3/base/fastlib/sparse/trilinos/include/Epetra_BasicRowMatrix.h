
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

#ifndef EPETRA_BASICROWMATRIX_H
#define EPETRA_BASICROWMATRIX_H

#include "Epetra_RowMatrix.h"
#include "Epetra_Object.h"
#include "Epetra_CompObject.h"
#include "Epetra_Map.h"
#include "Epetra_Comm.h"
#include "Epetra_SerialDenseVector.h"
#include "Epetra_IntSerialDenseVector.h"
#include "Epetra_MultiVector.h"

class Epetra_Vector;
class Epetra_Import;
class Epetra_Export;

//! Epetra_BasicRowMatrix: A class for simplifying the development of Epetra_RowMatrix adapters.

/*! The Epetra_BasicRowMatrix is an adapter class for Epetra_RowMatrix that implements most of the Epetra_RowMatrix
    methods using reasonable default implementations.  The Epetra_RowMatrix class has 39 pure virtual methods, requiring
    the adapter class to implement all of them. 
    Epetra_BasicRowMatrix has only 4 pure virtual methods that must be implemented (See Epetra_JadMatrix for an example):
<ol>
<li> ExtractMyRowCopy: Provide a row of values and indices for a specified local row.
<li> ExtractMyEntryView (const and non-const versions): Provide the memory address of the ith nonzero term stored on the
     calling processor, along with its corresponding local row and column index, where i goes from 0 to the NumMyNonzeros()-1.
     The order in which the nonzeros are traversed is not specified and is up to the adapter implementation.
<li> NumMyRowEntries: Provide the number of entries for a specified local row.
</ol>

     An alternative is possible if you do not want to provide a non-trivial implementation of the ExtraMyEntryView 
     methods (See Epetra_VbrRowMatrix for and example):
<ol>
<li> Implement ExtractMyRowCopy and NumMyRowEntries as above.
<li> Implement ExtractMyEntryView (both versions) returning a -1 integer code with no other executable code.
<li> Implement the RightScale and LeftScale methods non-trivially.
</ol>

In addition, most adapters will probably re-implement the Multiply() method and perhaps the Solve() method, although one or the other
may be implemented to return -1, signaling that there is no valid implementation.  By default, the Multiply() method is implemented using
ExtractMyRowCopy, which can usual be improved upon.  By default Solve() and ApplyInverse() are implemented to return -1 (not implemented).

All other implemented methods in Epetra_BasicRowMatrix should not exhibit a signficant performance degradation, either because they are relatively
small and fast, or because they are not a significant portion of the runtime for most codes.  All methods are virtual, so they can be re-implemented
by the adapter.

In addition to implementing the above methods, an adapter must inherit the Epetra_BasicRowMatrix interface and call the Epetra_BasicRowMatrix
constructor as part of the adapter constructor.  There are two constructors.  The first requires the user to pass in the RowMap and ColMap, both
of which are Epetra_Map objects.  On each processor the RowMap (ColMap) must contain the global IDs (GIDs) of the rows (columns) that the processor cares about.  
The first constructor requires only these two maps, assuming that the RowMap will also serve as the DomainMap and RangeMap.  In this case, the
RowMap must be 1-to-1, meaning that if a global ID appears on one processor, it appears only once on that processor and does not appear on any other
processor.  For many sparse matrix data structures, it is the case that a given row is completely owned by one processor and that the global matrix
is square.  The first constructor is for this situation.

The second constructor allows the caller to specify all four maps.  In this case the DomainMap, the layout of multivectors/vectors that are in the
domain of the matrix (the x vector if computing y = A*x), must be 1-to-1.  Also, the RangeMap, the layout of y must be 1-to-1.  The RowMap and ColMap
do not need to be 1-to-1, but the GIDs must be found in the RangeMap and DomainMap, respectively.

Note that Epetra_Operator is a base class for Epetra_RowMatrix, so any adapter
for Epetra_BasicRowMatrix (or Epetra_RowMatrix) is also an adapter for Epetra_Operator.

An example of how to provide an adapter for Epetra_BasicRowMatrix can be found by looking at Epetra_JadMatrix.

*/    

class Epetra_BasicRowMatrix: public Epetra_CompObject, public Epetra_Object, public virtual Epetra_RowMatrix  {
      
 public:

   //! @name Constructor/Destructor
  //@{ 
  //! Epetra_BasicRowMatrix constuctor.
  /* This constructor requires a valid Epetra_Comm object as its only argument.  The constructor will use Comm to build
     Epetra_Maps objects: RowMap, ColMap, DomainMap and RangeMap.  However, these will be zero-length (trivial) maps that
     \e must be reset by calling one of the two SetMap() methods listed below.
     \param Comm (In) An Epetra_Comm containing a valid Comm object.
  */
  Epetra_BasicRowMatrix(const Epetra_Comm & Comm);

  //! Epetra_BasicRowMatrix Destructor
  virtual ~Epetra_BasicRowMatrix();
  //@}
  
  //! @name Setup functions
  //@{ 
  //! Set maps (Version 1); call this function or the next, but not both.
  /* This method takes a row and column map.  On each processor these maps describe the global rows and columns, resp, 
     that the processor will care about.  Note that the ColMap does not have to be one-to-one.  In other words, a column ID can appear
     on more than one processor.  The RowMap \e must be 1-to-1.
     \param RowMap (In) An Epetra_Map containing on each processor a list of GIDs of rows that the processor cares about.
     \param ColMap (In) An Epetra_Map containing on each processor a list of GIDs of columns that the processor cares about.

     In this method, the domain and range maps are assumed to be the same as the row map.  Note that this requires that 
     the global matrix be square.  If the matrix is not square, or the domain vectors or range vectors do not have the same layout
     as the rows, then the second constructor should be called.
  */
  void SetMaps(const Epetra_Map & RowMap, const Epetra_Map & ColMap);

  //! Set maps (Version 2); call this function or the previous, but not both.
  /* This constructor takes a row, column, domain and range map.  On each processor these maps describe the global rows, columns, domain
     and range, resp, that the processor will care about.  The domain and range maps must be one-to-one, but note that the row and column
     maps do not have to be one-to-one.  In other words, a row ID can appear
     on more than one processor, as can a column ID.
     \param RowMap (In) An Epetra_Map containing on each processor a list of GIDs of rows that the processor cares about.
     \param ColMap (In) An Epetra_Map containing on each processor a list of GIDs of columns that the processor cares about.
     \param DomainMap (In) An Epetra_Map describing the distribution of domain vectors and multivectors.
     \param RangeMap (In) An Epetra_Map describing the distribution of range vectors and multivectors.

  */
  void SetMaps(const Epetra_Map & RowMap, const Epetra_Map & ColMap, 
	       const Epetra_Map & DomainMap, const Epetra_Map & RangeMap);

  //@}

  
  //! @name User-required implementation methods
  //@{ 

    //! Returns a copy of the specified local row in user-provided arrays.
    /*! 
    \param MyRow (In) - Local row to extract.
    \param Length (In) - Length of Values and Indices.
    \param NumEntries (Out) - Number of nonzero entries extracted.
    \param Values (Out) - Extracted values for this row.
    \param Indices (Out) - Extracted global column indices for the corresponding values.
	  
    \return Integer error code, set to 0 if successful, set to -1 if MyRow not valid, -2 if Length is too short (NumEntries will have required length).
  */
  virtual int ExtractMyRowCopy(int MyRow, int Length, int & NumEntries, double *Values, int * Indices) const = 0;

    //! Returns a reference to the ith entry in the matrix, along with its row and column index.
    /*! 
    \param CurEntry (In) - Index of local entry (from 0 to NumMyNonzeros()-1) to extract.
    \param Value (Out) - Extracted reference to current values.
    \param RowIndex (Out) - Row index for current entry.
    \param ColIndex (Out) - Column index for current entry.
	  
    \return Integer error code, set to 0 if successful, set to -1 if CurEntry not valid.
  */
    virtual int ExtractMyEntryView(int CurEntry, double * & Value, int & RowIndex, int & ColIndex) = 0;

    //! Returns a const reference to the ith entry in the matrix, along with its row and column index.
    /*! 
    \param CurEntry (In) - Index of local entry (from 0 to NumMyNonzeros()-1) to extract.
    \param Value (Out) - Extracted reference to current values.
    \param RowIndex (Out) - Row index for current entry.
    \param ColIndex (Out) - Column index for current entry.
	  
    \return Integer error code, set to 0 if successful, set to -1 if CurEntry not valid.
  */
    virtual int ExtractMyEntryView(int CurEntry, double const * & Value, int & RowIndex, int & ColIndex) const = 0;

    //! Return the current number of values stored for the specified local row.
    /*! Similar to NumMyEntries() except NumEntries is returned as an argument
      and error checking is done on the input value MyRow.
      \param MyRow (In) - Local row.
      \param NumEntries (Out) - Number of nonzero values.
      
      \return Integer error code, set to 0 if successful, set to -1 if MyRow not valid.
    */
    virtual int NumMyRowEntries(int MyRow, int & NumEntries) const = 0;
    //@}

    //! @name Computational methods
  //@{ 

    //! Returns the result of a Epetra_BasicRowMatrix multiplied by a Epetra_MultiVector X in Y.
    /*! 
    \param TransA (In) - If true, multiply by the transpose of matrix, otherwise just use matrix.
    \param X (Out) - An Epetra_MultiVector of dimension NumVectors to multiply with matrix.
    \param Y (Out) - An Epetra_MultiVector of dimension NumVectorscontaining result.

    \return Integer error code, set to 0 if successful.
  */
    virtual int Multiply(bool TransA, const Epetra_MultiVector& X, Epetra_MultiVector& Y) const;

    //! Returns the result of a Epetra_BasicRowMatrix solve with a Epetra_MultiVector X in Y (not implemented).
    /*! 
    \param Upper (In) - If true, solve Ux = y, otherwise solve Lx = y.
    \param Trans (In) - If true, solve transpose problem.
    \param UnitDiagonal (In) - If true, assume diagonal is unit (whether it's stored or not).
    \param X (In) - An Epetra_MultiVector of dimension NumVectors to solve for.
    \param Y (Out) - An Epetra_MultiVector of dimension NumVectors containing result.

    \return Integer error code, set to 0 if successful.
  */
    virtual int Solve(bool Upper, bool Trans, bool UnitDiagonal,
                      const Epetra_MultiVector& X,
                      Epetra_MultiVector& Y) const
    {
      (void)Upper;
      (void)Trans;
      (void)UnitDiagonal;
      (void)X;
      (void)Y;
      return(-1);
    }

    //! Returns a copy of the main diagonal in a user-provided vector.
    /*! 
    \param Diagonal (Out) - Extracted main diagonal.

    \return Integer error code, set to 0 if successful.
  */
    virtual int ExtractDiagonalCopy(Epetra_Vector & Diagonal) const;

    //! Computes the sum of absolute values of the rows of the Epetra_BasicRowMatrix, results returned in x.
    /*! The vector x will return such that x[i] will contain the inverse of sum of the absolute values of the 
        \e this matrix will be scaled such that A(i,j) = x(i)*A(i,j) where i denotes the global row number of A
        and j denotes the global column number of A.  Using the resulting vector from this function as input to LeftScale()
	will make the infinity norm of the resulting matrix exactly 1.
    \param x (Out) - An Epetra_Vector containing the row sums of the \e this matrix. 
	   \warning It is assumed that the distribution of x is the same as the rows of \e this.

    \return Integer error code, set to 0 if successful.
  */
    virtual int InvRowSums(Epetra_Vector& x) const;

    //! Scales the Epetra_BasicRowMatrix on the left with a Epetra_Vector x.
    /*! The \e this matrix will be scaled such that A(i,j) = x(i)*A(i,j) where i denotes the row number of A
        and j denotes the column number of A.
    \param x (In) - An Epetra_Vector to solve for.

    \return Integer error code, set to 0 if successful.
  */
    virtual int LeftScale(const Epetra_Vector& x);

    //! Computes the sum of absolute values of the columns of the Epetra_BasicRowMatrix, results returned in x.
    /*! The vector x will return such that x[j] will contain the inverse of sum of the absolute values of the 
        \e this matrix will be sca such that A(i,j) = x(j)*A(i,j) where i denotes the global row number of A
        and j denotes the global column number of A.  Using the resulting vector from this function as input to 
	RighttScale() will make the one norm of the resulting matrix exactly 1.
    \param x (Out) - An Epetra_Vector containing the column sums of the \e this matrix. 
	   \warning It is assumed that the distribution of x is the same as the rows of \e this.

    \return Integer error code, set to 0 if successful.
  */
    virtual int InvColSums(Epetra_Vector& x) const;

    //! Scales the Epetra_BasicRowMatrix on the right with a Epetra_Vector x.
    /*! The \e this matrix will be scaled such that A(i,j) = x(j)*A(i,j) where i denotes the global row number of A
        and j denotes the global column number of A.
    \param x (In) - The Epetra_Vector used for scaling \e this.

    \return Integer error code, set to 0 if successful.
  */
    virtual int RightScale(const Epetra_Vector& x);
  //@}

  //! @name Matrix Properties Query Methods
  //@{ 


    //! If FillComplete() has been called, this query returns true, otherwise it returns false, presently always returns true.
    virtual bool Filled() const {return(true);}

    //! If matrix is lower triangular, this query returns true, otherwise it returns false.
    bool LowerTriangular() const {if (!HaveNumericConstants_) ComputeNumericConstants(); return(LowerTriangular_);}

    //! If matrix is upper triangular, this query returns true, otherwise it returns false.
    virtual bool UpperTriangular() const {if (!HaveNumericConstants_) ComputeNumericConstants(); return(UpperTriangular_);}

  //@}
  
  //! @name Atribute access functions
  //@{ 

    //! Returns the infinity norm of the global matrix.
    /* Returns the quantity \f$ \| A \|_\infty\f$ such that
       \f[\| A \|_\infty = \max_{1\lei\lem} \sum_{j=1}^n |a_{ij}| \f].

     \warning This method is supported if and only if the Epetra_RowMatrix Object that was used to create this supports this method.

    */ 
    virtual double NormInf() const{if (!HaveNumericConstants_) ComputeNumericConstants(); return(NormInf_);}

    //! Returns the one norm of the global matrix.
    /* Returns the quantity \f$ \| A \|_1\f$ such that
       \f[\| A \|_1= \max_{1\lej\len} \sum_{i=1}^m |a_{ij}| \f].

     \warning This method is supported if and only if the Epetra_RowMatrix Object that was used to create this supports this method.

    */ 
    virtual double NormOne() const{if (!HaveNumericConstants_) ComputeNumericConstants(); return(NormOne_);}

    //! Returns the number of nonzero entries in the global matrix.
    /* Note that if the data decomposition is defined such that some nonzeros
       appear on multiple processors, then those nonzeros will be counted
       multiple times.
    */
    virtual int NumGlobalNonzeros() const{if (!HaveStructureConstants_) ComputeStructureConstants(); return(NumGlobalNonzeros_);}

    //! Returns the number of global matrix rows.
    virtual int NumGlobalRows() const {return(OperatorRangeMap().NumGlobalPoints());}

    //! Returns the number of global matrix columns.
    virtual int NumGlobalCols() const {return(OperatorDomainMap().NumGlobalPoints());}

    //! Returns the number of global nonzero diagonal entries.
    virtual int NumGlobalDiagonals() const{return(OperatorDomainMap().NumGlobalPoints());}
    
    //! Returns the number of nonzero entries in the calling processor's portion of the matrix.
    virtual int NumMyNonzeros() const{if (!HaveStructureConstants_) ComputeStructureConstants(); return(NumMyNonzeros_);}

    //! Returns the number of matrix rows owned by the calling processor.
    virtual int NumMyRows() const {return(OperatorRangeMap().NumMyPoints());}

    //! Returns the number of matrix columns owned by the calling processor.
    virtual int NumMyCols() const {return(RowMatrixColMap().NumMyPoints());}

    //! Returns the number of local nonzero diagonal entries.
    virtual int NumMyDiagonals() const {return(OperatorRangeMap().NumMyPoints());}

    //! Returns the maximum number of nonzero entries across all rows on this processor.
    virtual int MaxNumEntries() const{ if (!HaveStructureConstants_) ComputeStructureConstants(); return(MaxNumEntries_);}

    //! Returns the Epetra_Map object associated with the domain of this operator.
    virtual const Epetra_Map & OperatorDomainMap() const {return(OperatorDomainMap_);}

    //! Returns the Epetra_Map object associated with the range of this operator (same as domain).
    virtual const Epetra_Map & OperatorRangeMap() const  {return(OperatorRangeMap_);}

    //! Implement the Epetra_SrcDistObjec::Map() function.
    virtual const Epetra_BlockMap& Map() const {return(RowMatrixRowMap());}

    //! Returns the Row Map object needed for implementing Epetra_RowMatrix.
    virtual const Epetra_Map & RowMatrixRowMap() const {return(RowMatrixRowMap_);}

    //! Returns the Column Map object needed for implementing Epetra_RowMatrix.
    virtual const Epetra_Map & RowMatrixColMap() const {return(RowMatrixColMap_);}

    //! Returns the Epetra_Import object that contains the import operations for distributed operations.
    virtual const Epetra_Import * RowMatrixImporter() const {return(Importer_);}

    //! Returns a pointer to the Epetra_Comm communicator associated with this matrix.
    virtual const Epetra_Comm & Comm() const {return(*Comm_);}
  //@}
  
  
  //! @name I/O Methods
  //@{ 

  //! Print method
  virtual void Print(ostream & os) const;
  //@}

  //! @name Additional methods required to support the Epetra_RowMatrix interface
  //@{ 
    
    //! If set true, transpose of this operator will be applied.
    /*! This flag allows the transpose of the given operator to be used implicitly.  Setting this flag
        affects only the Apply() and ApplyInverse() methods.  If the implementation of this interface 
	does not support transpose use, this method should return a value of -1.
      
    \param UseTranspose (In) - If true, multiply by the transpose of operator, otherwise just use operator.

    \return Always returns 0.
  */
  virtual int SetUseTranspose(bool UseTranspose) {UseTranspose_ = UseTranspose; return(0);}

  //! Returns a character string describing the operator
  virtual const char* Label() const {return(Epetra_Object::Label());}
  
  //! Returns the result of a Epetra_RowMatrix applied to a Epetra_MultiVector X in Y.
  /*! 
    \param X (In) - A Epetra_MultiVector of dimension NumVectors to multiply with matrix.
    \param Y (Out) - A Epetra_MultiVector of dimension NumVectors containing result.
    
    \return Integer error code, set to 0 if successful.
  */
  virtual int Apply(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const {
    return(Multiply(Epetra_BasicRowMatrix::UseTranspose(), X, Y));}

    //! Returns the result of a Epetra_RowMatrix inverse applied to an Epetra_MultiVector X in Y.
    /*! 

    \param X (In) - A Epetra_MultiVector of dimension NumVectors to solve for.
    \param Y (Out) - A Epetra_MultiVector of dimension NumVectors containing result.

    \return Integer error code = -1.
    \warning This method is NOT supported.
  */
  virtual int ApplyInverse(const Epetra_MultiVector& X,
                           Epetra_MultiVector& Y) const
   {
     (void)X;
     (void)Y;
     return(-1);
   }

  //! Returns true because this class can compute an Inf-norm.
  bool HasNormInf() const {return(true);}
  
  //! Returns the current UseTranspose setting.
  virtual bool UseTranspose() const {return(UseTranspose_);}

  //@}
  
  //! @name Additional accessor methods
  //@{ 

  //! Returns the Epetra_Import object that contains the import operations for distributed operations, returns zero if none.
    /*! If RowMatrixColMap!=OperatorDomainMap, then this method returns a pointer to an Epetra_Import object that imports objects
        from an OperatorDomainMap layout to a RowMatrixColMap layout.  This operation is needed for sparse matrix-vector
	multiplication, y = Ax, to gather x elements for local multiplication operations.

	If RowMatrixColMap==OperatorDomainMap, then the pointer will be returned as 0.

    \return Raw pointer to importer.  This importer will be valid as long as the Epetra_RowMatrix object is valid.
  */
  virtual const Epetra_Import* Importer() const {return(Importer_);}
  
  //! Returns the Epetra_Export object that contains the export operations for distributed operations, returns zero if none.
    /*! If RowMatrixRowMap!=OperatorRangeMap, then this method returns a pointer to an Epetra_Export object that exports objects
        from an RowMatrixRowMap layout to a OperatorRangeMap layout.  This operation is needed for sparse matrix-vector
	multiplication, y = Ax, to scatter-add y elements generated during local multiplication operations.

	If RowMatrixRowMap==OperatorRangeMap, then the pointer will be returned as 0.  For a typical Epetra_RowMatrix object,
	this pointer will be zero since it is often the case that RowMatrixRowMap==OperatorRangeMap.

    \return Raw pointer to exporter.  This exporter will be valid as long as the Epetra_RowMatrix object is valid.
  */
  virtual const Epetra_Export* Exporter() const {return(Exporter_);}

  //@}

 protected:

  //! @name Post-construction modifications
  //@{ 
  //! Update the constants associated with the structure of the matrix: Call only if structure changes from the initial RowMatrix.
  /* Several constants are pre-computed to save excess computations.  However, if the structure of the
     problem changes, specifically if the nonzero count in any given row changes, then this function should be called
     to update these constants.
  */ 
  virtual void ComputeStructureConstants() const;
  //! Update the constants associated with the values of the matrix: Call only if values changes from the initial RowMatrix.
  /* Several numeric constants are pre-computed to save excess computations.  However, if the values of the
     problem change, then this function should be called to update these constants.
  */ 
  virtual void ComputeNumericConstants() const;
  //@}

  void Setup();
  void UpdateImportVector(int NumVectors) const;
  void UpdateExportVector(int NumVectors) const;
  void SetImportExport();
  Epetra_Comm * Comm_;
  Epetra_Map OperatorDomainMap_;
  Epetra_Map OperatorRangeMap_;
  Epetra_Map RowMatrixRowMap_;
  Epetra_Map RowMatrixColMap_;
  
  mutable int NumMyNonzeros_;
  mutable int NumGlobalNonzeros_;
  mutable int MaxNumEntries_;
  mutable double NormInf_;
  mutable double NormOne_;
  int NumMyRows_;
  int NumMyCols_;

  bool UseTranspose_;
  bool HasNormInf_;
  mutable bool LowerTriangular_;
  mutable bool UpperTriangular_;
  mutable bool HaveStructureConstants_;
  mutable bool HaveNumericConstants_;
  mutable bool HaveMaps_;
    

  mutable Epetra_MultiVector * ImportVector_;
  mutable Epetra_MultiVector * ExportVector_;
  Epetra_Import * Importer_;
  Epetra_Export * Exporter_;

};
#endif /* EPETRA_BASICROWMATRIX_H */

/*@HEADER
// ***********************************************************************
//
//       Ifpack: Object-Oriented Algebraic Preconditioner Package
//                 Copyright (2002) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
//
// This library is free software; you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as
// published by the Free Software Foundation; either version 2.1 of the
// License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
// USA
// Questions? Contact Michael A. Heroux (maherou@sandia.gov)
//
// ***********************************************************************
//@HEADER
*/

#ifndef IFPACK_REORDERFILTER_H
#define IFPACK_REORDERFILTER_H

#include "Ifpack_ConfigDefs.h"
#include "Epetra_RowMatrix.h"
#include "Teuchos_RefCountPtr.hpp"

class Epetra_Comm;
class Epetra_Map;
class Epetra_MultiVector;
class Epetra_Import;
class Epetra_BlockMap;
class Ifpack_Reordering;

//! Ifpack_ReorderFilter: a class for light-weight reorder of local rows and columns of an Epetra_RowMatrix.

/*!
Class Ifpack_ReorderFilter enables a light-weight construction of 
reordered matrices. 

This class is used in Ifpack_AdditiveSchwarz to reorder (if required
by the user) the localized matrix. As the localized matrix is defined
on a serial communicator only, all maps are trivial (as all elements
reside on the same process). This class does not attemp to define
properly reordered maps, hence it should not be used for distributed
matrices.

To improve the performances of Ifpack_AdditiveSchwarz, some
operations are not performed in the construction phase (like
for instance the computation of the 1-norm and infinite-norm,
of check whether the reordered matrix is lower/upper triangular or not).

\author Marzio Sala, SNL 9214.

\date Last modified: Oct-04.

*/

class Ifpack_ReorderFilter : public virtual Epetra_RowMatrix {

public:
  // Constructor.
  Ifpack_ReorderFilter(const Teuchos::RefCountPtr<Epetra_RowMatrix>& Matrix,
		       const Teuchos::RefCountPtr<Ifpack_Reordering>& Reordering);

  //! Copy constructor.
  Ifpack_ReorderFilter(const Ifpack_ReorderFilter& RHS);

  //! Destructor.
  virtual ~Ifpack_ReorderFilter() {};

  //! Operator assignment.
  Ifpack_ReorderFilter& operator=(const Ifpack_ReorderFilter& RHS);

  //! Returns the number of local row entries.
  virtual inline int NumMyRowEntries(int MyRow, int & NumEntries) const
  {
    return(Matrix()->NumMyRowEntries(MyRow, NumEntries));
  }

  //! Returns maximum num entries.
  virtual int MaxNumEntries() const
  {
    return(MaxNumEntries_);
  }
  
  // Extracts a copy of the given row for the reordered matrix.
  virtual int ExtractMyRowCopy(int MyRow, int Length, int & NumEntries, double *Values, int * Indices) const;

  //! Extracts a copy of the diagonal of the reordered matrix.
  virtual int ExtractDiagonalCopy(Epetra_Vector & Diagonal) const;

  //! Multiplies multi-vector X with the reordered matrix, returns result in Y.
  virtual int Multiply(bool TransA, const Epetra_MultiVector& X, 
		       Epetra_MultiVector& Y) const;

  //! Solve, not implemented.
  virtual int Solve(bool Upper, bool Trans, bool UnitDiagonal, 
		    const Epetra_MultiVector& X,
		    Epetra_MultiVector& Y) const;

  //! Applies the reordered matrix to multi-vector X, returns the result in Y.
  virtual int Apply(const Epetra_MultiVector& X,
		    Epetra_MultiVector& Y) const;

  //! Applies the inverse of \c this operator (not implemented).
  virtual int ApplyInverse(const Epetra_MultiVector& X,
			   Epetra_MultiVector& Y) const
  {
    return(-1);
  }

  //! Inverse of row sums (not implemented).
  virtual int InvRowSums(Epetra_Vector& x) const
  {
    return(-1);
  }

  //! Left scale of the matrix (not implemented).
  virtual int LeftScale(const Epetra_Vector& x)
  {
    return(-1);
  }

  //! Inverse of column sums (not implemented).
  virtual int InvColSums(Epetra_Vector& x) const
  {
    return(-1);
  }

  //! Right scale of the matrix (not implemented).
  virtual int RightScale(const Epetra_Vector& x) 
  {
    return(-1);
  }

  //! Returns \c true is the matrix called FillComplete().
  virtual bool Filled() const
  {
    return(Matrix()->Filled());
  }

  //! Returns the infinite-norm 
  virtual double NormInf() const
  {
    return(-1.0);
  }

  //! Returns the 1-norm 
  virtual double NormOne() const
  {
    return(-1.0);
  }

  //! Returns the number of global nonzero elements.
  virtual int NumGlobalNonzeros() const
  {
    return(Matrix()->NumGlobalNonzeros());
  }

  //! Returns the number of global rows.
  virtual int NumGlobalRows() const
  {
    return(Matrix()->NumGlobalRows());
  }

  //! Returns the number of global columns.
  virtual int NumGlobalCols() const
  {
    return(Matrix()->NumGlobalCols());
  }

  //! Returns the number of global diagonals.
  virtual int NumGlobalDiagonals() const
  {
    return(Matrix()->NumGlobalDiagonals());
  }

  //! Returns the number of local nonzero elements.
  virtual int NumMyNonzeros() const
  {
    return(Matrix()->NumMyNonzeros());
  }

  //! Returns the number of local rows.
  virtual int NumMyRows() const
  {
    return(Matrix()->NumMyRows());
  }

  //! Returns the number of local columns.
  virtual int NumMyCols() const
  {
    return(Matrix()->NumMyCols());
  }

  //! Returns the number of local diagonals.
  virtual int NumMyDiagonals() const
  {
    return(Matrix()->NumMyDiagonals());
  }

  //! Returns \c true is the reordered matrix is lower triangular 
  virtual bool LowerTriangular() const
  {
    return(false);
  }

  //! Returns \c true is the reordered matrix is upper triangular 
  virtual bool UpperTriangular() const
  {
    return(false);
  }

  //! Returns the row matrix of the non-reordered matrix.
  virtual const Epetra_Map & RowMatrixRowMap() const
  {
    return(Matrix()->RowMatrixRowMap());
  }

  //! Returns the column matrix of the non-reordered matrix.
  virtual const Epetra_Map & RowMatrixColMap() const
  {
    return(Matrix()->RowMatrixColMap());
  }

  //! Returns the importer of the non-reordered matrix.
  virtual const Epetra_Import * RowMatrixImporter() const
  {
    return(Matrix()->RowMatrixImporter());
  }

  //! Sets the use of the transpose.
  int SetUseTranspose(bool UseTranspose)
  {
    return(Matrix()->SetUseTranspose(UseTranspose));
  }

  //! Returns \c true if the transpose of \c this matrix is used.
  bool UseTranspose() const 
  {
    return(Matrix()->UseTranspose());
  }
  
  //! Returns \c true if \c this matrix has the infinite norm.
  bool HasNormInf() const
  {
    return(true);
  }

  //! Returns the communicator.
  const Epetra_Comm & Comm() const
  {
    return(Matrix()->Comm());
  }

  //! Returns the operator domain map of the non-reordered matrix.
  const Epetra_Map & OperatorDomainMap() const 
  {
    return(Matrix()->OperatorDomainMap());
  }

  //! Returns the operator domain range of the non-reordered matrix.
  const Epetra_Map & OperatorRangeMap() const 
  {
    return(Matrix()->OperatorRangeMap());
  }

  //! Returns the map of the non-reordered matrix.
  const Epetra_BlockMap& Map() const 
  {
    return(Matrix()->Map());
  }

  //! Returns the label of \c this object.
  const char* Label() const{
    return(Label_);
  }

  //! Returns a reference-counted pointer to the internally stored pointer to Epetra_RowMatrix.
  inline Teuchos::RefCountPtr<Epetra_RowMatrix> Matrix() const {
    return(A_);
  }

  //! Returns a reference-counted pointer to the internally stored pointer to Ifpack_Reordering..
  inline Teuchos::RefCountPtr<Ifpack_Reordering> Reordering() const {
    return(Reordering_);
  }

private:

  //! Pointer to the matrix to be preconditioned.
  Teuchos::RefCountPtr<Epetra_RowMatrix> A_;
  //! Pointer to the reordering to be used (already constructed).
  Teuchos::RefCountPtr<Ifpack_Reordering> Reordering_;

  //! Number of local rows of A_.
  int NumMyRows_;
  //! Maximum number of entries in A_.
  int MaxNumEntries_;
  //! Label for \c this object.
  char Label_[80];

};


#endif /* IFPACK_DROPFILTER_H */

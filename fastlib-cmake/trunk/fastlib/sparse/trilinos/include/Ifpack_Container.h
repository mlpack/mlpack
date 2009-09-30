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

#ifndef IFPACK_CONTAINER_H
#define IFPACK_CONTAINER_H

class Epetra_RowMatrix;
class Ifpack_Partitioner;
namespace Teuchos {
  class ParameterList;
}

//! Ifpack_Container: a pure virtual class for creating and solving local linear problems.

/*!
Class Ifpack_Container provides the abstract interfaces for 
containers. A "container" is an object that hosts all it is necessary
to create, populate, and solve local linear problems. The local
linear problem matrix, B, is a submatrix of the local components
of a distributed matrix, A. The idea of container is to
specify the rows of A that are contained in B, then extract B from A,
and compute all it is necessary to solve a linear system in B.
Then, set starting solution (if necessary) and right-hand side for B,
and solve the linear system in B.

<P>A container should be used in the following manner:
- Create an container object, specifying the number of rows of B.
- If necessary, set parameters for the solution using
  SetParameters().
- Initialize the container by calling Initialize().
- Specify the ID of the local rows of A that are contained in B,
  using ID().
- Prepare the linear system solver using Compute().
- set LHS and/or RHS elements using LHS() and  RHS().
- Solve the linear system using ApplyInverse().
- Get the componenets of the computed solution using LHS().

The number of vectors can be set using SetNumVectors(), and it
is defaulted to 1.

<P>Containers are currently used by class Ifpack_BlockRelaxation.

<P>Ifpack_Container is a pure virtual class.
Two concrete implementations are provided in classes
Ifpack_SparseContainer (that stores matrices in sparse the format
Epetra_CrsMatrix) and Ifpack_DenseContainer
(for relatively small matrices, as matrices are stored as
Epetra_SerialDenseMatrix's).

\note Still to do:
- Flops count has to be tested.

\author Marzio Sala, SNL 9214.

\date Last update Oct-04.
  
*/

class Ifpack_Container {

public:

  //! Destructor.
  virtual ~Ifpack_Container() {};

  //! Returns the number of rows of the matrix and LHS/RHS.
  virtual int NumRows() const = 0;

  //! Returns the number of vectors in LHS/RHS.
  virtual int NumVectors() const = 0;

  //! Sets the number of vectors for LHS/RHS.
  virtual int SetNumVectors(const int i) = 0;

  //! Returns the i-th component of the vector Vector of LHS.
  virtual double& LHS(const int i, const int Vector = 0) = 0;
  
  //! Returns the i-th component of the vector Vector of RHS.
  virtual double& RHS(const int i, const int Vector = 0) = 0;

  //! Returns the ID associated to local row i. 
  /*!
   * The set of (local) rows assigned to this container is defined
   * by calling ID(i) = j, where i (from 0 to NumRows()) indicates
   * the container-row, and j indicates the local row in the calling
   * process.
   *
   * This is usually used to recorder the local row ID (on calling process)
   * of the i-th row in the container.
   */
  virtual int& ID(const int i) = 0;

  //! Set the matrix element (row,col) to \c value.
  virtual int SetMatrixElement(const int row, const int col,
			       const double value) = 0;

  //! Initializes the container, by performing all operations that only require matrix structure.
  virtual int Initialize() = 0;

  //! Finalizes the linear system matrix and prepares for the application of the inverse.
  virtual int Compute(const Epetra_RowMatrix& A) = 0;

  //! Sets all necessary parameters.
  virtual int SetParameters(Teuchos::ParameterList& List) = 0;

  //! Returns \c true is the container has been successfully initialized.
  virtual bool IsInitialized() const = 0;

  //! Returns \c true is the container has been successfully computed.
  virtual bool IsComputed() const = 0;

  //! Apply the matrix to RHS, results are stored in LHS.
  virtual int Apply() = 0;

  //! Apply the inverse of the matrix to RHS, results are stored in LHS.
  virtual int ApplyInverse() = 0;

  //! Returns the label of \e this container.
  virtual const char* Label() const = 0;

  //! Returns the flops in Initialize().
  virtual double InitializeFlops() const = 0;

  //! Returns the flops in Compute().
  virtual double ComputeFlops() const = 0;

  //! Returns the flops in Apply().
  virtual double ApplyFlops() const = 0;

  //! Returns the flops in ApplyInverse().
  virtual double ApplyInverseFlops() const = 0;

  //! Prints out basic information about the container.
  virtual ostream& Print(std::ostream& os) const = 0;
};

inline ostream& operator<<(ostream& os, const Ifpack_Container& obj)
{
  return(obj.Print(os));
}

#endif // IFPACK_CONTAINER_H

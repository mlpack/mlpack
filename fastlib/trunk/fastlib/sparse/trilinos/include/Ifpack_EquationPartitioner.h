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

#ifndef IFPACK_EQUATIONPARTITIONER_H
#define IFPACK_EQUATIONPARTITIONER_H

#include "Ifpack_ConfigDefs.h"
#include "Ifpack_Partitioner.h"
#include "Ifpack_OverlappingPartitioner.h"
#include "Teuchos_ParameterList.hpp"
class Epetra_Comm;
class Ifpack_Graph;
class Epetra_Map;
class Epetra_BlockMap;
class Epetra_Import;

//! Ifpack_EquationPartitioner: A class to decompose an Ifpack_Graph so that each block will contain all the rows for a different equation.

/*!
Ifpack_EquationPartitioner enables a decomposition into blocks of equations.
Suppose that the input Ifpack_Graph is based on an Epetra_RowMatrix, whose
rows represent (U_i,V_i,P_i) for each grid node i. This partitioner
will decompose the graph into three subgraphs, each of them containing
the rows of U, then V, than P.

The number of equations is set as the number of local partitions.

\note It is required that NumRows % NumLocalParts() = 0.

\date Sep-04.
*/
class Ifpack_EquationPartitioner : public Ifpack_OverlappingPartitioner {

public:

  //! Constructor.
  Ifpack_EquationPartitioner(const Ifpack_Graph* Graph) :
    Ifpack_OverlappingPartitioner(Graph)
  {}

  //! Destructor.
  virtual ~Ifpack_EquationPartitioner() {};

  //! Sets all the parameters for the partitioner.
  int SetPartitionParameters(Teuchos::ParameterList& List)
  {
    return(0);
  }

  //! Computes the partitions. Returns 0 if successful.
  int ComputePartitions();

private:

};

#endif // IFPACK_EQUATIONPARTITIONER_H

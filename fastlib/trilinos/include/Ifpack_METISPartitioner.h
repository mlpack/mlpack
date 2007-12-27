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

#ifndef IFPACK_METISPARTITIONER_H
#define IFPACK_METISPARTITIONER_H

#include "Ifpack_ConfigDefs.h"
#include "Ifpack_Partitioner.h"
#include "Ifpack_OverlappingPartitioner.h"
#include "Teuchos_ParameterList.hpp"
class Ifpack_Graph;
class Epetra_Map;
class Epetra_BlockMap;
class Epetra_Import;

//! Ifpack_METISPartitioner: A class to decompose Ifpack_Graph's using METIS.
/*!
Class Ifpack_METISPartitioner enables the decomposition of the local
Ifpack_Graph's using METIS. In order to work properly, this class requires
IFPACK to be configured with option \c --enable-ifpack-metis.
Otherwise, this class will always create one partition.

\date Last modified: Oct-04.
*/

class Ifpack_METISPartitioner : public Ifpack_OverlappingPartitioner {

public:

  //! Constructor.
  Ifpack_METISPartitioner(const Ifpack_Graph* Graph) :
    Ifpack_OverlappingPartitioner(Graph),
    UseSymmetricGraph_(true)
  {}

  //! Destructor.
  virtual ~Ifpack_METISPartitioner() {};

  //! Sets all the parameters for the partitioner (none at moment).
  int SetPartitionParameters(Teuchos::ParameterList& List)
  {
    // `true' is the safest option, as singletons may still be
    // in the system, even after Ifpack_SingletonFilter (think
    // for example to an upper triangular matrix, which has one
    // singleton, and still has one after the elimination of 
    // the first one...)
    UseSymmetricGraph_ = List.get("partitioner: use symmetric graph", 
				  UseSymmetricGraph_);

    return(0);
  }

  //! Computes the partitions. Returns 0 if successful.
  int ComputePartitions();

private:
  bool UseSymmetricGraph_;

}; // class Ifpack_METISPartitioner

#endif // IFPACK_METISPARTITIONER_H

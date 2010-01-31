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

#ifndef IFPACK_OVERLAPPINGPARTITIONER_H
#define IFPACK_OVERLAPPINGPARTITIONER_H

#include "Ifpack_ConfigDefs.h"
#include "Ifpack_Partitioner.h"
#include "Teuchos_ParameterList.hpp"
class Epetra_Comm;
class Ifpack_Graph;
class Epetra_Map;
class Epetra_BlockMap;
class Epetra_Import;

/* \brief Ifpack_OverlappingPartitioner: A class to create overlapping
    partitions of a local graph.

Class Ifpack_OverlappingPartitioner enables the extension of 
non-overlapping partitions to an arbitrary value of overlap.
Note that overlap refers to the overlap among \e local parts,
and not the overlap among the processes.

Supported parameters are:
- \c "partitioner: local parts": the required number of parts;
- \c "partitioner: overlap": the required amount of overlap is set in 
  parameter. Default = 0 (integer).
- \c "partitioner: verbose": if \c true, information are reported on
  cout. Nothing is reported otherwise.

This class is a semi-virtual class, that contains the basic utilities
for derived classes Ifpack_LinearPartitioner, Ifpack_GreedyPartitioner,
Ifpack_METISPartitioner, and Ifpack_EquationPartitioner. Graphs in
input to one of these classes are supposed to contain no singletons.
Usually, this means that the graph is derived from an Epetra_RowMatrix,
that has been filtered using Ifpack_SingletonFilter.

\author Marzio Sala, SNL 9214.

\date Last update: Oct-04.
*/  
class Ifpack_OverlappingPartitioner : public Ifpack_Partitioner {

public:

  //! Constructor.
  Ifpack_OverlappingPartitioner(const Ifpack_Graph* Graph);

  //! Destructor.
  virtual ~Ifpack_OverlappingPartitioner();

  //! Returns the number of computed local partitions.
  int NumLocalParts() const 
  {
    return(NumLocalParts_);
  }

  //! Returns the overlapping level.
  int OverlappingLevel() const 
  {
    return(OverlappingLevel_);
  }

  //! Returns the local non-overlapping partition ID of the specified row.
  /*! Returns the non-overlapping partition ID of the specified row.
   \param 
   MyRow - (In) local row numbe

   \return
   Local ID of non-overlapping partition for \t MyRow.
   */
  int operator() (int MyRow) const
  {
    if ((MyRow < 0) || (MyRow > NumMyRows()))
      IFPACK_CHK_ERR(-1); // input value not valid

    return(Partition_[MyRow]);
  }

  //! Returns the local overlapping partition ID of the j-th node in partition i.
  int operator() (int i, int j) const
  {
    if ((i < 0) || (i >= NumLocalParts()))
      IFPACK_CHK_ERR(-1);

    if ((j < 0) || (j > (int)Parts_[i].size()))
      IFPACK_CHK_ERR(-2);

    return(Parts_[i][j]);
  }

  //! Returns the number of rows contained in specified partition.
  inline int NumRowsInPart(const int Part) const
  {
    return(Parts_[Part].size());
  }
    
  int RowsInPart(const int Part, int* List) const
  {
    for (int i = 0 ; i < NumRowsInPart(Part) ; ++i)
      List[i] = Parts_[Part][i];

    return(0);
  }
  
  const int* NonOverlappingPartition() const
  {
    return(&Partition_[0]);
  }

  //! Sets all the parameters for the partitioner.
  /*! The supported parameters are:
   * - \c "partitioner: overlap" (int, default = 0).
   * - \c "partitioner: local parts" (int, default = 1).
   * - \c "partitioner: print level" (int, default = 0).
   */
  virtual int SetParameters(Teuchos::ParameterList& List);

  //! Sets all the parameters for the partitioner.
  /*! This function is used by derived classes to set their own
   * parameters. These classes should not derive SetParameters(),
   * so that common parameters can be set just once.
   */
  virtual int SetPartitionParameters(Teuchos::ParameterList& List) = 0;

  //! Computes the partitions. Returns 0 if successful.
  virtual int Compute();

  //! Computes the partitions. Returns 0 if successful.
  virtual int ComputePartitions() = 0;

  //! Computes the partitions. Returns 0 if successful.
  virtual int ComputeOverlappingPartitions();
  
  //! Returns true if partitions have been computed successfully.
  bool IsComputed()
  {
    return(IsComputed_);
  }

  //! Prints basic information on iostream. This function is used by operator<<.
  virtual ostream& Print(std::ostream& os) const;

protected:
   
  //! Returns the number of local rows.
  int NumMyRows() const;
  //! Returns the number of local nonzero elements.
  int NumMyNonzeros() const;
  //! Returns the number of local rows.
  int NumGlobalRows() const;
  //! Returns the max number of local entries in a row.
  int MaxNumEntries() const;
  //! Returns the communicator object of Graph.
  const Epetra_Comm& Comm() const;
  //! Number of local subgraphs
  int NumLocalParts_;
  //! Partition_[i] contains the ID of non-overlapping part it belongs to
  std::vector<int> Partition_; 
  //! Parts_[i][j] is the ID of the j-th row contained in the (overlapping) 
  // partition i
  std::vector<std::vector<int> > Parts_;
  //! Reference to the graph to be partitioned
  const Ifpack_Graph* Graph_;
  //! Overlapping level.
  int OverlappingLevel_;
  //! If \c true,  the graph has been successfully partitioned.
  bool IsComputed_;
  //! If \c true, information are reported on cout.
  bool verbose_;

}; // class Ifpack_Partitioner

#endif // IFPACK_OVERLAPPINGPARTITIONER_H

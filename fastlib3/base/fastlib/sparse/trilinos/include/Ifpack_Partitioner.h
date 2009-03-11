#ifndef IFPACK_PARTITIONER_H
#define IFPACK_PARTITIONER_H

#include "Ifpack_ConfigDefs.h"
#include "Teuchos_ParameterList.hpp"
class Epetra_Comm;
class Ifpack_Graph;
class Epetra_Map;
class Epetra_BlockMap;
class Epetra_Import;

//! Ifpack_Partitioner: A class to decompose local Ifpack_Graph's.

/*!
 
  Class Ifpack_Partitioner enables the decomposition of a local
  Ifpack_Graph's. It is supposed that the graph refers to
  a localized matrix (that is, a matrix that has been filtered
  through Ifpack_LocalFilter).
  
  The overloaded operator (int i) can be used to extract the local partition
  ID of local row i.  
  
  The partitions created by Ifpack_Partitioner derived clased 
  are non-overlapping in graph sense. This means that each row
  (or, more approriately, vertex)
  of \c G is assigned to exactly one partition.

  Partitioner can be extended using the functionalities of class
  Ifpack_OverlappingPartitioner (itself derived from Ifpack_Partitioner.
  This class extends the non-overlapping partitions by the required
  amount of overlap, considering local nodes only (that is, this
  overlap do \e not modify the overlap among the processes).

  Ifpack_Partitioner is a pure virtual class. Concrete implementations
  are:
  - Ifpack_LinearPartitioner, which allows the decomposition of the
    rows of the graph in simple consecutive chunks;
  - Ifpack_METISPartitioner, which calls METIS to decompose the graph
    (this requires the configuration option --enable-ifpack-metis);
  - Ifpack_GreedyPartitioner, a simple greedy algorith;
  - Ifpack_EquationPartitioner, which creates \c NumPDEEqns parts
    (where \c NumPDEEqns is the number of equations in the linear
    system). It is supposed that all the equations referring to the 
    same grid node are ordered consecutively. Besides, the 
    number of equations per node must be constant in the domain.

  Generically, a constructor requires an Ifpack_Graph object. 
  Ifpack_Graph is a pure virtual class. Concrete implentations are:
  - Ifpack_Graph_Epetra_CrsGraph, a light-weight class to wrap 
    Epetra_CrsGraph objects as Ifpack_Graph objects;
  - Ifpack_Graph_Epetra_RowMatrix, a light-weight class to
    wrap Epetra_RowMatrix objects as Ifpack_Graph objects.
  
  <P>An example of use is an Ifpack_Partitioner derived class is as follows:  
  \code
#include "Ifpack_Partitioner.h"
#include "Ifpack_LinearPartitioner.h"
#include "Ifpack_Graph.h"
#include "Ifpack_Graph_Epetra_CrsGraph.h"
...
Epetra_CrsMatrix* A;         // A is filled
// create the wrapper from Epetra_CrsGraph
Ifpack_Graph* Graph = new Ifpack_Graph_Epetra_CrsGraph(A);

// we aim to create non-overlapping partitions only
Ifpack_Partitioner Partitioner(Graph);

Ifpack_Partitioner* Partitioner;
Partitioner = new Ifpack_Graph_Epetra_CrsGraph(&A);

// we want 16 local parts
List.set("partitioner: local parts", 16);
// and an overlap of 0 among the local parts (default option)
List.set("partitioner: overlap", 0);

// decompose the graph
Partitioner.Create(List);

// now Graph can be deleted, as Partitioner contains all the
// necessary information to use the partitions
delete Graph;

// we can get the number of parts actually created...
int NumParts = Partitioner.NumParts();

// ... and the number of rows in each of them
for (int i = 0 ; i < NumParts ; ++i) {
  cout << "rows in " << i << "=" << Partitioner.RowsInPart(i);
}  

// .. and, for non-overlapping partitions only, the partition ID 
// for each local row simply using:
for (int i = 0 ; i < A->NumMyRows() ; ++i)
  cout << "Partition[" << i <<"] = " << Partitioner(i) << endl;

\endcode
  
When overlapping partitiones are created, the user can get the 
row ID contained in each partition as follows:
\code
for (int i = 0 ; i < NumParts ; ++i) {
  for (int j = 0 ; j < Partitioner.RowsInPart(i) ; ++j) {
    cout << "Partition " << i << ", contains local row "
         << Partitioner(i,j) << endl;
  }
}  
\endcode
  
Ifpack_Partitioner is used to create the subblocks in 
Ifpack_BlockJacobi, Ifpack_BlockGaussSeidel, and 
Ifpack_BlockSymGaussSeidel.

\author Marzio Sala, SNL 9214.

\date Last modified on Nov-04.

*/  
class Ifpack_Partitioner {

public:

  //! Destructor.
  virtual ~Ifpack_Partitioner() {};

  //! Returns the number of computed local partitions.
  virtual int NumLocalParts() const = 0;

  //! Returns the overlapping level.
  virtual int OverlappingLevel() const = 0;

  //! Returns the local non-overlapping partition ID of the specified row.
  /*! Returns the non-overlapping partition ID of the specified row.
   \param 
   MyRow - (In) local row numbe

   \return
   Local ID of non-overlapping partition for \t MyRow.
   */
  virtual int operator() (int MyRow) const = 0;

  //! Returns the local overlapping partition ID of the j-th node in partition i.
  virtual int operator() (int i, int j) const = 0;

  //! Returns the number of rows contained in specified partition.
  virtual int NumRowsInPart(const int Part) const = 0;
    
  //! Copies into List the rows in the (overlapping) partition Part.
  virtual int RowsInPart(const int Part, int* List) const = 0;
  
  //! Returns a pointer to the integer vector containing the non-overlapping partition ID of each local row.
  virtual const int* NonOverlappingPartition() const = 0;

  //! Sets all the parameters for the partitioner.
  virtual int SetParameters(Teuchos::ParameterList& List) = 0;

  //! Computes the partitions. Returns 0 if successful.
  virtual int Compute() = 0;

  //! Returns true if partitions have been computed successfully.
  virtual bool IsComputed() = 0;

  //! Prints basic information about the partitioning object.
  virtual ostream& Print(std::ostream& os) const = 0;

}; // class Ifpack_Partitioner

inline ostream& operator<<(ostream& os, const Ifpack_Partitioner& obj)
{
  return(obj.Print(os));
}

#endif // IFPACK_PARTITIONER_H

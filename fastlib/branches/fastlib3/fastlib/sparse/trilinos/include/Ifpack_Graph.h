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

#ifndef IFPACK_GRAPH_H
#define IFPACK_GRAPH_H
class Epetra_Comm;

//! Ifpack_Graph: a pure virtual class that defines graphs for IFPACK.
/*!
Class Ifpack_Graph defines the abstract interface to use graphs in
IFPACK. This class contains all the functions that are required by
IFPACK classes.

\author Marzio Sala, SNL 9214.

\date Last modified on Nov-04.

*/

class Ifpack_Graph {

public:
    
  //! Destructor.
  virtual ~Ifpack_Graph() {};

  //! Returns the number of local rows.
  virtual int NumMyRows() const = 0;

  //! Returns the number of local columns.
  virtual int NumMyCols() const = 0;

  //! Returns the number of global rows.
  virtual int NumGlobalRows() const = 0;

  //! Returns the number of global columns.
  virtual int NumGlobalCols() const = 0;

  //! Returns the maximun number of entries for row.
  virtual int MaxMyNumEntries() const = 0;

  //! Returns the number of local nonzero entries.
  virtual int NumMyNonzeros() const = 0;

  //! Returns \c true is graph is filled.
  virtual bool Filled() const = 0;

  //! Returns the global row ID of input local row.
  virtual int GRID(int) const = 0;

  //! Returns the global column ID of input local column.
  virtual int GCID(int) const = 0;
  
  //! Returns the local row ID of input global row.
  virtual int LRID(int) const = 0;

  //! Returns the local column ID of input global column.
  virtual int LCID(int) const = 0;

  //! Extracts a copy of input local row.
  virtual int ExtractMyRowCopy(int MyRow, int LenOfIndices, 
			       int &NumIndices, int *Indices) const = 0;

  //! Returns the communicator object of the graph.
  virtual const Epetra_Comm& Comm() const = 0;

  //! Prints basic information about the graph object.
  virtual ostream& Print(std::ostream& os) const = 0;

};

inline ostream& operator<<(ostream& os, const Ifpack_Graph& obj)
{
  return(obj.Print(os));
}

#endif // iFPACK_GRAPH_H

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

#ifndef EPETRA_OFFSETINDEX_H
#define EPETRA_OFFSETINDEX_H

#include "Epetra_Object.h"

class Epetra_Import;
class Epetra_Export;
class Epetra_CrsGraph;
class Epetra_Distributor;

//! Epetra_OffsetIndex: This class builds index for efficient mapping of data from one Epetra_CrsGraph based object to another.

/*! Epetra_OffsetIndex generates and index of offsets allowing direct access to data
 for Import/Export operations on Epetra_CrsGraph based objects such as Epetra_CrsMatrix.
*/

class Epetra_OffsetIndex: public Epetra_Object {
    
 public:

  //! Constructs a Epetra_OffsetIndex object from the graphs and an importer.
  Epetra_OffsetIndex( const Epetra_CrsGraph & SourceGraph,
                      const Epetra_CrsGraph & TargetGraph,
                      Epetra_Import & Importer );
  
  //! Constructs a Epetra_OffsetIndex object from the graphs and an exporter.
  Epetra_OffsetIndex( const Epetra_CrsGraph & SourceGraph,
                      const Epetra_CrsGraph & TargetGraph,
                      Epetra_Export & Exporter );

  //! Epetra_OffsetIndex copy constructor. 
  Epetra_OffsetIndex(const Epetra_OffsetIndex & Indexor);
  
  //! Epetra_OffsetIndex destructor.
  virtual ~Epetra_OffsetIndex(void);

  //! @name Print object to an output stream
  //@{ 
  virtual void Print(ostream & os) const;
  //@}

  //! Accessor
  int ** SameOffsets() const { return SameOffsets_; }
 
  //! Accessor
  int ** PermuteOffsets() const { return PermuteOffsets_; }
 
  //! Accessor
  int ** RemoteOffsets() const { return RemoteOffsets_; }
 
 private:

  void GenerateLocalOffsets_( const Epetra_CrsGraph & SourceGraph,
                              const Epetra_CrsGraph & TargetGraph,
                              const int * PermuteLIDs );

  void GenerateRemoteOffsets_( const Epetra_CrsGraph & SourceGraph,
                               const Epetra_CrsGraph & TargetGraph,
                               const int * ExportLIDs,
                               const int * RemoteLIDs,
                               Epetra_Distributor & Distor );

  //! Epetra_OffsetIndex copy constructor. 
  Epetra_OffsetIndex & operator=(const Epetra_OffsetIndex & Indexor);
 public:

  int NumSame_;
  int ** SameOffsets_;
  int NumPermute_;
  int ** PermuteOffsets_;
  int NumExport_;
  int NumRemote_;
  int ** RemoteOffsets_;

  bool DataOwned_;
};

#endif /* EPETRA_OFFSETINDEX_H */

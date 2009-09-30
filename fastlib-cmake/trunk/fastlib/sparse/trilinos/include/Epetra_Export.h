
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

#ifndef EPETRA_EXPORT_H
#define EPETRA_EXPORT_H

#include "Epetra_Object.h"
#include "Epetra_BlockMap.h"

class Epetra_Distributor;

//! Epetra_Export: This class builds an export object for efficient exporting of off-processor elements.

/*! Epetra_Export is used to construct a communication plan that can be called repeatedly by computational
    classes such the Epetra matrix, vector and multivector classes to efficiently send data to a target processor.

    This class currently has one constructor, taking two Epetra_Map or Epetra_BlockMap objects.  The
    first map specifies the global IDs that are owned by the calling processor.  The second map specifies
    the global IDs of elements that we want to export to later.
*/

class Epetra_Export: public Epetra_Object {
    
  public:

  //! Constructs a Epetra_Export object from the source and target maps.
  /*! This constructor builds an Epetra_Export object by comparing the GID lists of the source and
      target maps.
      \param SourceMap (In) Map containing the GIDs from which data should be exported from each processor to
             the target map whenever an export operation is performed using this exporter.
      \param  TargetMap (In) Map containing the GIDs that should be used for exporting data.

      \warning Note that the TargetMap \e must have GIDs uniquely owned, each GID of the target map can occur only once.

Builds an export object that will transfer objects built with SourceMap to objects built with TargetMap.

    A Epetra_Export object categorizes the elements of the target map into three sets as follows:
    <ol>
    <li> All elements in the target map that have the same GID as the corresponding element of the source map, 
         starting with the first 
         element in the target map, going up to the first element that is different from the source map.  The number of
	 these IDs is returned by NumSameIDs().
    <li> All elements that are local to the processor, but are not part of the first set of elements.  These elements
         have GIDs that are owned by the calling processor, but at least the first element of this list is permuted.
	 Even if subsequent elements are not permuted, they are included in this list.  The number of permuted elements
	 is returned by NumPermutedIDs().  The list of elements (local IDs) in the source map that are permuted can be
	 found in the list PermuteFromLIDs().  The list of elements (local IDs) in the target map that are the new locations
	 of the source elements can be found in the list PermuteToLIDs().
    <li> All remaining elements of the target map correspond to global IDs that are owned by remote processors.  The number 
         of these elements is returned by NumRemoteIDs() and the list of these is returned by RemoteLIDs().
    </ol>

Given the above information, the Epetra_Export constructor builds a list of elements that must be communicated to other
processors as a result of export requests.  The number of exported elements (where multiple sends of the same element
to different processors is counted) is returned by NumExportIDs().  The local IDs to be sent are returned by the list 
ExportLIDs().  The processors to which each of the elements will be sent in returned in a list of the same length by 
ExportPIDs().

The total number of elements that will be sent by the calling processor is returned by NumSend().  The total number of
elements that will be received is returned by NumRecv().


The following example illustrates the basic concepts.

Assume we have 3 processors and 9 global elements with each processor owning 3 elements as follows
\verbatim
 PE 0 Elements |  PE 1 Elements  |  PE 2 Elements
    0  1  2          3  4  5           6  7  8
\endverbatim

The above layout essentially defines the target map argument of the export object.

This could correspond to a 9-entry forcing vector with the first three entries on PE 0, and so on.  
Suppose that the entries of this forcing vector are computed by integrating over linear "hat" functions:

\verbatim
^  ^  ^  ^  ^  ^  ^  ^  ^
 \/ \/ \/ \/ \/ \/ \/ \/ 
 /\ /\ /\ /\ /\ /\ /\ /\
+--+--+--+--+--+--+--+--+
0  1  2  3  4  5  6  7  8


\endverbatim

In this case, PE 0 will make contributions to entries 0 through 3, PE 1 will make contributions to entries 2 through
6 and PE 2 will make contributions to entries 5 through 8.  A convenient way to compute these contributions is to create
a forcing vector with replicated entries for the shared contributions.  Specifically the following SourceMap works for
this scenario:

\verbatim

    PE 0 Elements    |  PE 1 Elements    |  PE 2 Elements
     0  1  2  3         2  3  4  5  6        5  6  7  8
\endverbatim

A vector constructed using this SourceMap can be used to collect each processor's contributions to the forcing vector.
Note that the ordering of the elements on each processor is not unique, but has been chosen for illustration.

With these two maps passed into the Epetra_Export constructor, we get the following attribute definitions:

On PE 0:

\verbatim
NumSameIDs      = 3

NumPermuteIDs   = 0
PermuteToLIDs   = 0
PermuteFromLIDs = 0

NumRemoteIDs    = 1
RemoteLIDs      = [2]

NumExportIDs    = 1
ExportLIDs      = [3]
ExportPIDs      = [1]

NumSend         = 1
NumRecv         = 1

\endverbatim

On PE 1:

\verbatim
NumSameIDs      = 0

NumPermuteIDs   = 3
PermuteToLIDs   = [0, 1, 2]
PermuteFromLIDs = [1, 2, 3]

NumRemoteIDs    = 2
RemoteLIDs      = [0, 2]

NumExportIDs    = 2
ExportLIDs      = [0, 4]
ExportPIDs      = [0, 2]

NumSend         = 2
NumRecv         = 2

\endverbatim

On PE 2:

\verbatim
NumSameIDs      = 0

NumPermuteIDs   = 3
PermuteToLIDs   = [0, 1, 2]
PermuteFromLIDs = [1, 2, 3]

NumRemoteIDs    = 1
RemoteLIDs      = [0]

NumExportIDs    = 1
ExportLIDs      = [0]
ExportPIDs      = [1]

NumSend         = 1
NumRecv         = 1

\endverbatim


<b> Using Epetra_Export Objects </b>

Once a Epetra_Export object has been constructed, it can be used by any of the Epetra classes that support distributed global
objects, namely Epetra_Vector, Epetra_MultiVector, Epetra_CrsGraph, Epetra_CrsMatrix and Epetra_VbrMatrix.  
All of these classes have Export and Export methods that will fill new objects whose distribution is described by
the target map, taking elements from the source object whose distribution is described by the source map.  Details of usage
for each class is given in the appropriate class documentation.

In the above example, if x_integrate is constructed using the SourceMap and then filled with local contributions, and x_force
is constructed using the target map, the following operation will fill x_force with the combined results of x_integrate:
\verbatim
x_force.Export(x_integrate, exporter, Add);
\endverbatim
The third argument above tells the export operation to add results that come from multiple processors for the same GID.

Epetra_Export objects can also be used by Import operations to perform the reverse operation.  For example, if x_force in the
above example had boundary conditions that should be sent to processors that share a boundary element, the following operation
would send replicated values to x_integrate:
\verbatim
x_integrate.Import(x_force, exporter, Insert);
\endverbatim
At the end of this operation, x_integrate would have replicated values from x_force of entries 2 and 3 on PEs 0 and 1, 
and entries 5 and 6 on PEs 1 and 2.

  */ 
  Epetra_Export( const Epetra_BlockMap & SourceMap, const Epetra_BlockMap & TargetMap );
  
  //! Epetra_Export copy constructor. 
  Epetra_Export(const Epetra_Export& Exporter);
  
  //! Epetra_Export destructor.
  
  virtual ~Epetra_Export(void);
  //! Returns the number of elements that are identical between the source and target maps, up to the first different ID
  int NumSameIDs() const {return(NumSameIDs_);};

  //! Returns the number of elements that are local to the calling processor, but not part of the first NumSameIDs() elements.
  int NumPermuteIDs() const {return(NumPermuteIDs_);};

  //! List of elements in the source map that are permuted.
  int * PermuteFromLIDs () const {return(PermuteFromLIDs_);};
  //! List of elements in the target map that are permuted.
  int * PermuteToLIDs () const {return(PermuteToLIDs_);};

  //! Returns the number of elements that are not on the calling processor.
  int NumRemoteIDs() const {return(NumRemoteIDs_);};
  
  //! List of elements in the target map that are coming from other processors.
  int * RemoteLIDs() const {return(RemoteLIDs_);};

  //! Returns the number of elements that must be sent by the calling processor to other processors.
  int  NumExportIDs () const {return(NumExportIDs_);};

  //! List of elements that will be sent to other processors.
  int * ExportLIDs () const {return(ExportLIDs_);};

  //! List of processors to which elements will be sent, ExportLIDs() [i] will be sent to processor ExportPIDs() [i].
  int * ExportPIDs () const {return(ExportPIDs_);};

  //! Total number of elements to be sent.
  int NumSend() const {return(NumSend_);};

  //! Total number of elements to be received.
  int NumRecv() const {return(NumRecv_);};

  //! Returns the SourceMap used to construct this exporter
  const Epetra_BlockMap & SourceMap() const {return(SourceMap_);};

  //! Returns the TargetMap used to construct this exporter
  const Epetra_BlockMap & TargetMap() const {return(TargetMap_);};

  Epetra_Distributor & Distributor() const {return(*Distor_);};

  //! @name Print object to an output stream
  //@{ 
  virtual void Print(ostream & os) const;
  //@}
 protected:

 friend class Epetra_BlockMap;

 private:
 Epetra_Export& operator=(const Epetra_Export& src);

  Epetra_BlockMap TargetMap_;
  Epetra_BlockMap SourceMap_;

  int  NumSameIDs_;
  int  NumPermuteIDs_;
  int * PermuteToLIDs_;
  int * PermuteFromLIDs_;
  int  NumRemoteIDs_;
  int * RemoteLIDs_;

  int  NumExportIDs_;
  int * ExportLIDs_;
  int * ExportPIDs_;

  int NumSend_;
  int NumRecv_;

  Epetra_Distributor * Distor_;
  

};

#endif /* EPETRA_EXPORT_H */

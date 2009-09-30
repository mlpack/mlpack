
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

#ifndef EPETRA_SRCDISTOBJECT_H
#define EPETRA_SRCDISTOBJECT_H
class Epetra_BlockMap;


//! Epetra_SrcDistObject: A class for supporting flexible source distributed objects for import/export operations.

/*! The Epetra_SrcDistObject is a base class for all Epetra distributed global objects that are potential 
    source objects for the general Epetra_DistObject class.  It provides a way to send a very general distributed
    object as the potential source object for an import or export object.  For example, it is possible to pass
    an Epetra_RowMatrix object as the source object for an import/export where the target is an Epetra_CrsMatrix, or
    an Epetra_CrsGraph (where the RowMatrix values will be ignored).

*/

//==========================================================================
class Epetra_SrcDistObject {

  public:
    //! @name Destructor
  //@{ 
  //! Epetra_SrcDistObject destructor.  
  virtual ~Epetra_SrcDistObject() {};
  //@}

  
  //! @name Attribute accessor methods
  //@{ 
  //! Returns a reference to the Epetra_BlockMap for this object.
  virtual const Epetra_BlockMap & Map() const = 0;
};

#endif /* EPETRA_SRCDISTOBJECT_H */

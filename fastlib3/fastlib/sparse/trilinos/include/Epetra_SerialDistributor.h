
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

#ifndef EPETRA_SERIALDISTRIBUTOR_H
#define EPETRA_SERIALDISTRIBUTOR_H

#include "Epetra_Object.h"
#include "Epetra_Distributor.h"
class Epetra_SerialComm;

//! Epetra_SerialDistributor:  The Epetra Serial implementation of the Epetra_Distributor Gather/Scatter Setup Class.
/*! The Epetra_SerialDistributor class is an Serial implement of Epetra_Distributor that is essentially a trivial class
    since a serial machine is a trivial parallel machine.
  An Epetra_SerialDistributor object is actually produced by calling a method in the Epetra_SerialComm class.
  
*/

class Epetra_SerialDistributor: public Epetra_Object, public virtual Epetra_Distributor {
    
  public:

    //! @name Constructor/Destructor
  //@{ 

  //! Constructor.
  Epetra_SerialDistributor(const Epetra_SerialComm & Comm);

  //! Epetra_SerialDistributor Copy Constructor.
  Epetra_SerialDistributor(const Epetra_SerialDistributor & Plan);

  //! Clone method
  Epetra_Distributor * Clone(){return(dynamic_cast<Epetra_Distributor *>(new Epetra_SerialDistributor(*this)));};

  //! Epetra_Comm Destructor.
  virtual ~Epetra_SerialDistributor();
  //@}

  
  int CreateFromSends( const int & NumExportIDs,
                       const int * ExportPIDs,
		       bool Deterministic,
                       int & NumRemoteIDs );

  int CreateFromRecvs( const int & NumRemoteIDs,
                       const int * RemoteGIDs,
                       const int * RemotePIDs,
		       bool Deterministic,
                       int & NumExportIDs,
                       int *& ExportGIDs,
                       int *& ExportPIDs);


  int Do       (char * export_objs,
                int obj_size,
                int & len_import_objs,
                char *& import_objs);
  int DoReverse(char * export_objs,
                int obj_size,
                int & len_import_objs,
                char *& import_objs);

  int DoPosts(char * export_objs,
              int obj_size,
              int & len_import_objs,
              char *& import_objs);
  int DoWaits();

  int DoReversePosts(char * export_objs,
                     int obj_size,
                     int & len_import_objs,
                     char *& import_objs);
  int DoReverseWaits();


  int Do       (char * export_objs,
                int obj_size,
                int *& sizes,
                int & len_import_objs,
                char *& import_objs);
  int DoReverse(char * export_objs,
                int obj_size,
                int *& sizes,
                int & len_import_objs,
                char *& import_objs);

  int DoPosts(char * export_objs,
              int obj_size,
              int *& sizes,
              int & len_import_objs,
              char *& import_objs);

  int DoReversePosts(char * export_objs,
                     int obj_size,
                     int *& sizes,
                     int & len_import_objs,
                     char *& import_objs);


  virtual void Print(ostream & os) const;

 private:
  int nrecvs_;
  int nsends_;
};
#endif /* EPETRA_SERIALDISTRIBUTOR_H */

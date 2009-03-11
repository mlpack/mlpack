
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

#ifndef EPETRA_DISTRIBUTOR_H
#define EPETRA_DISTRIBUTOR_H

//! Epetra_Distributor:  The Epetra Gather/Scatter Setup Base Class.
/*! The Epetra_Distributor class is an interface that encapsulates the general
  information and services needed for other Epetra classes to perform gather/scatter
  operations on a parallel computer.
  An Epetra_Distributor object is actually produced by calling a method in the Epetra_Comm class.
  
  Epetra_Distributor has default implementations, via Epetra_SerialDistributor and 
  Epetra_MpiDistributor, for both serial execution and MPI
  distributed memory execution.  It is meant to insulate the user from
  the specifics of communication that are not required for normal
  manipulation of linear algebra objects..
*/

#include "Epetra_Object.h"
class Epetra_Distributor {
    
  public:
    //! @name Constructor and Destructor
  //@{ 
  //! Epetra_Distributor clone constructor.
  virtual Epetra_Distributor * Clone() = 0;
  //! Epetra_Distributor Destructor.
  virtual ~Epetra_Distributor(){};
  //@}

  
  //! @name Gather/Scatter Constructors
  //@{ 
  //! Create Distributor object using list of process IDs to which we export
  /*! Take a list of Process IDs and construct a plan for efficiently scattering to these processes.
      Return the number of IDs being sent to me.
    \param NumExportIDs In
           Number of IDs that need to be sent from this processor.
    \param ExportPIDs In
           List of processors that will get the exported IDs.
    \param Deterministic In
           No op.
    \param NumRemoteIDs Out
           Number of IDs this processor will be receiving.
  */
  virtual int CreateFromSends( const int & NumExportIDs,
                               const int * ExportPIDs,
			       bool Deterministic,
                               int & NumRemoteIDs ) = 0;

  //! Create Distributor object using list of Remote global IDs and corresponding PIDs
  /*! Take a list of global IDs and construct a plan for efficiently scattering to these processes.
      Return the number and list of IDs being sent by me.
    \param NumRemoteIDs In
           Number of IDs this processor will be receiving.
    \param RemoteGIDs In
           List of IDs that this processor wants.
    \param RemotePIDs In
           List of processors that will send the remote IDs.
    \param Deterministic In
           No op.
    \param NumExportIDs Out
           Number of IDs that need to be sent from this processor.
    \param ExportPIDs Out
           List of processors that will get the exported IDs.
  */
  virtual int CreateFromRecvs( const int & NumRemoteIDs,
                               const int * RemoteGIDs,
			       const int * RemotePIDs,
			       bool Deterministic,
			       int & NumExportIDs,
			       int *& ExportGIDs,
			       int *& ExportPIDs) = 0;
  //@}

  //! @name Execute Gather/Scatter Operations (Constant size objects)
  //@{ 

  //! Execute plan on buffer of export objects in a single step
  virtual int Do( char * export_objs,
                  int obj_size,
                  int & len_import_objs,
                  char *& import_objs) = 0;

  //! Execute reverse of plan on buffer of export objects in a single step
  virtual int DoReverse( char * export_objs,
                         int obj_size,
                         int & len_import_objs,
                         char *& import_objs ) = 0;

  //! Post buffer of export objects (can do other local work before executing Waits)
  virtual int DoPosts( char * export_objs, 
                       int obj_size,
                       int & len_import_objs,
                       char *& import_objs ) = 0;

  //! Wait on a set of posts
  virtual int DoWaits() = 0;

  //! Do reverse post of buffer of export objects (can do other local work before executing Waits)
  virtual int DoReversePosts( char * export_objs,
                              int obj_size,
                              int & len_import_objs,
                              char *& import_objs) = 0;

  //! Wait on a reverse set of posts
  virtual int DoReverseWaits() = 0;
  //@}

  //! @name Execute Gather/Scatter Operations (Non-constant size objects)
  //@{ 

  //! Execute plan on buffer of export objects in a single step (object size may vary)
  virtual int Do( char * export_objs,
                  int obj_size,
                  int *& sizes,
                  int & len_import_objs,
                  char *& import_objs) = 0;

  //! Execute reverse of plan on buffer of export objects in a single step (object size may vary)
  virtual int DoReverse( char * export_objs,
                         int obj_size,
                         int *& sizes,
                         int & len_import_objs,
                         char *& import_objs) = 0;

  //! Post buffer of export objects (can do other local work before executing Waits)
  virtual int DoPosts( char * export_objs,
                       int obj_size,
                       int *& sizes,
                       int & len_import_objs,
                       char *& import_objs) = 0;

  //! Do reverse post of buffer of export objects (can do other local work before executing Waits)
  virtual int DoReversePosts( char * export_objs,
                              int obj_size,
                              int *& sizes,
                              int & len_import_objs,
                              char *& import_objs) = 0;

  //@}

  //! @name Print object to an output stream
  //@{ 
  virtual void Print(ostream & os) const = 0;
  //@}
};
#endif /* EPETRA_DISTRIBUTOR_H */

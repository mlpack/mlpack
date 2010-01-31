
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

#ifndef EPETRA_SERIALCOMMDATA_H
#define EPETRA_SERIALCOMMDATA_H

#include "Epetra_Data.h"

//! Epetra_SerialCommData:  The Epetra Serial Communication Data Class.
/*! The Epetra_SerialCommData class is an implementation detail of Epetra_SerialComm.
    It is reference-counted, and can be shared by multiple Epetra_SerialComm instances. 
		It derives from Epetra_Data, and inherits reference-counting from it.
*/

class Epetra_SerialCommData : public Epetra_Data {
	friend class Epetra_SerialComm;
 private:
  //! @name Constructor/Destructor Methods
  //@{ 

  //! Epetra_SerialCommData Default Constructor.
  Epetra_SerialCommData();

  //! Epetra_SerialCommData Destructor.
  ~Epetra_SerialCommData();

  //@}

	int MyPID_;
	int NumProc_;

	// these are intentionally declared but not defined. See Epetra Developer's Guide for details.
  Epetra_SerialCommData(const Epetra_SerialCommData & CommData);
	Epetra_SerialCommData& operator=(const Epetra_SerialCommData & CommData);
  
};
#endif /* EPETRA_SERIALCOMMDATA_H */

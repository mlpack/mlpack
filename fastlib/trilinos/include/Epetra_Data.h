
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

#ifndef EPETRA_DATA_H
#define EPETRA_DATA_H

//! Epetra_Data:  The Epetra Base Data Class.
/*! The Epetra_Data class is a base class for all Epetra Data Classes.
	  It provides a mechanism so that one data object can be shared by multiple
		class instances. However, it is meant only to be used internally by 
		another Epetra class. It does not provide smart pointer like capabilities.
		Incrementing and decrementing the reference count, and deleting the 
		data class instance (if necessary), are duties of the Epetra class 
		utilizing Epetra_Data.

		All of Epetra_Data's methods are protected. This is because Epetra_Data 
		should never be used directly. Rather, a class that derives from
		Epetra_Data should be used instead. For example, Epetra_MpiCommData or 
		Epetra_BlockMapData.

		DEVELOPER NOTES: 
		(1) Any class that inherits from Epetra_Data may need to define an 
		assignment operator, if it adds pointers. Epetra_Data doesn't have any, 
		and so the default (compiler-generated) assignment operator is good enough. 
		(2) The behavior of a derived class is left up to the 
		implementer(s) of that class. As such, it cannot be assumed that 
		just because a class inherits from Epetra_Data, that it supports copy 
		construction or assignment, or that it will perform as expected. 
*/

class Epetra_Data {
 protected:
   //! @name Constructor/Destructor Methods
  //@{ 

  //! Epetra_Data Serial Constructor.
  Epetra_Data();

  //! Epetra_Data Copy Constructor.
  /*! Reference count will be set to 1 on new instance.*/
  Epetra_Data(const Epetra_Data & Data);

  //! Epetra_Data Destructor.
  virtual ~Epetra_Data();

  //@}

  //! @name Reference-Counting Methods
	//@{ 

	//! Increment reference count
	void IncrementReferenceCount();

	//! Decrement reference count
	void DecrementReferenceCount();

	//! Get reference count
	int ReferenceCount() const;

	//@}

	int ReferenceCount_;
  
};

#endif /* EPETRA_DATA_H */

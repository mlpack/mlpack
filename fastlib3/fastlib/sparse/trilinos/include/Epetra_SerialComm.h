
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

#ifndef EPETRA_SERIALCOMM_H
#define EPETRA_SERIALCOMM_H

#include "Epetra_Object.h"
#include "Epetra_Comm.h"
#include "Epetra_SerialCommData.h"
class Epetra_Distributor;

//! Epetra_SerialComm:  The Epetra Serial Communication Class.
/*! The Epetra_SerialComm class is an implementation of Epetra_Comm, providing the general
  information and services needed for other Epetra classes to run on a serial computer.
*/

class Epetra_SerialComm: public Epetra_Object, public virtual Epetra_Comm {
    
  public:
    //! @name Constructor/Destructor Methods
  //@{ 

  //! Epetra_SerialComm Serial Constructor.
  /*! Builds an instance of a serial communicator.  Even
    if the application is running in parallel via MPI, this communicator
    will execute in serial.  The access functions return the number of
    processors to be 1 and the processor ID to be 0.
  */
  Epetra_SerialComm();


  //! Epetra_SerialComm Copy Constructor.
  /*! Makes an exact copy of an existing Epetra_SerialComm instance.
  */
  Epetra_SerialComm(const Epetra_SerialComm& Comm);

	//! Clone method.
	Epetra_Comm * Clone() const {
		return(dynamic_cast<Epetra_Comm *>(new Epetra_SerialComm(*this)));
	};

  //! Epetra_SerialComm Destructor.
  /*! Completely deletes a Epetra_SerialComm object.  
    \warning Note:  All objects that depend
    on a Epetra_SerialComm instance should be destroyed prior to calling this
    function.
  */
  virtual ~Epetra_SerialComm();
  //@}

  //! @name Barrier Methods
  //@{ 
  //! Epetra_SerialComm Barrier function.
  /*! A no-op for a serial communicator.
  */
  void Barrier() const;
  //@}

  //! @name Broadcast Methods
  //@{ 
  //! Epetra_SerialComm Broadcast function.
  /*! A no-op for a serial communicator.
    \param MyVals InOut
           On entry, the root processor contains the list of values.  On exit,
	   all processors will have the same list of values.  Note that values must be
	   allocated on all processor before the broadcast.
    \param Count In
           On entry, contains the length of the list of MyVals.
    \param Root In
           On entry, contains the processor from which all processors will receive a copy of MyVals.
  */

  int Broadcast(double * MyVals, int Count, int Root) const;

  //! Epetra_SerialComm Broadcast function.
  /*! A no-op for a serial communicator.
    \param MyVals InOut
           On entry, the root processor contains the list of values.  On exit,
	   all processors will have the same list of values.  Note that values must be
	   allocated on all processor before the broadcast.
    \param Count In
           On entry, contains the length of the list of MyVals.
    \param Root In
           On entry, contains the processor from which all processors will receive a copy of MyVals.
  */

  int Broadcast(int * MyVals, int Count, int Root) const;

  //! Epetra_SerialComm Broadcast function.
  /*! A no-op for a serial communicator.
    \param MyVals InOut
           On entry, the root processor contains the list of values.  On exit,
	   all processors will have the same list of values.  Note that values must be
	   allocated on all processor before the broadcast.
    \param Count In
           On entry, contains the length of the list of MyVals.
    \param Root In
           On entry, contains the processor from which all processors will receive a copy of MyVals.
  */

  int Broadcast(long * MyVals, int Count, int Root) const;
  //@}

  //! @name Gather Methods
  //@{ 
  //! Epetra_SerialComm All Gather function.
  /*! A copy for a serial communicator.
    \param MyVals In
           On entry, contains the list of values, to be sent to all processors.
    \param AllVals Out
           On exit, contains the list of values from all processors. Must by of size NumProc*Count.
    \param Count In
           On entry, contains the length of the list of MyVals.
  */

  int GatherAll(double * MyVals, double * AllVals, int Count) const;

  //! Epetra_SerialComm All Gather function.
  /*! A copy for a serial communicator.
    \param MyVals In
           On entry, contains the list of values, to be sent to all processors.
    \param AllVals Out
           On exit, contains the list of values from all processors. Must by of size NumProc*Count.
    \param Count In
           On entry, contains the length of the list of MyVals.
  */

  int GatherAll(int * MyVals, int * AllVals, int Count) const;

  //! Epetra_SerialComm All Gather function.
  /*! A copy for a serial communicator.
    \param MyVals In
           On entry, contains the list of values, to be sent to all processors.
    \param AllVals Out
           On exit, contains the list of values from all processors. Must by of size NumProc*Count.
    \param Count In
           On entry, contains the length of the list of MyVals.
  */

  int GatherAll(long * MyVals, long * AllVals, int Count) const;
  //@}

  //! @name Sum Methods
  //@{ 
  //! Epetra_SerialComm Global Sum function.
  /*! A copy for a serial communicator.
    \param PartialSums In
           On entry, contains the list of values, usually partial sums computed locally,
	   to be summed across all processors.
    \param GlobalSums Out
           On exit, contains the list of values summed across all processors.
    \param Count In
           On entry, contains the length of the list of values.
  */

  int SumAll(double * PartialSums, double * GlobalSums, int Count) const;

  //! Epetra_SerialComm Global Sum function.
  /*! A copy for a serial communicator.
    \param PartialSums In
           On entry, contains the list of values, usually partial sums computed locally,
	   to be summed across all processors.
    \param GlobalSums Out
           On exit, contains the list of values summed across all processors.
    \param Count In
           On entry, contains the length of the list of values.
  */
  int SumAll(int * PartialSums, int * GlobalSums, int Count) const;

  //! Epetra_SerialComm Global Sum function.
  /*! A copy for a serial communicator.
    \param PartialSums In
           On entry, contains the list of values, usually partial sums computed locally,
	   to be summed across all processors.
    \param GlobalSums Out
           On exit, contains the list of values summed across all processors.
    \param Count In
           On entry, contains the length of the list of values.
  */
  int SumAll(long * PartialSums, long * GlobalSums, int Count) const;
  //@}

  //! @name Max/Min Methods
  //@{ 
  //! Epetra_SerialComm Global Max function.
  /*! A copy for a serial communicator.
    \param PartialMaxs In
           On entry, contains the list of values, usually partial maxs computed locally,
					 using these Partial Maxs, the max across all processors will be computed.
    \param GlobalMaxs Out
           On exit, contains the list of maxs computed across all processors.
    \param Count In
           On entry, contains the length of the list of values.
  */
  int MaxAll(double * PartialMaxs, double * GlobalMaxs, int Count) const;

  //! Epetra_SerialComm Global Max function.
  /*! A copy for a serial communicator.
    \param PartialMaxs In
           On entry, contains the list of values, usually partial maxs computed locally;
					 using these Partial Maxs, the max across all processors will be computed.
    \param GlobalMaxs Out
           On exit, contains the list of maxs computed across all processors.
    \param Count In
           On entry, contains the length of the list of values.
  */
  int MaxAll(int * PartialMaxs, int * GlobalMaxs, int Count) const;

  //! Epetra_SerialComm Global Max function.
  /*! A copy for a serial communicator.
    \param PartialMaxs In
           On entry, contains the list of values, usually partial maxs computed locally;
					 using these Partial Maxs, the max across all processors will be computed.
    \param GlobalMaxs Out
           On exit, contains the list of maxs computed across all processors.
    \param Count In
           On entry, contains the length of the list of values.
  */
  int MaxAll(long * PartialMaxs, long * GlobalMaxs, int Count) const;

  //! Epetra_SerialComm Global Min function.
  /*! A copy for a serial communicator.
    \param PartialMins In
           On entry, contains the list of values, usually partial mins computed locally;
					 using these Partial Mins, the min across all processors will be computed.
    \param GlobalMins Out
           On exit, contains the list of mins computed across all processors.
    \param Count In
           On entry, contains the length of the list of values.
  */
  int MinAll(double * PartialMins, double * GlobalMins, int Count) const;

  //! Epetra_SerialComm Global Min function.
  /*! A copy for a serial communicator.
    \param PartialMins In
           On entry, contains the list of values, usually partial mins computed locally;
					 using these Partial Mins, the min across all processors will be computed.
    \param GlobalMins Out
           On exit, contains the list of mins computed across all processors.
    \param Count In
           On entry, contains the length of the list of values.
  */
  int MinAll(int * PartialMins, int * GlobalMins, int Count) const;

  //! Epetra_SerialComm Global Min function.
  /*! A copy for a serial communicator.
    \param PartialMins In
           On entry, contains the list of values, usually partial mins computed locally;
					 using these Partial Mins, the min across all processors will be computed.
    \param GlobalMins Out
           On exit, contains the list of mins computed across all processors.
    \param Count In
           On entry, contains the length of the list of values.
  */
  int MinAll(long * PartialMins, long * GlobalMins, int Count) const;
  //@}

  //! @name Parallel Prefix Methods
  //@{ 
  //! Epetra_SerialComm Scan Sum function.
  /*! A copy for a serial communicator.
    \param MyVals In
           On entry, contains the list of values to be summed across all processors.
    \param ScanSums Out
           On exit, contains the list of values summed across processors 0 through i.
    \param Count In
           On entry, contains the length of the list of values.
  */
  int ScanSum(double * MyVals, double * ScanSums, int Count) const;

  //! Epetra_SerialComm Scan Sum function.
  /*! A copy for a serial communicator.
    \param MyVals In
           On entry, contains the list of values to be summed across all processors.
    \param ScanSums Out
           On exit, contains the list of values summed across processors 0 through i.
    \param Count In
           On entry, contains the length of the list of values.
  */
  int ScanSum(int * MyVals, int * ScanSums, int Count) const;

  //! Epetra_SerialComm Scan Sum function.
  /*! A copy for a serial communicator.
    \param MyVals In
           On entry, contains the list of values to be summed across all processors.
    \param ScanSums Out
           On exit, contains the list of values summed across processors 0 through i.
    \param Count In
           On entry, contains the length of the list of values.
  */
  int ScanSum(long * MyVals, long * ScanSums, int Count) const;
  //@}

  //! @name Attribute Accessor Methods
  //@{ 
  
  //! Return my process ID. 
  /*! In MPI mode returns the rank of the calling process.  In serial mode
    returns 0.
  */
  int MyPID() const {return(SerialCommData_->MyPID_);};
  
  //! Returns total number of processes (always returns 1 for SerialComm). 
  int NumProc() const {return(SerialCommData_->NumProc_);};

  //@}

  //! @name Gather/Scatter and Directory Constructors
  //@{ 
  //! Create a distributor object.
  Epetra_Distributor * CreateDistributor() const;
  //! Create a directory object for the given Epetra_BlockMap.
  Epetra_Directory * CreateDirectory(const Epetra_BlockMap & Map) const;
  //@}

  //! @name Print object to an output stream
  //@{ 
  //! Print method that implements Epetra_Object virtual Print method
  inline void Print(ostream & os) const {
		os << "::Processor "<< MyPID()<<" of " << NumProc() << " total processors."; 
		return; 
	}
  //! Print method that implements Epetra_Comm virtual PrintInfo method
  void PrintInfo(ostream & os) const {
		Epetra_SerialComm::Print(os);
		return;
	};
  //@}

  //! @name Expert Users and Developers Only
  //@{ 

	//! Returns the reference count of SerialCommData.
	/*! (Intended for testing purposes.) */
	int ReferenceCount() const;

	//! Returns a pointer to the SerialCommData instance this SerialComm uses. 
	/*! (Intended for developer use only for testing purposes.) */
	const Epetra_SerialCommData * DataPtr() const {return(SerialCommData_);};

  //@}

	//! Assignment Operator
	Epetra_SerialComm & operator=(const Epetra_SerialComm & Comm);

 private:

	void CleanupData();
	Epetra_SerialCommData * SerialCommData_;
  
};
#endif /* EPETRA_SERIALCOMM_H */

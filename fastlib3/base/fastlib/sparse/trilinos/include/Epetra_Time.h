
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

#ifndef EPETRA_TIME_H
#define EPETRA_TIME_H

//! Epetra_Time:  The Epetra Timing Class.
/*! The Epetra_Time class is a wrapper that encapsulates the general
  information needed getting timing information.  Currently it return
  the elapsed time for each calling processor..
  A Epetra_Comm object is required for building all Epetra_Time objects.
  
  Epetra_Time support both serial execution and (via MPI) parallel 
  distributed memory execution.  It is meant to insulate the user from
  the specifics of timing across a variety of platforms.
*/

#include "Epetra_Object.h"
#include "Epetra_Comm.h"

#ifdef EPETRA_MPI
#include "mpi.h"
#elif ICL
#include <time.h>
#else
#include <sys/time.h>
#ifndef MINGW
#include <sys/resource.h>
#endif
#endif

class Epetra_Time: public Epetra_Object {
    
  public:
  //! Epetra_Time Constructor.
  /*! Creates a Epetra_Time instance. This instance can be queried for
      elapsed time on the calling processor.  StartTime is also set
      for use with the ElapsedTime function.
  */
  Epetra_Time(const Epetra_Comm & Comm);

  //! Epetra_Time Copy Constructor.
  /*! Makes an exact copy of an existing Epetra_Time instance.
  */
  Epetra_Time(const Epetra_Time& Time);

  //! Epetra_Time wall-clock time function.
  /*! Returns the wall-clock time in seconds.  A code section can be 
      timed by putting it between two calls to WallTime and taking the
      difference of the times.
  */
  double WallTime(void) const;

  //! Epetra_Time function to reset the start time for a timer object.
  /*! Resets the start time for the timer object to the current time
      A code section can be 
      timed by putting it between a call to ResetStartTime and ElapsedTime.
  */
  void ResetStartTime(void);

  //! Epetra_Time elapsed time function.
  /*! Returns the elapsed time in seconds since the timer object was
      constructed, or since the ResetStartTime function was called. 
      A code section can be 
      timed by putting it between the Epetra_Time constructor and a call to 
      ElapsedTime, or between a call to ResetStartTime and ElapsedTime.
  */
  double ElapsedTime(void) const;

  //! Epetra_Time Destructor.
  /*! Completely deletes a Epetra_Time object.  
  */
  virtual ~Epetra_Time(void);

  Epetra_Time& operator=(const Epetra_Time& src)
    {
      StartTime_ = src.StartTime_;
      Comm_ = src.Comm_;
      return( *this );
    }

 private:

  double StartTime_;
  const Epetra_Comm * Comm_;
  
};

#endif /* EPETRA_TIME_H */

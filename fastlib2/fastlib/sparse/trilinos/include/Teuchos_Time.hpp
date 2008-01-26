// @HEADER
// ***********************************************************************
// 
//                    Teuchos: Common Tools Package
//                 Copyright (2004) Sandia Corporation
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
// @HEADER

// Kris
// 07.08.03 -- Move into Teuchos package/namespace

#ifndef _TEUCHOS_TIME_HPP_
#define _TEUCHOS_TIME_HPP_

/*! \file Teuchos_Time.hpp
    \brief Basic wall-clock timer class
*/

#include "Teuchos_ConfigDefs.hpp"

#include <ctime>
#ifdef HAVE_MPI
#include "mpi.h"
#else
#ifndef ICL
#include <sys/time.h>
#ifndef MINGW
#include <sys/resource.h>
#endif
#endif
#endif

namespace Teuchos 
{

  /** \brief Basic wall-clock timer class. 
   *
   *  To time a section of code, place it in between calls to start() and stop(). 
   *
   *  \note For std::exception safety and correct behavior in reentrant code, this class should
   * generally be used only through the Teuchos::TimeMonitor mechanism. 
   *
   */

  /* ADDITIONAL COMMENTS:
   * Initial version by Mike Heroux and Kris Kampshoff. 
   * Modified as follows by Kevin Long, 9/29/03:
   * <ul>
   * <li> There is no need to define explicit copy ctor and dtor for this class.
   * <li> The wallTime() method returns the same value for every instance of this class, so
   * it would be best to make it a static method.
   * <li> Removed the communicator data member. Cross-processor comparisons of timings
   * can be done by the TimeMonitor.
   * </ul>
   */ 

  class Time
  {

  public:
    /** \brief Construct with a descriptive name */
    Time(const std::string& name, bool start = false);
  
    /** \brief Returns current wall-clock time in seconds.*/
    static double wallTime();
  
    /** \brief Starts the timer */
    void start(bool reset = false);

    /** \brief Stops the timer */
    double stop();

    /** \brief Returns the total time accumulated by this timer. <b>This should be called
     * only when the clock is stopped.</b> */
    double totalElapsedTime(bool readCurrentTime = false) const;

    /** \brief Resets the cummulative time and number of times this timer
     * has been called.  Does not affect any other state. */
    void reset() {totalTime_ = 0; numCalls_ = 0;}

    /** \brief Indicates if this timer is currently running, i.e., if it has been started but
     * not yet stopped. 
     *
     *	It is necessary to know if a timer is running to avoid incorrectly starting or 
     *  stopping in reentrant code. */
    bool isRunning() const {return isRunning_;}

    /** \brief Return the name of this timer */
    const std::string& name() const {return name_;}

    /** \brief Increment the number of times this timer has been called */
    void incrementNumCalls() {numCalls_++;}

    /** \brief Return the number of times this timer has been called */
    int numCalls() const {return numCalls_;}

  private:
    double startTime_;

    double totalTime_;

    bool isRunning_;

    std::string name_;

    int numCalls_;
  };

} // namespace Teuchos

#endif // TEUCHOS_TIME_HPP_

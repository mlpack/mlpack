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

#ifndef TEUCHOS_ERRORPOLLING_H
#define TEUCHOS_ERRORPOLLING_H

#include "Teuchos_ConfigDefs.hpp"
#include "Teuchos_TestForException.hpp"

/*! \defgroup ErrorPolling_grp Utility code for synchronizing std::exception detection across processors. 
*/
//@{

namespace Teuchos
{
  class MPIComm;

  /** \brief ErrorPolling provides utilities for establishing agreement
   * between processors on whether an std::exception has been detected on any one
   * processor.
   *
   * The two functions must be used in a coordinated way. The simplest use
   * case is to embed a call to reportFailure() whenever an std::exception is
   * detected at the top-level try/catch block, and then to do a call to
   * pollForFailures() whenever it is desired to check for off-processor
   * errors before proceeding. The macro

    \code
    TEUCHOS_TEST_FOR_FAILURE(comm);
    \endcode  

   * calls pollForFailures() and throws an std::exception if the return value is
   * true.
   *
   * Polling is a collective operation (an MPI_Reduce) and so incurs some
   * performance overhead. It can be disabled with a call to 
   * \code
   * Teuchos::ErrorPolling::disable();
   * \endcode 
   * IMPORTANT: all processors must agree on whether collective error checking
   * is enabled or disabled. If there are inconsistent states, the reduction
   * operations in pollForFailures() will hang because some processors cannot be 
   * contacted. 
   */
  class ErrorPolling
  {
  public:
    /** Call this function upon catching an std::exception in order to
     * inform other processors of the error. This function will do an
     * AllReduce in conjunction with calls to either this function or
     * its partner, pollForFailures(), on the other processors. This
     * procedure has the effect of communicating to the other
     * processors that an std::exception has been detected on this one. */
    static void reportFailure(const MPIComm& comm);
    
    /** Call this function after std::exception-free completion of a
     * try/catch block. This function will do an AllReduce in
     * conjunction with calls to either this function or its partner,
     * reportFailure(), on the other processors. If a failure has been
     * reported by another processor, the call to pollForFailures()
     * will return true and an std::exception can be thrown. */
    static bool pollForFailures(const MPIComm& comm);
    
    /** Activate error polling */
    static void enable() {isActive()=true;}

    /** Disable error polling */
    static void disable() {isActive()=false;}

  private:
    /** Set or check whether error polling is active */
    static bool& isActive() {static bool rtn = true; return rtn;}
  };

  /** 
   * This macro polls all processors in the given communicator to find
   * out whether an error has been reported by a call to 
   * ErrorPolling::reportFailure(comm).
   * 
   * @param comm [in] The communicator on which polling will be done
   */
#define TEUCHOS_POLL_FOR_FAILURES(comm)                                  \
  TEST_FOR_EXCEPTION(Teuchos::ErrorPolling::pollForFailures(comm), \
                     std::runtime_error,                                     \
                     "off-processor error detected by proc=" << (comm).getRank());
}

//@}

#endif

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

#ifndef TEUCHOS_PERFORMANCEMONITORBASE_H
#define TEUCHOS_PERFORMANCEMONITORBASE_H

/*! \file Teuchos_PerformanceMonitorBase.hpp
  \brief Provides common capabilities for collecting and reporting
  performance data across processors
*/

#include "Teuchos_ConfigDefs.hpp"
#include "Teuchos_MPIComm.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_Array.hpp"
#include "Teuchos_PerformanceMonitorUtils.hpp"
#include "Teuchos_TableFormat.hpp"

namespace Teuchos
{
 
/** \brief Provides common capabilities for collecting and reporting
 * performance data across processors.
 *
 * PerformanceMonitorBase is templated on a counter type (which might be a
 * timer or a flop counter). The common capability of the counter type is a
 * counter for the number of calls. Derived counter types can supply
 * additional features.
 *
 * A PerformanceMonitorBase will increment its call counter upon
 * every ctor call. Derived types might do more upon construction or
 * destruction; for example, a timer will start upon construction
 * and stop upon destruction.
 *
 * The class keeps a static list of all counters created using
 * the getNewCounter() method during the course of a run. Counts
 * from this list can then be printed out at the end of the run. 
 *
 * The minimum requirements on the counter for use in the PerformanceMonitorBase
 * are the following methods:
 * \code
 * // add one to number of calls 
 * void incrementNumCalls() 
 * // return the number of calls
 * int numCalls() const 
 * // indicate whether the counter is already running
 * bool isRunning() const 
 * \endcode
 */
template <class T> 
class PerformanceMonitorBase
{
public:

  /** \brief Construct with a counter. */
  PerformanceMonitorBase(T& counter, bool reset=false)
    : counter_(counter), isRecursiveCall_(counter_.isRunning())
    {
      counter_.incrementNumCalls();
    }

  /** \brief The dtor for the base class does nothing. */
  virtual ~PerformanceMonitorBase() {}
    
  /** \brief Create a new counter with the specified name and append it to a
   * global list of counters of this type.
   *
   * New counters should usually be created in this way rather than through a
   * direct ctor call so that they can be appended to the list.
   */
  static RCP<T> getNewCounter(const std::string& name)
    {
      RCP<T> rtn = rcp(new T(name), true);
      counters().append(rtn);
      return rtn;
    }

  /** \brief Get the format that will be used to print a summary of
   * results.
   */
  static TableFormat& format()
    {
      static RCP<TableFormat> rtn=rcp(new TableFormat()); 
      return *rtn; 
    }

protected:
    
  /** \brief Access to the counter. */
  const T& counter() const { return counter_; }
    
  /** \brief Access to the counter. */
  T& counter() { return counter_; }

  /** \brief Indicate whether the current call is recursive.
   *
   * This can matter in cases such as timing where we don't want to start and
   * stop timers multiple times within a single call stack.
   */
  bool isRecursiveCall() const { return isRecursiveCall_; }
      
  /** \brief Use the "Meyers Trick" to create static data safely. */
  static Array<RCP<T> >& counters() 
    {
      static Array<RCP<T> > rtn;
      return rtn;
    }
    
private:
    
  T& counter_;
    
  bool isRecursiveCall_;
    
};

  
} // namespace Teuchos


#endif

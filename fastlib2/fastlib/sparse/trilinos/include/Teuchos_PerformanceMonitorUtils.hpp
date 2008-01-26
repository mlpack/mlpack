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

#ifndef TEUCHOS_PERFORMANCEMONITORUTILS_H
#define TEUCHOS_PERFORMANCEMONITORUTILS_H

/*! \file Teuchos_PerformanceMonitorUtils.hpp
    \brief Provides common capabilities for collecting and reporting
    performance data across processors
*/

#include "Teuchos_ConfigDefs.hpp"
#include "Teuchos_MPIComm.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_Array.hpp"

namespace Teuchos
{
  /**
   * \brief Types of reduction operations on performance metrics.
   * Add other operations if you like. 
   */
  enum EMetricReduction {ELocal, ETotal, EMin, EMax, EAvg} ;

  

  /** 
   * \brief Provides common capabilities for collecting and reporting
   * performance data across processors
   */
  class PerformanceMonitorUtils
    {
    public:
 
      /** 
       * \brief Synchronizes lists of metric names (e.g., timer names)
       * across processors. This is necessary because some functions
       * may not have been invoked on some processors, so that their
       * named timers/counters aren't created on that processor. 
       * This function does a set union of all names created on all processors.
       * It is called by synchValues().
       *
       * \param comm [in] the communicator over which name lists are
       * being synchronized
       * \param localNames [in] the names appearing on the local processor
       * \param allNames [out] the set union of name lists from all processors
       */
      static void synchNames(const MPIComm& comm,
                             const Array<std::string>& localNames,
                             Array<std::string>& allNames);

      /** 
       * \brief Creates zero values for metrics absent on this
       * processor but present on other processors. This function uses
       * a call to synchNames() to inform this processor 
       * of the existence of other metrics on other processors. 
       *
       * \param comm [in] the communicator over which name lists are
       * being synchronized
       * \param localNames [in] the names appearing on the local processor
       * \param localValues [in] the values appearing on the local processor
       * \param allNames [out] the set union of name lists from all processors
       * \param allValues [out] the metric values from all processors
       */
      static void synchValues(const MPIComm& comm,
                              const Array<std::string>& localNames,
                              const Array<Array<double> >& localValues,
                              Array<std::string>& allNames,
                              Array<Array<double> >& allValues);

      /** \brief Compute reduced performance metrics across processors, for
       * example, min, max, or total times or flop counts. 
       * \param comm [in] The MPIComm object representing the communicator
       * on which the reduction is to be done.
       * \param reductionType [in] the reduction operation to be performed
       * \param localVals [in] The metrics on this processor (<b>after</b>
       * synchronization through a call to synchValues())
       * \param reducedVals [out] The reduced metrics
       */
      static void reduce(const MPIComm& comm,
                         const EMetricReduction& reductionType,
                         const Array<double>& localVals,
                         Array<double>& reducedVals);

    };

}
#endif

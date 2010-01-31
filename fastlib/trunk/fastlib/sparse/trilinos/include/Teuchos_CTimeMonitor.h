/* @HEADER
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
*/

#ifndef TEUCHOS_CTIMEMONITOR_H
#define TEUCHOS_CTIMEMONITOR_H

/*! \file Teuchos_CTimeMonitor.hpp
    \brief Timer functions for C that starts and stops timers for C code.
*/

#ifdef __cplusplus
extern "C" {
#endif

/** \brief Start a timer with a given name and ID.
 *
 * \param  timerName
 *           [in] Globally unique null-terminated string name of the timer.
 *           This is only significant on the first call.
 *
 * \param  timerID
 *           [in] On first call, <tt>timerID</tt> should be less than
 *           <tt>0</tt> On future calls, it should be what was returned by the
 *           first call.
 *
 * \returns on first call <tt>returnVal</tt> gives the ID of a newly created
 * timer of the given globally unique name <tt>timerName</tt>.  On future
 * calls, <tt>returnVal==timerID</tt>.
 *
 * \note You can not start the same timer more than once.  You must stop a
 * timer with <tt>Teuchos_stopTimer()</tt> before you can call this function
 * to start it again.
 */
int Teuchos_startTimer( char timerName[], int timerID );

/** \brief Stop a timer that was started with <tt>Teuchos_startTimer()</tt>.
 *
 * \param  timerID
 *           [in] Must be the ID returned from a prior call to
 *           <tt>Teuchos_startTimer()</tt>.
 *
 *
 * <b>Preconditions:</b><ul>
 * <li><tt>timerID >= 0</tt> and it must have been
 *   created by a prior call to <tt>Teuchos_startTimer()</tt>.
 * </ul>
 *
 * \note It is okay to stop a timer more than once (i.e. stop a timer that is
 * not running).  But, the timer must actually exist.
 */
void Teuchos_stopTimer( int timerID );

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* TEUCHOS_CTIMEMONITOR_H */

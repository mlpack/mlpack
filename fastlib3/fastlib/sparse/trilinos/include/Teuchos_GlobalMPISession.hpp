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

#ifndef TEUCHOS_GLOBAL_MPI_SESSION_HPP
#define TEUCHOS_GLOBAL_MPI_SESSION_HPP

/*! \file Teuchos_MPISession.hpp
    \brief A MPI utilities class, providing methods for initializing,
	finalizing, and querying the global MPI session
*/
#include "Teuchos_ConfigDefs.hpp"

#ifdef HAVE_MPI
#include "mpi.h"
#endif

namespace Teuchos {

/** \brief This class provides methods for initializing, finalizing, and
 * querying the global MPI session.
 *
 * This class is primarilly designed to insulate basic <tt>main()</tt>
 * program type of code from having to know if MPI is enabled or not.
 *
 * ToDo: Give examples!
 */
class GlobalMPISession
{
public:
  
  //! @name Public constructor and destructor 
  //@{
  
  /** \brief Calls <tt>MPI_Init()</tt> if MPI is enabled.
   *
   * \param argc  [in] Argment passed into <tt>main(argc,argv)</tt>
   * \param argv  [in] Argment passed into <tt>main(argc,argv)</tt>
   * \param out   [in] If <tt>out!=NULL</tt>, then a small message on each
   *              processor will be printed to this stream.  The default is <tt>&std::cout</tt>.
   *
   * If the option <tt>--teuchos-suppress-startup-banner</tt> is found, the
   * this option will be removed from <tt>argv[]</tt> before being passed to
   * <tt>MPI_Init(...)</tt> and the startup output message to <tt>*out</tt>
   * will be suppressed.
   *
   * <b>Warning!</b> This constructor can only be called once per
   * executable or an error is printed to <tt>*out</tt> and an std::exception will
   * be thrown!
   */
  GlobalMPISession( int* argc, char*** argv, std::ostream *out = &std::cout );
  
  /** \brief Calls <tt>MPI_Finalize()</tt> if MPI is enabled.
   */
  ~GlobalMPISession();
    
  //@}
    
  //! @name Static functions 
  //@{

  /** \breif Return if MPI is initialized or not. */
  static bool mpiIsInitialized();

  /** \breif Return if MPI has already been finalized. */
  static bool mpiIsFinalized();
  
  /** \brief Returns the process rank relative to <tt>MPI_COMM_WORLD</tt>
   *
   * Returns <tt>0</tt> if MPI is not enabled.
   *
   * Note, this function can be called even if the above constructor was never
   * called so it is safe to use no matter how <tt>MPI_Init()</tt> got called
   * (but it must have been called somewhere).
   */
  static int getRank();

  /** \brief Returns the number of processors relative to
   * <tt>MPI_COMM_WORLD</tt>
   *
   * Returns <tt>1</tt> if MPI is not enabled.
   *
   * Note, this function can be called even if the above constructor was never
   * called so it is safe to use no matter how <tt>MPI_Init()</tt> got called
   * (but it must have been called somewhere).
   */
  static int getNProc();
  
  //@}
  
private:
  
  static bool haveMPIState_;
  static bool mpiIsFinalized_;
  static int rank_;
  static int nProc_;

  static void initialize( std::ostream *out );

};

} // namespace Teuchos

#endif // TEUCHOS_GLOBAL_MPI_SESSION_HPP

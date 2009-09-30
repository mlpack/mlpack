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

#ifndef TEUCHOS_DEFAULT_COMM_HPP
#define TEUCHOS_DEFAULT_COMM_HPP

#include "Teuchos_RCP.hpp"
#include "Teuchos_DefaultSerialComm.hpp"
#ifdef HAVE_MPI
#  include "Teuchos_DefaultMpiComm.hpp"
#endif

namespace Teuchos {

/** \brief Returns a default global communicator appropriate for the
 * enviroment.
 *
 * If HAVE_MPI is defined, then an instance of <tt>MpiComm</tt> will be
 * created from <tt>MPI_COMM_WORLD</tt>.  Otherwise, a <tt>SerialComm</tt>
 * is returned.
 */
template<typename Ordinal>
class DefaultComm {
public:

  /** \brief Return the default glocal communicator.
   *
   * Note that this function can not be called until after MPI has been
   * initialized if MPI is expected!
   */
  static Teuchos::RCP<const Comm<Ordinal> > getComm();

  /** \brief Return a serial comm if the input comm in null.
   */
  static Teuchos::RCP<const Comm<Ordinal> >
  getDefaultSerialComm( const Teuchos::RCP<const Comm<Ordinal> > &comm );

private:

  static Teuchos::RCP<const Comm<Ordinal> > comm_;
  static Teuchos::RCP<const Comm<Ordinal> > defaultSerialComm_;

};

// ///////////////////////////
// Template Implementations

template<typename Ordinal>
Teuchos::RCP<const Teuchos::Comm<Ordinal> >
DefaultComm<Ordinal>::getComm()
{
  if(!comm_.get()) {
#ifdef HAVE_MPI
    comm_ = rcp(new MpiComm<Ordinal>(opaqueWrapper((MPI_Comm)MPI_COMM_WORLD)));
#else // HAVE_MPI    
    comm_ = rcp(new SerialComm<Ordinal>());
#endif // HAVE_MPI    
  }
  return comm_;
}

template<typename Ordinal>
Teuchos::RCP<const Teuchos::Comm<Ordinal> >
DefaultComm<Ordinal>::getDefaultSerialComm(
  const Teuchos::RCP<const Comm<Ordinal> > &comm
  )
{
  if( comm.get() )
    return comm;
  else
    return defaultSerialComm_;
}

template<typename Ordinal>
Teuchos::RCP<const Teuchos::Comm<Ordinal> >
DefaultComm<Ordinal>::comm_ = Teuchos::null;

template<typename Ordinal>
Teuchos::RCP<const Teuchos::Comm<Ordinal> >
DefaultComm<Ordinal>::defaultSerialComm_
= Teuchos::rcp(new Teuchos::SerialComm<Ordinal>());

} // namespace Teuchos

#endif // TEUCHOS_DEFAULT_COMM_HPP

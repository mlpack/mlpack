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

#ifndef TEUCHOS_RCP_SHAREDPTR_CONVERSIONS_HPP
#define TEUCHOS_RCP_SHAREDPTR_CONVERSIONS_HPP

#include "Teuchos_RCPBoostSharedPtrConversionsDecl.hpp"
#include "Teuchos_RCP.hpp"

template<class T>
Teuchos::RCP<T>
Teuchos::rcp( const boost::shared_ptr<T> &sptr )
{
  if(sptr.get()) {
    const RCPDeleter<T>
      *rcpd = boost::get_deleter<RCPDeleter<T> >(sptr);
    if(rcpd)
      return rcpd->ptr();
    return rcp(sptr.get(),DeallocBoostSharedPtr<T>(sptr),true);
  }
  return null;
}

template<class T>
boost::shared_ptr<T>
Teuchos::shared_pointer( const RCP<T> &rcp )
{
  if(rcp.get()) {
    const DeallocBoostSharedPtr<T>
      *dbsp = get_optional_dealloc<DeallocBoostSharedPtr<T> >(rcp);
    if(dbsp)
      return dbsp->ptr();
    return boost::shared_ptr<T>(rcp.get(),RCPDeleter<T>(rcp));
  }
  return boost::shared_ptr<T>();
}

#endif	// TEUCHOS_RCP_SHAREDPTR_CONVERSIONS_HPP

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

#ifndef TEUCHOS_PARAMETER_LIST_ACCEPTOR_DEFAULT_BASE_HPP
#define TEUCHOS_PARAMETER_LIST_ACCEPTOR_DEFAULT_BASE_HPP

#include "Teuchos_ParameterListAcceptor.hpp"
#include "Teuchos_RCP.hpp"


namespace Teuchos {


/** \brief Intermediate node base class for objects that accept parameter lists
 * that implements some of the needed behavior automatically.
 *
 * Subclasses just need to implement <tt>setParameterList()</tt> and
 * <tt>getValidParameters()</tt>.  The underlying paraemeter list is accessed
 * using the protected members <tt>setMyParamList()</tt> and
 * <tt>getMyParamList()</tt>.
 */
class ParameterListAcceptorDefaultBase : virtual public ParameterListAcceptor {
public:

  /** \name Overridden from ParameterListAcceptor */
  //@{

  /** \brief . */
  RCP<ParameterList> getParameterList();
  /** \brief . */
  RCP<ParameterList> unsetParameterList();
  /** \brief . */
  RCP<const ParameterList> getParameterList() const;

  //@}

protected:

  /** \name Protected accessors to actual parameter list object. */
  //@{

  /** \brief . */
  void setMyParamList( const RCP<ParameterList> &paramList );

  /** \brief . */
  RCP<ParameterList> getMyParamList();

  /** \brief . */
  RCP<const ParameterList> getMyParamList() const;

  //@}

private:

  RCP<ParameterList> paramList_;

};


//
// Inline definitions
//


inline
void ParameterListAcceptorDefaultBase::setMyParamList(
  const RCP<ParameterList> &paramList
  )
{
  paramList_ = paramList;
}


inline
RCP<ParameterList>
ParameterListAcceptorDefaultBase::getMyParamList()
{
  return paramList_;
}


inline
RCP<const ParameterList>
ParameterListAcceptorDefaultBase::getMyParamList() const
{
  return paramList_;
}


} // end namespace Teuchos


#endif // TEUCHOS_PARAMETER_LIST_ACCEPTOR_DEFAULT_BASE_HPP

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

#ifndef TEUCHOS_PARAMETER_LIST_ACCEPTOR_HPP
#define TEUCHOS_PARAMETER_LIST_ACCEPTOR_HPP

#include "Teuchos_ConfigDefs.hpp"

namespace Teuchos {

class ParameterList;
template<class T> class RCP;

/** \brief Base class objects that can accept a parameter list.
 *
 * ToDo: Finish Documentation!
 */
class ParameterListAcceptor {
public:

  /** \brief . */
  virtual ~ParameterListAcceptor();

  //! @name Pure virtual functions that must be overridden in subclasses 
  //@{

  /** \brief Set parameters from a parameter list and return with default values.
   *
   * \param  paramList [in] On input contains the parameters set by the client.
   *                   Note that <tt>*paramList</tt> may have parameters set to their
   *                   default values added while the list is being parsed either right
   *                   away or later.
   *
   * <b>Preconditions:</b><ul>
   * <li><tt>paramList.get() != NULL</tt>
   * </ul>
   *
   * <b>Postconditions:</b><ul>
   * <li><tt>this->getParameterList().get() == paramList.get()</tt>
   * </ul>
   *
   * This is parameter list is "remembered" by <tt>*this</tt> object until it is
   * unset using <tt>unsetParameterList()</tt>.
   *
   * <b>Note:</b> When this parameter list is passed in it is assumed that the
   * client has finished setting all of the values that they want to set so
   * that the list is completely ready to read (and be validated) by
   * <tt>*this</tt> object.  If the client is to change this parameter list by
   * adding new options or changing the value of current options, the behavior
   * of <tt>*this</tt> object is undefined.  This is because, the object may
   * read the options from <tt>*paramList</tt> right away or may wait to read
   * some options until a later time.  There should be no expectation that if
   * an option is changed by the client that this will automatically be
   * recognized by <tt>*this</tt> object.  To change even one parameter, this
   * function must be called again, with the entire sublist.
   */
  virtual void setParameterList(RCP<ParameterList> const& paramList) = 0;

  /** \brief Get the parameter list that was set using <tt>setParameterList()</tt>.
   */
  virtual RCP<ParameterList> getParameterList() = 0;

  /** \brief Unset the parameter list that was set using <tt>setParameterList()</tt>.
   *
   * This just means that the parameter list that was set using
   * <tt>setParameterList()</tt> is detached from this object.  This does not
   * mean that the effect of the parameters is undone.
   *
   * <b>Postconditions:</b><ul>
   * <li><tt>this->getParameterList().get() == NULL</tt>
   * </ul>
   */
  virtual RCP<ParameterList> unsetParameterList() = 0;

  //@}

  //! @name Virtual functions with default implementation 
  //@{

  /** \brief Get const version of the parameter list that was set using <tt>setParameterList()</tt>.
   *
   * The default implementation returns:
   \code
   return const_cast<ParameterListAcceptor*>(this)->getParameterList();
   \endcode
   */
  virtual RCP<const ParameterList> getParameterList() const;

  /** \brief Return a const parameter list of all of the valid parameters that
   * <tt>this->setParameterList(...)</tt> will accept.
   *
   * The default implementation returns <tt>Teuchos::null</tt>.
   */
  virtual RCP<const ParameterList> getValidParameters() const;

  //@}

};

} // end namespace Teuchos

#endif // TEUCHOS_PARAMETER_LIST_ACCEPTOR_HPP

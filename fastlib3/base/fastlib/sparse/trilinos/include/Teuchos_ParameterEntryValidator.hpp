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


#ifndef TEUCHOS_PARAMETER_ENTRY_VALIDATOR_H
#define TEUCHOS_PARAMETER_ENTRY_VALIDATOR_H

#include "Teuchos_RCP.hpp"
#include "Teuchos_Array.hpp"

namespace Teuchos {


#ifndef DOXYGEN_SHOULD_SKIP_THIS
class ParameterEntry;
#endif


/** \brief Abstract interface for an object that can validate a
 *  ParameterEntry's value.
 *
 * Not only can a validator validate and entry but it can also help to set
 * and/or adjust the default value.
 */
class ParameterEntryValidator {
public:

  /** \brief . */
  virtual ~ParameterEntryValidator() {}

  /** \brief Print documentation for this parameter.
   *
   * \param  docString 
   *           [in] (Multi-line) documentation std::string.
   * \param  out
   *           [out] The std::ostream used for the output
   *
   * The purpose of this function is to augment what is in <tt>docString</tt>
   * with some description of what valid values this parameter validator will
   * accept.
   */
  virtual void printDoc(
    std::string         const& docString
    ,std::ostream            & out
    ) const = 0;

  /** \brief Return an array of strings of valid values if applicable.
   *
   * If there is no such array of std::string values that makes since, just return
   * <tt>return.get()==NULL</tt>.
   *
   * The returned strings must not contain any newlines (i.e. no <tt>'\n'</tt>
   * characters) and must be short enough to fit on one line and be readable.
   */
  virtual Teuchos::RCP<const Array<std::string> >
  validStringValues() const = 0;

  /** \brief Validate a parameter entry value and throw std::exception (with a
   * great error message) if validation fails.
   *
   * \param  entry
   *            [in] The ParameterEntry who's type and value is being validated
   * \param  paramName
   *            [in] The name of the ParameterEntry that is used to build error messages.
   * \param  sublistName
   *            [in] The name of the ParameterList that <tt>paramName</tt> exists in
   *            that is used to build error messages.
   */
  virtual void validate(
    ParameterEntry  const& entry,
    std::string const& paramName,
    std::string const& sublistName
    ) const = 0;

  /** \brief Validate and perhaps modify a parameter entry's value.
   *
   * \param  paramName
   *            [in] The name of the ParameterEntry that is used to build error messages.
   * \param  sublistName
   *            [in] The name of the ParameterList that <tt>paramName</tt> exists in
   *            that is used to build error messages.
   * \param  entry
   *            [in/out] The ParameterEntry who's type and value is being validated and
   *            perhaps even changed as a result of calling this function.
   *
   * The default implementation simply calls <tt>this->validate()</tt>.
   */
  virtual void validateAndModify(
    std::string const& paramName,
    std::string const& sublistName,
    ParameterEntry * entry
    ) const
    {
      TEST_FOR_EXCEPT(0==entry);
      this->validate(*entry,paramName,sublistName);
    }
  
};


} // namespace Teuchos


#endif // TEUCHOS_PARAMETER_ENTRY_VALIDATOR_H

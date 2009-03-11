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

#ifndef TEUCHOS_XML_PARAMETER_LIST_HELPERS_HPP
#define TEUCHOS_XML_PARAMETER_LIST_HELPERS_HPP

/*! \file Teuchos_XMLParameterListHelpers.hpp \brief Simple helper functions
     that make it easy to read and write XML to and from a parameterlist.
*/

#include "Teuchos_ParameterList.hpp"

namespace Teuchos {

/** \brief Reads XML parameters from a file and updates those already in the
 * given parameter list.
 *
 * \param  xmlFileName  [in] The file name containing XML parameter list specification.
 * \param  paramList    [in/out]  On input, <tt>*paramList</tt> may be empty or contain some
 *                      parameters and sublists.  On output, parameters and sublist from
 *                      the file <tt>xmlFileName</tt> will be set or overide those in
 *                      <tt>*paramList</tt>.
 *
 * \ingroup XML
 */
void updateParametersFromXmlFile(
  const std::string            &xmlFileName
  ,Teuchos::ParameterList      *paramList
  );

/** \brief Reads XML parameters from a std::string and updates those already in the
 * given parameter list.
 *
 * \param  xmlStr       [in] String containing XML parameter list specification.
 * \param  paramList    [in/out]  On input, <tt>*paramList</tt> may be empty or contain some
 *                      parameters and sublists.  On output, parameters and sublist from
 *                      the file <tt>xmlStr</tt> will be set or overide those in
 *                      <tt>*paramList</tt>.
 *
 * \ingroup XML
 */
void updateParametersFromXmlString(
  const std::string            &xmlStr
  ,Teuchos::ParameterList      *paramList
  );

/** \brief Write parameters and sublists in XML format to an std::ostream.
 *
 * \param  paramList    [in]  Contains the parameters and sublists that will be written
 *                      to file.
 * \param  xmlOut       [in] The stream that will get the XML output.
 *
 * \ingroup XML
 */
void writeParameterListToXmlOStream(
  const Teuchos::ParameterList      &paramList
  ,std::ostream                     &xmlOut
  );

/** \brief Write parameters and sublist to an XML file.
 *
 * \param  paramList    [in]  Contains the parameters and sublists that will be written
 *                      to file.
 * \param  xmlFileName  [in] The file name that will be create to contain the XML version
 *                      of the parameter list specification.
 *
 * \ingroup XML
 */
void writeParameterListToXmlFile(
  const Teuchos::ParameterList      &paramList
  ,const std::string                &xmlFileName
  );

} // namespace Teuchos

#endif // TEUCHOS_XML_PARAMETER_LIST_HELPERS_HPP

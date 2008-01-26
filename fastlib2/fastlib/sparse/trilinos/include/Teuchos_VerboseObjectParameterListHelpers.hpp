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

#ifndef TEUCHOS_VERBOSE_OBJECT_PARAMETER_LIST_HELPERS_HPP
#define TEUCHOS_VERBOSE_OBJECT_PARAMETER_LIST_HELPERS_HPP

#include "Teuchos_VerboseObject.hpp"
#include "Teuchos_ParameterList.hpp"


namespace Teuchos {


/** \brief Return the sublist of valid parameters for the "VerboseObject"
 * sublist.
 *
 * This function need not be directly called by clients since the function
 * <tt>setupVerboseObjectSublist()</tt> sets up the sublist automatically.
 *
 * \relates VerboseObject
 */
RCP<const ParameterList> getValidVerboseObjectSublist();


/** \brief Setup a sublist called "VerboseObject" in the given parameter list.
 *
 * \param paramList
 *          [in/out] The parameter list hat the "VerboseObject" sublist will
 *          be added to
 *
 * <b>Preconditions:</b><ul>
 * <li><tt>paramList!=0</tt>
 * </ul>
 *
 * \relates VerboseObject
 */
void setupVerboseObjectSublist( ParameterList* paramList );

/** \brief Read the parameters in the "VerboseObject" sublist and set them on
 * the given VerboseObject.
 *
 * \param paramList
 *          [in/out] On input, contains the user's parameter list for the
 *          given objet for which "VerboseObject" can be a sublist of.
 * \param oStream
 *          [out] The oStream object to be used.  On output,
 *          <tt>oStream->get()!=0</tt> if an output stream was specified by
 *          the parameter sublist.
 * \param verbLevel
 *          [out] The verbosity level to be used.  On output,
 *          <tt>*verbLevel</tt> gives the verbosity level set in the parameter
 *          list.  If no verbosity level was set, then a value of
 *          <tt>*verbLevel==VERB_DEFAULT</tt> will be set on return.
 *
 * <b>Preconditions:</b><ul>
 * <li><tt>oStream!=0</tt>
 * <li><tt>verbLevel!=0</tt>
 * </ul>
 *
 * \relates VerboseObject
 */
void readVerboseObjectSublist(
  ParameterList* paramList,
  RCP<FancyOStream> *oStream, EVerbosityLevel *verbLevel
  );


/** \brief Read the parameters in the "VerboseObject" sublist and set them on
 * the given VerboseObject.
 *
 * \param paramList
 *          [in/out] On input, contains the user's parameter list for the
 *          given objet for which "VerboseObject" can be a sublist of.
 * \param verboseObject
 *          [in/out] The verbose object that will have its verbosity level
 *          and/or output stream set.
 *
 * This function just calls the above nontemplated
 * <tt>readVerboseObjectSublist()</tt> to validate and and read the verbosity
 * and output stream from the "VerboseObject" sublist.
 *
 * \relates VerboseObject
 */
template<class ObjectType>
void readVerboseObjectSublist(
  ParameterList* paramList, VerboseObject<ObjectType> *verboseObject
  );


} // namespace Teuchos


// /////////////////////////////////
// Implementations


template<class ObjectType>
void Teuchos::readVerboseObjectSublist(
  ParameterList* paramList, VerboseObject<ObjectType> *verboseObject
  )
{
  TEST_FOR_EXCEPT(0==paramList);
  TEST_FOR_EXCEPT(0==verboseObject);
  const EVerbosityLevel bogusVerbLevel = static_cast<EVerbosityLevel>(-50);
  RCP<FancyOStream> oStream = null;
  EVerbosityLevel verbLevel = bogusVerbLevel;
  readVerboseObjectSublist(paramList,&oStream,&verbLevel);
  verboseObject->setOverridingOStream(oStream);
  verboseObject->setOverridingVerbLevel(verbLevel);
}


#endif // TEUCHOS_VERBOSE_OBJECT_PARAMETER_LIST_HELPERS_HPP

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

#ifndef TEUCHOS_STANDARD_PARAMETER_ENTRY_VALIDATORS_H
#define TEUCHOS_STANDARD_PARAMETER_ENTRY_VALIDATORS_H

#include "Teuchos_ParameterEntryValidator.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_ParameterListExceptions.hpp"
#include "Teuchos_VerbosityLevel.hpp"
#include "Teuchos_Array.hpp"
#include "Teuchos_StrUtils.hpp"
#include "Teuchos_TypeNameTraits.hpp"


namespace Teuchos {


/** \brief Standard implementation of a ParameterEntryValidator that maps from
 * a list of strings to some integral type value.
 *
 * Objects of this type are meant to be used as both abstract objects passed
 * to <tt>Teuchos::ParameterList</tt> objects to be used to validate parameter
 * types and values, and to be used by the code that reads parameter values.
 * Having a single definition for the types of valids input and outputs for a
 * parameter value makes it easier to write error free validated code.
 */
template<class IntegralType>
class StringToIntegralParameterEntryValidator : public ParameterEntryValidator {
public:

  /** \name Constructors */
  //@{

  /** \brief Construct with a mapping from strings to ordinals <tt>0</tt> to
   * </tt>n-1</tt>.
   *
   * \param  strings
   *             [in] Array of unique std::string names.
   * \param  defaultParameterName
   *             [in] The default name of the parameter (used in error messages)
   */
  StringToIntegralParameterEntryValidator(
    Array<std::string> const& strings,
    std::string const& defaultParameterName
    );

  /** \brief Construct with a mapping from strings to aribitrary typed
   * integral values.
   *
   * \param  strings
   *             [in] Array of unique std::string names.
   * \param  integralValues
   *            [in] Array that gives the integral values associated with
   *            <tt>strings[]</tt>
   * \param  defaultParameterName
   *             [in] The default name of the parameter (used in error messages)
   *
   * <b>Preconditions:</b><ul>
   * <li> <tt>strings.size() == integralValues.size()</tt>
   * </ul>
   */
  StringToIntegralParameterEntryValidator(
    Array<std::string> const& strings,
    Array<IntegralType> const& integralValues, 
    std::string const& defaultParameterName
    );

  /** \brief Construct with a mapping from strings (with documentation) to
   * aribitrary typed integral values.
   *
   * \param  strings
   *             [in] Array of unique std::string names.
   * \param  stringsDocs
   *             [in] Array of documentation strings for each std::string value.
   * \param  integralValues
   *            [in] Array that gives the integral values associated with
   *            <tt>strings[]</tt>
   * \param  defaultParameterName
   *             [in] The default name of the parameter (used in error messages)
   *
   * <b>Preconditions:</b><ul>
   * <li> <tt>strings.size() == stringDocs.size()</tt>
   * <li> <tt>strings.size() == integralValues.size()</tt>
   * </ul>
   */
  StringToIntegralParameterEntryValidator(
    Array<std::string> const& strings,
    Array<std::string> const& stringsDocs,
    Array<IntegralType> const& integralValues, 
    std::string const& defaultParameterName
    );

  //@}

  /** \name Local non-virtual validated lookup functions */
  //@{

  /** \brief Perform a mapping from a std::string value to its integral value.
   *
   * \param  str  [in] String that is being used to lookup the corresponding
   *              integral value.
   * \param  paramName
   *              [in] Optional name that will be used to generate error messages.
   *
   * If the std::string name <tt>str</tt> does not exist, the an std::exception will be
   * thrown with a very descriptive error message.
   */
  IntegralType getIntegralValue(
    const std::string &str, const std::string &paramName = "",
    const std::string &sublistName = ""
    ) const;

  /** \brief Perform a mapping from a std::string value embedded in a
   * <tt>ParameterEntry</tt> object and return its associated integral value.
   *
   * \param  entry
   *              [in] The std::string entry.
   * \param  paramName
   *              [in] Optional name that will be used to generate error messages.
   * \param  sublistName
   *              [in] The name of the sublist.
   * \param  activeQuery
   *              [in] If true, then this lookup will be recored as an active query
   *              which will turn the <tt>isUsed</tt> bool to <tt>true</tt>.
   */
  IntegralType getIntegralValue(
    const ParameterEntry &entry, const std::string &paramName = "",
    const std::string &sublistName = "", const bool activeQuery = true
    ) const;

  /** \brief Get and validate a std::string value embedded in a
   * <tt>ParameterEntry</tt> object.
   *
   *
   * \param  entry
   *              [in] The std::string entry.
   * \param  paramName
   *              [in] Optional name that will be used to generate error messages.
   * \param  sublistName
   *              [in] The name of the sublist.
   * \param  activeQuery
   *              [in] If true, then this lookup will be recored as an active query
   *              which will turn the <tt>isUsed</tt> bool to <tt>true</tt>.
   */
  std::string getStringValue(
    const ParameterEntry &entry, const std::string &paramName = "",
    const std::string &sublistName = "", const bool activeQuery = true
    ) const;

  /** \brief Lookup a parameter from a parameter list, perform a mapping from
   * a std::string value embedded in the <tt>ParameterEntry</tt> object and return
   * its associated integral value.
   */
  IntegralType getIntegralValue(
    ParameterList &paramList, const std::string &paramName,
    const std::string &defaultValue
    ) const;

  /** \brief Lookup a parameter from a parameter list, validate the std::string
   * value, and return the std::string value.
   */
  std::string getStringValue(
    ParameterList &paramList, const std::string &paramName,
    const std::string &defaultValue
    ) const;

  /** \brief Validate the std::string and pass it on..
   *
   * \param  str  [in] String that is being used to lookup the corresponding
   *              integral value.
   * \param  name [in] Optional name that will be used to generate error messages.
   *
   * If the std::string name <tt>str</tt> does not exist, the an std::exception will be
   * thrown with a very descriptive error message.
   */
  std::string validateString(
    const std::string &str, const std::string &paramName = "",
    const std::string &sublistName = ""
    ) const;

  //@}

  /** \name Overridden from ParameterEntryValidator */
  //@{

  /** \brief . */
  void printDoc(
    std::string const& docString,
    std::ostream & out
    ) const;

  /** \brief . */
  Teuchos::RCP<const Array<std::string> >
  validStringValues() const;

  /** \brief . */
  void validate(
    ParameterEntry const& entry,
    std::string const& paramName,
    std::string const& sublistName
    ) const;

  //@}

private:

  typedef std::map<std::string,IntegralType> map_t;
  std::string defaultParameterName_;
  std::string validValues_;
  RCP<const Array<std::string> > validStringValues_;
  RCP<const Array<std::string> > validStringValuesDocs_;
  map_t map_;

  void setValidValues(
    Array<std::string> const& strings,
    Array<std::string> const* stringsDocs = NULL
    );

  // Not defined and not to be called.
  StringToIntegralParameterEntryValidator();

};


/** \brief Nonmember constructor (see implementation).
 *
 * \relates StringToIntegralParameterEntryValidator
 */
template<class IntegralType>
RCP<StringToIntegralParameterEntryValidator<IntegralType> >
stringToIntegralParameterEntryValidator(
  Array<std::string> const& strings,
  std::string const& defaultParameterName
  );


/** \brief Nonmember constructor (see implementation).
 *
 * \relates StringToIntegralParameterEntryValidator
 */
template<class IntegralType>
RCP<StringToIntegralParameterEntryValidator<IntegralType> >
stringToIntegralParameterEntryValidator(
  Array<std::string> const& strings,
  Array<IntegralType> const& integralValues, 
  std::string const& defaultParameterName
  );


/** \brief Nonmember constructor (see implementation).
 *
 * \relates StringToIntegralParameterEntryValidator
 */
template<class IntegralType>
RCP<StringToIntegralParameterEntryValidator<IntegralType> >
stringToIntegralParameterEntryValidator(
  Array<std::string> const& strings,
  Array<std::string> const& stringsDocs,
  Array<IntegralType> const& integralValues, 
  std::string const& defaultParameterName
  );


/** \brief Set up a std::string parameter that will use an embedded validator to
 * allow the extraction of an integral value.
 *
 * The function <tt>getIntegralValue()</tt> can then be used to extract the
 * integral value of the std::string parameter.  In this case, the integral value
 * return will just be the zero-based index of the std::string value in the list
 * <tt>strings</tt>.
 *
 * \relates ParameterList
 */
template<class IntegralType>
void setStringToIntegralParameter(
  std::string const& paramName,
  std::string const& defaultValue,
  std::string const& docString,
  Array<std::string> const& strings,
  ParameterList * paramList
  );


/** \brief Set up a std::string parameter that will use an embedded validator to
 * allow the extraction of an integral value from a list of integral values.
 *
 * The function <tt>getIntegralValue()</tt> can then be used to extract the
 * integral value of the std::string parameter.  In this case, the integral value
 * return will just be the zero-based index of the std::string value in the list
 * <tt>strings</tt>.
 *
 * \relates ParameterList
 */
template<class IntegralType>
void setStringToIntegralParameter(
  std::string const& paramName,
  std::string const& defaultValue,
  std::string const& docString,
  Array<std::string> const& strings,
  Array<IntegralType> const& integralValues, 
  ParameterList * paramList
  );


/** \brief Set up a std::string parameter with documentation strings for each valid
 * value that will use an embedded validator to allow the extraction of an
 * integral value from a list of integral values.
 *
 * The function <tt>getIntegralValue()</tt> can then be used to extract the
 * integral value of the std::string parameter.  In this case, the integral value
 * return will just be the zero-based index of the std::string value in the list
 * <tt>strings</tt>.
 *
 * \relates ParameterList
 */
template<class IntegralType>
void setStringToIntegralParameter(
  std::string const& paramName,
  std::string const& defaultValue,
  std::string const& docString,
  Array<std::string> const& strings,
  Array<std::string> const& stringsDocs,
  Array<IntegralType> const& integralValues, 
  ParameterList * paramList
  );


/** \brief Get an integral value for a parameter that is assumed to already be set.
 *
 * This function does a dynamic cast to get the underlying valiator of type
 * StringToIntegralParameterEntryValidator<IntegralType>.  If this dynamic
 * cast failes then an <tt>Exceptions::InvalidParameterType</tt> std::exception is
 * thrown with an excellent error message.
 *
 * \relates ParameterList
 */
template<class IntegralType>
IntegralType getIntegralValue(
  ParameterList const& paramList, std::string const& paramName
  );


/** \brief Get a std::string value for a parameter that is assumed to already be set.
 *
 * This function does a dynamic cast to get the underlying valiator of type
 * StringToIntegralParameterEntryValidator<IntegralValue>.  The default type
 * for IntegralValue is int.  If this dynamic cast failes then an
 * <tt>Exceptions::InvalidParameterType</tt> std::exception is thrown with an
 * excellent error message.
 *
 * \relates ParameterList
 */
template<class IntegralType>
std::string getStringValue(
  ParameterList const& paramList, std::string const& paramName
  );


/** \brief Get a StringToIntegralParameterEntryValidator<IntegralType> object out of
 * a ParameterEntry object.
 *
 * This function with thrown of the validator does not exist.
 */
template<class IntegralType>
RCP<const StringToIntegralParameterEntryValidator<IntegralType> >
getStringToIntegralParameterEntryValidator(
  ParameterEntry const& entry, ParameterList const& paramList,
  std::string const& paramName
  );


/** \brief Return the std::string name of the verbosity level as it is accepted by the
 * verbosity level parameter.
 *
 * \relates EVerbosityLevel
 */
std::string getVerbosityLevelParameterValueName(
  const EVerbosityLevel verbLevel
  );


/** \brief Return a validator for <tt>EVerbosityLevel</tt>.
 *
 * \relates EVerbosityLevel
 */
RCP<StringToIntegralParameterEntryValidator<EVerbosityLevel> >
verbosityLevelParameterEntryValidator(std::string const& defaultParameterName);


/** \brief Standard implementation of a ParameterEntryValidator that accepts
 * numbers from a number of different formats and converts them to numbers in
 * another format.
 *
 * Objects of this type are meant to be used as both abstract objects passed
 * to <tt>Teuchos::ParameterList</tt> objects to be used to validate parameter
 * types and values, and to be used by the code that reads parameter values.
 * Having a single definition for the types of valids input and outputs for a
 * parameter value makes it easier to write error-free validated code.
 */
class AnyNumberParameterEntryValidator : public ParameterEntryValidator {
public:

  /** \name Public types */
  //@{

  /** \brief Determines what type is the preferred type. */
  enum EPreferredType { PREFER_INT, PREFER_DOUBLE, PREFER_STRING };


  /** \brief Determines the types that are accepted.  */
  class AcceptedTypes {
  public:
    /** \brief Allow all types or not on construction. */
    AcceptedTypes( bool allowAllTypesByDefault = true )
      :allowInt_(allowAllTypesByDefault),allowDouble_(allowAllTypesByDefault),
       allowString_(allowAllTypesByDefault)
      {}
    /** \brief Set allow an <tt>int</tt> value or not */
    AcceptedTypes& allowInt( bool _allowInt )
      { allowInt_ = _allowInt; return *this; }
    /** \brief Set allow a <tt>double</tt> value or not */
    AcceptedTypes& allowDouble( bool _allowDouble )
      { allowDouble_ = _allowDouble; return *this; }
    /** \brief Set allow an <tt>std::string</tt> value or not */
    AcceptedTypes& allowString( bool _allowString )
      { allowString_ = _allowString; return *this; }
    /** \brief Allow an <tt>int</tt> value? */
    bool allowInt() const { return allowInt_; }
    /** \brief Allow an <tt>double</tt> value? */
    bool allowDouble() const { return allowDouble_; }
    /** \brief Allow an <tt>std::string</tt> value? */
    bool allowString() const { return allowString_; }
  private:
    bool  allowInt_;
    bool  allowDouble_;
    bool  allowString_;
  };



  //@}

  /** \name Constructors */
  //@{

  /** \brief Construct with a preferrded type of double and accept all
   * types.
   */
  AnyNumberParameterEntryValidator();

  /** \brief Construct with allowed input and output types and the preferred
   * type.
   *
   * \param preferredType
   *          [in] Determines the preferred type.  This enum value is used to 
   *          set the default value in the override <tt>validateAndModify()</tt>.
   * \param acceptedType
   *          [in] Determines the types that are allowed in the parameter list.
   */
  AnyNumberParameterEntryValidator(
    EPreferredType const preferredType,
    AcceptedTypes const& acceptedTypes
    );

  //@}

  /** \name Local non-virtual validated lookup functions */
  //@{

  /** \brief Get an integer value from a parameter entry. */
  int getInt(
    const ParameterEntry &entry, const std::string &paramName = "",
    const std::string &sublistName = "", const bool activeQuery = true
    ) const;

  /** \brief Get a double value from a parameter entry. */
  double getDouble(
    const ParameterEntry &entry, const std::string &paramName = "",
    const std::string &sublistName = "", const bool activeQuery = true
    ) const;

  /** \brief Get a std::string value from a parameter entry. */
  std::string getString(
    const ParameterEntry &entry, const std::string &paramName = "",
    const std::string &sublistName = "", const bool activeQuery = true
    ) const;

  /** \brief Lookup parameter from a parameter list and return as an int
   * value.
   */
  int getInt(
    ParameterList &paramList, const std::string &paramName,
    const int defaultValue
    ) const;

  /** \brief Lookup parameter from a parameter list and return as an double
   * value.
   */
  double getDouble(
    ParameterList &paramList, const std::string &paramName,
    const double defaultValue
    ) const;

  /** \brief Lookup parameter from a parameter list and return as an std::string
   * value.
   */
  std::string getString(
    ParameterList &paramList, const std::string &paramName,
    const std::string &defaultValue
    ) const;
  
  //@}

  /** \name Overridden from ParameterEntryValidator */
  //@{

  /** \brief . */
  void printDoc(
    std::string const& docString,
    std::ostream & out
    ) const;

  /** \brief . */
  Teuchos::RCP<const Array<std::string> >
  validStringValues() const;

  /** \brief . */
  void validate(
    ParameterEntry const& entry,
    std::string const& paramName,
    std::string const& sublistName
    ) const;

  /** \brief . */
  void validateAndModify(
    std::string const& paramName,
    std::string const& sublistName,
    ParameterEntry * entry
    ) const;

  //@}

private:

  // ////////////////////////////
  // Private data members

  EPreferredType preferredType_;
  const AcceptedTypes acceptedTypes_;
  std::string acceptedTypesString_;

  // ////////////////////////////
  // Private member functions

  void finishInitialization();

  void throwTypeError(
    ParameterEntry const& entry,
    std::string const& paramName,
    std::string const& sublistName
    ) const;

};


// Nonmember helper functions


/** \brief Nonmember constructor AnyNumberParameterEntryValidator.
 *
 * \relates AnyNumberParameterEntryValidator
 */
RCP<AnyNumberParameterEntryValidator>
anyNumberParameterEntryValidator(
  AnyNumberParameterEntryValidator::EPreferredType const preferredType,
  AnyNumberParameterEntryValidator::AcceptedTypes const& acceptedTypes
  );


/** \brief Set an integer parameter that allows for (nearly) any input
 * parameter type that is convertible to an int.
 *
 * \relates ParameterList
 */
void setIntParameter(
  std::string const& paramName,
  int const value, std::string const& docString,
  ParameterList *paramList,
  AnyNumberParameterEntryValidator::AcceptedTypes const& acceptedTypes
  = AnyNumberParameterEntryValidator::AcceptedTypes()
  );


/** \brief Set an double parameter that allows for (nearly) any input
 * parameter type that is convertible to a double.
 *
 * \relates ParameterList
 */
void setDoubleParameter(
  std::string const& paramName,
  double const& value, std::string const& docString,
  ParameterList *paramList,
  AnyNumberParameterEntryValidator::AcceptedTypes const& acceptedTypes
  = AnyNumberParameterEntryValidator::AcceptedTypes()
  );


/** \brief Set an numeric parameter preferred as a std::string that allows for
 * (nearly) any input parameter type that is convertible to a std::string.
 *
 * \relates ParameterList
 */
void setNumericStringParameter(
  std::string const& paramName,
  std::string const& value, std::string const& docString,
  ParameterList *paramList,
  AnyNumberParameterEntryValidator::AcceptedTypes const& acceptedTypes
  = AnyNumberParameterEntryValidator::AcceptedTypes()
  );


/** \brief Get an integer parameter.
 *
 * If the underlying parameter type is already an integer, then all is good.
 * However, if it is not, then a AnyNumberParameterEntryValidator object is
 * looked for to extract the type correctly.  If no validator is attached to
 * the entry, then a new AnyNumberParameterEntryValidator object will be
 * created that that will allow the conversion from any supported type.
 *
 * The parameter must exist or an <tt>Exceptions::InvalidParameterName</tt>
 * object will be thrown.  The parameters type must be acceptable, or an
 * <tt>Exceptions::InvalidParameterType</tt> object will be thown.
 *
 * \relates ParameterList
 */
int getIntParameter(
  ParameterList const& paramList, std::string const& paramName
  );


/** \brief Get double integer parameter.
 *
 * If the underlying parameter type is already a double, then all is good.
 * However, if it is not, then a AnyNumberParameterEntryValidator object is
 * looked for to extract the type correctly.  If no validator is attached to
 * the entry, then a new AnyNumberParameterEntryValidator object will be
 * created that that will allow the conversion from any supported type.
 *
 * The parameter must exist or an <tt>Exceptions::InvalidParameterName</tt>
 * object will be thrown.  The parameters type must be acceptable, or an
 * <tt>Exceptions::InvalidParameterType</tt> object will be thown.
 *
 * \relates ParameterList
 */
double getDoubleParameter(
  ParameterList const& paramList,
  std::string const& paramName
  );


/** \brief Get std::string numeric parameter.
 *
 * If the underlying parameter type is already a std::string, then all is good.
 * However, if it is not, then a AnyNumberParameterEntryValidator object is
 * looked for to extract the type correctly.  If no validator is attached to
 * the entry, then a new AnyNumberParameterEntryValidator object will be
 * created that that will allow the conversion from any supported type.
 *
 * The parameter must exist or an <tt>Exceptions::InvalidParameterName</tt>
 * object will be thrown.  The parameters type must be acceptable, or an
 * <tt>Exceptions::InvalidParameterType</tt> object will be thown.
 *
 * \relates ParameterList
 */
std::string getNumericStringParameter(
  ParameterList const& paramList,
  std::string const& paramName
  );


// ///////////////////////////
// Implementations


//
// StringToIntegralParameterEntryValidator
//


// Constructors


template<class IntegralType>
StringToIntegralParameterEntryValidator<IntegralType>::StringToIntegralParameterEntryValidator(
  Array<std::string> const& strings, std::string const& defaultParameterName
  )
  :defaultParameterName_(defaultParameterName)
{
  typedef typename map_t::value_type val_t;
  for( int i = 0; i < static_cast<int>(strings.size()); ++i ) {
    const bool unique = map_.insert( val_t( strings[i], i ) ).second;
    TEST_FOR_EXCEPTION(
      !unique, std::logic_error
      ,"Error, the std::string \"" << strings[i] << "\" is a duplicate for parameter \""
      << defaultParameterName_ << "\"."
      );
  }
  setValidValues(strings);
}


template<class IntegralType>
StringToIntegralParameterEntryValidator<IntegralType>::StringToIntegralParameterEntryValidator(
  Array<std::string> const& strings, Array<IntegralType> const& integralValues 
  ,std::string const& defaultParameterName
  )
  :defaultParameterName_(defaultParameterName)
{
#ifdef TEUCHOS_DEBUG
  TEST_FOR_EXCEPT( strings.size() != integralValues.size() );
#endif
  typedef typename map_t::value_type val_t;
  for( int i = 0; i < static_cast<int>(strings.size()); ++i ) {
    const bool unique = map_.insert( val_t( strings[i], integralValues[i] ) ).second;
    TEST_FOR_EXCEPTION(
      !unique, std::logic_error
      ,"Error, the std::string \"" << strings[i] << "\" is a duplicate for parameter \""
      << defaultParameterName_ << "\""
      );
  }
  setValidValues(strings);
}


template<class IntegralType>
StringToIntegralParameterEntryValidator<IntegralType>::StringToIntegralParameterEntryValidator(
  Array<std::string>    const& strings
  ,Array<std::string>   const& stringsDocs
  ,Array<IntegralType>  const& integralValues 
  ,std::string          const& defaultParameterName
  )
  :defaultParameterName_(defaultParameterName)
{
#ifdef TEUCHOS_DEBUG
  TEST_FOR_EXCEPT( strings.size() != stringsDocs.size() );
  TEST_FOR_EXCEPT( strings.size() != integralValues.size() );
#endif
  typedef typename map_t::value_type val_t;
  for( int i = 0; i < static_cast<int>(strings.size()); ++i ) {
    const bool unique = map_.insert( val_t( strings[i], integralValues[i] ) ).second;
    TEST_FOR_EXCEPTION(
      !unique, std::logic_error
      ,"Error, the std::string \"" << strings[i] << "\" is a duplicate for parameter \""
      << defaultParameterName_ << "\""
      );
  }
  setValidValues(strings,&stringsDocs);
}


// Lookup functions


template<class IntegralType>
IntegralType
StringToIntegralParameterEntryValidator<IntegralType>::getIntegralValue(
  const std::string &str, const std::string &paramName
  ,const std::string &sublistName
  ) const
{
  typename map_t::const_iterator itr = map_.find(str);
  TEST_FOR_EXCEPTION_PURE_MSG(
    itr == map_.end(), Exceptions::InvalidParameterValue
    ,"Error, the value \"" << str << "\" is not recognized for the parameter \""
    << ( paramName.length() ? paramName : defaultParameterName_ ) << "\""
    << "\nin the sublist \"" << sublistName << "\"."
    << "\n\nValid values include:"
    << "\n  {\n"
    << validValues_
    << "  }"
    );
  return (*itr).second;	
}


template<class IntegralType>
IntegralType
StringToIntegralParameterEntryValidator<IntegralType>::getIntegralValue(
  const ParameterEntry &entry, const std::string &paramName
  ,const std::string &sublistName, const bool activeQuery
  ) const
{
  const bool validType = ( entry.getAny(activeQuery).type() == typeid(std::string) );
  TEST_FOR_EXCEPTION_PURE_MSG(
    !validType, Exceptions::InvalidParameterType
    ,"Error, the parameter {paramName=\""<<(paramName.length()?paramName:defaultParameterName_)
    << "\",type=\""<<entry.getAny(activeQuery).typeName()<<"\"}"
    << "\nin the sublist \"" << sublistName << "\""
    << "\nhas the wrong type."
    << "\n\nThe correct type is \"string\"!"
    );
  const std::string
    &strValue = any_cast<std::string>(entry.getAny(activeQuery)); // This cast should not fail!
  return getIntegralValue(strValue,paramName,sublistName); // This will validate the value and throw!
}


template<class IntegralType>
std::string
StringToIntegralParameterEntryValidator<IntegralType>::getStringValue(
  const ParameterEntry &entry, const std::string &paramName
  ,const std::string &sublistName, const bool activeQuery
  ) const
{
  // Validate the parameter's type and value
  this->getIntegralValue(entry,paramName,sublistName,activeQuery);
  // Return the std::string value which is now validated!
  return any_cast<std::string>(entry.getAny(activeQuery)); // This cast should not fail!
}


template<class IntegralType>
IntegralType
StringToIntegralParameterEntryValidator<IntegralType>::getIntegralValue(
  ParameterList &paramList, const std::string &paramName
  ,const std::string &defaultValue
  ) const
{
  const std::string
    &strValue = paramList.get(paramName,defaultValue);
  return getIntegralValue(strValue,paramName,paramList.name());
}


template<class IntegralType>
std::string
StringToIntegralParameterEntryValidator<IntegralType>::getStringValue(
  ParameterList &paramList, const std::string &paramName
  ,const std::string &defaultValue
  ) const
{
  const std::string
    &strValue = paramList.get(paramName,defaultValue);
  getIntegralValue(strValue,paramName,paramList.name()); // Validate!
  return strValue;
}


template<class IntegralType>
std::string
StringToIntegralParameterEntryValidator<IntegralType>::validateString(
  const std::string &str, const std::string &paramName
  ,const std::string &sublistName
  ) const
{
  getIntegralValue(str,paramName,sublistName); // Validate!
  return str;
}


// Overridden from ParameterEntryValidator


template<class IntegralType>
void StringToIntegralParameterEntryValidator<IntegralType>::printDoc(
  std::string         const& docString
  ,std::ostream            & out
  ) const
{
  StrUtils::printLines(out,"# ",docString);
  out << "#   Valid std::string values:\n";
  out << "#     {\n";
  if(validStringValuesDocs_.get()) {
    for( int i = 0; i < static_cast<int>(validStringValues_->size()); ++i ) {
      out << "#       \"" << (*validStringValues_)[i] << "\"\n";
      StrUtils::printLines(out,"#          ",(*validStringValuesDocs_)[i] );
    }
  }
  else {
    StrUtils::printLines(out,"#   ",validValues_);
    // Note: Above validValues_ has for initial spaces already so indent should
    // be correct!
  }
  out << "#     }\n";
}


template<class IntegralType>
Teuchos::RCP<const Array<std::string> >
StringToIntegralParameterEntryValidator<IntegralType>::validStringValues() const
{
  return validStringValues_;
}


template<class IntegralType>
void StringToIntegralParameterEntryValidator<IntegralType>::validate(
  ParameterEntry  const& entry
  ,std::string    const& paramName
  ,std::string    const& sublistName
  ) const
{
  this->getIntegralValue(entry,paramName,sublistName,false);
}


// private


template<class IntegralType>
void StringToIntegralParameterEntryValidator<IntegralType>::setValidValues(
  Array<std::string>   const& strings
  ,Array<std::string>  const* stringsDocs
  )
{
  validStringValues_ = rcp(new Array<std::string>(strings));
  if(stringsDocs)
    validStringValuesDocs_ = rcp(new Array<std::string>(*stringsDocs));
  // Here I build the list of valid values in the same order as passed in by
  // the client!
  std::ostringstream oss;
  typename map_t::const_iterator itr = map_.begin();
  for( int i = 0; i < static_cast<int>(strings.size()); ++i ) {
    oss << "    \""<<strings[i]<<"\"\n";
  }
  // Note: Above four spaces is designed for the error output above.
  validValues_ = oss.str();
}


} // namespace Teuchos


//
// Nonmember function implementations for StringToIntegralParameterEntryValidator
//


template<class IntegralType>
inline
Teuchos::RCP<Teuchos::StringToIntegralParameterEntryValidator<IntegralType> >
Teuchos::stringToIntegralParameterEntryValidator(
  Array<std::string> const& strings,
  std::string const& defaultParameterName
  )
{
  return rcp(
    new StringToIntegralParameterEntryValidator<IntegralType>(
      strings, defaultParameterName
      )
    );
}


template<class IntegralType>
inline
Teuchos::RCP<Teuchos::StringToIntegralParameterEntryValidator<IntegralType> >
Teuchos::stringToIntegralParameterEntryValidator(
  Array<std::string> const& strings,
  Array<IntegralType> const& integralValues, 
  std::string const& defaultParameterName
  )
{
  return rcp(
    new StringToIntegralParameterEntryValidator<IntegralType>(
      strings, integralValues, defaultParameterName
      )
    );
}


template<class IntegralType>
inline
Teuchos::RCP< Teuchos::StringToIntegralParameterEntryValidator<IntegralType> >
Teuchos::stringToIntegralParameterEntryValidator(
  Array<std::string> const& strings,
  Array<std::string> const& stringsDocs,
  Array<IntegralType> const& integralValues, 
  std::string const& defaultParameterName
  )
{
  return rcp(
    new StringToIntegralParameterEntryValidator<IntegralType>(
      strings, stringsDocs, integralValues, defaultParameterName
      )
    );
}


template<class IntegralType>
void Teuchos::setStringToIntegralParameter(
  std::string const& paramName,
  std::string const& defaultValue,
  std::string const& docString,
  Array<std::string> const& strings,
  ParameterList * paramList
  )
{
  TEST_FOR_EXCEPT(0==paramList);
  paramList->set(
    paramName, defaultValue, docString,
    stringToIntegralParameterEntryValidator<IntegralType>(
      strings, paramName
      )
    );
}


template<class IntegralType>
void Teuchos::setStringToIntegralParameter(
  std::string const& paramName,
  std::string const& defaultValue,
  std::string const& docString,
  Array<std::string> const& strings,
  Array<IntegralType> const& integralValues, 
  ParameterList * paramList
  )
{
  TEST_FOR_EXCEPT(0==paramList);
  paramList->set(
    paramName, defaultValue, docString,
    stringToIntegralParameterEntryValidator<IntegralType>(
      strings, integralValues, paramName
      )
    );
}


template<class IntegralType>
void Teuchos::setStringToIntegralParameter(
  std::string const& paramName,
  std::string const& defaultValue,
  std::string const& docString,
  Array<std::string> const& strings,
  Array<std::string> const& stringsDocs,
  Array<IntegralType> const& integralValues, 
  ParameterList * paramList
  )

{
  TEST_FOR_EXCEPT(0==paramList);
  paramList->set(
    paramName, defaultValue, docString,
    stringToIntegralParameterEntryValidator<IntegralType>(
      strings, stringsDocs, integralValues, paramName
      )
    );
}


template<class IntegralType>
IntegralType Teuchos::getIntegralValue(
  ParameterList const& paramList, std::string const& paramName
  )
{
  const ParameterEntry &entry = paramList.getEntry(paramName);
  RCP<const StringToIntegralParameterEntryValidator<IntegralType> >
    integralValidator = getStringToIntegralParameterEntryValidator<IntegralType>(
      entry, paramList, paramName
      );
  return integralValidator->getIntegralValue(
    entry, paramName, paramList.name(), true
    );
}


template<class IntegralType>
std::string Teuchos::getStringValue(
  ParameterList const& paramList, std::string const& paramName
  )
{
  const ParameterEntry &entry = paramList.getEntry(paramName);
  RCP<const StringToIntegralParameterEntryValidator<IntegralType> >
    integralValidator = getStringToIntegralParameterEntryValidator<IntegralType>(
      entry, paramList, paramName
      );
  return integralValidator->getStringValue(
    entry, paramName, paramList.name(), true
    );
}


template<class IntegralType>
Teuchos::RCP<const Teuchos::StringToIntegralParameterEntryValidator<IntegralType> >
Teuchos::getStringToIntegralParameterEntryValidator(
  ParameterEntry const& entry, ParameterList const& paramList,
  std::string const& paramName
  )
{
  RCP<const ParameterEntryValidator>
    validator = entry.validator();
  TEST_FOR_EXCEPTION_PURE_MSG(
    is_null(validator), Exceptions::InvalidParameterType,
    "Error!  The parameter \""<<paramName<<"\" exists\n"
    "in the parameter (sub)list \""<<paramList.name()<<"\"\n"
    "but it does not contain any validator needed to extract\n"
    "an integral value of type \""<<TypeNameTraits<IntegralType>::name()<<"\"!"
    );
  RCP<const StringToIntegralParameterEntryValidator<IntegralType> >
    integralValidator
    =
    rcp_dynamic_cast<const StringToIntegralParameterEntryValidator<IntegralType> >(
      validator
      );
  TEST_FOR_EXCEPTION_PURE_MSG(
    is_null(integralValidator), Exceptions::InvalidParameterType,
    "Error!  The parameter \""<<paramName<<"\" exists\n"
    "in the parameter (sub)list \""<<paramList.name()<<"\"\n"
    "but it contains the wrong type of validator.  The expected validator type\n"
    "is \""<<TypeNameTraits<StringToIntegralParameterEntryValidator<IntegralType> >::name()<<"\"\n"
    "but the contained validator type is \""<<typeName(*validator)<<"\"!"
    );
  return integralValidator;
}


#endif // TEUCHOS_STANDARD_PARAMETER_ENTRY_VALIDATORS_H

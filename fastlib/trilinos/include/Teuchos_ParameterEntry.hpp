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


#ifndef TEUCHOS_PARAMETER_ENTRY_H
#define TEUCHOS_PARAMETER_ENTRY_H

/*! \file Teuchos_ParameterEntry.hpp
    \brief Object held as the "value" in the Teuchos::ParameterList std::map.
*/

#include "Teuchos_ConfigDefs.hpp"
#include "Teuchos_any.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterEntryValidator.hpp"

namespace Teuchos {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
class ParameterList; // another parameter type (forward declaration)
#endif

/*! \brief This object is held as the "value" in the Teuchos::ParameterList std::map.

    This structure holds a \c Teuchos::any value and information on the status of this
    parameter (isUsed, isDefault, etc.).  The type of parameter is chosen through the
    templated Set/Get methods.
*/
class ParameterEntry {
public:

  //! @name Constructors/Destructor 
  //@{

  //! Default Constructor
  ParameterEntry();
  
  //! Copy constructor
  ParameterEntry(const ParameterEntry& source);

  //! Templated constructor
  template<typename T>
  explicit ParameterEntry(
    T value, bool isDefault = false, bool isList = false,
    const std::string &docString = "",
    RCP<const ParameterEntryValidator> const& validator = null
    );

  //! Destructor
  ~ParameterEntry();

  //@}

  //! @name Set Methods 
  //@{

  //! Replace the current parameter entry with \c source.
  ParameterEntry& operator=(const ParameterEntry& source);

  /*! \brief Templated set method that uses the input value type to determine the type of parameter.  
      
      \note <ul>
	    <li> Invalidates any previous values stored by this object although it doesn't necessarily erase them.  
            <li> Resets 'isUsed' functionality.  
	    </ul>
  */
  template<typename T>
  void setValue(
    T value, bool isDefault = false,
    const std::string &docString = "",
    RCP<const ParameterEntryValidator> const& validator = null
    );

  /*! \brief Set the value as an any object.
  *
  * This wipes all other data including documentation strings.
  *
  * Warning! Do not use function ths to set a sublist!
  */
  void setAnyValue(
    const any &value, bool isDefault = false
    );

  /*! \brief Set the validator. */
  void setValidator(
    RCP<const ParameterEntryValidator> const& validator
    );

  /*! \brief Set the documentation std::string. */
  void setDocString(const std::string &docString);

  //! Create a parameter entry that is an empty list.
  ParameterList& setList(
    bool isDefault = false,
    const std::string &docString = ""
    );

  //@}

  //! @name Get Methods 
  //@{
   
  /*! \brief Templated get method that uses the input pointer type to determine the type of parameter to return.  

      \note This method will cast the value to the type requested.  If that type is incorrect, 
	    an std::exception will be thrown by the any_cast.
  */
  template<typename T>
  T& getValue(T *ptr) const;

  /*! \brief Direct access to the Teuchos::any data value underlying this
   *  object. The bool argument \c activeQry (default: true) indicates that the 
   *  call to getAny() will set the isUsed() value of the ParameterEntry to true.
   */
  any& getAny(bool activeQry = true);

  /*! \brief Constant direct access to the Teuchos::any data value underlying this
   *  object. The bool argument \c activeQry (default: true) indicates that the 
   *  call to getAny() will set the isUsed() value of the ParameterEntry to true.
   */
  const any& getAny(bool activeQry = true) const;

  //@}

  //! @name Attribute/Query Methods 
  //@{
  
  //! Return whether or not the value has been used; i.e., whether or not the value has been retrieved via a get function.
  bool isUsed() const;

  //! Return whether or not the value itself is a list.
  bool isList() const;

  //! Test the type of the data being contained.
  template <typename T>
  bool isType() const;

  //! Indicate whether this entry takes on the default value.
  bool isDefault() const;

  //! Return the (optional) documentation std::string
  std::string docString() const;

  //! Return the (optional) validator object
  RCP<const ParameterEntryValidator> validator() const;

  //@}

  //! @name I/O Methods 

  /*! \brief Output a non-list parameter to the given output stream.  

      The parameter is followed by "[default]" if it is the default value given through a 
      Set method.  Otherwise, if the parameter was unused (not accessed through a Get method), 
      it will be followed by "[unused]".  This function is called by the "std::ostream& operator<<". 
  */
  std::ostream& leftshift(std::ostream& os, bool printFlags = true) const;

  //@}

private:

  //! Reset the entry
  void reset();
  
  //! Templated Datatype
  any val_;

  //! Has this parameter been accessed by a "get" function?
  mutable bool isUsed_;

  //! Was this parameter a default value assigned by a "get" function?
  mutable bool isDefault_;

  //! Optional documentation field
  std::string  docString_;

  //! Optional validator object
  RCP<const ParameterEntryValidator> validator_;

};

/*! \relates ParameterEntry 
    \brief A templated helper function for returning the value of type \c T held in the ParameterEntry object,
    where the type \c T can be specified in the call.  This is an easier way to call the getValue method
    in the ParameterEntry class, since the user does not have to pass in a pointer of type \c T.
*/
template<typename T>
inline T& getValue( const ParameterEntry &entry )
{
  return entry.getValue((T*)NULL);
}

/*! \relates ParameterEntry 
    \brief Returns true if two ParameterEntry objects are equal.
*/
inline bool operator==(const ParameterEntry& e1, const ParameterEntry& e2) 
{ 
  return (
    e1.getAny() == e2.getAny()
    && e1.isList()== e2.isList()
    && e1.isUsed() == e2.isUsed()
    && e1.isDefault() == e2.isDefault()
    );
}

/*! \relates ParameterEntry 
    \brief Returns true if two ParameterEntry objects are <b>not</b> equal.
*/
inline bool operator!=(const ParameterEntry& e1, const ParameterEntry& e2) 
{ 
  return !( e1 == e2 );
}

/*! \relates ParameterEntry 
    \brief Output stream operator for handling the printing of parameter entries.  
*/
inline std::ostream& operator<<(std::ostream& os, const ParameterEntry& e) 
{ 
  return e.leftshift(os);
}

// ///////////////////////////////////////////
// Inline and Template Function Definitions

// Constructor/Destructor

template<typename T>
inline
ParameterEntry::ParameterEntry(
  T value, bool isDefault, bool isList
  ,const std::string &docString
  ,RCP<const ParameterEntryValidator> const& validator
  )
  : val_(value),
    isUsed_(false),
    isDefault_(isDefault),
    docString_(docString),
    validator_(validator)
{}

inline
ParameterEntry::~ParameterEntry()
{}

// Set Methods

template<typename T>
inline
void ParameterEntry::setValue(
  T value, bool isDefault, const std::string &docString
  ,RCP<const ParameterEntryValidator> const& validator
  )
{
  val_ = value;
  isDefault_ = isDefault;
  if(docString.length())
    docString_ = docString;
  if(validator.get())
    validator_ = validator;
}

// Get Methods

template<typename T>
inline
T& ParameterEntry::getValue(T *ptr) const
{
  isUsed_ = true;
  return const_cast<T&>(Teuchos::any_cast<T>( val_ ));
}

inline
any& ParameterEntry::getAny(bool activeQry)
{ 
  if (activeQry == true) {
    isUsed_ = true;
  }
  return val_; 
}

inline
const any& ParameterEntry::getAny(bool activeQry) const
{ 
  if (activeQry == true) {
    isUsed_ = true;
  }
  return val_; 
}

// Attribute Methods

inline
bool ParameterEntry::isUsed() const
{ return isUsed_; }

template <typename T>
inline
bool ParameterEntry::isType() const
{
  bool match = ( val_.type() == typeid(T) );
#ifdef HAVE_SHARED
  // For shared libraries, the above may resolve as false because one
  // of the types is not fully qualified.  The following check will
  // compensate for that.
  if ( !match )
    match = ( strcmp(val_.type().name(), typeid(T).name()) == 0 );
#endif
  return match;
}

inline
bool ParameterEntry::isDefault() const
{ return isDefault_; }

inline
std::string ParameterEntry::docString() const
{ return docString_; }

inline
RCP<const ParameterEntryValidator>
ParameterEntry::validator() const
{ return validator_; }

} // namespace Teuchos

#endif

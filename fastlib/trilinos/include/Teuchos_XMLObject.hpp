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

#ifndef Teuchos_XMLOBJECT_H
#define Teuchos_XMLOBJECT_H

/*! \file Teuchos_XMLObject.hpp
    \brief An object representation of a subset of XML data
*/

#include "Teuchos_XMLObjectImplem.hpp"
#include "Teuchos_Utils.hpp"

namespace Teuchos
{

//! Thrown when attempting to parse an empty XML std::string.
class EmptyXMLError : public std::runtime_error
{public: EmptyXMLError(const std::string& what_arg) : std::runtime_error(what_arg) {}};

/** \ingroup XML 
 * \brief Representation of an XML data tree. XMLObject is a ref-counted
 * handle to a XMLObjectImplem object, allowing storage by reference.
 */
class XMLObject
{
public:

  //! @name Constructors 
  //@{

  //! Empty constructor
  XMLObject() : ptr_() {;}

  //! Construct using a node labeled by tag
  XMLObject(const std::string& tag);

  /**
   * \brief Construct with a pointer to the low-level representation. 
   *
   * This is used to allow construction of an XMLObject from the
   * XMLObjectImplem* return value of ExceptionBase::toXML().
   */
  XMLObject(XMLObjectImplem* ptr);
  //@}	

  //! @name Copy methods 
  //@{

  //! Make a deep copy of this object
  XMLObject deepCopy() const ;
  //@}

  //! @name Data Access methods 
  //@{

  //! Return the tag of the current node
  const std::string& getTag() const {return ptr_->getTag();}

  //! Find out if the current node has an attribute of the specified name
  bool hasAttribute(const std::string& name) const 
    {return ptr_->hasAttribute(name);}

  //! Return the value of the attribute with the specified name
  const std::string& getAttribute(const std::string& name) const 
    {return ptr_->getAttribute(name);}

  //! Get an attribute, throwing an std::exception if it is not found
  const std::string& getRequired(const std::string& name) const ;

  //! Get a required attribute, returning it as a double
  double getRequiredDouble(const std::string& name) const 
    {return std::atof(getRequired(name).c_str());}

  //! Get a required attribute, returning it as an int
  int getRequiredInt(const std::string& name) const 
    {return std::atoi(getRequired(name).c_str());}

  //! Get a required attribute, returning it as a bool
  bool getRequiredBool(const std::string& name) const ;


  /** \brief Get an attribute, assigning a default value if the requested
   * attribute does not exist */
  std::string getWithDefault(const std::string& name, 
                        const std::string& defaultValue) const ;

  //! Return the number of child nodes owned by this node
  int numChildren() const {return ptr_->numChildren();}

  //! Return the i-th child node 
  const XMLObject& getChild(int i) const {return ptr_->getChild(i);}

  //! Return the number of lines of character content stored in this node 
  int numContentLines() const {return ptr_->numContentLines();}

  //! Return the i-th line of character content stored in this node
  const std::string& getContentLine(int i) const {return ptr_->getContentLine(i);}

  //! Represent this node and its children as a std::string
  std::string toString() const {return ptr_->toString();}

  //! Print this node and its children to stream with the given indentation
  void print(std::ostream& os, int indent) const
    {ptr_->print(os, indent);}


  //! Write the header for this object to a std::string
  std::string header() const {return ptr_->header();}

  //! Write the header for this object to a std::string
  std::string terminatedHeader() const {return ptr_->terminatedHeader();}

  //! Write the footer for this object to a std::string
  std::string footer() const {return ptr_->footer();}

  //! Find out if a node is empty
  bool isEmpty() const { return ptr_.get()==0;}

  //! Check that a tag is equal to an expected std::string
  void checkTag(const std::string& expected) const ;
  //@}
	
  //! @name Tree-Assembly methods 
  //@{

  //! Add an attribute to the current node's atribute list
  void addAttribute(const std::string& name, const std::string& value)
    {ptr_->addAttribute(name, value);}
			
  //! Add a double as an attribute
  void addDouble(const std::string& name, double val)
    {addAttribute(name, Teuchos::toString(val));}

  //! Add an int as an attribute
  void addInt(const std::string& name, int val)
    {addAttribute(name, Teuchos::toString(val));}

  //! Add a bool as an attribute
  void addBool(const std::string& name, bool val)
    {addAttribute(name, Teuchos::toString(val));}
			
  //! Add a child node to the node
  void addChild(const XMLObject& child)
    {ptr_->addChild(child);}

  //! Add a line of character content
  void addContent(const std::string& contentLine)
    {ptr_->addContent(contentLine);}
  //@}
	
private:
  RCP<XMLObjectImplem> ptr_;
};

/** \relates XMLObject 
    \brief Write XMLObject to \c os stream 
*/
inline std::ostream& operator<<(std::ostream& os, const XMLObject& xml)
{
  xml.print(os, 0);
  return os;
}

/** \relates XMLObject 
    \brief Write XMLObject to std::string 
*/
inline std::string toString(const XMLObject& xml)
{
  return xml.toString();
}

} // namespace Teuchos

#endif


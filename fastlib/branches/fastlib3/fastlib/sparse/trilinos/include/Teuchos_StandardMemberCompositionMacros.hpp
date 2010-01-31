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

// ////////////////////////////////////////////////////////////
// Teuchos_StandardMemberCompositionMacros.hpp

#ifndef TEUCHOS_STANDARD_MEMBER_COMPOSITION_MACROS_H
#define TEUCHOS_STANDARD_MEMBER_COMPOSITION_MACROS_H

/*! \file Teuchos_StandardMemberCompositionMacros.hpp \brief Macro
	that adds <<std member comp>> members as attribute members for any
	class.
*/
#include "Teuchos_ConfigDefs.hpp"

/** \brief Macro that adds <<std member comp>> attributes to any class
 * 
 * For example, if you want to include a <<std member comp>> attribute
 * as a member object of type MyClass with the name my_attribute you
 * would include the macro in the public section of YourClass
 * declaration as follows:
 \verbatim

  class YourClass {
  public:
    STANDARD_MEMBER_COMPOSITION_MEMBERS( MyClass, my_attribute );
  };
 \endverbatim
 * This macro adds the following data member to the class declaration:
 \verbatim
	private:
		MyClass my_attribute_;
 \endverbatim
 * and the following methods to your class declaration:
 \verbatim
  public:
  void my_attribute( const My_Class & my_attribute )
    { my_attribute_ = my_attribute; }
  const My_Class& my_attribute() const
    { return my_attribute_; }
 \endverbatim
 * The advantage of using this type of declaration is that it saves
 * you a lot of typing and space.  Later if you need to override these
 * operations you can just implement the member functions by hand.
 */
#define STANDARD_MEMBER_COMPOSITION_MEMBERS( TYPE, NAME )\
  void NAME ( const TYPE & NAME ) { NAME ## _ = NAME ; }\
  const TYPE& NAME() const { return NAME ## _; }\
private:\
  TYPE NAME ## _;\
public: \
  typedef ::Teuchos::DummyDummyClass NAME ## DummyDummyClass_t

// Note: Above, the 'using Teuchos::DummyDummyClass' statement is just there
// to allow (and require) the macro use of the form:
//
//  STANDARD_MEMBER_COMPOSITION_MEMBERS( MyClass, my_attribute );
//
// which allows a semicolon at the end!
//

#endif	// TEUCHOS_STANDARD_MEMBER_COMPOSITION_MACROS_H

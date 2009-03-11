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

#ifndef TEUCHOS_STANDARD_COMPOSITION_MACROS_HPP
#define TEUCHOS_STANDARD_COMPOSITION_MACROS_HPP

/*! \file Teuchos_StandardCompositionMacros.hpp \brief Macro that adds
	<<std comp>> members as attribute members for any class.
*/

#include "Teuchos_RCP.hpp"

/** \brief Macro that adds <<std comp>> members for a composition association.
 *
 * This form is for when the object being held will have const attributes
 * the same as the <tt>this</tt> object.
 *
 * For example, if you want to include a <<std comp>> association
 * with an non-const object of type MyClass of the name my_object you
 * would include the macro in the public section of YourClass
 * declaration as follows:
 *
 \verbatim
 class YourClass {
 public:
   STANDARD_COMPOSITION_MEMBERS( MyClass, my_object );
 };
 \endverbatim
 *
 * Note that the macro adds the following data member
 * to the class declaration:<br>
 \verbatim
 private:
   Teuchos::RCP< TYPE > NAME_;		
 \endverbatim
 *
 * \ingroup StandardContainmentMacros_grp
 */
#define STANDARD_COMPOSITION_MEMBERS( TYPE, NAME ) \
	void set_ ## NAME (const Teuchos::RCP< TYPE >& NAME ) \
	{	NAME ## _ = NAME ; } \
	Teuchos::RCP< TYPE > get_ ## NAME() const \
	{	return NAME ## _; } \
	TYPE& NAME() \
	{	return *NAME ## _; } \
	const TYPE& NAME() const \
	{	return *NAME ## _; } \
private: \
	Teuchos::RCP< TYPE > NAME ## _; \
public: \
  typedef Teuchos::RCP< TYPE > NAME ## _ptr_t

/** \breif Macro that adds <<std comp>> members for a composition association.
 *
 * This form is for when the object being held will have non-const attributes
 * irrespective of the const of <tt>this</tt>.
 *
 * For example, if you want to include a <<std comp>> association
 * with an non-const object of type MyClass of the name my_object you
 * would include the macro in the public section of YourClass
 * declaration as follows:
 *
 \verbatim
 class YourClass {
 public:
   STANDARD_NONCONST_COMPOSITION_MEMBERS( MyClass, my_object );
 };
 \endverbatim
 *
 * Note that the macro adds the following data member
 * to the class declaration:<br>
 \verbatim
 private:
   Teuchos::RCP< TYPE > NAME_;		
 \endverbatim
 *
 * \ingroup StandardContainmentMacros_grp
 */
#define STANDARD_NONCONST_COMPOSITION_MEMBERS( TYPE, NAME ) \
	void set_ ## NAME ( const Teuchos::RCP< TYPE >& NAME ) \
	{	NAME ## _ = NAME ; } \
	Teuchos::RCP< TYPE > get_ ## NAME() const \
	{	return NAME ## _; } \
	TYPE& NAME() const \
	{	return *NAME ## _; } \
private: \
	Teuchos::RCP< TYPE > NAME ## _; \
public: \
  typedef Teuchos::RCP< TYPE > NAME ## _ptr_t

/** \brief Macro that adds <<std comp>> members for a composition association
 * where the contained object is always constant.
 *
 * This form is for when the object being held will have const attributes
 * irrespective of the const of <tt>this</tt>.
 *
 * For example, if you want to include a <<std comp>> association
 * with a const object of type MyClass of the name my_object you
 * would include the macro in the public section of YourClass
 * declaration as follows:
 *
 \verbatim
 class YourClass {
 public:
   STANDARD_CONST_COMPOSITION_MEMBERS( MyClass, my_object );
 };
 \endverbatim
 *
 * Note that the macro adds the following data member
 * to the class declaration:<br>
 \verbatim
 private:
   NAME_ptr_t NAME_;		
 \endverbatim
 *
 * \ingroup StandardContainmentMacros_grp
 */
#define STANDARD_CONST_COMPOSITION_MEMBERS( TYPE, NAME ) \
public: \
	void set_ ## NAME ( const Teuchos::RCP< const TYPE >& NAME ) \
	{	NAME ## _ = NAME ; } \
	Teuchos::RCP< const TYPE > get_ ## NAME() const \
	{	return NAME ## _; } \
	const TYPE& NAME() const \
	{	return *NAME ## _; } \
private: \
	Teuchos::RCP< const TYPE > NAME ## _; \
public: \
  typedef Teuchos::RCP< const TYPE > NAME ## _ptr_t

#endif	// TEUCHOS_STANDARD_COMPOSITION_MACROS_HPP

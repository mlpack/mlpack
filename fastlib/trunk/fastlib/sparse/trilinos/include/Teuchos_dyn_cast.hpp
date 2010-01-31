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

#ifndef TEUCHOS_DYN_CAST_HPP
#define TEUCHOS_DYN_CAST_HPP

#include "Teuchos_TypeNameTraits.hpp"

namespace Teuchos {

/** \brief Exception class for bad cast.

\ingroup teuchos_language_support_grp

We create this class so that we may throw a bad_cast when appropriate and
still use the TEST_FOR_EXCEPTION macro.  We recommend users try to catch a
bad_cast.
*/
class m_bad_cast : public std::bad_cast {
	std::string msg;
public:
	explicit m_bad_cast(const std::string&  what_arg ) : msg(what_arg) {}
	virtual ~m_bad_cast() throw() {}
	virtual const char* what() const throw() { return msg.data(); }
};

// Throw the std::exception <tt>std::invalid_argument</tt> for below functions
void dyn_cast_throw_exception(
  const std::string   &type_from_name
  ,const std::string  &type_from_concr_name
  ,const std::string  &type_to_name
  );

/** \brief Dynamic casting utility function meant to replace
 * <tt>dynamic_cast<T&></tt> by throwing a better documented error
 * message.
 *
 * \ingroup teuchos_language_support_grp
 *
 * Existing uses of the built-in <tt>dynamic_cast<T&>()</tt> operator
 * such as:
 
 \code
 C &c = dynamic_cast<C&>(a);
 \endcode

 * are easily replaced as:

 \code
 C &c = dyn_cast<C>(a);
 \endcode

 * and that is it.  One could write a perl script to do this
 * automatically.
 *
 * This utility function is designed to cast an object reference of
 * type <tt>T_From</tt> to type <tt>T_To</tt> and if the cast fails at
 * runtime then an std::exception (derived from <tt>std::bad_cast</tt>) is
 * thrown that contains a very good error message.
 *
 * Consider the following class hierarchy:

 \code
 class A {};
 class B : public A {};
 class C : public A {};
 \endcode
 *
 * Now consider the following program:
 \code
  int main( int argc, char* argv[] ) {
    B b;
    A &a = b;
    try {
      std::cout << "\nTrying: dynamic_cast<C&>(a);\n";
      dynamic_cast<C&>(a);
    }
    catch( const std::bad_cast &e ) {
      std::cout << "\nCaught std::bad_cast std::exception e where e.what() = \"" << e.what() << "\"\n";
    }
    try {
      std::cout << "\nTrying: Teuchos::dyn_cast<C>(a);\n";
      Teuchos::dyn_cast<C>(a);
    }
    catch( const std::bad_cast &e ) {
      std::cout << "\nCaught std::bad_cast std::exception e where e.what() = \"" << e.what() << "\"\n";
    }
  	return 0;
  }
 \endcode
 
 * The above program will print something that looks like (compiled
 * with g++ for example):

 \verbatim

  Trying: dynamic_cast<C&>(a);

  Caught std::bad_cast std::exception e where e.what() = "St8bad_cast"

  Trying: Teuchos::dyn_cast<C>(a);

  Caught std::bad_cast std::exception e where e.what() = "../../../../packages/teuchos/src/Teuchos_dyn_cast.cpp:46: true:
  dyn_cast<1C>(1A) : Error, the object with the concrete type '1B' (passed in through the interface type '1A')  does
  not support the interface '1C' and the dynamic cast failed!"

 \endverbatim
 
 * The above program shows that the standard implementation of
 * <tt>dynamic_cast<T&>()</tt> does not return any useful debugging
 * information at all but the templated function
 * <tt>Teuchos::dyn_cast<T>()</tt> returns all kinds of useful
 * information.  The generated error message gives the type of the
 * interface that the object was passed in as (i.e. <tt>A</tt>), what
 * the actual concrete type of the object is (i.e. <tt>B</tt>) and
 * what type is trying to be dynamically casted to (i.e. <tt>C</tt>).
 * This type of information is extremely valuable when trying to track
 * down these type of runtime dynamic casting errors.  In some cases
 * (such as with <tt>gdb</tt>), debuggers do not even give the type of
 * concrete object so this function is very important on these
 * platforms.  In many cases, a debugger does not even need to be
 * opened to diagnose what the problem is and how to fix it.
 *
 * Note that this function is inlined and does not incur any
 * significant runtime performance penalty over the raw
 * <tt>dynamic_cast<T&>()</tt> operator.
 */
template <class T_To, class T_From>
inline
T_To& dyn_cast(T_From &from)
{
  T_To *to_ = dynamic_cast<T_To*>(&from);
  if(!to_)
    dyn_cast_throw_exception(
      TypeNameTraits<T_From>::name()
      ,typeName(from)
      ,TypeNameTraits<T_To>::name()
      );
  return *to_;
}

} // namespace Teuchos

#endif // TEUCHOS_DYN_CAST_HPP

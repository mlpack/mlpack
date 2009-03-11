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

#ifndef _TEUCHOS_TYPE_NAME_TRAITS_HPP_
#define _TEUCHOS_TYPE_NAME_TRAITS_HPP_

/*! \file Teuchos_TypeNameTraits.hpp
 \brief Defines basic traits returning the
    name of a type in a portable and readable way.
*/

#include "Teuchos_ConfigDefs.hpp"

#if defined(HAVE_GCC_ABI_DEMANGLE) && defined(HAVE_TEUCHOS_DEMANGLE)
#  include <cxxabi.h>
#endif

#ifdef HAVE_TEUCHOS_ARPREC
#include "mp/mpreal.h"
#endif

#ifdef HAVE_TEUCHOS_GNU_MP
#include "gmp.h"
#include "gmpxx.h"
#endif


namespace  Teuchos {


/** \brief Demangle a C++ name if valid.
 *
 * The name must have come from <tt>typeid(...).name()</tt> in order to be
 * valid name to pass to this function.
 *
 * \ingroup teuchos_language_support_grp
 */
std::string demangleName( const std::string &mangledName );


/** \brief Template function for returning the demangled name of an object.
 *
 * \ingroup teuchos_language_support_grp
 */
template<typename T>
std::string typeName( const T &t )
{
  return demangleName(typeid(t).name());
}

/** \brief Default traits class that just returns
 * <tt>typeid(T).name()</tt>.
 *
 * \ingroup teuchos_language_support_grp
 */
template<typename T>
class TypeNameTraits {
public:
  static std::string name()
    {
      return demangleName(typeid(T).name());
    }
};


#define TEUCHOS_TYPE_NAME_TRAITS_BUILTIN_TYPE_SPECIALIZATION(TYPE) \
template<> \
class TypeNameTraits<TYPE> { \
public: \
  static std::string name() { return (#TYPE); } \
} \

TEUCHOS_TYPE_NAME_TRAITS_BUILTIN_TYPE_SPECIALIZATION(bool);
TEUCHOS_TYPE_NAME_TRAITS_BUILTIN_TYPE_SPECIALIZATION(char);
TEUCHOS_TYPE_NAME_TRAITS_BUILTIN_TYPE_SPECIALIZATION(int);
TEUCHOS_TYPE_NAME_TRAITS_BUILTIN_TYPE_SPECIALIZATION(short int);
TEUCHOS_TYPE_NAME_TRAITS_BUILTIN_TYPE_SPECIALIZATION(long int);
TEUCHOS_TYPE_NAME_TRAITS_BUILTIN_TYPE_SPECIALIZATION(float);
TEUCHOS_TYPE_NAME_TRAITS_BUILTIN_TYPE_SPECIALIZATION(double);


template<typename T>
class TypeNameTraits<T*> {
public:
  static std::string name() { return TypeNameTraits<T>::name() + "*"; }
};


template<>
class TypeNameTraits<std::string> {
public:
  static std::string name() { return "string"; }
};


template<typename T>
class TypeNameTraits<std::vector<T> > {
public:
  static std::string name() { return "vector<"+TypeNameTraits<T>::name()+">"; }
};


#if defined(HAVE_COMPLEX) && defined(HAVE_TEUCHOS_COMPLEX)


template<typename T>
class TypeNameTraits<std::complex<T> > {
public:
  static std::string name() { return "complex<"+TypeNameTraits<T>::name()+">"; }
};


#endif // defined(HAVE_COMPLEX) && defined(HAVE_TEUCHOS_COMPLEX)
 

} // namespace Teuchos


#endif // _TEUCHOS_TYPE_NAME_TRAITS_HPP_

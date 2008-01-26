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

#ifndef TEUCHOS_AS_HPP
#define TEUCHOS_AS_HPP

#include "Teuchos_TestForException.hpp"


namespace Teuchos {


/** \brief Default traits class for all conversions of value types.
 *
 * This class should never be called directly by clients.  Instead, use the
 * <tt>as()</tt> and <tt>asSafe()</tt> template functions.
 *
 * This default traits class simply does an implicit type conversion.
 * Therefore, any conversions that are built into the language and are safe do
 * not need a traits class specialization and should not generate any compiler
 * warnings.  For example, the conversions <tt>float</tt> to <tt>double</tt>,
 * <tt>short type</tt> to <tt>type</tt>, <tt>type</tt> to <tt>long type</tt>,
 * and an enum value to <tt>int</tt> are all always value preserving and
 * should never result in a compiler warning or any aberrant runtime behavior.
 *
 * All other conversions that cause compiler warnings and/or could result in
 * aberrant runtime behavior (e.g. <tt>type</tt> to and from <tt>unsigned
 * type</tt>, to and from floating point and integral types, etc.), or do not
 * have compiler defined conversions (e.g. <tt>std::string</tt> to
 * <tt>int</tt>, <tt>double</tt> etc.) should be given specializations of this
 * class template.  If an unsafe or non-supported conversion is requested by
 * a client (i.e. through <tt>as()</tt> or <tt>asSafe()</tt>) then this
 * default traits class will be instantiated and the compiler will either
 * generate a warning message (if the conversion is supported but is unsafe)
 * or will not compile the code (if the conversion is not supported by default
 * in C++).  When this happens, a specialization can be added or the client
 * code can be changed to avoid the conversion.
 *
 * \ingroup teuchos_language_support_grp
 */
template<class TypeTo, class TypeFrom>
class ValueTypeConversionTraits {
public:
  static TypeTo convert( const TypeFrom t )
    {
      return t;
      // This default implementation is just an implicit conversion and will
      // generate compiler warning on dangerous conversions.
    }
  static TypeTo safeConvert( const TypeFrom t )
    {
      return t;
      // This default implementation is just an implicit conversion and will
      // generate compiler warning on dangerous conversions.  No checking can
      // be done by default; only specializations can define meaningful and
      // portable runtime checks of conversions.
    }
};


/** \brief Perform an debug-enabled checked conversion from one value type
 * object to another.
 *
 * This function is used as:

 \code
    TypeTo myConversion( const TypeFrom& a )
    {
      return Teuchos::as<TypeTo>(a);
    }
 \endcode 

 * This is just an interface function what calls the traits class
 * <tt>ValueTypeConversionTraits</tt> to perform the actual conversion.  All
 * specializations of behavior is done through specializations of the
 * <tt>ValueTypeConversionTraits</tt> class (which should be done in the
 * <tt>Teuchos</tt> namespace).
 *
 * When debug checking is turned on (e.g. when the <tt>TEUCHOS_DEBUG</tt>
 * macro is defined by the <tt>--enable-teuchos-debug</tt> configure option),
 * then the checked conversion function
 * <tt>ValueTypeConversionTraits<TypeTo,TypeFrom>::safeConvert(t)</tt> is
 * called.  When debug checking is not turned on, the unchecked
 * <tt>ValueTypeConversionTraits<TypeTo,TypeFrom>::convert(t)</tt> function is
 * called.
 *
 * For cases where the checking should always be done (i.e. to validate user
 * data), use the <tt>asSafe()</tt> version of this function.
 *
 * \ingroup teuchos_language_support_grp
 */
template<class TypeTo, class TypeFrom>
inline TypeTo as( const TypeFrom& t )
{
#ifdef TEUCHOS_DEBUG
  return ValueTypeConversionTraits<TypeTo,TypeFrom>::safeConvert(t);
#else
  return ValueTypeConversionTraits<TypeTo,TypeFrom>::convert(t);
#endif
}


/** \brief Perform an always checked conversion from one value type object to
 * another.
 *
 * This function is used as:

 \code
    TypeTo mySafeConversion( const TypeFrom& a )
    {
      return Teuchos::asSafe<TypeTo>(a);
    }
 \endcode 

 * This is just an interface function what calls the traits class
 * <tt>ValueTypeConversionTraits</tt> to perform the actual conversion.  All
 * specializations of behavior is done through specializations of the
 * <tt>ValueTypeConversionTraits</tt> class (which should be done in the
 * <tt>Teuchos</tt> namespace).
 *
 * This function always calls
 * <tt>ValueTypeConversionTraits<TypeTo,TypeFrom>::safeConvert(t)</tt>
 * independent of whether <tt>TEUCHOS_DEBUG</tt> is defined or not, which
 * ensures that the conversions are always runtime checked and therefore well
 * defined.
 *
 * For cases where the checking should only be done in a debug build, use the
 * the <tt>as()</tt> version of this function.
 *
 * \ingroup teuchos_language_support_grp
 */
template<class TypeTo, class TypeFrom>
inline TypeTo asSafe( const TypeFrom& t )
{
  return ValueTypeConversionTraits<TypeTo,TypeFrom>::safeConvert(t);
}


//
// Standard specializations of ValueTypeConversionTraits
//


// ToDo: Add specializations as needed!


} // end namespace Teuchos


#endif // TEUCHOS_AS_HPP

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

// ////////////////////////////////////////////////////////////////////////
// Teuchos_PrimitiveTypeTraits.hpp

#ifndef TEUCHOS_PRIMITIVE_TYPE_TRAITS_H
#define TEUCHOS_PRIMITIVE_TYPE_TRAITS_H

#include "Teuchos_TestForException.hpp"
#ifdef HAVE_TEUCHOS_GNU_MP
#include "gmp.h"
#include "gmpxx.h"
#endif

/** \file Teuchos_PrimitiveTypeTraits.hpp
	\brief Declaration of a templated traits class for decomposing an
		object into an array of primitive objects
 */

namespace Teuchos {

/** \brief A templated traits class for decomposing object into an
 * array of primitive objects.
 *
 * The idea behind this traits class it that it allows an object of
 * semi-std::complex structure to be externalized into an array of
 * primitive data types.
 *
 * This default traits class works just fine for types that are
 * already primitive.
 */
template <class T> class PrimitiveTypeTraits {
public:
  /** \brief . */
  typedef T  primitiveType;
  /** \brief . */
  static int numPrimitiveObjs() { return 1; }
  /** \brief . */
  static void extractPrimitiveObjs(
    const T                &obj
    ,const int             numPrimitiveObjs
    ,primitiveType         primitiveObjs[]
    )
    {
#ifdef TEUCHOS_DEBUG
      TEST_FOR_EXCEPTION( numPrimitiveObjs!=1 || primitiveObjs==NULL, std::invalid_argument, "Error!" );
#endif
      primitiveObjs[0] = obj;
    }
  /** \brief . */
  static void loadPrimitiveObjs(
    const int              numPrimitiveObjs
    ,const primitiveType   primitiveObjs[]
    ,T                     *obj
    )
    {
#ifdef TEUCHOS_DEBUG
      TEST_FOR_EXCEPTION( numPrimitiveObjs!=1 || primitiveObjs==NULL, std::invalid_argument, "Error!" );
#endif
      *obj = primitiveObjs[0];
    }
};

#if defined(HAVE_COMPLEX) && defined(HAVE_TEUCHOS_COMPLEX)

/** \brief Partial specialization of <tt>PrimitiveTypeTraits</tt> for <tt>std::complex</tt>.
 */
template <class T> class PrimitiveTypeTraits< std::complex<T> > {
public:
  /** \brief . */
  typedef T  primitiveType;
  /** \brief . */
  static int numPrimitiveObjs() { return 2; }
  /** \brief . */
  static void extractPrimitiveObjs(
    const std::complex<T>  &obj
    ,const int             numPrimitiveObjs
    ,primitiveType         primitiveObjs[]
    )
    {
#ifdef TEUCHOS_DEBUG
      TEST_FOR_EXCEPTION( numPrimitiveObjs!=2 || primitiveObjs==NULL, std::invalid_argument, "Error!" );
#endif
      primitiveObjs[0] = obj.real();
      primitiveObjs[1] = obj.imag();
    }
  /** \brief . */
  static void loadPrimitiveObjs(
    const int              numPrimitiveObjs
    ,const primitiveType   primitiveObjs[]
    ,std::complex<T>       *obj
    )
    {
#ifdef TEUCHOS_DEBUG
      TEST_FOR_EXCEPTION( numPrimitiveObjs!=2 || primitiveObjs==NULL, std::invalid_argument, "Error!" );
#endif
      *obj = std::complex<T>( primitiveObjs[0], primitiveObjs[1] );
    }
};

#endif // defined(HAVE_COMPLEX) && defined(HAVE_TEUCHOS_COMPLEX)

#ifdef HAVE_TEUCHOS_GNU_MP

/** \brief Full specialization of <tt>PrimitiveTypeTraits</tt> for <tt>mpf_class</tt>.
 *
 * Note: This class is not complete yet!
 */
template <> class PrimitiveTypeTraits<mpf_class> {
public:
  /** \brief . */
  typedef double  primitiveType; // Just a guess!
  /** \brief . */
  static int numPrimitiveObjs() { return 10; } // Just a guess!
  /** \brief . */
  static void extractPrimitiveObjs(
    const mpf_class        &obj
    ,const int             numPrimitiveObjs
    ,primitiveType         primitiveObjs[]
    )
    {
#ifdef TEUCHOS_DEBUG
      TEST_FOR_EXCEPTION( numPrimitiveObjs!=10 || primitiveObjs==NULL, std::invalid_argument, "Error!" );
#endif
			TEST_FOR_EXCEPT(true); // ToDo: Implement
    }
  /** \brief . */
  static void loadPrimitiveObjs(
    const int              numPrimitiveObjs
    ,const primitiveType   primitiveObjs[]
    ,mpf_class             *obj
    )
    {
#ifdef TEUCHOS_DEBUG
      TEST_FOR_EXCEPTION( numPrimitiveObjs!=10 || primitiveObjs==NULL, std::invalid_argument, "Error!" );
#endif
			TEST_FOR_EXCEPT(true); // ToDo: Implement
    }
};

#endif // HAVE_TEUCHOS_GNU_MP


} // namespace Teuchos

#endif // TEUCHOS_PRIMITIVE_TYPE_TRAITS_H

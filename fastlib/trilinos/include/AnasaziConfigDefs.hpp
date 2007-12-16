// @HEADER
// ***********************************************************************
//
//                 Anasazi: Block Eigensolvers Package
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

/*! \file AnasaziConfigDefs.hpp
  \brief Anasazi header file which uses auto-configuration information to include
  necessary C++ headers
*/

#ifndef ANASAZI_CONFIGDEFS_HPP
#define ANASAZI_CONFIGDEFS_HPP

#ifndef __cplusplus
#define __cplusplus
#endif

#ifdef HAVE_CONFIG_H

/*
 * The macros PACKAGE, PACKAGE_NAME, etc, get defined for each package and need to
 * be undef'd here to avoid warnings when this file is included from another package.
 * KL 11/25/02
 */
#ifdef PACKAGE
#undef PACKAGE
#endif

#ifdef PACKAGE_NAME
#undef PACKAGE_NAME
#endif

#ifdef PACKAGE_BUGREPORT
#undef PACKAGE_BUGREPORT
#endif

#ifdef PACKAGE_STRING
#undef PACKAGE_STRING
#endif

#ifdef PACKAGE_TARNAME
#undef PACKAGE_TARNAME
#endif

#ifdef PACKAGE_VERSION
#undef PACKAGE_VERSION
#endif

#ifdef VERSION
#undef VERSION
#endif

#include <Anasazi_config.h>

#ifdef HAVE_MPI
#ifndef EPETRA_MPI
#define EPETRA_MPI
#endif
#endif

#ifdef HAVE_CSTDLIB
#include <cstdlib>
#else
#include <stdlib.h>
#endif

#ifdef HAVE_CSTDIO
#include <cstdio>
#else
#include <stdio.h>
#endif

#ifdef HAVE_STRING
#include <string>
#else
#include <string.h>
#endif

#ifdef HAVE_VECTOR
#include <vector>
#else
#include <vector.h>
#endif

#ifdef HAVE_NUMERIC
#include <numeric>
#else
#include <algo.h>
#endif

#ifdef HAVE_COMPLEX
#include <complex>
#else
#include <complex.h>
#endif

#ifdef HAVE_IOSTREAM
#include <iostream>
#else
#include <iostream.h>
#endif

#ifdef HAVE_ITERATOR
#include <iterator>
#else
#include <iterator.h>
#endif

#if HAVE_STDEXCEPT
#include <stdexcept>
#elif HAVE_STDEXCEPT_H
#include <stdexcept.h>
#endif

#ifndef JANUS_STLPORT
#ifdef HAVE_CMATH
#include <cmath>
#else
#include <math.h>
#endif
#else /* JANUS_STLPORT */
#include <math.h>
#endif /* JANUS_STLPORT */

#else /*HAVE_CONFIG_H is not defined*/

#include <iterator>
#include <iostream>
#include <string>

#if defined(SGI) || defined(SGI64) || defined(SGI32) || defined(CPLANT) || defined (TFLOP)

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#else

#include <cstdlib>
#include <cstdio>
#include <cmath>

#endif

#include <vector>
#include <map>
#include <deque>
#include <algorithm>
#include <numeric>

#endif /*HAVE_CONFIG_H*/

/* Define some macros */
#define ANASAZI_MAX(x,y) (( (x) > (y) ) ? (x)  : (y) )     /* max function  */
#define ANASAZI_MIN(x,y) (( (x) < (y) ) ? (x)  : (y) )     /* min function  */
#define ANASAZI_SGN(x)   (( (x) < 0.0 ) ? -1.0 : 1.0 )     /* sign function */

/*
 * Anasazi_Version() method 
 */
namespace Anasazi {
  std::string Anasazi_Version();
}

#endif /*ANASAZI_CONFIGDEFS_HPP*/

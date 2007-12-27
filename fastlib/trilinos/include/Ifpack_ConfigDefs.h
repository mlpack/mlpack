/*@HEADER
// ***********************************************************************
// 
//       Ifpack: Object-Oriented Algebraic Preconditioner Package
//                 Copyright (2002) Sandia Corporation
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
//@HEADER
*/

#ifndef _IFPACK_CONFIGDEFS_H_
#define _IFPACK_CONFIGDEFS_H_

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

#include <Ifpack_config.h>

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

#ifdef HAVE_MPI

#ifndef EPETRA_MPI
#define EPETRA_MPI
#endif

#endif

/******************************************************************************
 *   Choose header file flavor: either ANSI-style (no .h, e.g. <iostream>) or
 * old-style (with .h, e.g., <iostream.h>). 
 * KL 9/26/03
 *****************************************************************************/

#if HAVE_CSTDIO
#include <cstdio>
#elif HAVE_STDIO_H
#include <stdio.h>
#else
#error "Found neither cstdio nor stdio.h"
#endif

#if HAVE_STRING
#include <string>
#elif HAVE_STRING_H
#include <string.h>
#else
#error "Found neither string nor string.h"
#endif

#if HAVE_IOSTREAM
#include <iostream>
#elif HAVE_IOSTREAM_H
#include <iostream.h>
#else
#error "Found neither iostream nor iostream.h"
#endif

#ifdef HAVE_ALGORITHM
#include <algorithm>
#elif defined(HAVE_ALGO_H)
#include <algo.h>
#elif defined(HAVE_ALGORITHM_H)
#include <algorithm.h>
#else
#error "Did not find algorithm, algo.h or algorithm.h"
#endif

#if HAVE_VECTOR
#include <vector>
#elif HAVE_IOSTREAM_H
#include <vector.h>
#else
#error "Found neither vector nor vector.h"
#endif

#if defined(TFLOP)
#ifdef HAVE_STRING
using std::string;
#endif
#ifdef HAVE_IOSTREAM
using std::istream;
using std::ostream;
using std::cerr;
using std::cout;
using std::endl;
#endif
#else /* NOT TFLOP */
using namespace std;
#endif /* defined(TFLOP) */

#endif

// prints out an error message if variable is not zero,
// and returns this value.
#define IFPACK_CHK_ERR(ifpack_err) \
{ if (ifpack_err < 0) { \
  std::cerr << "IFPACK ERROR " << ifpack_err << ", " \
    << __FILE__ << ", line " << __LINE__ << std::endl; \
    return(ifpack_err);  } }

// prints out an error message if variable is not zero,
// and returns void
#define IFPACK_CHK_ERRV(ifpack_err) \
{ if (ifpack_err < 0) { \
  std::cerr << "IFPACK ERROR " << ifpack_err << ", " \
    << __FILE__ << ", line " << __LINE__ << std::endl; \
    return;  } }
// prints out an error message and returns
#define IFPACK_RETURN(ifpack_err) \
{ if (ifpack_err < 0) { \
  std::cerr << "IFPACK ERROR " << ifpack_err << ", " \
    << __FILE__ << ", line " << __LINE__ << std::endl; \
		       } return(ifpack_err); }

#define IFPACK_SGN(x) (((x) < 0.0) ? -1.0 : 1.0)  /* sign function */
#define IFPACK_ABS(x) (((x) > 0.0) ? (x) : (-x))  /* abs function */

#endif /*_IFPACK_CONFIGDEFS_H_*/

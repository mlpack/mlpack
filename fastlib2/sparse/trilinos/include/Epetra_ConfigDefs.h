
//@HEADER
/*
************************************************************************

              Epetra: Linear Algebra Services Package 
                Copyright (2001) Sandia Corporation

Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
license for use of this work by or on behalf of the U.S. Government.

This library is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation; either version 2.1 of the
License, or (at your option) any later version.
 
This library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.
 
You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
USA
Questions? Contact Michael A. Heroux (maherou@sandia.gov) 

************************************************************************
*/
//@HEADER

#ifndef EPETRA_CONFIGDEFS_H
#define EPETRA_CONFIGDEFS_H

#ifndef __cplusplus
#define __cplusplus
#endif

#define EPETRA_MAX(x,y) (( (x) > (y) ) ? x : y)     /* max function  */
#define EPETRA_MIN(x,y) (( (x) < (y) ) ? x : y)     /* min function  */
#define EPETRA_SGN(x) (((x) < 0.0) ? -1.0 : 1.0)  /* sign function */

const double Epetra_MinDouble = 1.0E-100;
const double Epetra_MaxDouble = 1.0E+100;
const double Epetra_Overflow = 1.79E308; // Used to test if equilibration should be done.
const double Epetra_Underflow = 2.23E-308;

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

#include "Epetra_config.h"

#ifdef HAVE_MPI
#ifndef EPETRA_MPI
#define EPETRA_MPI
#endif
#endif

#ifdef HAVE_CSTDLIB
#include <cstdlib>
#endif
#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif

#ifdef HAVE_CSTDIO
#include <cstdio>
using std::sprintf;
#endif
#ifdef HAVE_STDIO_H
#include <stdio.h>
#endif

#ifdef HAVE_CASSERT
#include <cassert>
#else
#include <assert.h>
#endif

#ifdef HAVE_STRING
#include <string>
#endif
#ifdef HAVE_STRING_H
#include <string.h>
#endif

#ifdef HAVE_IOSTREAM
#include <iostream>
#else
#include <iostream.h>
#endif

#if HAVE_SSTREAM
#include <sstream>
#endif

#ifdef HAVE_CMATH
#include <cmath>
using std::rand;
#endif
#ifdef HAVE_MATH_H
#include <math.h>
#endif

#ifdef HAVE_IOMANIP
#include <iomanip>
#else
#include <iomanip.h>
#endif

#ifdef HAVE_STRING
using std::string;
#endif

#ifdef HAVE_IOSTREAM
using std::istream;
using std::ostream;
using std::cerr;
using std::cout;
using std::endl;
using std::flush;
#endif

/*-----------------------------------------------------------------------
  Must refine the following up to #else HAVE_CONFIG_H is not defined
  -----------------------------------------------------------------------*/

#ifdef EPETRA_SIMULATE_BOOL
#ifdef bool
#undef bool
#endif
#ifdef true
#undef true
#endif
#ifdef false
#undef false
#endif

#define bool int
#define true 1
#define false 0

#endif

#ifndef HAVE_FORMAT_IO
const bool Epetra_FormatStdout = false; // Set true if the ostream << operator should format output
#else
const bool Epetra_FormatStdout = true;
#endif

// Define DefultTracebackMode (HAVE_WARNING_MESSAGES and HAVE_FATAL_MESSAGES can be defined
// via the configure script command line)

#ifdef HAVE_WARNING_MESSAGES
const int DefaultTracebackMode = 2;
#elif defined HAVE_FATAL_MESSAGES
const int DefaultTracebackMode = 1;
#else
const int DefaultTracebackMode = 0;
#endif

#else /*HAVE_CONFIG_H is not defined*/

#ifndef __cplusplus
#define __cplusplus
#endif

#if defined(SGI) || defined(SGI64) || defined(SGI32) || defined(CPLANT)

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <math.h>
#include <string>

using std::string;
using std::istream;
using std::ostream;
using std::cerr;
using std::cout;
using std::endl;
using std::flush;

#else

#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <iostream>
#include <cmath>
#include <string>

using std::string;
using std::istream;
using std::ostream;
using std::cerr;
using std::cout;
using std::endl;
using std::flush;

#endif



#ifdef EPETRA_SIMULATE_BOOL
#ifdef bool
#undef bool
#endif
#ifdef true
#undef true
#endif
#ifdef false
#undef false
#endif

#define bool int
#define true 1
#define false 0

#endif

const bool Epetra_FormatStdout = true; // Set true if the ostream << operator should format output
const int DefaultTracebackMode = 1;

#endif /*HAVE_CONFIG_H*/

// Delete any previous definition of EPETRA_NO_ERROR_REPORTS

#ifdef EPETRA_CHK_ERR
#undef EPETRA_CHK_ERR
#endif
#ifdef EPETRA_CHK_PTR
#undef EPETRA_CHK_PTR
#endif
#ifdef EPETRA_CHK_REF
#undef EPETRA_CHK_REF
#endif

// Great little macro obtained from Alan Williams (modified for dynamic switching on/off)

#define EPETRA_CHK_ERR(a) { { int epetra_err = a; \
                              if ((epetra_err < 0 && Epetra_Object::GetTracebackMode() > 0) || \
                                  (epetra_err > 0 && Epetra_Object::GetTracebackMode() > 1)) { \
                      Epetra_Object::GetTracebackStream() << "Epetra ERROR " << epetra_err << ", " \
                           << __FILE__ << ", line " << __LINE__ << endl; }\
                      if (epetra_err != 0) return(epetra_err);  }\
                   }

// Extension of same macro for pointer, returns zero if bad

#define EPETRA_CHK_PTR(a) { if (a == 0 && Epetra_Object::GetTracebackMode() > 0) { \
                      Epetra_Object::GetTracebackStream() << "Epetra returning zero pointer " << ", " \
                           << __FILE__ << ", line " << __LINE__ << endl; } \
                      return(a); \
                   }
// Extension of same macro for reference, returns a default reference

#define EPETRA_CHK_REF(a) { if (Epetra_Object::GetTracebackMode() > 0) {\
                      Epetra_Object::GetTracebackStream() << "Epetra returning default reference " << ", " \
                           << __FILE__ << ", line " << __LINE__ << endl; } \
                      return(a); \
                   }

#endif /* EPETRA_CONFIGDEFS_H */


/*@HEADER
// ***********************************************************************
//
//        AztecOO: An Object-Oriented Aztec Linear Solver Package
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

#ifndef AZTECOO_CONFIGDEFS_H
#define AZTECOO_CONFIGDEFS_H

#ifdef HAVE_CONFIG_H

/*
 * The macros PACKAGE, PACKAGE_NAME, etc, get defined for each package
 * and need to be undef'd here to avoid warnings when this file is
 * included from another package.  KL 11/25/02
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

#include "AztecOO_config.h"

#ifdef HAVE_MATH_H
#include <math.h>
#endif

#ifdef __cplusplus

#ifdef HAVE_IOSTREAM
#include <iostream>
#else
#error "must have <iostream>"
#endif

#ifdef HAVE_FSTREAM
#include <fstream>
#else
#error "must have <fstream>"
#endif

#ifdef HAVE_SSTREAM
#include <sstream>
#else
#error "must have <sstream>"
#endif

#ifdef HAVE_STRING
#include <string>
#endif

#ifdef HAVE_CMATH
#include <cmath>
#endif

#ifdef HAVE_IOMANIP
#include <iomanip>
#else
#include <iomanip.h>
#endif

#endif /*__cplusplus */
#endif /*HAVE_CONFIG_H*/
#endif /* AZTECOO_CONFIGDEFS_H */

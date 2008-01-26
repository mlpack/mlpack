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

#ifndef _AZOO_printf_h_
#define _AZOO_printf_h_

#ifdef __cplusplus
extern "C" {
#endif

/*********************************************************************/
/* Declarations for functions that will be passed as function-pointers
   and called from C code in Aztec. These are not user-functions.
*/
extern int AZOO_printf_out(const char* format, ...);
extern void AZOO_flush_out();

extern int AZOO_printf_err(const char* format, ...);
/*********************************************************************/

#ifdef __cplusplus
}

#include <iostream>

/** C++ users can use this function to set a destination stream
   for the output that Aztec would normally send to stdout.
*/
void AZOO_set_stream_out(std::ostream& ostrm);

/** C++ users can use this function to set a destination stream
   for the output that Aztec would normally send to stderr.
*/
void AZOO_set_stream_err(std::ostream& ostrm);
#endif

#endif


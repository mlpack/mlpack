/* @HEADER
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
*/

#ifndef TEUCHOS_EXIT_H
#define TEUCHOS_EXIT_H

/*! \file Teuchos_exit.h \brief Macros and helper functions for replacing raw
calls to <tt>exit(int)</tt> in C and C++ code.
*/

#include "Teuchos_ConfigDefs.hpp"

/*! \defgroup Teuchos_exit_grp Utility code for replacing calls to exit() with macros that result in thrown exceptions. 

\ingroup teuchos_language_support_grp
*/
/* @{ */

#ifdef __cplusplus
extern "C" {
#endif

/** \brief Function with C linkage that rases a C++ exception.
 *
 * \param  file_and_line  [in] Null-terminated string that gives the file name and line number where
 *                        the error occured.
 * \param  msg            [in] Null-terminated string that gives some extra message that will be
 *                        embedded in the thrown exception.
 * \param  error_code     The error code that would have been passed to 'exit(...)'
 *
 * <b>Note:</b> This function should not be called directly but instead by
 * using the macro <tt>TEUCHOS_EXIT()</tt> or <tt>TEUCHOS_MSG_EXIT()</tt>
 */
void Teuchos_exit_helper(
  char       file[]
  ,int       line
  ,char      msg[]
  ,int       error_code
  );

#ifdef __cplusplus
}
#endif

/** \brief Macro to replace call to <tt>exit(...)</tt>.
 *
 * This macro calls the function <tt>Teuchos_exit_helper()</tt> which the file
 * name and line number where this macro is used and results in a C++
 * exception to be thrown with a good error message.
 */
#define TEUCHOS_EXIT(ERROR_CODE) Teuchos_exit_helper( __FILE__, __LINE__, 0, ERROR_CODE )

/** \brief Macro to replace call to <tt>exit(...)</tt> and add a message string.
 *
 * This macro calls the function <tt>Teuchos_exit_helper()</tt> which the file
 * name and line number where this macro is used and results in a C++
 * exception to be thrown with a good error message.
 */
#define TEUCHOS_MSG_EXIT(MSG,ERROR_CODE) Teuchos_exit_helper(  __FILE__, __LINE__, MSG, ERROR_CODE )

/* @} */

#endif /* TEUCHOS_EXIT_H */

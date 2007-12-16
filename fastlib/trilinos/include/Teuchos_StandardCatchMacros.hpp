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

#ifndef TEUCHOS_STANDARD_CATCH_MACROS_HPP
#define TEUCHOS_STANDARD_CATCH_MACROS_HPP

#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_FancyOStream.hpp"
#include "Teuchos_TypeNameTraits.hpp"

/** \brief Simple macro that catches and reports standard exceptions and other exceptions.
 *
 * \ingroup teuchos_language_support_grp
 *
 * This macro should be used to write simple <tt>main()</tt> program functions
 * wrapped in a try statement as:

 \code

 int main(...)
 {
   bool verbose = true;
   bool success = true;
   try {
     ...
   }
   TEUCHOS_STANDARD_CATCH_STATEMENTS(verbose,std::cerr,success);
   return ( success ? 0 : 1 );
 }
 \endcode
 */
#define TEUCHOS_STANDARD_CATCH_STATEMENTS(VERBOSE,ERR_STREAM,SUCCESS_FLAG) \
  catch( const std::exception &excpt ) { \
    if((VERBOSE)) { \
      std::ostringstream oss; \
      oss \
        << "\np="<<::Teuchos::GlobalMPISession::getRank()<<": *** Caught standard std::exception of type \'" \
        <<Teuchos::typeName(excpt)<<"\' :\n\n"; \
        Teuchos::OSTab(oss).o() << excpt.what() << std::endl; \
        std::cout << std::flush; \
      (ERR_STREAM) << oss.str(); \
    (SUCCESS_FLAG) = false; \
    } \
  } \
  catch( const int &excpt_code ) { \
    if((VERBOSE)) { \
      std::ostringstream oss; \
      oss \
        << "\np="<<::Teuchos::GlobalMPISession::getRank() \
        << ": *** Caught an integer std::exception with value = " \
        << excpt_code << std::endl; \
        std::cout << std::flush; \
      (ERR_STREAM) << oss.str(); \
    (SUCCESS_FLAG) = false; \
    } \
  } \
  catch( ... ) { \
    if((VERBOSE)) { \
      std::ostringstream oss; \
      oss << "\np="<<::Teuchos::GlobalMPISession::getRank()<<": *** Caught an unknown exception\n"; \
      std::cout << std::flush; \
      (ERR_STREAM) << oss.str(); \
      (SUCCESS_FLAG) = false; \
    } \
  }

#endif // TEUCHOS_STANDARD_CATCH_MACROS_HPP

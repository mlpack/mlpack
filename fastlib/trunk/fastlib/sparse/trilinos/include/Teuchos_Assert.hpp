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

#ifndef TEUCHOS_ASSERT_HPP
#define TEUCHOS_ASSERT_HPP


#include "Teuchos_TestForException.hpp"


/** \brief This macro is throws when an assert fails.
 *
 * \note <tt>The std::exception</tt> thrown is <tt>std::logic_error</tt>.
 *
 * \ingroup TestForException_grp
 */
#define TEUCHOS_ASSERT(assertion_test) TEST_FOR_EXCEPT(!(assertion_test))


/** \brief This macro asserts that an integral number fallis in the range
 * <tt>[lower_inclusive,upper_exclusive)</tt>
 *
 * \note <tt>The std::exception</tt> thrown is <tt>std::logic_error</tt>.
 *
 * \ingroup TestForException_grp
 */
#define TEUCHOS_ASSERT_IN_RANGE_UPPER_EXCLUSIVE( index, lower_inclusive, upper_exclusive ) \
  { \
    TEST_FOR_EXCEPTION( \
      !( (lower_inclusive) <= (index) && (index) < (upper_exclusive) ), \
      std::out_of_range, \
      "Error, the index " #index " = " << (index) << " does not fall in the range" \
      "["<<(lower_inclusive)<<","<<(upper_exclusive)<<")!" ); \
  }


/** \brief This macro is checks that to numbers are equal and if not then
 * throws an exception with a good error message.
 *
 * \note The <tt>std::exception</tt> thrown is <tt>std::logic_error</tt>.
 *
 * \ingroup TestForException_grp
 */
#define TEUCHOS_ASSERT_EQUALITY( val1, val2 ) \
  { \
    TEST_FOR_EXCEPTION( \
      (val1) != (val2), std::out_of_range, \
      "Error, (" #val1 " = " << (val1) << ") != (" #val2 " = " << (val2) << ")!" ); \
  }


#endif // TEUCHOS_ASSERT_HPP

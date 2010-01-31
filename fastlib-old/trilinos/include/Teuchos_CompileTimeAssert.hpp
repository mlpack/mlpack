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

#ifndef TEUCHOS_COMPILE_TIME_ASSERT_HPP
#define TEUCHOS_COMPILE_TIME_ASSERT_HPP

/*! \file Teuchos_CompileTimeAssert.hpp
    \brief Template classes for testing assertions at compile time.
*/

#include "Teuchos_ConfigDefs.hpp"

namespace Teuchos {

/*! \defgroup CompileTimeAssert_grp  Template classes for testing assertions at compile time.
 \ingroup teuchos_language_support_grp
*/
///@{

/// If instantiated (for Test!=0) then this should not compile!
template <int Test>
class CompileTimeAssert {
	int compile_time_assert_failed[Test-1000]; // Should not compile if instantiated!
};

/// If instantiated (i.e. Test==0) then this will compile!
template <>
class CompileTimeAssert<0> {};

///@}

} // namespace Teuchos

#endif // TEUCHOS_COMPILE_TIME_ASSERT_HPP

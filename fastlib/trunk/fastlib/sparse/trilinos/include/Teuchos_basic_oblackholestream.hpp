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

#ifndef TEUCHOS_BASIC_O_BLACK_HOLE_STREAM_H
#define TEUCHOS_BASIC_O_BLACK_HOLE_STREAM_H

#include "Teuchos_ConfigDefs.hpp"

namespace Teuchos {

/** \brief <tt>basic_ostream<></tt> subclass that does nothing but discard output.
 *
 * \ingroup teuchos_outputting_grp
 *
 * Use the class anytime you must pass an <tt>basic_ostream<></tt> object
 * but don't want the output for any reason.
 *
 * This subclass just sets the stream buffer to NULL and that is all you need to do!
 */
template<typename _CharT, typename _Traits>
class basic_oblackholestream
	: virtual public std::basic_ostream<_CharT, _Traits>
{
public:
  /** \brief . */
	explicit basic_oblackholestream() : std::basic_ostream<_CharT, _Traits>(NULL) {}
}; // end class basic_oblackholestream

} // end namespace Teuchos

#endif // TEUCHOS_BASIC_O_BLACK_HOLE_STREAM_H

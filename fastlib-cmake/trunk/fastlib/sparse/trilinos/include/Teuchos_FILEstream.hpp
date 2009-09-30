// @HEADER
// ***********************************************************************
// 
//                    Teuchos: Common Tools Package
//                 Copyright (2006) Sandia Corporation
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

#ifndef TEUCHOS_FILESTREAM_H
#define TEUCHOS_FILESTREAM_H

//! \file Teuchos_FILEstream.hpp

#include <streambuf>

namespace Teuchos
{

  //! Teuchos::FILEstream: Combined C FILE and C++ stream

  /*! Teuchos::FILEstream is a class that defines an object that is
      simultaneously a C FILE object and a C++ stream object.  The
      utility of this class is in connecting existing C++ code that
      uses streams and C code that uses FILEs.  An important example
      of this situation is the python wrappers for Trilinos packages.
      Trilinos is of course written primarily in C++, but the python
      wrappers must interface to the python C API.  Wrappers for
      Trilinos methods or operators that expect a stream can be given
      a Teuchos::FILEstream, which then behaves as a FILE within the
      python C API.  This is a low-level object that should not be
      needed at the user level.
   */

  class FILEstream : public std::streambuf {

  public:

    //! Constructor

    /*! The only constructor for Teuchos:FILEstream, and it requires a
        pointer to a C FILE struct.
     */
    FILEstream(std::FILE* file): self_file(file) {}

  protected:

    std::streambuf::int_type overflow(std::streambuf::int_type c) {
      return std::fputc(c, self_file) == EOF?
	std::streambuf::traits_type::eof(): c;
    }

    FILE* self_file;
  };
}

#endif

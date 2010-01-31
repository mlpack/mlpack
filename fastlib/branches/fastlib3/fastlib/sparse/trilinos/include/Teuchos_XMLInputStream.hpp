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

#ifndef TEUCHOS_XMLINPUTSTREAM_H
#define TEUCHOS_XMLINPUTSTREAM_H

/*! \file Teuchos_XMLInputStream.hpp
    \brief A base class for defining a XML input stream
*/

#include "Teuchos_ConfigDefs.hpp"

namespace Teuchos
{
  /** \brief XMLInputStream represents an XML input stream
  *	that can be used by a XMLInputSource
  */
  class XMLInputStream
    {
    public:
      /** \brief Constructor */
      XMLInputStream(){;}

      /** \brief Destructor */
      inline virtual ~XMLInputStream(){;}

      /** \brief Read up to maxToRead bytes from the stream */
      virtual unsigned int readBytes(unsigned char* const toFill,
                                     const unsigned int maxToRead) = 0 ;

      /** \brief Identify current position */
      virtual unsigned int curPos() const ;

    };
}
#endif


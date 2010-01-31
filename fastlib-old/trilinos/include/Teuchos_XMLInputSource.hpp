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

#ifndef TEUCHOS_XMLINPUTSOURCE_H
#define TEUCHOS_XMLINPUTSOURCE_H

/*! \file Teuchos_XMLInputSource.hpp
    \brief A base class for defining a source of XML input.
*/

#include "Teuchos_ConfigDefs.hpp"
#include "Teuchos_XMLObject.hpp"
#include "Teuchos_XMLInputStream.hpp"

namespace Teuchos
{
  /** 
   * \brief XMLInputSource represents a source of XML input that can be parsed
   * to produce an XMLObject. 
   *
   * \note 
   *	<ul>
   *	<li>The source might be a file, a socket, a
   * std::string. The XMLObject is created with a call to the getObject() method.
   *
   *    <li> The source gets its data from a XMLInputStream object that is
   * created (internally) to work with this source.
   *
   *    <li> getObject() is implemented with EXPAT if Teuchos is configured with
   * <tt>--enable-expat</tt>.
   *	</ul>
   */
  class XMLInputSource
    {
    public:
      /** \brief Empty constructor */
      XMLInputSource(){;}

      /** \brief Destructor */
      virtual ~XMLInputSource(){;}

      /** \brief Virtual input source interface */
      virtual RCP<XMLInputStream> stream() const = 0 ;

      /** \brief Get an object by invoking the TreeBuildingXMLHandler on the
       * input data */
      XMLObject getObject() const ;

    };

}
#endif


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

#ifndef TEUCHOS_LIBXML2HANDLERADAPTER_H
#define TEUCHOS_LIBXML2HANDLERADAPTER_H

/*! \file Teuchos_Libxml2HandlerAdapter.hpp
    \brief libxml2 adapter for the TreeBuildingXMLHandler
*/

#include "Teuchos_ConfigDefs.hpp"

#ifdef HAVE_TEUCHOS_LIBXML2
#include "Teuchos_TreeBuildingXMLHandler.hpp"
#include "Teuchos_RCP.hpp"

#include <libxml/parser.h>

extern "C"
{
  /** \ingroup libXML2 callback for start of an XML element. */
  void xmlSAX2StartElement(void* context,
                           const xmlChar* name,
                           const xmlChar** attr);

  /** \ingroup libXML2 callback for end of an XML element. */
  void xmlSAX2EndElement(void* context,
                         const xmlChar* name);

  /** \ingroup libXML2 callback for character data. */
  void xmlSAX2Characters(void* context,
                         const xmlChar* s,
                         int len);
};

#endif


#endif

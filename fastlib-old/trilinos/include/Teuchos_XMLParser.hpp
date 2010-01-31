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

#ifndef TEUCHOS_XMLPARSER_H
#define TEUCHOS_XMLPARSER_H

/*! \file Teuchos_XMLParser.hpp
    \brief A class providing a simple XML parser. Methods can be overloaded 
           to exploit external XML parsing libraries.
*/

#include "Teuchos_ConfigDefs.hpp"
#include "Teuchos_XMLObject.hpp"
#include "Teuchos_XMLInputStream.hpp"

namespace Teuchos
{
  /** 
   * \brief XMLParser consumes characters from an XMLInputStream object,
   * parsing the XML and using a TreeBuildingXMLHandler to construct an
   * XMLObject.
   */
  class XMLParser
    {
    public:
     
      /** \brief Constructor */
      XMLParser(RCP<XMLInputStream> is) : _is(is) {;}
      
      /** \brief Destructor */
      ~XMLParser(){;}
      
      /** \brief Consume the XMLInputStream to build an XMLObject. */
      XMLObject parse();
    private:
      RCP<XMLInputStream> _is;
      Teuchos::map<std::string,string> _entities;
      
      /** \brief Determine whether \c c matches the <tt>Letter</tt> production according to the XML specification.*/
      inline static bool isLetter(unsigned char c);
      /** \brief Determine whether \c c matches the <tt>NameChar</tt> production according to the XML specification.*/
      inline static bool isNameChar(unsigned char c);
      /** \brief Determine whether \c c matches the <tt>Char</tt> production according to the XML specification.*/
      inline static bool isChar(unsigned char c);
      /** \brief Determine whether \c c matches the <tt>Space</tt> production according to the XML specification.*/
      inline static bool isSpace(unsigned char c);

      /** \brief Consume a <tt>ETag</tt> production according to the XML specification.
       *  <tt>getETag</tt> throws an std::exception if the input does not match the production rule.
       *  
       *  @param tag
       *         [out] On output, will be set to the tag name of the closing tag.
       */
      void getETag(std::string &tag);

      /** \brief Consume a <tt>STag</tt> production according to the XML specification.
       *  <tt>getSTag</tt> throws an std::exception if the input does not match the production rule.
       *  
       *  @param lookahead
       *         [in] Contains the first character of the tag name.
       * 
       *  @param tag
       *         [out] On output, will be set to the tag name of the opening tag.
       * 
       *  @param attrs
       *         [out] On output, contains the attributes of the tag.
       *
       *  @param emptytag
       *         [out] On output, specifies if this was an empty element tag.
       *
       */
      void getSTag(unsigned char lookahead, std::string &tag, Teuchos::map<std::string,string> &attrs, bool &emptytag);

      /** \brief Consume a <tt>Comment</tt> production according to the XML specification.
       *  <tt>getComment</tt> throws an std::exception if the input does not match the production rule.
       */
      void getComment();

      /** \brief Consumes a <tt>Space</tt> (block of whitepace) production according to the XML specification.
       *
       *  @param lookahead
       *         [out] On output, specifies the first character after the whitespace.
       *
       *  @return Returns non-zero if the input stream was exhausted while reading whitespace.
       */
      int getSpace(unsigned char &lookahead);

      /** \brief Consumes a <tt>Reference</tt> production according to the XML specification.
       *
       *  @param refstr
       *         [out] On output, specifies the decoded reference.
       *
       */
      void getReference(std::string &refstr);

      /** \brief Determines if the next character on the stream 
       *
       *  @param cexp
       *         [in] The expected character.
       *
       *  @return Returns non-zero if the next character on the stream is not \c cexp.
       * 
       */
      int assertChar(unsigned char cexp);
    };
}

#endif

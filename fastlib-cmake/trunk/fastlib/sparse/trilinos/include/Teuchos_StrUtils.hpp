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

#ifndef TEUCHOS_STRUTILS_H
#define TEUCHOS_STRUTILS_H

/*! \file Teuchos_StrUtils.hpp
    \brief A std::string utilities class for Teuchos
*/

#include "Teuchos_ConfigDefs.hpp"

#include "Teuchos_Utils.hpp"
#include "Teuchos_Array.hpp"

namespace Teuchos
{
  /** 
   * \brief Provides std::string manipulation utilities that are not provided in the
   * standard C++ std::string class.
   */
  class StrUtils
    {
    public:
      /** \brief Read a file, putting each line into a std::string */
      static Array<std::string> readFile(std::istream& is, char comment);

      /** \brief Split an input std::string that contains newlines into an array
 	 of strings, one for each line */
      static Array<std::string> splitIntoLines(const std::string& input);

      /** \brief Tokenize a file into whitespace-delimited tokens */
      static Array<Array<std::string> > tokenizeFile(std::istream& is, char comment);

      /** \brief Read a single line into a std::string */
      static bool readLine(std::istream& is, std::string& line);

      static Array<std::string> stringTokenizer(const std::string& str);

      static Array<std::string> getTokensPlusWhitespace(const std::string& str);

      static std::string reassembleFromTokens(const Array<std::string>& tokens, int iStart=0);

      static void splitList(const std::string& bigstring, Array<std::string>& elements);

      static int findNextWhitespace(const std::string& str, int offset);

      static int findNextNonWhitespace(const std::string& str, int offset);


      static  std::string varSubstitute(const std::string& rawLine,
                                   const std::string& varName,
                                   const std::string& varValue);

      static std::string varTableSubstitute(const std::string& rawLine,
                                       const Array<std::string>& varNames,
                                       const Array<std::string>& varValues);

      static std::string envSubstitute(const std::string& line);

      /** \brief Find the substring before a specified substring. For example,
       * before("abcdefghij", "gh") returns "abcdef". */
      static std::string before(const std::string& str, const std::string& sub);

      /** \brief Find the substring before a specified character. For example,
       * before("abcdefghij", 'g') returns "abcdef". */
      static std::string before(const std::string& str, char sub);

      /** \brief Find the substring after a specified substring. For example,
       * before("abcdefghij", "gh") returns "ij". */
      static std::string after(const std::string& str, const std::string& sub);

      /** \brief Find the position at which a substring first occurs. For example,
       * find("abcdefghij", "gh") returns 6. */
      static int find(const std::string& str, const std::string& sub);

      /** \brief Returns true if a std::string consists entirely of whitespace */
      static bool isWhite(const std::string& str);

      /** \brief Convert unprintable non-null characters to whitespace */
      static std::string fixUnprintableCharacters(const std::string& str);

      /** \brief Returns true if a std::string has any non-whitespace */
      static bool isNonWhite(const std::string& str) {return !isWhite(str);}

      /** \brief Returns the std::string between two delimiting strings, and returns
       * by reference the strings before and after the delimiters.
       *
       * For example, between("abcdefghij", "c", "g", front, back)
       * returns "def" and sets front to "ab", back to "hij". */
      static std::string between(const std::string& str, const std::string& begin,
                            const std::string& end, std::string& front, std::string& back);

      /** \brief Returns the substring between two positions. 
       *
       *  For example, subString("abcdefghij", 2, 5) returns "cde". */
      static std::string subString(const std::string& str, int begin, int end);

      static std::string readFromStream(std::istream& is);

      /** \brief Converts a std::string to all upper case */
      static std::string allCaps(const std::string& str);

      /** \brief Returns the double value of a std::string. */
      static double atof(const std::string& str);

      /** \brief Returns the int value of a std::string. */
      static int atoi(const std::string& str);

      /** \brief Print lines with prefix first. */
      static std::ostream& printLines(
        std::ostream             &os
        ,const std::string       &linePrefix
        ,const std::string       &lines
        );

    };

}

#endif

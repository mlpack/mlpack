/* MLPACK 0.2
 *
 * Copyright (c) 2008, 2009 Alexander Gray,
 *                          Garry Boyer,
 *                          Ryan Riegel,
 *                          Nikolaos Vasiloglou,
 *                          Dongryeol Lee,
 *                          Chip Mappus, 
 *                          Nishant Mehta,
 *                          Hua Ouyang,
 *                          Parikshit Ram,
 *                          Long Tran,
 *                          Wee Chin Wong
 *
 * Copyright (c) 2008, 2009 Georgia Institute of Technology
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
 * 02110-1301, USA.
 */
/**
 * @file tokenizer.h
 *
 * Simple tokenizer function for std::strings.
 *
 */

#include "../base/base.h"

#include <string>
#include <algorithm>
#include <vector>
#include <iostream>

/**
 * Simple tokenizer function for std::strings.
 *
 * Runs in O( str.length() * ( delimeters.length() + stopon.length() ) ).
 *
 * @param str string to tokenize
 * @param delimeters delimeters used to separate tokens
 * @param result vector used to store tokens
 * @param pos position in str to start from
 * @param stopon a character at which to stop processing
 * @param stopat specifies the maximum number of tokens to find
 */
void tokenizeString( const std::string& str, const std::string& delimeters,
    std::vector<std::string>& result, index_t pos=0,
    const std::string& stopon="", index_t stopat=0 ) {

  result.reserve(stopat);

  size_t last = pos;

  // Allow for numeric_limits<index_t>::max() tokens
  // Minimum is INT_MAX (sufficient?)
  if( !stopat )
    --stopat;
  for( ; pos < str.length() && stopat; ++pos ) {

    for( std::string::const_iterator stopit = stopon.begin();
        stopit < stopon.end(); ++stopit ) {
      if( str[pos] == *stopit ) {
        if( stopat && last < pos )
          result.push_back(
              str.substr( last, pos - last )
              );
        return;
      }
    }
    
    for( size_t dpos = 0; dpos < delimeters.length(); ++dpos ) {
      if( str[pos] == delimeters[dpos] ) {
        if( last == pos )
          ++last;
        else {
          result.push_back( 
              str.substr( last, pos - last )
              );
          last = pos+1;
          --stopat;
        }
        break;
      }
    }

  }

  // Grab the last token
  if( stopat && last < str.length() )
    result.push_back(
        str.substr( last, pos - last )
        );
}


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

#ifndef COLLECTIONS_TOKENIZESTRING_H
#define COLLECTIONS_TOKENIZESTRING_H

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
 * @param delimiters delimeters used to separate tokens
 * @param result vector used to store tokens
 * @param pos position in str to start from
 * @param stopon a character at which to stop processing
 * @param stopat specifies the maximum number of tokens to find
 * @param save_last whether or not to save the last token, if stopping on a
 *         stopon character
 */
void tokenizeString( const std::string& str, const std::string& delimiters,
    std::vector<std::string>& result, index_t pos = 0,
    const std::string& stopon = "", index_t stopat = 0, bool save_last = false );


#endif

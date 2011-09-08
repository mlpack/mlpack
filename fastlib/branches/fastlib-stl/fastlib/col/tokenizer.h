/**
 * @file tokenizer.h
 *
 * Simple tokenizer function for std::strings.
 *
 */

#ifndef COLLECTIONS_TOKENIZESTRING_H
#define COLLECTIONS_TOKENIZESTRING_H

#include "../base/common.h"

#include <string>
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
    std::vector<std::string>& result, size_t pos = 0,
    const std::string& stopon = "", size_t stopat = 0, bool save_last = false );


#endif

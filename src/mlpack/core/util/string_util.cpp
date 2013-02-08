/**
 * @file string_util.cpp
 *
 * Defines methods useful for formatting output.
 */
#include "string_util.hpp"

using namespace mlpack;
using namespace mlpack::util;

//! A utility function that replaces all all newlines with a number of spaces
//! depending on the indentation level.
std::string mlpack::util::Indent(std::string input)
{
  // Tab the first line.
  input.insert(0, 1, ' ');
  input.insert(0, 1, ' ');

  // Get the character sequence to replace all newline characters.
  std::string tabbedNewline("\n  ");

  // Replace all newline characters with the precomputed character sequence.
  size_t start_pos = 0;
  while((start_pos = input.find("\n", start_pos)) != std::string::npos) {
      input.replace(start_pos, 1, tabbedNewline);
      start_pos += tabbedNewline.length();
  }

  return input;
}

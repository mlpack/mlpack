/**
 * @file string_util.cpp
 *
 * Defines methods useful for formatting output.
 */
#include "string_util.hpp"

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

//! A utility function that replaces all all newlines with a number of spaces
//! depending on the indentation level.
string mlpack::util::Indent(string input)
{
  // Tab the first line.
  input.insert(0, 1, ' ');
  input.insert(0, 1, ' ');

  // Get the character sequence to replace all newline characters.
  std::string tabbedNewline("\n  ");

  // Replace all newline characters with the precomputed character sequence.
  size_t startPos = 0;
  while ((startPos = input.find("\n", startPos)) != string::npos)
  {
    input.replace(startPos, 1, tabbedNewline);
    startPos += tabbedNewline.length();
  }

  return input;
}

/**
 * @file string_util.cpp
 * @author Trironk Kiatkungwanglai
 * @author Ryan Birmingham
 *
 * Defines methods useful for formatting output.
 *
 * This file is part of mlpack 1.0.12.
 *
 * mlpack is free software; you may redstribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "string_util.hpp"

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

//! A utility function that replaces all all newlines with a number of spaces
//! depending on the indentation level.
string mlpack::util::Indent(string input, const size_t howManyTabs)
{
  // For each declared...
  string standardTab = "  ";
  string bigTab = "";
  for (size_t ind = 0; ind < howManyTabs; ind++)
  {
    // Increase amount tabbed on later lines.
    bigTab += standardTab;

    // Add indentation to first line.
    input.insert(0, 1, ' ');
    input.insert(0, 1, ' ');
  }

  // Create the character sequence to replace all newline characters.
  std::string tabbedNewline("\n" + bigTab);

  // Replace all newline characters with the precomputed character sequence.
  size_t startPos = 0;
  while ((startPos = input.find("\n", startPos)) != string::npos)
  {
    // Don't replace the last newline.
    if (startPos == input.length() - 1)
      break;

    input.replace(startPos, 1, tabbedNewline);
    startPos += tabbedNewline.length();
  }

  return input;
}

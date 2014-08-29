/**
 * @file string_util.cpp
 * @author Trironk Kiatkungwanglai
 * @author Ryan Birmingham
 *
 * Defines methods useful for formatting output.
 *
 * This file is part of MLPACK 1.0.10.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
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

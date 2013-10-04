/**
 * @file string_util.cpp
 *
 * Defines methods useful for formatting output.
 *
 * This file is part of MLPACK 1.0.7.
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

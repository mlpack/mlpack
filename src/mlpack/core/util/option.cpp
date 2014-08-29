/**
 * @file option.cpp
 * @author Ryan Curtin
 *
 * Implementation of the ProgramDoc class.  The class registers itself with CLI
 * when constructed.
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
#include "cli.hpp"
#include "option.hpp"

#include <string>

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

/**
 * Construct a ProgramDoc object.  When constructed, it will register itself
 * with CLI.  A fatal error will be thrown if more than one is constructed.
 *
 * @param programName Short string representing the name of the program.
 * @param documentation Long string containing documentation on how to use the
 *    program and what it is.  No newline characters are necessary; this is
 *    taken care of by CLI later.
 * @param defaultModule Name of the default module.
 */
ProgramDoc::ProgramDoc(const std::string& programName,
                       const std::string& documentation) :
    programName(programName),
    documentation(documentation)
{
  // Register this with CLI.
  CLI::RegisterProgramDoc(this);
}

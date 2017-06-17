/**
 * @file option.cpp
 * @author Ryan Curtin
 *
 * Implementation of the ProgramDoc class.  The class registers itself with CLI
 * when constructed.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
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

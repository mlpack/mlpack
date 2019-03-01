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
#include "program_doc.hpp"

#include <string>

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

/**
 * Construct a ProgramDoc object.  When constructed, it will register itself
 * with CLI.  A fatal error will be thrown if more than one is constructed.
 *
 * @param defaultModule Name of the default module.
 * @param shortDocumentation A short two-sentence description of the program,
 *     what it does, and what it is useful for.
 * @param documentation Long string containing documentation on how to use the
 *     program and what it is.  No newline characters are necessary; this is
 *     taken care of by CLI later.
 * @param seeAlso A set of pairs of strings with useful "see also"
 *     information; each pair is <description, url>.
 */
ProgramDoc::ProgramDoc(
    const std::string& programName,
    const std::string& shortDocumentation,
    const std::function<std::string()>& documentation,
    const std::vector<std::pair<std::string, std::string>>& seeAlso) :
    programName(programName),
    shortDocumentation(shortDocumentation),
    documentation(documentation),
    seeAlso(seeAlso)
{
  // Register this with CLI.
  CLI::RegisterProgramDoc(this);
}

/**
 * Construct an empty ProgramDoc object.
 */
ProgramDoc::ProgramDoc()
{
  CLI::RegisterProgramDoc(this);
}

/**
 * @file core/util/program_doc.cpp
 * @author Yashwant Singh Parihar
 * @author Ryan Curtin
 *
 * Implementation of the mutiple classes.  The classes registers itself with IO
 * when constructed.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "io.hpp"
#include "program_doc.hpp"

#include <string>

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

/**
 * Construct a ProgramName object.  When constructed, it will register itself
 * with IO.  A fatal error will be thrown if more than one is constructed.
 *
 * @param defaultModule Name of the default module.
 */
ProgramName::ProgramName(
    const std::string programName) :
    programName(std::move(programName))
{
  // Register this with IO.
  IO::RegisterProgramName(this);
}

/**
 * Construct an empty ProgramName object.
 */
ProgramName::ProgramName()
{
  IO::RegisterProgramName(this);
}

/**
 * Construct a ShortDescription object.  When constructed, it will register
 * itself with IO.  A fatal error will be thrown if more than one is
 * constructed.
 *
 * @param shortDescription A short two-sentence description of the program,
 *     what it does, and what it is useful for.
 */
ShortDescription::ShortDescription(
    const std::string shortDescription) :
    shortDescription(std::move(shortDescription))
{
  // Register this with IO.
  IO::RegisterShortDescription(this);
}

/**
 * Construct an empty ShortDescription object.
 */
ShortDescription::ShortDescription()
{
  IO::RegisterShortDescription(this);
}

/**
 * Construct a LongDescription object. When constructed, it will register itself
 * with IO.  A fatal error will be thrown if more than one is constructed.
 *
 * @param longDescription Long string containing documentation on 
 *     what it is.  No newline characters are necessary; this is
 *     taken care of by IO later.
 */
LongDescription::LongDescription(
    const std::function<std::string()> longDescription) :
    longDescription(std::move(longDescription))
{
  // Register this with IO.
  IO::RegisterLongDescription(this);
}

/**
 * Construct an empty LongDescription object.
 */
LongDescription::LongDescription()
{
  IO::RegisterLongDescription(this);
}

/**
 * Construct a Example object.  When constructed, it will register itself
 * with IO.  A fatal error will be thrown if more than one is constructed.
 *
 * @param example Documentation on how to use the program.
 */
Example::Example(
    const std::function<std::string()> example) :
    example(std::move(example))
{
  // Register this with IO.
  IO::RegisterExample(this);
}

/**
 * Construct an empty Example object.
 */
Example::Example()
{
  IO::RegisterExample(this);
}

/**
 * Construct a SeeAlso object.  When constructed, it will register itself
 * with IO.  A fatal error will be thrown if more than one is constructed.
 *
 * @param seeAlso A set of pairs of strings with useful "see also"
 *     information; each pair is <description, url>.
 */
SeeAlso::SeeAlso(
    const std::string description, const std::string link) :
    description(std::move(description)),
    link(std::move(link))
{
  // Register this with IO.
  IO::RegisterSeeAlso(this);
}

/**
 * Construct an empty SeeAlso object.
 */
SeeAlso::SeeAlso()
{
  IO::RegisterSeeAlso(this);
}

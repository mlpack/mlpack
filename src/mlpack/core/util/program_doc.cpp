/**
 * @file core/util/program_doc.cpp
 * @author Yashwant Singh Parihar
 * @author Ryan Curtin
 *
 * Implementation of mutiple classes that store information related to a binding.
 * The classes register themselves with IO when constructed.
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
 * @param programName Name of the binding.
 */
ProgramName::ProgramName(const std::string& programName)
{
  // Register this with IO.
  IO::GetSingleton().doc.programName = std::move(programName);
}

/**
 * Construct a ShortDescription object.  When constructed, it will register
 * itself with IO.  A fatal error will be thrown if more than one is
 * constructed.
 *
 * @param shortDescription A short two-sentence description of the binding,
 *     what it does, and what it is useful for.
 */
ShortDescription::ShortDescription(const std::string& shortDescription)
{
  // Register this with IO.
  IO::GetSingleton().doc.shortDescription = std::move(shortDescription);
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
    const std::function<std::string()>& longDescription)
{
  // Register this with IO.
  IO::GetSingleton().doc.longDescription = std::move(longDescription);
}

/**
 * Construct a Example object.  When constructed, it will register itself
 * with IO.
 *
 * @param example Documentation on how to use the binding.
 */
Example::Example(
    const std::function<std::string()>& example)
{
  // Register this with IO.
  IO::GetSingleton().doc.example.push_back(std::move(example));
}

/**
 * Construct a SeeAlso object.  When constructed, it will register itself
 * with IO.
 *
 * @param description Description of SeeAlso.
 * @param link Link of SeeAlso.
 */
SeeAlso::SeeAlso(
    const std::string& description, const std::string& link)
{
  // Register this with IO.
  IO::GetSingleton().doc.seeAlso.push_back(make_pair(description, link));
}

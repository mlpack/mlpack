/**
 * @file core/util/program_doc.hpp
 * @author Yashwant Singh Parihar
 * @author Matthew Amidon
 *
 * The structure used to store a program's name, documentation, example and
 * see also.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_UTIL_PROGRAM_DOC_HPP
#define MLPACK_CORE_UTIL_PROGRAM_DOC_HPP

namespace mlpack {
namespace util {

/**
 * A static object whose constructor registers program documentation with the
 * IO class.  This should not be used outside of IO itself, and you should use
 * these BINDING_PNAME(), BINDING_SHORT_DESC(), BINDING_LONG_DESC(), 
 * BINDING_EXAMPLE() and BINDING_SEE_ALSO() macros to declare these objects.
 * Only correspond object should ever exist.
 *
 * @see core/util/io.hpp, mlpack::IO
 */
class ProgramName
{
 public:
  /**
   * Construct a ProgramName object.  When constructed, it will register itself
   * with IO.  A fatal error will be thrown if more than one is constructed.
   *
   * @param programName Name of the default module.
   */
  ProgramName(const std::string& programName);

  /**
   * Construct an empty ProgramName object.  (This is not meant to be used!)
   */
  ProgramName();
  
  std::string programName;
};

class ShortDescription
{
 public:
  /**
   * Construct a ShortDescription object.  When constructed, it will register
   * itself with IO.  A fatal error will be thrown if more than one is
   * constructed.
   *
   * @param shortDescription A short two-sentence description of the program,
   *     what it does, and what it is useful for.
   */
  ShortDescription(const std::string& shortDescription);

  /**
   * Construct an empty ShortDescription object.
   * (This is not meant to be used!)
   */
  ShortDescription();
  
  std::string shortDescription;
};

class LongDescription
{
 public:
  /**
   * Construct a LongDescription object. When constructed, it will register itself
   * with IO.  A fatal error will be thrown if more than one is constructed.
   *
   * @param longDescription Long string containing documentation on 
   *     what it is.  No newline characters are necessary; this is
   *     taken care of by IO later.
   */
  LongDescription(const std::function<std::string()>& longDescription);

  /**
   * Construct an empty LongDescription object.  (This is not meant to be used!)
   */
  LongDescription();
  
  std::function<std::string()> longDescription;
};

class Example
{
 public:
  /**
   * Construct a Example object.  When constructed, it will register itself
   * with IO.  A fatal error will be thrown if more than one is constructed.
   *
   * @param example Documentation on how to use the program.
   */
  Example(const std::function<std::string()>& example);

  /**
   * Construct an empty Example object.  (This is not meant to be used!)
   */
  Example();
  
  std::function<std::string()> example;
};

class SeeAlso
{
 public:
  /**
   * Construct a SeeAlso object.  When constructed, it will register itself
   * with IO.  A fatal error will be thrown if more than one is constructed.
   *
   * @param description Description of SeeAlso.
   * @param link Link of SeeAlso.
   */
  SeeAlso(const std::string& description, const std::string& link);

  /**
   * Construct an empty SeeAlso object.  (This is not meant to be used!)
   */
  SeeAlso();

  std::string description;
  std::string link;
};

} // namespace util
} // namespace mlpack

#endif

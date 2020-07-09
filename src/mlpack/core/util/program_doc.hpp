/**
 * @file core/util/program_doc.hpp
 * @author Matthew Amidon
 *
 * The structure used to store a program's name and documentation.
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
 * the PROGRAM_INFO() macro to declare these objects.  Only one ProgramDoc
 * object should ever exist.
 *
 * @see core/util/io.hpp, mlpack::IO
 */
class ProgramDoc
{
 public:
  /**
   * Construct a ProgramDoc object.  When constructed, it will register itself
   * with IO, and when the user calls --help (or whatever the option is named
   * for the given binding type), the given function that returns a std::string
   * will be returned.
   *
   * @param programName Short string representing the name of the program.
   * @param shortDocumentation A short two-sentence description of the program,
   *     what it does, and what it is useful for.
   * @param documentation Long string containing documentation on how to use the
   *     program and what it is.  No newline characters are necessary; this is
   *     taken care of by IO later.
   * @param seeAlso A set of pairs of strings with useful "see also"
   *     information; each pair is <description, url>.
   */
  ProgramDoc(const std::string programName,
             const std::string shortDocumentation,
             const std::function<std::string()> documentation,
             const std::vector<std::pair<std::string, std::string>> seeAlso);

  /**
   * Construct an empty ProgramDoc object.  (This is not meant to be used!)
   */
  ProgramDoc();

  //! The name of the program.
  std::string programName;
  //! The short documentation for the program.
  std::string shortDocumentation;
  //! Documentation for what the program does.
  std::function<std::string()> documentation;
  //! Set of see also information.
  std::vector<std::pair<std::string, std::string>> seeAlso;
};

} // namespace util
} // namespace mlpack

#endif

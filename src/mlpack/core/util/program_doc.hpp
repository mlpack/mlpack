#ifndef MLPACK_CORE_UTIL_PROGRAM_DOC_HPP
#define MLPACK_CORE_UTIL_PROGRAM_DOC_HPP

namespace mlpack {
namespace util {

/**
 * A static object whose constructor registers program documentation with the
 * CLI class.  This should not be used outside of CLI itself, and you should use
 * the PROGRAM_INFO() macro to declare these objects.  Only one ProgramDoc
 * object should ever exist.
 *
 * @see core/util/cli.hpp, mlpack::CLI
 */
class ProgramDoc
{
 public:
  /**
   * Construct a ProgramDoc object.  When constructed, it will register itself
   * with CLI, and when the user calls --help (or whatever the option is named
   * for the given binding type), the given function that returns a std::string
   * will be returned.
   *
   * @param programName Short string representing the name of the program.
   * @param documentation Long string containing documentation on how to use the
   *     program and what it is.  No newline characters are necessary; this is
   *     taken care of by CLI later.
   */
  ProgramDoc(const std::string& programName,
             const std::function<std::string()>& documentation);

  //! The name of the program.
  std::string programName;
  //! Documentation for what the program does.
  std::function<std::string()> documentation;
};

} // namespace util
} // namespace mlpack

#endif

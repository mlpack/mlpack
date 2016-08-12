/**
 * @file option.hpp
 * @author Matthew Amidon
 *
 * Definition of the Option class, which is used to define parameters which are
 * used by CLI.  The ProgramDoc class also resides here.
 */
#ifndef MLPACK_CORE_UTIL_OPTION_HPP
#define MLPACK_CORE_UTIL_OPTION_HPP

#include <string>

#include "cli.hpp"

namespace mlpack {
namespace util {

/**
 * A static object whose constructor registers a parameter with the CLI class.
 * This should not be used outside of CLI itself, and you should use the
 * PARAM_FLAG(), PARAM_DOUBLE(), PARAM_INT(), PARAM_STRING(), or other similar
 * macros to declare these objects instead of declaring them directly.
 *
 * @see core/io/cli.hpp, mlpack::CLI
 */
template<typename N>
class Option
{
 public:
  /**
   * Construct an Option object.  When constructed, it will register
   * itself with CLI.
   *
   * @param ignoreTemplate Whether or not the template type matters for this
   *     option.  Essentially differs options with no value (flags) from those
   *     that do, and thus require a type.
   * @param defaultValue Default value this parameter will be initialized to.
   * @param identifier The name of the option (no dashes in front; for --help,
   *      we would pass "help").
   * @param description A short string describing the option.
   * @param alias Short name of the parameter.
   * @param required Whether or not the option is required at runtime.
   * @param input Whether or not the option is an input option.
   */
  Option(const bool ignoreTemplate,
         const N defaultValue,
         const std::string& identifier,
         const std::string& description,
         const std::string& alias,
         const bool required = false,
         const bool input = true);

  /**
   * Constructs an Option object.  When constructed, it will register a flag
   * with CLI.
   *
   * @param identifier The name of the option (no dashes in front); for --help
   *     we would pass "help".
   * @param description A short string describing the option.
   * @param alias Short name of the parameter.
   */
  Option(const std::string& identifier,
         const std::string& description,
         const std::string& alias);
};

/**
 * A specialization for Armadillo matrices.
 */
template<>
class Option<arma::mat>
{
 public:
  /**
   * Construct an Option matrix object.  When constructed, it will register
   * itself with CLI.  The default is always an empty matrix.
   *
   * @param identifier The name of the option (no dashes in front; for --help,
   *      we would pass "help").
   * @param description A short string describing the option.
   * @param alias Short name of the parameter.
   * @param required Whether or not the option is required at runtime.
   * @param input Whether or not the option is an input option.
   * @param transpose Whether or not the matrix should be transposed on a
   *      load/save operation.
   */
  Option(const std::string& identifier,
         const std::string& description,
         const std::string& alias,
         const bool required = false,
         const bool input = true,
         const bool transpose = false);
};

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
   * with CLI.
   *
   * @param programName Short string representing the name of the program.
   * @param documentation Long string containing documentation on how to use the
   *     program and what it is.  No newline characters are necessary; this is
   *     taken care of by CLI later.
   */
  ProgramDoc(const std::string& programName,
             const std::string& documentation);

  //! The name of the program.
  std::string programName;
  //! Documentation for what the program does.
  std::string documentation;
};

} // namespace util
} // namespace mlpack

// For implementations of templated functions
#include "option_impl.hpp"

#endif

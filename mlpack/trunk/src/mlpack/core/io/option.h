/***
 * @file option.h
 * @author Matthew Amidon
 *
 * The Option class is used to facilitate compile-time instantiation
 * of parameters (or more specifically, at program start up).  This should
 * not be used outside of CLI itself.  The ProgramDoc class is used to facilitate
 * compile-time instantiation of overall program documentation (basically, what
 * the program is called and some basic outline of the program).
 */
#ifndef MLPACK_CLI_OPTCLIN_H
#define MLPACK_CLI_OPTCLIN_H

#include <string>

namespace mlpack {

template<typename N>
class Option {
 public:
  /***
   * Construct an Option object.  When constructed, it will register
   * itself with CLI.
   *
   * @param ignoreTemplate Whether or not the template type matters for
   *       this option.  Essentially differs options with no value (flags)
   *       from those that do, and thus require a type.
   * @param defaultValue Default value this parameter will be initialized to.
   * @param identifier The name of the option (no dashes in front; for
   *       --help, we would pass "help").
   * @param description A short string describing the option.
   * @param parent Full pathname of the parent module that "owns" this option.
   *        The default is evaluates to the root node.
   * @param required Whether or not the option is required at runtime.
   */
  Option(bool ignoreTemplate,
         N defaultValue,
         const char* identifier,
         const char* description,
         const char* parent = NULL,
         bool required = false);

  /**
   * Constructs an Option object.  When constructed, it will register a flag
   * with CLI.
   *
   * @param identifier The name of the option (no dashes in front); for
   * 	--help, we would pass "help"
   * @param descriptoin A short string describing the option.
   * @param parent Full pathname of the parent module that "owns" this
   *    option.  The default evaluates to the root node.
   */
  Option(const char* identifier,
         const char* description,
         const char* parent = NULL);
};

class ProgramDoc {
 public:
  /***
   * Construct a ProgramDoc object.  When constructed, it will register itself
   * with CLI.
   *
   * @param programName Short string representing the name of the program.
   * @param documentation Long string containing documentation on how to use the
   *    program and what it is.  No newline characters are necessary; this is
   *    taken care of by CLI later.
   */
  ProgramDoc(std::string programName, std::string documentation,
    std::string defaultModule);

  std::string programName;
  std::string documentation;
  std::string defaultModule;
};

}; // namespace mlpack

// For implementations of templated functions
#include "option_impl.h"

#endif

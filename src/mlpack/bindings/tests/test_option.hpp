/**
 * @file test_option.hpp
 * @author Ryan Curtin
 *
 * Definition of the TestOption class, which is used to define parameters for
 * CLI for use inside of mlpack_test.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_BINDINGS_TESTS_TEST_OPTION_HPP
#define MLPACK_CORE_BINDINGS_TESTS_TEST_OPTION_HPP

#include <string>

#include <mlpack/core/util/cli.hpp>
#include "get_printable_param.hpp"
#include "get_param.hpp"
#include "get_allocated_memory.hpp"
#include "delete_allocated_memory.hpp"

namespace mlpack {
namespace bindings {
namespace tests {

// Defined in mlpack_main.hpp.
extern std::string programName;

/**
 * A static object whose constructor registers a parameter with the CLI class.
 * This should not be used outside of CLI itself, and you should use the
 * PARAM_FLAG(), PARAM_DOUBLE(), PARAM_INT(), PARAM_STRING(), or other similar
 * macros to declare these objects instead of declaring them directly.
 *
 * @see core/util/cli.hpp, mlpack::CLI
 */
template<typename N>
class TestOption
{
 public:
  /**
   * Construct an Option object.  When constructed, it will register
   * itself with CLI.
   *
   * @param defaultValue Default value this parameter will be initialized to
   *      (for flags, this should be false, for instance).
   * @param identifier The name of the option (no dashes in front; for --help,
   *      we would pass "help").
   * @param description A short string describing the option.
   * @param alias Short name of the parameter. "" for no alias.
   * @param cppName Name of the C++ type of this parameter (i.e. "int").
   * @param required Whether or not the option is required at runtime.
   * @param input Whether or not the option is an input option.
   * @param noTranspose If the parameter is a matrix and this is true, then the
   *      matrix will not be transposed on loading.
   * @param testName Name of the test (used for identifiying which binding test 
   *      this option belongs to)
   */
  TestOption(const N defaultValue,
             const std::string& identifier,
             const std::string& description,
             const std::string& alias,
             const std::string& cppName,
             const bool required = false,
             const bool input = true,
             const bool noTranspose = false,
             const std::string& testName = "")
  {
    // Create the ParamData object to give to CLI.
    util::ParamData data;

    data.desc = description;
    data.name = identifier;
    data.tname = TYPENAME(N);
    data.alias = alias[0];
    data.wasPassed = false;
    data.noTranspose = noTranspose;
    data.required = required;
    data.input = input;
    data.loaded = false;
    data.cppType = cppName;
    data.value = boost::any(defaultValue);
    data.persistent = false;

    const std::string tname = data.tname;

    CLI::RestoreSettings(testName, false);

    // Set some function pointers that we need.
    CLI::GetSingleton().functionMap[tname]["GetPrintableParam"] =
        &GetPrintableParam<N>;
    CLI::GetSingleton().functionMap[tname]["GetParam"] = &GetParam<N>;
    CLI::GetSingleton().functionMap[tname]["GetAllocatedMemory"] =
        &GetAllocatedMemory<N>;
    CLI::GetSingleton().functionMap[tname]["DeleteAllocatedMemory"] =
        &DeleteAllocatedMemory<N>;

    CLI::Add(std::move(data));

    // If this is an output option, set it as passed.
    if (!input)
      CLI::SetPassed(identifier);

    CLI::StoreSettings(testName);
    CLI::ClearSettings();
  }
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

} // namespace tests
} // namespace bindings
} // namespace mlpack

#endif

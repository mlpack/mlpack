/**
 * @file bindings/tests/test_option.hpp
 * @author Ryan Curtin
 *
 * Definition of the TestOption class, which is used to define parameters for
 * IO for use inside of mlpack_test.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_BINDINGS_TESTS_TEST_OPTION_HPP
#define MLPACK_CORE_BINDINGS_TESTS_TEST_OPTION_HPP

#include <string>

#include <mlpack/core/util/io.hpp>
#include "get_printable_param.hpp"
#include "get_param.hpp"
#include "get_allocated_memory.hpp"
#include "delete_allocated_memory.hpp"
#include "test_function_map.hpp"

namespace mlpack {
namespace bindings {
namespace tests {

// Defined in mlpack_main.hpp.
extern std::string programName;

/**
 * A static object whose constructor registers a parameter with the IO class.
 * This should not be used outside of IO itself, and you should use the
 * PARAM_FLAG(), PARAM_DOUBLE(), PARAM_INT(), PARAM_STRING(), or other similar
 * macros to declare these objects instead of declaring them directly.
 *
 * @see core/util/io.hpp, mlpack::IO
 */
template<typename N>
class TestOption
{
 public:
  /**
   * Construct an Option object.  When constructed, it will register
   * itself with IO.
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
             const std::string& bindingName = "")
  {
    // Create the ParamData object to give to IO.
    util::ParamData data;

    data.desc = description;
    data.name = identifier;
    data.tname = TYPENAME(N);
    data.alias = alias[0];
    // If this is an output option, set it as passed.
    data.wasPassed = !input;
    data.noTranspose = noTranspose;
    data.required = required;
    data.input = input;
    data.loaded = false;
    data.cppType = cppName;
    data.value = defaultValue;

    const std::string tname = data.tname;

    // Set the function pointers in the test function map singleton.  We don't
    // register them with IO::AddFunction(), because these will be restored as
    // part of the test binding fixture; see
    // `src/mlpack/tests/main_tests/main_test_fixture.hpp`.
    TestFunctionMap::RegisterFunction(tname, "GetPrintableParam",
        &GetPrintableParam<N>);
    TestFunctionMap::RegisterFunction(tname, "GetParam", &GetParam<N>);
    TestFunctionMap::RegisterFunction(tname, "GetAllocatedMemory",
        &GetAllocatedMemory<N>);
    TestFunctionMap::RegisterFunction(tname, "DeleteAllocatedMemory",
        &DeleteAllocatedMemory<N>);

    IO::AddParameter(bindingName, std::move(data));
  }
};

} // namespace tests
} // namespace bindings
} // namespace mlpack

#endif

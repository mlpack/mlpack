/**
 * @file bindings/cli/cli_option.hpp
 * @author Matthew Amidon
 *
 * Definition of the Option class, which is used to define parameters which are
 * used by CLI.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_BINDINGS_CLI_CLI_OPTION_HPP
#define MLPACK_CORE_BINDINGS_CLI_CLI_OPTION_HPP

#include <string>

#include <mlpack/core/util/io.hpp>
#include "parameter_type.hpp"
#include "add_to_cli11.hpp"
#include "default_param.hpp"
#include "output_param.hpp"
#include "get_printable_param.hpp"
#include "string_type_param.hpp"
#include "get_param.hpp"
#include "get_raw_param.hpp"
#include "map_parameter_name.hpp"
#include "set_param.hpp"
#include "get_printable_param_name.hpp"
#include "get_printable_param_value.hpp"
#include "get_allocated_memory.hpp"
#include "delete_allocated_memory.hpp"
#include "in_place_copy.hpp"

namespace mlpack {
namespace bindings {
namespace cli {

/**
 * A static object whose constructor registers a parameter with the IO class.
 * This should not be used outside of IO itself, and you should use the
 * PARAM_FLAG(), PARAM_DOUBLE(), PARAM_INT(), PARAM_STRING(), or other similar
 * macros to declare these objects instead of declaring them directly.
 *
 * @see core/util/io.hpp, mlpack::IO
 */
template<typename N>
class CLIOption
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
   * @param bindingName Name of the binding that this option is for.  If empty,
   *      then it will be added to every binding.
   */
  CLIOption(const N defaultValue,
            const std::string& identifier,
            const std::string& description,
            const std::string& alias,
            const std::string& cppName,
            const bool required = false,
            const bool input = true,
            const bool noTranspose = false,
            const std::string& bindingName = "")
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

    // Apply default value.
    if (std::is_same<typename std::remove_pointer<N>::type,
                     typename ParameterType<typename
                         std::remove_pointer<N>::type>::type>::value)
    {
      data.value = defaultValue;
    }
    else
    {
      typename ParameterType<typename std::remove_pointer<N>::type>::type tmp;
      data.value = std::tuple<N, decltype(tmp)>(defaultValue, tmp);
    }

    const std::string tname = data.tname;
    const std::string cliName = MapParameterName<
        typename std::remove_pointer<N>::type>(identifier);
    std::string progOptId = (alias[0] != '\0') ?
        "-" + std::string(1, alias[0]) + ",--" + cliName : "--" + cliName;

    // Set some function pointers that we need.
    IO::AddFunction(tname, "DefaultParam", &DefaultParam<N>);
    IO::AddFunction(tname, "OutputParam", &OutputParam<N>);
    IO::AddFunction(tname, "GetPrintableParam", &GetPrintableParam<N>);
    IO::AddFunction(tname, "StringTypeParam", &StringTypeParam<N>);
    IO::AddFunction(tname, "GetParam", &GetParam<N>);
    IO::AddFunction(tname, "GetRawParam", &GetRawParam<N>);
    IO::AddFunction(tname, "AddToCLI11", &AddToCLI11<N>);
    IO::AddFunction(tname, "MapParameterName", &MapParameterName<N>);
    IO::AddFunction(tname, "GetPrintableParamName", &GetPrintableParamName<N>);
    IO::AddFunction(tname, "GetPrintableParamValue",
        &GetPrintableParamValue<N>);
    IO::AddFunction(tname, "GetAllocatedMemory", &GetAllocatedMemory<N>);
    IO::AddFunction(tname, "DeleteAllocatedMemory", &DeleteAllocatedMemory<N>);
    IO::AddFunction(tname, "InPlaceCopy", &InPlaceCopy<N>);

    IO::AddParameter(bindingName, std::move(data));
  }
};

} // namespace cli
} // namespace bindings
} // namespace mlpack

#endif

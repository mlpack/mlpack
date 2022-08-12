/**
 * @file bindings/julia/julia_option.hpp
 * @author Ryan Curtin
 *
 * The Julia option type.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_JULIA_JULIA_OPTION_HPP
#define MLPACK_BINDINGS_JULIA_JULIA_OPTION_HPP

#include <mlpack/core/util/param_data.hpp>
#include "get_param.hpp"
#include "get_printable_param.hpp"
#include "print_param_defn.hpp"
#include "print_input_param.hpp"
#include "print_input_processing.hpp"
#include "print_output_processing.hpp"
#include "print_doc.hpp"
#include "print_model_type_import.hpp"
#include "default_param.hpp"

namespace mlpack {
namespace bindings {
namespace julia {

/**
 * The Julia option class.
 */
template<typename T>
class JuliaOption
{
 public:
  /**
   * Construct a JuliaOption object.  When constructed, it will register itself
   * with IO. The testName parameter is not used and added for compatibility
   * reasons.
   */
  JuliaOption(const T defaultValue,
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
    data.tname = TYPENAME(T);
    data.alias = alias[0];
    data.wasPassed = false;
    data.noTranspose = noTranspose;
    data.required = required;
    data.input = input;
    data.loaded = false;
    data.cppType = cppName;

    // Every parameter we'll get from Julia will have the correct type.
    data.value = defaultValue;

    // Set the function pointers that we'll need.  All of these function
    // pointers will be used by both the program that generates the pyx, and
    // also the binding itself.  (The binding itself will only use GetParam,
    // GetPrintableParam, and GetRawParam.)
    IO::AddFunction(data.tname, "GetParam", &GetParam<T>);
    IO::AddFunction(data.tname, "GetPrintableParam", &GetPrintableParam<T>);

    // These are used by the jl generator.
    IO::AddFunction(data.tname, "PrintParamDefn", &PrintParamDefn<T>);
    IO::AddFunction(data.tname, "PrintInputParam", &PrintInputParam<T>);
    IO::AddFunction(data.tname, "PrintOutputProcessing",
        &PrintOutputProcessing<T>);
    IO::AddFunction(data.tname, "PrintInputProcessing",
        &PrintInputProcessing<T>);
    IO::AddFunction(data.tname, "PrintDoc", &PrintDoc<T>);
    IO::AddFunction(data.tname, "PrintModelTypeImport",
        &PrintModelTypeImport<T>);

    // This is needed for the Markdown binding output.
    IO::AddFunction(data.tname, "DefaultParam", &DefaultParam<T>);

    // Add the ParamData object.
    IO::AddParameter(bindingName, std::move(data));
  }
};

} // namespace julia
} // namespace bindings
} // namespace mlpack

#endif

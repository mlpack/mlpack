/**
 * @file py_option.hpp
 * @author Ryan Curtin
 *
 * The Python option type.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_PYTHON_PY_OPTION_HPP
#define MLPACK_BINDINGS_PYTHON_PY_OPTION_HPP

#include <mlpack/core/util/param_data.hpp>
#include "default_param.hpp"
#include "get_param.hpp"
#include "get_printable_param.hpp"
#include "print_class_defn.hpp"
#include "print_defn.hpp"
#include "print_doc.hpp"
#include "print_input_processing.hpp"
#include "print_output_processing.hpp"
#include "import_decl.hpp"

namespace mlpack {
namespace bindings {
namespace python {

// Defined in mlpack_main.hpp.
extern std::string programName;

/**
 * The Python option class.
 */
template<typename T>
class PyOption
{
 public:
  /**
   * Construct a PyOption object.  When constructed, it will register itself
   * with CLI. The testName parameter is not used and added for compatibility 
   * reasons.
   */
  PyOption(const T defaultValue,
           const std::string& identifier,
           const std::string& description,
           const std::string& alias,
           const std::string& cppName,
           const bool required = false,
           const bool input = true,
           const bool noTranspose = false,
           const std::string& /*testName*/ = "")
  {
    // Create the ParamData object to give to CLI.
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
    // Only "verbose" and "copy_all_inputs" will be persistent.
    if (identifier == "verbose" || identifier == "copy_all_inputs")
      data.persistent = true;
    else
      data.persistent = false;
    data.cppType = cppName;

    // Every parameter we'll get from Python will have the correct type.
    data.value = boost::any(defaultValue);

    // Restore the parameters for this program.
    if (identifier != "verbose" && identifier != "copy_all_inputs")
      CLI::RestoreSettings(programName, false);

    // Set the function pointers that we'll need.  All of these function
    // pointers will be used by both the program that generates the pyx, and
    // also the binding itself.  (The binding itself will only use GetParam,
    // GetPrintableParam, and GetRawParam.)
    CLI::GetSingleton().functionMap[data.tname]["GetParam"] = &GetParam<T>;
    CLI::GetSingleton().functionMap[data.tname]["GetPrintableParam"] =
        &GetPrintableParam<T>;

    CLI::GetSingleton().functionMap[data.tname]["DefaultParam"] =
        &DefaultParam<T>;

    // These are used by the pyx generator.
    CLI::GetSingleton().functionMap[data.tname]["PrintClassDefn"] =
        &PrintClassDefn<T>;
    CLI::GetSingleton().functionMap[data.tname]["PrintDefn"] = &PrintDefn<T>;
    CLI::GetSingleton().functionMap[data.tname]["PrintDoc"] = &PrintDoc<T>;
    CLI::GetSingleton().functionMap[data.tname]["PrintOutputProcessing"] =
        &PrintOutputProcessing<T>;
    CLI::GetSingleton().functionMap[data.tname]["PrintInputProcessing"] =
        &PrintInputProcessing<T>;
    CLI::GetSingleton().functionMap[data.tname]["ImportDecl"] = &ImportDecl<T>;

    // Add the ParamData object, then store.  This is necessary because we may
    // import more than one .so that uses CLI, so we have to keep the options
    // separate.  programName is a global variable from mlpack_main.hpp.
    CLI::Add(std::move(data));
    if (identifier != "verbose" && identifier != "copy_all_inputs")
      CLI::StoreSettings(programName);
    CLI::ClearSettings();
  }
};

} // namespace python
} // namespace bindings
} // namespace mlpack

#endif

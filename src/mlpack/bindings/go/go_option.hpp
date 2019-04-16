/**
 * @file go_option.hpp
 * @author Yasmine Dumouchel
 *
 * The Go option type.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_GOLANG_GO_OPTION_HPP
#define MLPACK_BINDINGS_GOLANG_GO_OPTION_HPP

#include <mlpack/core/util/param_data.hpp>
#include "get_param.hpp"
#include "get_type.hpp"
#include "default_param.hpp"
#include "get_printable_param.hpp"
#include "print_defn_input.hpp"
#include "print_defn_output.hpp"
#include "print_doc.hpp"
#include "print_import_decl.hpp"
#include "print_input_processing.hpp"
#include "print_method_config.hpp"
#include "print_method_init.hpp"
#include "print_model_util.hpp"
#include "print_output_processing.hpp"

namespace mlpack {
namespace bindings {
namespace go {

// Defined in mlpack_main.hpp.
extern std::string programName;

/**
 * The Go option class.
 */
template<typename T>
class GoOption
{
 public:
  /**
   * Construct a GoOption object.  When constructed, it will register itself
   * with CLI. The testName parameter is not used and added for compatibility
   * reasons.
   */
  GoOption(const T defaultValue,
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
    if (identifier == "verbose" /*|| identifier == "copy_all_inputs"*/)
      data.persistent = true;
    else
      data.persistent = false;
    data.cppType = cppName;

    data.value = boost::any(defaultValue);

    // Restore the parameters for this program.
    if (identifier != "verbose" /*&& identifier != "copy_all_inputs"*/)
      CLI::RestoreSettings(programName, false);

    // Set the function pointers that we'll need.  All of these function
    // pointers will be used by both the program that generates the .cpp,
    // the .h, and the .go binding files.
    CLI::GetSingleton().functionMap[data.tname]["GetParam"] = &GetParam<T>;
    CLI::GetSingleton().functionMap[data.tname]["GetPrintableParam"] =
        &GetPrintableParam<T>;

    CLI::GetSingleton().functionMap[data.tname]["DefaultParam"] =
        &DefaultParam<T>;

    CLI::GetSingleton().functionMap[data.tname]["PrintModelUtilCPP"] =
        &PrintModelUtilCPP<T>;
    CLI::GetSingleton().functionMap[data.tname]["PrintModelUtilH"] =
        &PrintModelUtilH<T>;
    CLI::GetSingleton().functionMap[data.tname]["PrintModelUtilGo"] =
        &PrintModelUtilGo<T>;
    CLI::GetSingleton().functionMap[data.tname]["PrintDefnInput"] =
        &PrintDefnInput<T>;
    CLI::GetSingleton().functionMap[data.tname]["PrintDefnOutput"] =
        &PrintDefnOutput<T>;
    CLI::GetSingleton().functionMap[data.tname]["PrintDoc"] = &PrintDoc<T>;
    CLI::GetSingleton().functionMap[data.tname]["PrintOutputProcessing"] =
        &PrintOutputProcessing<T>;
    CLI::GetSingleton().functionMap[data.tname]["PrintMethodConfig"] =
        &PrintMethodConfig<T>;
    CLI::GetSingleton().functionMap[data.tname]["PrintMethodInit"] =
        &PrintMethodInit<T>;
    CLI::GetSingleton().functionMap[data.tname]["ImportDecl"] =
        &ImportDecl<T>;
    CLI::GetSingleton().functionMap[data.tname]["PrintInputProcessing"] =
        &PrintInputProcessing<T>;
    CLI::GetSingleton().functionMap[data.tname]["GetType"] = &GetType<T>;

    // Add the ParamData object, then store.  This is necessary because we may
    // import more than one .so that uses CLI, so we have to keep the options
    // separate.  programName is a global variable from mlpack_main.hpp.
    CLI::Add(std::move(data));
    if (identifier != "verbose" /*&& identifier != "copy_all_inputs"*/)
      CLI::StoreSettings(programName);
    CLI::ClearSettings();
  }
};

} // namespace go
} // namespace bindings
} // namespace mlpack

#endif

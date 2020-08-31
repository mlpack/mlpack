/**
 * @file bindings/R/R_option.hpp
 * @author Yashwant Singh Parihar
 *
 * The R option type.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_R_R_OPTION_HPP
#define MLPACK_BINDINGS_R_R_OPTION_HPP
#include <mlpack/core/util/param_data.hpp>
#include "get_param.hpp"
#include "get_printable_param.hpp"
#include "print_input_param.hpp"
#include "print_input_processing.hpp"
#include "print_output_processing.hpp"
#include "print_doc.hpp"
#include "print_serialize_util.hpp"

namespace mlpack {
namespace bindings {
namespace r {

/**
 * The R option class.
 */
template<typename T>
class ROption
{
 public:
  /**
   * Construct a ROption object.  When constructed, it will register itself
   * with IO. The testName parameter is not used and added for compatibility
   * reasons.
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
   * @param * (testName) Is not used and added for compatibility reasons.
   */
  ROption(const T defaultValue,
          const std::string& identifier,
          const std::string& description,
          const std::string& alias,
          const std::string& cppName,
          const bool required = false,
          const bool input = true,
          const bool noTranspose = false,
          const std::string& /* testName */ = "")
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

    // Only "verbose" will be persistent.
    if (identifier == "verbose")
      data.persistent = true;
    else
      data.persistent = false;
    data.cppType = cppName;

    // Every parameter we'll get from R will have the correct type.
    data.value = boost::any(defaultValue);

    // Restore the parameters for this program.
    if (identifier != "verbose")
      IO::RestoreSettings(IO::ProgramName(), false);

    // Set the function pointers that we'll need.  All of these function
    // pointers will be used by both the program that generates the R, and
    // also the binding itself.  (The binding itself will only use GetParam,
    // GetPrintableParam, and GetRawParam.)
    IO::GetSingleton().functionMap[data.tname]["GetParam"] = &GetParam<T>;
    IO::GetSingleton().functionMap[data.tname]["GetPrintableParam"] =
        &GetPrintableParam<T>;

    // These are used by the R generator.
    IO::GetSingleton().functionMap[data.tname]["PrintDoc"] = &PrintDoc<T>;
    IO::GetSingleton().functionMap[data.tname]["PrintInputParam"] =
        &PrintInputParam<T>;
    IO::GetSingleton().functionMap[data.tname]["PrintOutputProcessing"] =
        &PrintOutputProcessing<T>;
    IO::GetSingleton().functionMap[data.tname]["PrintInputProcessing"] =
        &PrintInputProcessing<T>;
    IO::GetSingleton().functionMap[data.tname]["PrintSerializeUtil"] =
        &PrintSerializeUtil<T>;

    // Add the ParamData object, then store.  This is necessary because we may
    // import more than one .so or .o that uses IO, so we have to keep the
    // options separate.  programName is a global variable from mlpack_main.hpp.
    IO::Add(std::move(data));
    if (identifier != "verbose")
      IO::StoreSettings(IO::ProgramName());
    IO::ClearSettings();
  }
};

} // namespace r
} // namespace bindings
} // namespace mlpack

#endif

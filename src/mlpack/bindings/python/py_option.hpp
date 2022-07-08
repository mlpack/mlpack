/**
 * @file bindings/python/py_option.hpp
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
#include "is_serializable.hpp"

namespace mlpack {
namespace bindings {
namespace python {

/**
 * The Python option class.
 */
template<typename T>
class PyOption
{
 public:
  /**
   * Construct a PyOption object.  When constructed, it will register itself
   * with IO.
   */
  PyOption(const T defaultValue,
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

    // Every parameter we'll get from Python will have the correct type.
    data.value = defaultValue;

    // Set the function pointers that we'll need.  All of these function
    // pointers will be used by both the program that generates the pyx, and
    // also the binding itself.  (The binding itself will only use GetParam,
    // GetPrintableParam, and GetRawParam.)
    IO::AddFunction(data.tname, "GetParam", &GetParam<T>);
    IO::AddFunction(data.tname, "GetPrintableParam", &GetPrintableParam<T>);
    IO::AddFunction(data.tname, "DefaultParam", &DefaultParam<T>);

    // These are used by the pyx generator.
    IO::AddFunction(data.tname, "PrintClassDefn", &PrintClassDefn<T>);
    IO::AddFunction(data.tname, "PrintDefn", &PrintDefn<T>);
    IO::AddFunction(data.tname, "PrintDoc", &PrintDoc<T>);
    IO::AddFunction(data.tname, "PrintOutputProcessing",
        &PrintOutputProcessing<T>);
    IO::AddFunction(data.tname, "PrintInputProcessing",
        &PrintInputProcessing<T>);
    IO::AddFunction(data.tname, "ImportDecl", &ImportDecl<T>);
    IO::AddFunction(data.tname, "IsSerializable", &IsSerializable<T>);

    // Add the ParamData object to the IO class for the correct binding name.
    IO::AddParameter(bindingName, std::move(data));
  }
};

} // namespace python
} // namespace bindings
} // namespace mlpack

#endif

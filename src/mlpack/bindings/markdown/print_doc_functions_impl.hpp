/**
 * @file print_doc_functions_impl.hpp
 * @author Ryan Curtin
 *
 * Call out to different printing functionality for different binding languages.
 * If a new binding is added, this code must be modified.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_MARKDOWN_PRINT_DOC_FUNCTIONS_IMPL_HPP
#define MLPACK_BINDINGS_MARKDOWN_PRINT_DOC_FUNCTIONS_IMPL_HPP

#include "print_doc_functions.hpp"
#include "binding_info.hpp"

#include <mlpack/bindings/cli/print_doc_functions.hpp>
#include <mlpack/bindings/python/print_doc_functions.hpp>

namespace mlpack {
namespace bindings {
namespace markdown {

/**
 * Given the name of the binding, print the name for the current language (as
 * given by BindingInfo).
 */
inline std::string GetBindingName(const std::string& bindingName)
{
  if (BindingInfo::Language() == "cli")
  {
    return cli::GetBindingName(bindingName);
  }
  else if (BindingInfo::Language() == "python")
  {
    return python::GetBindingName(bindingName);
  }
  else
  {
    throw std::invalid_argument("PrintValue(): unknown "
        "BindingInfo::Language(): " + BindingInfo::Language() + "!");
  }
}

/**
 * Given a parameter type, print the corresponding value.
 */
template<typename T>
inline std::string PrintValue(const T& value, bool quotes)
{
  if (BindingInfo::Language() == "cli")
  {
    return cli::PrintValue(value, quotes);
  }
  else if (BindingInfo::Language() == "python")
  {
    return python::PrintValue(value, quotes);
  }
  else
  {
    throw std::invalid_argument("PrintValue(): unknown "
        "BindingInfo::Language(): " + BindingInfo::Language() + "!");
  }
}

/**
 * Given a parameter name, print its corresponding default value.
 */
inline std::string PrintDefault(const std::string& paramName)
{
  if (CLI::Parameters().count(paramName) == 0)
    throw std::invalid_argument("unknown parameter" + paramName + "!");

  const util::ParamData& d = CLI::Parameters()[paramName];

  std::ostringstream oss;

  if (d.required)
  {
    oss << "**--**";
  }
  else
  {
    if (BindingInfo::Language() == "cli")
    {
      oss << cli::PrintDefault(paramName);
    }
    else if (BindingInfo::Language() == "python")
    {
      oss << python::PrintDefault(paramName);
    }
    else
    {
      throw std::invalid_argument("PrintDefault: unknown "
          "BindingInfo::Language(): " + BindingInfo::Language() + "!");
    }
  }

  return oss.str();
}

/**
 * Print a dataset type parameter (add .csv and return).
 */
inline std::string PrintDataset(const std::string& dataset)
{
  if (BindingInfo::Language() == "cli")
  {
    return cli::PrintDataset(dataset);
  }
  else if (BindingInfo::Language() == "python")
  {
    return python::PrintDataset(dataset);
  }
  else
  {
    throw std::invalid_argument("PrintDataset(): unknown "
        "BindingInfo::Language(): " + BindingInfo::Language() + "!");
  }
}

/**
 * Print a model type parameter (add .bin and return).
 */
inline std::string PrintModel(const std::string& model)
{
  if (BindingInfo::Language() == "cli")
  {
    return cli::PrintModel(model);
  }
  else if (BindingInfo::Language() == "python")
  {
    return python::PrintModel(model);
  }
  else
  {
    throw std::invalid_argument("PrintModel(): unknown "
        "BindingInfo::Language(): " + BindingInfo::Language() + "!");
  }
}

/**
 * Given a program name and arguments for it, print what its invocation would
 * be.
 */
template<typename... Args>
std::string ProgramCall(const std::string& programName, Args... args)
{
  std::string s = "```\n";
  if (BindingInfo::Language() == "cli")
  {
    s += cli::ProgramCall(GetBindingName(programName), args...);
  }
  else if (BindingInfo::Language() == "python")
  {
    s += python::ProgramCall(GetBindingName(programName), args...);
  }
  else
  {
    throw std::invalid_argument("ProgramCall(): unknown "
        "BindingInfo::Language(): " + BindingInfo::Language() + "!");
  }
  s += "\n```";
  return s;
}

/**
 * Print what a user would type to invoke the given option name.  Note that the
 * name *must* exist in the CLI module.  (Note that because of the way
 * ProgramInfo is structured, this doesn't mean that all of the PARAM_*()
 * declarataions need to come before the PROGRAM_INFO() declaration.)
 */
inline std::string ParamString(const std::string& paramName)
{
  // These functions always put a '' around the parameter, so we will skip that
  // bit.
  std::string s;
  if (BindingInfo::Language() == "cli")
  {
    // The CLI bindings put a '' around the parameter, so skip that...
    s = cli::ParamString(paramName);
  }
  else if (BindingInfo::Language() == "python")
  {
    s = python::ParamString(paramName);
  }
  else
  {
    throw std::invalid_argument("ParamString(): unknown "
        "BindingInfo::Language(): " + BindingInfo::Language() + "!");
  }

  return s.substr(1, s.size() - 2);
}

/**
 * Print the user-encountered type of an option.
 */
inline std::string ParamType(const util::ParamData& d)
{
  std::string output;
  CLI::GetSingleton().functionMap[d.tname]["GetPrintableType"](d, NULL,
      &output);
  return output;
}

template<typename T>
inline bool IgnoreCheck(const T& t)
{
  if (BindingInfo::Language() == "cli")
  {
    return cli::IgnoreCheck(t);
  }
  else if (BindingInfo::Language() == "python")
  {
    return python::IgnoreCheck(t);
  }
  else
  {
    throw std::invalid_argument("IgnoreCheck(): unknown "
        "BindingInfo::Language(): " + BindingInfo::Language() + "!");
  }
}

} // namespace markdown
} // namespace bindings
} // namespace mlpack

#endif

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
#include "print_type_doc.hpp"
#include "get_printable_type.hpp"

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
 * Print the name of the given language.
 */
inline std::string PrintLanguage(const std::string& language)
{
  if (language == "cli")
  {
    return "CLI";
  }
  else if (language == "python")
  {
    return "Python";
  }
  else
  {
    throw std::invalid_argument("PrintLanguage(): unknown "
        "BindingInfo::Language(): " + BindingInfo::Language() + "!");
  }
}

/**
 * Print any imports that need to be done before using the binding.
 */
inline std::string PrintImport(const std::string& bindingName)
{
  if (BindingInfo::Language() == "cli")
  {
    return cli::PrintImport(bindingName);
  }
  else if (BindingInfo::Language() == "python")
  {
    return python::PrintImport(bindingName);
  }
  else
  {
    throw std::invalid_argument("PrintImport(): unknown "
        "BindingInfo::Language(): " + BindingInfo::Language() + "!");
  }
}

/**
 * Print any special information about output options.
 */
inline std::string PrintOutputOptionInfo()
{
  if (BindingInfo::Language() == "cli")
  {
    return cli::PrintOutputOptionInfo();
  }
  else if (BindingInfo::Language() == "python")
  {
    return python::PrintOutputOptionInfo();
  }
  else
  {
    throw std::invalid_argument("PrintOutputOptionInfo(): unknown "
        "BindingInfo::Language(): " + BindingInfo::Language() + "!");
  }
}

namespace priv {

// We'll need a fake class for printing model type documentation.
class mlpackModel
{
 public:
  // Fake serialization to make SFINAE work right for this type.
  template<typename Archive>
  void serialize(Archive&, const unsigned int) {}
};

} // namespace priv

/**
 * Print details about the different types for a language.
 */
inline std::string PrintTypeDocs()
{
  std::ostringstream oss;
  oss << "<div id=\"" << BindingInfo::Language()
      << "\" class=\"language-types\" markdown=\"1\">" << std::endl;
  oss << "## data formats" << std::endl;
  oss << "{: .language-types-h2 #" << BindingInfo::Language()
      << "_data-formats }" << std::endl;
  oss << std::endl;

  // Iterate through each of the types that we care about.
  oss << "mlpack bindings for " << PrintLanguage(BindingInfo::Language())
      << " take and return a restricted set of types, for simplicity.  These "
      << "include primitive types, matrix/vector types, categorical matrix "
      << "types, and model types.  Each type is detailed below." << std::endl;
  oss << std::endl;

  // Create fake ParamData to pass around.
  util::ParamData data;
  data.desc = "fake";
  data.name = "fake";
  data.tname = std::string(typeid(int).name());
  data.cppType = "int";
  data.alias = 'f';
  data.wasPassed = false;
  data.noTranspose = true;
  data.required = false;
  data.input = true;
  data.loaded = false;
  data.persistent = false;
  data.value = boost::any(int(0));

  oss << " - `" << GetPrintableType<int>(data) << "`: "
      << PrintTypeDoc<int>(data) << std::endl;

  data.tname = std::string(typeid(double).name());
  data.cppType = "double";
  data.value = boost::any(double(0.0));

  oss << " - `" << GetPrintableType<double>(data) << "`: "
      << PrintTypeDoc<double>(data) << std::endl;

  data.tname = std::string(typeid(bool).name());
  data.cppType = "double";
  data.value = boost::any(bool(0.0));

  oss << " - `" << GetPrintableType<bool>(data) << "`: "
      << PrintTypeDoc<bool>(data) << std::endl;

  data.tname = std::string(typeid(std::string).name());
  data.cppType = "std::string";
  data.value = boost::any(std::string(""));

  oss << " - `" << GetPrintableType<std::string>(data) << "`: "
      << PrintTypeDoc<std::string>(data) << std::endl;

  data.tname = std::string(typeid(std::vector<int>).name());
  data.cppType = "std::vector<int>";
  data.value = boost::any(std::vector<int>());

  oss << " - `" << GetPrintableType<std::vector<int>>(data) << "`: "
      << PrintTypeDoc<std::vector<int>>(data) << std::endl;

  data.tname = std::string(typeid(std::vector<std::string>).name());
  data.cppType = "std::vector<std::string>";
  data.value = boost::any(std::vector<std::string>());

  oss << " - `" << GetPrintableType<std::vector<std::string>>(data) << "`: "
      << PrintTypeDoc<std::vector<std::string>>(data) << std::endl;

  data.tname = std::string(typeid(arma::mat).name());
  data.cppType = "arma::mat";
  data.value = boost::any(arma::mat());

  oss << " - `" << GetPrintableType<arma::mat>(data) << "`: "
      << PrintTypeDoc<arma::mat>(data) << std::endl;

  data.tname = std::string(typeid(arma::Mat<size_t>).name());
  data.cppType = "arma::Mat<size_t>";
  data.value = boost::any(arma::Mat<size_t>());

  oss << " - `" << GetPrintableType<arma::Mat<size_t>>(data) << "`: "
      << PrintTypeDoc<arma::Mat<size_t>>(data) << std::endl;

  data.tname = std::string(typeid(arma::rowvec).name());
  data.cppType = "arma::rowvec";
  data.value = boost::any(arma::rowvec());

  oss << " - `" << GetPrintableType<arma::rowvec>(data) << "`: "
      << PrintTypeDoc<arma::rowvec>(data) << std::endl;

  data.tname = std::string(typeid(arma::Row<size_t>).name());
  data.cppType = "arma::Row<size_t>";
  data.value = boost::any(arma::Row<size_t>());

  oss << " - `" << GetPrintableType<arma::Row<size_t>>(data) << "`: "
      << PrintTypeDoc<arma::Row<size_t>>(data) << std::endl;

  data.tname = std::string(typeid(arma::vec).name());
  data.cppType = "arma::vec";
  data.value = boost::any(arma::vec());

  oss << " - `" << GetPrintableType<arma::vec>(data) << "`: "
      << PrintTypeDoc<arma::vec>(data) << std::endl;

  data.tname = std::string(typeid(arma::Col<size_t>).name());
  data.cppType = "arma::Col<size_t>";
  data.value = boost::any(arma::Col<size_t>());

  oss << " - `" << GetPrintableType<arma::Col<size_t>>(data) << "`: "
      << PrintTypeDoc<arma::Col<size_t>>(data) << std::endl;

  data.tname =
      std::string(typeid(std::tuple<data::DatasetInfo, arma::mat>).name());
  data.cppType = "std::tuple<data::DatasetInfo, arma::mat>";
  data.value = boost::any(std::tuple<data::DatasetInfo, arma::mat>());

  oss << " - `"
      << GetPrintableType<std::tuple<data::DatasetInfo, arma::mat>>(data)
      << "`: " << PrintTypeDoc<std::tuple<data::DatasetInfo, arma::mat>>(data)
      << std::endl;

  data.tname = std::string(typeid(priv::mlpackModel).name());
  data.cppType = "priv::mlpackModel";
  data.value = boost::any(new priv::mlpackModel());

  oss << " - `" << GetPrintableType<priv::mlpackModel*>(data) << "`: "
      << PrintTypeDoc<priv::mlpackModel*>(data) << std::endl;

  // Clean up memory.
  delete boost::any_cast<priv::mlpackModel*>(data.value);

  oss << std::endl << "</div>" << std::endl;

  return oss.str();
}

/**
 * Given a parameter type, print the corresponding value.
 */
template<typename T>
inline std::string PrintValue(const T& value, bool quotes)
{
  std::string result;
  if (BindingInfo::Language() == "cli")
  {
    result = cli::PrintValue(value, quotes);
  }
  else if (BindingInfo::Language() == "python")
  {
    result = python::PrintValue(value, quotes);
  }
  else
  {
    throw std::invalid_argument("PrintValue(): unknown "
        "BindingInfo::Language(): " + BindingInfo::Language() + "!");
  }

  return "`" + result + "`";
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
  std::string result;
  if (BindingInfo::Language() == "cli")
  {
    result = cli::PrintDataset(dataset);
  }
  else if (BindingInfo::Language() == "python")
  {
    result = python::PrintDataset(dataset);
  }
  else
  {
    throw std::invalid_argument("PrintDataset(): unknown "
        "BindingInfo::Language(): " + BindingInfo::Language() + "!");
  }

  return "`" + result + "`";
}

/**
 * Print a model type parameter.
 */
inline std::string PrintModel(const std::string& model)
{
  std::string result;
  if (BindingInfo::Language() == "cli")
  {
    result = cli::PrintModel(model);
  }
  else if (BindingInfo::Language() == "python")
  {
    result = python::PrintModel(model);
  }
  else
  {
    throw std::invalid_argument("PrintModel(): unknown "
        "BindingInfo::Language(): " + BindingInfo::Language() + "!");
  }

  return "`" + result + "`";
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

  return "`" + s.substr(1, s.size() - 2) + "`";
}

/**
 * Print the user-encountered type of an option.
 */
inline std::string ParamType(const util::ParamData& d)
{
  std::string output;
  CLI::GetSingleton().functionMap[d.tname]["GetPrintableType"](d, NULL,
      &output);
  return "`" + output + "`";
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

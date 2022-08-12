/**
 * @file bindings/markdown/print_doc_functions_impl.hpp
 * @author Ryan Curtin
 *
 * Calls out to different printing functionality for different binding languages.
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
#include <mlpack/bindings/python/wrapper_functions.hpp>
#include <mlpack/bindings/julia/print_doc_functions.hpp>
#include <mlpack/bindings/go/print_doc_functions.hpp>
#include <mlpack/bindings/R/print_doc_functions.hpp>

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
  else if (BindingInfo::Language() == "julia")
  {
    return julia::GetBindingName(bindingName);
  }
  else if (BindingInfo::Language() == "go")
  {
    return go::GetBindingName(bindingName);
  }
  else if (BindingInfo::Language() == "r")
  {
    return r::GetBindingName(bindingName);
  }
  else
  {
    throw std::invalid_argument("GetBindingName(): unknown "
        "BindingInfo::Language(): " + BindingInfo::Language() + "!");
  }
}

/**
 * Given the name of the binding, print the name for the wrapper for
 * current language.
 */
inline std::string GetWrapperName(const std::string& bindingName)
{
  if (BindingInfo::Language() == "python")
  {
    return "class " + python::GetClassName(bindingName);
  }
  else
  {
    throw std::invalid_argument("GetWrapperName(): unknown "
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
  else if (language == "julia")
  {
    return "Julia";
  }
  else if (language == "go")
  {
    return "Go";
  }
  else if (language == "r")
  {
    return "R";
  }
  else
  {
    throw std::invalid_argument("PrintLanguage(): unknown "
        "BindingInfo::Language(): " + BindingInfo::Language() + "!");
  }
}

/**
 * Print any import that needs to be done before using the binding.
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
  else if (BindingInfo::Language() == "julia")
  {
    return julia::PrintImport(bindingName);
  }
  else if (BindingInfo::Language() == "go")
  {
    return go::PrintImport();
  }
  else if (BindingInfo::Language() == "r")
  {
    return r::PrintImport();
  }
  else
  {
    throw std::invalid_argument("PrintImport(): unknown "
        "BindingInfo::Language(): " + BindingInfo::Language() + "!");
  }
}

/**
 * Print any special information about input options.
 */
inline std::string PrintInputOptionInfo()
{
  if (BindingInfo::Language() == "cli")
  {
    return cli::PrintInputOptionInfo();
  }
  else if (BindingInfo::Language() == "python")
  {
    return python::PrintInputOptionInfo();
  }
  else if (BindingInfo::Language() == "julia")
  {
    return julia::PrintInputOptionInfo();
  }
  else if (BindingInfo::Language() == "go")
  {
    return go::PrintInputOptionInfo();
  }
  else if (BindingInfo::Language() == "r")
  {
    return r::PrintInputOptionInfo();
  }
  else
  {
    throw std::invalid_argument("PrintInputOptionInfo(): unknown "
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
  else if (BindingInfo::Language() == "julia")
  {
    return julia::PrintOutputOptionInfo();
  }
  else if (BindingInfo::Language() == "go")
  {
    return go::PrintOutputOptionInfo();
  }
  else if (BindingInfo::Language() == "r")
  {
    return r::PrintOutputOptionInfo();
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

// Utility function that returns the first word (as delimited by spaces) of a
// string.
inline std::string ToUnderscores(const std::string& str)
{
  std::string ret(str);
  std::replace(ret.begin(), ret.end(), ' ', '_');
  std::replace(ret.begin(), ret.end(), '{', '_');
  std::replace(ret.begin(), ret.end(), '}', '_');
  return ret;
}

/**
 * Print details about the different types of a language.
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
      << "types, and model types. Each type is detailed below." << std::endl;
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
  data.value = MLPACK_ANY(int(0));

  std::string type = GetPrintableType<int>(data);
  oss << " - `" << type << "`{: #doc_" << BindingInfo::Language() << "_"
      << ToUnderscores(type) << " }: " << PrintTypeDoc<int>(data) << std::endl;

  data.tname = std::string(typeid(double).name());
  data.cppType = "double";
  data.value = MLPACK_ANY(double(0.0));

  type = GetPrintableType<double>(data);
  oss << " - `" << type << "`{: #doc_" << BindingInfo::Language() << "_"
      << ToUnderscores(type) << " }: " << PrintTypeDoc<double>(data)
      << std::endl;

  data.tname = std::string(typeid(bool).name());
  data.cppType = "double";
  data.value = MLPACK_ANY(bool(0.0));

  type = GetPrintableType<bool>(data);
  oss << " - `" << type << "`{: #doc_" << BindingInfo::Language() << "_"
      << ToUnderscores(type) << " }: " << PrintTypeDoc<bool>(data) << std::endl;

  data.tname = std::string(typeid(std::string).name());
  data.cppType = "std::string";
  data.value = MLPACK_ANY(std::string(""));

  type = GetPrintableType<std::string>(data);
  oss << " - `" << type << "`{: #doc_" << BindingInfo::Language() << "_"
      << ToUnderscores(type) << " }: " << PrintTypeDoc<std::string>(data)
      << std::endl;

  data.tname = std::string(typeid(std::vector<int>).name());
  data.cppType = "std::vector<int>";
  data.value = MLPACK_ANY(std::vector<int>());

  type = GetPrintableType<std::vector<int>>(data);
  oss << " - `" << type << "`{: #doc_" << BindingInfo::Language() << "_"
      << ToUnderscores(type) << " }: " << PrintTypeDoc<std::vector<int>>(data)
      << std::endl;

  data.tname = std::string(typeid(std::vector<std::string>).name());
  data.cppType = "std::vector<std::string>";
  data.value = MLPACK_ANY(std::vector<std::string>());

  type = GetPrintableType<std::vector<std::string>>(data);
  oss << " - `" << type << "`{: " << "#doc_" << BindingInfo::Language() << "_"
      << ToUnderscores(type) << " }: "
      << PrintTypeDoc<std::vector<std::string>>(data) << std::endl;

  data.tname = std::string(typeid(arma::mat).name());
  data.cppType = "arma::mat";
  data.value = MLPACK_ANY(arma::mat());

  type = GetPrintableType<arma::mat>(data);
  oss << " - `" << type << "`{: #doc_" << BindingInfo::Language() << "_"
      << ToUnderscores(type) << " }: " << PrintTypeDoc<arma::mat>(data)
      << std::endl;

  data.tname = std::string(typeid(arma::Mat<size_t>).name());
  data.cppType = "arma::Mat<size_t>";
  data.value = MLPACK_ANY(arma::Mat<size_t>());

  type = GetPrintableType<arma::Mat<size_t>>(data);
  oss << " - `" << type << "`{: #doc_" << BindingInfo::Language() << "_"
      << ToUnderscores(type) << " }: " << PrintTypeDoc<arma::Mat<size_t>>(data)
      << std::endl;

  data.tname = std::string(typeid(arma::rowvec).name());
  data.cppType = "arma::rowvec";
  data.value = MLPACK_ANY(arma::rowvec());
  const std::string& rowType = GetPrintableType<arma::rowvec>(data);

  oss << " - `" << rowType << "`{: #doc_" << BindingInfo::Language() << "_"
      << ToUnderscores(rowType) << " }: " << PrintTypeDoc<arma::rowvec>(data)
      << std::endl;

  data.tname = std::string(typeid(arma::Row<size_t>).name());
  data.cppType = "arma::Row<size_t>";
  data.value = MLPACK_ANY(arma::Row<size_t>());
  const std::string& urowType = GetPrintableType<arma::Row<size_t>>(data);

  oss << " - `" << urowType << "`{: #doc_" << BindingInfo::Language() << "_"
      << ToUnderscores(urowType) << " }: "
      << PrintTypeDoc<arma::Row<size_t>>(data)
      << std::endl;

  data.tname = std::string(typeid(arma::vec).name());
  data.cppType = "arma::vec";
  data.value = MLPACK_ANY(arma::vec());
  const std::string& colType = GetPrintableType<arma::vec>(data);

  // For some languages there is no distinction between column and row vectors.
  // If that is the case, then don't print both.
  if (colType != rowType)
  {
    oss << " - `" << colType << "`{: #doc_" << BindingInfo::Language() << "_"
        << ToUnderscores(colType) << " }: " << PrintTypeDoc<arma::vec>(data)
        << std::endl;
  }

  data.tname = std::string(typeid(arma::Col<size_t>).name());
  data.cppType = "arma::Col<size_t>";
  data.value = MLPACK_ANY(arma::Col<size_t>());
  const std::string& ucolType = GetPrintableType<arma::Col<size_t>>(data);

  // For some languages there is no distinction between column and row vectors.
  // If that is the case, then don't print both.
  if (ucolType != urowType)
  {
    oss << " - `" << ucolType << "`{ #doc_" << BindingInfo::Language() << "_"
        << ToUnderscores(ucolType) << " }: "
        << PrintTypeDoc<arma::Col<size_t>>(data) << std::endl;
  }

  data.tname =
      std::string(typeid(std::tuple<data::DatasetInfo, arma::mat>).name());
  data.cppType = "std::tuple<data::DatasetInfo, arma::mat>";
  data.value = MLPACK_ANY(std::tuple<data::DatasetInfo, arma::mat>());

  type = GetPrintableType<std::tuple<data::DatasetInfo, arma::mat>>(data);
  oss << " - `" << type << "`{: #doc_" << BindingInfo::Language() << "_"
      << ToUnderscores(type) << " }: "
      << PrintTypeDoc<std::tuple<data::DatasetInfo, arma::mat>>(data)
      << std::endl;

  data.tname = std::string(typeid(priv::mlpackModel).name());
  data.cppType = "mlpackModel";
  data.value = MLPACK_ANY(new priv::mlpackModel());

  type = GetPrintableType<priv::mlpackModel*>(data);
  oss << " - `" << type << "`{: #doc_" << BindingInfo::Language()
      << "_model }: " << PrintTypeDoc<priv::mlpackModel*>(data) << std::endl;

  // Clean up memory.
  delete MLPACK_ANY_CAST<priv::mlpackModel*>(data.value);

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
  else if (BindingInfo::Language() == "julia")
  {
    result = julia::PrintValue(value, quotes);
  }
  else if (BindingInfo::Language() == "go")
  {
    result = go::PrintValue(value, quotes);
  }
  else if (BindingInfo::Language() == "r")
  {
    result = r::PrintValue(value, quotes);
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
inline std::string PrintDefault(const std::string& bindingName,
                                const std::string& paramName)
{
  util::Params p = IO::Parameters(bindingName);
  if (p.Parameters().count(paramName) == 0)
    throw std::invalid_argument("unknown parameter" + paramName + "!");

  util::ParamData& d = p.Parameters()[paramName];

  std::ostringstream oss;

  if (d.required)
  {
    oss << "**--**";
  }
  else
  {
    if (BindingInfo::Language() == "cli")
    {
      oss << cli::PrintDefault(bindingName, paramName);
    }
    else if (BindingInfo::Language() == "python")
    {
      oss << python::PrintDefault(bindingName, paramName);
    }
    else if (BindingInfo::Language() == "julia")
    {
      oss << julia::PrintDefault(bindingName, paramName);
    }
    else if (BindingInfo::Language() == "go")
    {
      oss << go::PrintDefault(bindingName, paramName);
    }
    else if (BindingInfo::Language() == "r")
    {
      oss << r::PrintDefault(bindingName, paramName);
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
  else if (BindingInfo::Language() == "julia")
  {
    result = julia::PrintDataset(dataset);
  }
  else if (BindingInfo::Language() == "go")
  {
    result = go::PrintDataset(dataset);
  }
  else if (BindingInfo::Language() == "r")
  {
    result = r::PrintDataset(dataset);
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
  else if (BindingInfo::Language() == "julia")
  {
    result = julia::PrintModel(model);
  }
  else if (BindingInfo::Language() == "go")
  {
    result = go::PrintModel(model);
  }
  else if (BindingInfo::Language() == "r")
  {
    result = r::PrintModel(model);
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
  std::string s;
  if (BindingInfo::Language() == "cli")
  {
    s += "```bash\n";
    s += cli::ProgramCall(programName, args...);
  }
  else if (BindingInfo::Language() == "python")
  {
    s += "```python\n";
    s += python::ProgramCall(programName, args...);
  }
  else if (BindingInfo::Language() == "julia")
  {
    // Julia's ProgramCall() with a set of arguments will automatically enclose
    // the text in Markdown code, so we don't need to.
    s += julia::ProgramCall(programName, args...);
  }
  else if (BindingInfo::Language() == "go")
  {
    s += "```go\n";
    s += go::ProgramCall(programName, args...);
  }
  else if (BindingInfo::Language() == "r")
  {
    s += "```R\n";
    s += r::ProgramCall(true, programName, args...);
  }
  else
  {
    throw std::invalid_argument("ProgramCall(): unknown "
        "BindingInfo::Language(): " + BindingInfo::Language() + "!");
  }

  // Close the Markdown code block, but only if we opened one.
  if (BindingInfo::Language() != "julia")
    s += "\n```";
  return s;
}

/**
 * Given a program name, print a call assuming that all arguments are specified.
 */
inline std::string ProgramCall(const std::string& programName)
{
  std::string s = "```";
  util::Params p = IO::Parameters(programName);
  if (BindingInfo::Language() == "cli")
  {
    // Strip non-CLI options.
    p.Parameters().erase("copy_all_inputs");
    p.Parameters().erase("check_input_matrices");

    s += "bash\n";
    std::string import = PrintImport(GetBindingName(programName));
    if (import.size() > 0)
      s += "$ " + import + "\n";
    s += cli::ProgramCall(p, programName);
  }
  else if (BindingInfo::Language() == "python")
  {
    // Strip non-Python options.
    p.Parameters().erase("help");
    p.Parameters().erase("info");
    p.Parameters().erase("version");

    s += "python\n";
    std::string import = PrintImport(programName);
    if (import.size() > 0)
      s += ">>> " + import + "\n";
    s += python::ProgramCall(p, programName);
  }
  else if (BindingInfo::Language() == "julia")
  {
    // Strip non-Julia options.
    p.Parameters().erase("help");
    p.Parameters().erase("info");
    p.Parameters().erase("version");
    p.Parameters().erase("copy_all_inputs");
    p.Parameters().erase("check_input_matrices");

    s += "julia\n";
    std::string import = PrintImport(programName);
    if (import.size() > 0)
      s += "julia> " + import + "\n";
    s += julia::ProgramCall(p, programName);
  }
  else if (BindingInfo::Language() == "go")
  {
    // Strip non-Go options.
    p.Parameters().erase("help");
    p.Parameters().erase("info");
    p.Parameters().erase("version");
    p.Parameters().erase("copy_all_inputs");
    p.Parameters().erase("check_input_matrices");

    s += "go\n";
    std::string import = PrintImport(programName);
    if (import.size() > 0)
      s += import + "\n";
    s += go::ProgramCall(p, programName);
  }
  else if (BindingInfo::Language() == "r")
  {
    // Strip non-R options.
    p.Parameters().erase("help");
    p.Parameters().erase("info");
    p.Parameters().erase("version");
    p.Parameters().erase("copy_all_inputs");
    p.Parameters().erase("check_input_matrices");

    s += "R\n";
    std::string import = PrintImport(programName);
    if (import.size() > 0)
      s += "R> " + import + "\n";
    s += r::ProgramCall(p, programName);
  }
  else
  {
    throw std::invalid_argument("ProgramCall(): unknown "
        "BindingInfo::Language(): " + BindingInfo::Language() + "!");
  }
  s += "\n```\n";
  return s;
}

/**
 * Print what a user would type to invoke the given option name.  Note that the
 * name *must* exist in the CLI module.  (Note that because of the way
 * BINDING_LONG_DESC() and BINDING_EXAMPLE() is structured, this doesn't mean
 * that all of the PARAM_*() declarataions need to come before
 * BINDING_LONG_DESC() and BINDING_EXAMPLE() declaration.)
 */
inline std::string ParamString(const std::string& bindingName,
                               const std::string& paramName)
{
  // These functions always put a '' around the parameter, so we will skip that
  // bit.
  std::string s;
  if (BindingInfo::Language() == "cli")
  {
    // The CLI bindings put a '' around the parameter, so skip that...
    s = cli::ParamString(bindingName, paramName);
  }
  else if (BindingInfo::Language() == "python")
  {
    s = python::ParamString(paramName);
  }
  else if (BindingInfo::Language() == "julia")
  {
    s = julia::ParamString(paramName);
  }
  else if (BindingInfo::Language() == "go")
  {
    s = go::ParamString(paramName);
  }
  else if (BindingInfo::Language() == "r")
  {
    s = r::ParamString(paramName);
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
inline std::string ParamType(util::Params& p, util::ParamData& d)
{
  std::string output;
  p.functionMap[d.tname]["GetPrintableType"](d, NULL, &output);
  // We want to make this a link to the type documentation.
  std::string anchorType = output;
  bool result;
  p.functionMap[d.tname]["IsSerializable"](d, NULL, &result);
  if (result)
    anchorType = "model";

  return "[`" + output + "`](#doc_" + BindingInfo::Language() + "_" +
      ToUnderscores(anchorType) + ")";
}

inline std::string ImportExtLib()
{
  std::string s;
  if (BindingInfo::Language() == "python")
  {
    s = python::ImportExtLib();
  }
  else
  {
    throw std::invalid_argument("ImportExtLib(): unknown "
        "BindingInfo::Language(): " + BindingInfo::Language() + "!");
  }

  return s;
}

inline std::string ImportSplit()
{
  std::string s;
  if (BindingInfo::Language() == "python")
  {
    s = python::ImportSplit();
  }
  else
  {
    throw std::invalid_argument("ImportSplit(): unknown "
        "BindingInfo::Language(): " + BindingInfo::Language() + "!");
  }

  return s;
}

inline std::string ImportThis(const std::string& groupName)
{
  std::string s;
  if (BindingInfo::Language() == "python")
  {
    s = python::ImportThis(groupName);
  }
  else
  {
    throw std::invalid_argument("ImportThis(): unknown "
        "BindingInfo::Language(): " + BindingInfo::Language() + "!");
  }

  return s;
}

inline std::string SplitTrainTest(const std::string& datasetName,
                                  const std::string& labelName,
                                  const std::string& trainDataset,
                                  const std::string& trainLabels,
                                  const std::string& testDataset,
                                  const std::string& testLabels,
                                  const std::string& splitRatio)
{
  std::string s;
  if (BindingInfo::Language() == "python")
  {
    s = python::SplitTrainTest(datasetName, labelName,
        trainDataset, trainLabels, testDataset, testLabels, splitRatio);
  }
  else
  {
    throw std::invalid_argument("SplitTrainTest(): unknown "
        "BindingInfo::Language(): " + BindingInfo::Language() + "!");
  }

  return s;
}

inline std::string GetDataset(const std::string& datasetName,
                              const std::string& url)
{
  std::string s;
  if (BindingInfo::Language() == "python")
  {
    s = python::GetDataset(datasetName, url);
  }
  else
  {
    throw std::invalid_argument("GetDataset(): unknown "
        "BindingInfo::Language(): " + BindingInfo::Language() + "!");
  }

  return s;
}

template<typename... Args>
std::string CreateObject(const std::string& bindingName,
                         const std::string& objectName,
                         const std::string& groupName,
                         Args... args)
{
  std::string s;
  if (BindingInfo::Language() == "python")
  {
    s = python::CreateObject(bindingName, objectName,
        groupName, args...);
  }
  else
  {
    throw std::invalid_argument("CreateObject(): unknown "
        "BindingInfo::Language(): " + BindingInfo::Language() + "!");
  }

  return s;
}

inline std::string CreateObject(const std::string& bindingName,
                                const std::string& objectName,
                                const std::string& groupName)
{
  std::string s;
  if (BindingInfo::Language() == "python")
  {
    s = python::CreateObject(bindingName, objectName, groupName);
  }
  else
  {
    throw std::invalid_argument("CreateObject(): unknown "
        "BindingInfo::Language(): " + BindingInfo::Language() + "!");
  }

  return s;
}

template<typename... Args>
std::string CallMethod(const std::string& bindingName,
                       const std::string& objectName,
                       const std::string& methodName,
                       Args... args)
{
  std::string s;
  if (BindingInfo::Language() == "python")
  {
    s = python::CallMethod(bindingName, objectName,
        methodName, args...);
  }
  else
  {
    throw std::invalid_argument("CallMethod(): unknown "
        "BindingInfo::Language(): " + BindingInfo::Language() + "!");
  }

  return s;
}

template<typename T>
inline bool IgnoreCheck(const std::string& bindingName, const T& t)
{
  if (BindingInfo::Language() == "cli")
  {
    return cli::IgnoreCheck(t);
  }
  else if (BindingInfo::Language() == "python")
  {
    return python::IgnoreCheck(bindingName, t);
  }
  else if (BindingInfo::Language() == "julia")
  {
    return julia::IgnoreCheck(bindingName, t);
  }
  else if (BindingInfo::Language() == "go")
  {
    return go::IgnoreCheck(bindingName, t);
  }
  else if (BindingInfo::Language() == "r")
  {
    return r::IgnoreCheck(bindingName, t);
  }
  else
  {
    throw std::invalid_argument("IgnoreCheck(): unknown "
        "BindingInfo::Language(): " + BindingInfo::Language() + "!");
  }
}

inline std::string GetMappedName(const std::string& methodName)
{
  if (BindingInfo::Language() == "python")
  {
    return python::GetMappedName(methodName);
  }
  else
  {
    throw std::invalid_argument("GetMappedName(): unknown "
        "BindingInfo::Language(): " + BindingInfo::Language() + "!");
  }
}

inline std::string GetWrapperLink(const std::string& bindingName)
{
  if (BindingInfo::Language() == "python")
  {
    return "class-" + bindingName;
  }
  else
  {
    throw std::invalid_argument("GetWrapperLink(): unknown "
        "BindingInfo::Language(): " + BindingInfo::Language() + "!");
  }
}

} // namespace markdown
} // namespace bindings
} // namespace mlpack

#endif

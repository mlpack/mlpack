/**
 * @file print_doc_functions_impl.hpp
 * @author Vasyl Teliman
 *
 * Implementation of functions to generate documentation
 */
#ifndef MLPACK_BINDINGS_JAVA_PRINT_DOC_FUNCTIONS_IMPL_HPP
#define MLPACK_BINDINGS_JAVA_PRINT_DOC_FUNCTIONS_IMPL_HPP

#include <sstream>

namespace mlpack {
namespace bindings {
namespace java {

/**
 * Base case for template recursion below
 */
inline void PrintInputCallParam(std::ostream&)
{}

/**
 * Generate input arguments for function call
 */
template <typename T, typename... Args>
void PrintInputCallParam(std::ostream& os, const std::string& name,
    const T& value, const Args&... args)
{
  const util::ParamData& param = CLI::Parameters().at(name);
  if (param.input)
  {
    os << std::endl;
    if (param.cppType == "bool")
    {
      os << "    .put(\"" << name << "\", true)";
    }
    else if (param.cppType == "arma::mat" ||
        param.cppType == "arma::vec" ||
        param.cppType == "arma::rowvec" ||
        param.cppType == "std::tuple<mlpack::data::DatasetInfo, arma::mat>" ||
        param.cppType == "arma::Mat<size_t>" ||
        param.cppType == "arma::Row<size_t>" ||
        param.cppType == "arma::Col<size_t>")
    {
      os << "    .put(\"" << name << "\", Nd4j.createFromNpyFile(new File(\""
         << value << ".npy\"))";
    }
    else if (param.cppType == "std::string")
    {
      os << "    .put(\"" << name << "\", \"" << value << "\")";
    }
    else
    {
      os << "    .put(\"" << name << "\", " << value << ")";
    }
  }

  PrintInputCallParam(os, args...);
}

/**
 * Base case for template recursion below
 */
inline void PrintOutputCallParam(std::ostream&)
{}

/**
 * Print output values for a function call
 */
template <typename T, typename... Args>
void PrintOutputCallParam(std::ostream& os, const std::string& name,
    const T& /* value */, const Args&... args)
{
  const util::ParamData& param = CLI::Parameters().at(name);
  if (!param.input)
  {
    os << std::endl;
    std::string type;
    CLI::GetSingleton().functionMap[param.tname]["GetJavaType"]
        (param, nullptr, (void*) &type);

    os << type << " " << name << " = params.get(\"" << name << "\", "
       << type << ".class);";
  }

  PrintOutputCallParam(os, args...);
}

/**
 * Generate documentation for string parameter
 */
inline std::string ParamString(const std::string& paramName)
{
  return "{@code " + paramName + '}';
}

/**
 * Generate documentation for value parameter
 */
template<typename T>
inline std::string PrintValue(const T& value, bool quotes)
{
  std::ostringstream oss;
  oss << std::boolalpha;
  const char* quote = quotes ? "'" : "";
  oss << "{@code " << quote << value << quote << "}";
  return oss.str();
}

/**
 * Generate documentation for dataset parameter
 */
inline std::string PrintDataset(const std::string& datasetName)
{
  return "{@code " + datasetName + '}';
}

/**
 * Generate documentation for model parameter
 */
inline std::string PrintModel(const std::string& modelName)
{
  return "{@code " + modelName + '}';
}

/**
 * Generate documentation string for function calls
 */
template<typename... Args>
std::string ProgramCall(const std::string& /* programName */, Args... args)
{
  std::ostringstream oss;
  oss << std::boolalpha;

  oss << "<pre>" << std::endl
      << "{@code" << std::endl;

  oss << "Params params = new Params()";

  PrintInputCallParam(oss, args...);

  oss << ";" << std::endl
      << std::endl
      << "run(params);" << std::endl;

  PrintOutputCallParam(oss, args...);

  oss << std::endl
      << "}" << std::endl
      << "</pre>" << std::endl;

  return oss.str();
}

/**
 * Generate function call without arguments
 */
inline std::string ProgramCall(const std::string& /* programName */)
{
  std::ostringstream oss;

  oss << "<pre>" << std::endl
      << "{@code" << std::endl
      << "Params params = new Params();" << std::endl
      << "run(params);" << std::endl
      << "}" << std::endl
      << "</pre>" << std::endl;

  return oss.str();
}

/**
 * Ignore check only for output parameters
 */
inline bool IgnoreCheck(const std::string& paramName)
{
  return !CLI::Parameters()[paramName].input;
}

/**
 * Ignore check if at least one parameter is output parameter
 */
inline bool IgnoreCheck(const std::vector<std::string>& constraints)
{
  for (size_t i = 0; i < constraints.size(); ++i)
  {
    if (!CLI::Parameters()[constraints[i]].input)
      return true;
  }

  return false;
}

/**
 * Ignore check if at least one parameter is output parameter
 */
inline bool IgnoreCheck(
  const std::vector<std::pair<std::string, bool>>& constraints,
  const std::string& paramName)
{
  for (size_t i = 0; i < constraints.size(); ++i)
  {
    if (!CLI::Parameters()[constraints[i].first].input)
      return true;
  }

  return !CLI::Parameters()[paramName].input;
}

} // namespace java
} // namespace bindings
} // namespace mlpack

#endif

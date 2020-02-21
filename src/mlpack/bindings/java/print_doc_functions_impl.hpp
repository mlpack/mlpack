#ifndef MLPACK_BINDINGS_JAVA_PRINT_DOC_FUNCTIONS_IMPL_HPP
#define MLPACK_BINDINGS_JAVA_PRINT_DOC_FUNCTIONS_IMPL_HPP

#include <sstream>

namespace mlpack {
namespace bindings {
namespace java {

inline void PrintInputCallParam(std::ostream&)
{}

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

inline void PrintOutputCallParam(std::ostream&)
{}

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

inline std::string ParamString(const std::string& paramName)
{
  return "{@code " + paramName + '}';
}

template<typename T>
inline std::string PrintValue(const T& value, bool quotes)
{
  std::ostringstream oss;
  oss << std::boolalpha;
  const char* quote = quotes ? "'" : "";
  oss << "{@code " << quote << value << quote << "}";
  return oss.str();
}

inline std::string PrintDataset(const std::string& datasetName)
{
  return "{@code " + datasetName + '}';
}

inline std::string PrintModel(const std::string& modelName)
{
  return "{@code " + modelName + '}';
}

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

inline bool IgnoreCheck(const std::string& paramName)
{
  return !CLI::Parameters()[paramName].input;
}

inline bool IgnoreCheck(const std::vector<std::string>& constraints)
{
  for (size_t i = 0; i < constraints.size(); ++i)
  {
    if (!CLI::Parameters()[constraints[i]].input)
      return true;
  }

  return false;
}

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

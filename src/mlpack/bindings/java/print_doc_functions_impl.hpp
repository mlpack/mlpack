#ifndef MLPACK_BINDINGS_JAVA_PRINT_DOC_FUNCTIONS_IMPL_HPP
#define MLPACK_BINDINGS_JULIA_PRINT_DOC_FUNCTIONS_IMPL_HPP

namespace mlpack {
namespace bindings {
namespace java {

inline std::string ParamString(const std::string& paramName)
{
  return "{@code " + paramName + '}';
}

template<typename T>
inline std::string PrintValue(const T& value, bool quotes) 
{
  std::ostringstream oss;
  if (quotes) oss << "`";
  oss << value;
  if (quotes) oss << "`";
  return oss.str();
}

template<>
inline std::string PrintValue(const bool& value, bool quotes) 
{
  std::string result = value ? "true" : "false";
  if (quotes) result = "`" + std::move(result) + "`";
  return result;
}

inline std::string PrintDataset(const std::string& datasetName) 
{
  return "`" + datasetName + "`";
}

inline std::string PrintModel(const std::string& modelName)
{
  return "`" + modelName + "`";
}

template<typename... Args>
std::string ProgramCall(const std::string& programName, Args... args)
{
  // TODO: finish it
  return "`" + programName + "(<args>)`";
}

inline std::string ProgramCall(const std::string& programName)
{
  return "`" + programName + "()`";
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

}
}
}

#endif

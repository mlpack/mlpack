/**
 * @file utilities/save_restore_utility_impl.hpp
 * @author Neil Slagle
 *
 * The SaveRestoreUtility provides helper functions in saving and
 *   restoring models.  The current output file type is XML.
 *
 * @experimental
 */
#ifndef SAVE_RESTORE_MODEL_HPP
#error "Do not include this header directly."
#endif

using namespace mlpack;
using namespace mlpack::utilities;

template<typename T>
T& SaveRestoreUtility::LoadParameter(T& t, std::string name)
{
  std::map<std::string, std::string>::iterator it = parameters.find(name);
  if (it != parameters.end())
  {
    std::string value = (*it).second;
    std::istringstream input (value);
    input >> t;
    return t;
  }
  else
  {
    Log::Fatal << "Missing the correct name\n";
  }
}
template<typename T>
void SaveRestoreUtility::SaveParameter(T& t, std::string name)
{
  std::ostringstream output;
  output << t;
  parameters[name] = output.str();
}

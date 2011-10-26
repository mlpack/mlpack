#ifndef __MLPACK_CORE_MODEL_SAVE_RESTORE_MODEL_HPP
#error "Do not include this header directly."
#endif

namespace mlpack {
namespace model {

template<typename T>
T& SaveRestoreModel::loadParameter (T& t, std::string name)
{
  std::map<std::string, std::string>::iterator it = parameters.find (name);
  if (it != parameters.end ())
  {
    std::string value = (*it).second;
    std::istringstream input (value);
    input >> t;
    return t;
  }
  else
  {
    errx (1, "Missing the correct name\n");
  }
}
template<typename T>
void SaveRestoreModel::saveParameter (T& t, std::string name)
{
  std::ostringstream output;
  output << t;
  parameters[name] = output.str();
}

}; // namespace model
}; // namespace mlpack


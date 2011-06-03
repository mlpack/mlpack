#ifndef MLPACK_IO_IO_IMPL_H
#define MLPACK_IO_IO_IMPL_H

//Include option.h here because it requires IO but is also templated
#include "option.h"

/**
* @brief Adds a parameter to IO, making 
*   it accessibile via GetValue & CheckValue.
*
* @tparam T The type of the parameter.
* @param identifier The name of the parameter, eg foo in bar/foo.
* @param description A string description of the parameter.
* @param parent The name of the parent of the parameter, 
*   eg bar/foo in bar/foo/buzz.
* @param required If required, the program will refuse to run 
*   unless the parameter is specified.
*/
template<typename T>
void IO::Add(const char* identifier, 
             const char* description, 
             const char* parent, 
             bool required) {

  po::options_description& desc = IO::GetSingleton().desc;
  //Generate the full pathname and insert the node into the hierarchy
  std::string tmp = TYPENAME(T);
  std::string path = IO::GetSingleton().ManageHierarchy(identifier, parent, 
    tmp, description);

  //Add the option to boost program_options
  desc.add_options()
    (path.c_str(), po::value<T>(),  description);
  //If the option is required, add it to the required options list
  if (required) 
    GetSingleton().requiredOptions.push_front(path);
  return;
}


/**
* @brief Returns the value of the specified parameter.  
*   If the parameter is unspecified, an undefined but 
*   more or less valid value is returned.
*
* @tparam T The type of the parameter.
* @param identifier The full pathname of the parameter.
*
* @return The value of the parameter.  
*   Use IO::CheckValue to determine if it's valid.
*/
template<typename T>
T& IO::GetValue(const char* identifier) {
  //Used to ensure we have a valid value
  T tmp;
  //Used to index into the globalValues map
  std::string key = std::string(identifier);
  std::map<std::string, boost::any>& gmap = GetSingleton().globalValues;
  po::variables_map& vmap = GetSingleton().vmap;
  //If we have the option, set it's value
  if (vmap.count(key) && !gmap.count(key)) {
    gmap[key] = boost::any(vmap[identifier].as<T>());
  }

  //We may have whatever is on the commandline, but what if
  //The programmer has made modifications?
  if (!gmap.count(key)) {//The programmer hasn't done anything, lets register it
    gmap[key] = boost::any(tmp);
    *boost::any_cast<T>(&gmap[key]) = tmp;
  }
  tmp =*boost::any_cast<T>(&gmap[key]);
  return *boost::any_cast<T>(&gmap[key]);
}

#endif 

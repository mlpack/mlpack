/**
 * @file cli_impl.hpp
 * @author Matthew Amidon
 *
 * Implementation of templated functions of the CLI class.
 */
#ifndef __MLPACK_CORE_IO_CLI_HPP
#error "Do not include this file directly."
#endif

#ifndef __MLPACK_CORE_IO_CLI_IMPL_HPP
#define __MLPACK_CORE_IO_CLI_IMPL_HPP

// Include option.hpp here because it requires CLI but is also templated.
#include "option.hpp"

namespace mlpack {

/**
 * @brief Adds a parameter to CLI, making it accessibile via GetParam &
 *     CheckValue.
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
void CLI::Add(const char* identifier,
             const char* description,
             const char* parent,
             bool required)
{

  po::options_description& desc = CLI::GetSingleton().desc;
  std::string path = identifier;

  // Add the option to boost program_options.
  desc.add_options()
    (path.c_str(), po::value<T>(),  description);
  // If the option is required, add it to the required options list.
  if (required)
    GetSingleton().requiredOptions.push_front(path);
}


/**
 * @brief Returns the value of the specified parameter.
 *   If the parameter is unspecified, an undefined but
 *   more or less valid value is returned.
 *
 * @tparam T The type of the parameter.
 * @param identifier The full pathname of the parameter.
 *
 * @return The value of the parameter.  Use CLI::CheckValue to determine if it's
 *     valid.
 */
template<typename T>
T& CLI::GetParam(const char* identifier)
{
  // Used to ensure we have a valid value.
  T tmp = T();

  // Used to index into the globalValues map.
  std::string key = std::string(identifier);
  gmap_t& gmap = GetSingleton().globalValues;
  //po::variables_map& vmap = GetSingleton().vmap;

  // We may have whatever is on the commandline, but what if the programmer has
  // made modifications?
  if (!gmap.count(key))
  {
    // The programmer hasn't done anything; register it.
    gmap[key] = ParamData();
    gmap[key].value = boost::any(tmp);
    *boost::any_cast<T>(&gmap[key].value) = tmp;
  }

  return *boost::any_cast<T>(&gmap[key].value);
}

}; // namespace mlpack

#endif

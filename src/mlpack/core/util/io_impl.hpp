/**
 * @file core/util/io_impl.hpp
 * @author Matthew Amidon
 *
 * Implementation of the IO module for parsing parameters.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_UTIL_IO_IMPL_HPP
#define MLPACK_CORE_UTIL_IO_IMPL_HPP

#include "io.hpp"
#include "log.hpp"
#include "hyphenate_string.hpp"

namespace mlpack {

/* Constructors, Destructors, Copy */
/* Make the constructor private, to preclude unauthorized instances */
inline IO::IO()
{
  return;
}

// Private copy constructor; don't want copies floating around.
inline IO::IO(const IO& /* other */)
{
  return;
}

// Private copy operator; don't want copies floating around.
inline IO& IO::operator=(const IO& /* other */) { return *this; }

inline void IO::AddParameter(const std::string& bindingName,
                             util::ParamData&& data)
{
  // Temporarily define color code escape sequences.
  #ifndef _WIN32
    #define BASH_RED "\033[0;31m"
    #define BASH_CLEAR "\033[0m"
  #else
    #define BASH_RED ""
    #define BASH_CLEAR ""
  #endif

  // Temporary outstream object for detecting duplicate identifiers.
  util::PrefixedOutStream outstr(MLPACK_CERR_STREAM,
        BASH_RED "[FATAL] " BASH_CLEAR, false, true /* fatal */);

  #undef BASH_RED
  #undef BASH_CLEAR

  // Define identifier and alias maps.
  std::map<std::string, util::ParamData>& bindingParams =
      GetSingleton().parameters[bindingName];
  std::map<char, std::string>& bindingAliases =
      GetSingleton().aliases[bindingName];

  // If found in current map, print fatal error and terminate the program, but
  // only if the parameter is not a global parameter.
  if (bindingParams.count(data.name) && bindingName != "")
  {
    outstr << "Parameter '" << data.name << "' ('" << data.alias << "') "
           << "is defined multiple times with the same identifiers."
           << std::endl;
  }
  else if (bindingParams.count(data.name) && bindingName == "")
  {
    // It already exists; no need to add it again.
    return;
  }

  // Check for duplicate aliases.
  if (data.alias != '\0' && bindingAliases.count(data.alias))
  {
    outstr << "Parameter '" << data.name << " ('" << data.alias << "') "
           << "is defined multiple times with the same alias." << std::endl;
  }

  // Add the alias, if necessary.
#ifndef MLPACK_NO_STD_MUTEX
  std::lock_guard<std::mutex> lock(GetSingleton().mapMutex);
#endif
  if (data.alias != '\0')
    bindingAliases[data.alias] = data.name;

  bindingParams[data.name] = std::move(data);
}

/**
 * Add a function to the function map.
 *
 * @param type Type that this function should be called for.
 * @param name Name of the function.
 * @param func Function to call.
 */
inline void IO::AddFunction(const std::string& type,
                            const std::string& name,
                            void (*func)(util::ParamData&, const void*, void*))
{
#ifndef MLPACK_NO_STD_MUTEX
  std::lock_guard<std::mutex> lock(GetSingleton().mapMutex);
#endif
  GetSingleton().functionMap[type][name] = func;
}

/**
 * Add a user-friendly name for a binding.
 *
 * @param bindingName Name of the binding to add the user-friendly name for.
 * @param name User-friendly name.
 */
inline void IO::AddBindingName(const std::string& bindingName,
                               const std::string& name)
{
#ifndef MLPACK_NO_STD_MUTEX
  std::lock_guard<std::mutex> lock(GetSingleton().mapMutex);
#endif
  GetSingleton().docs[bindingName].name = name;
}

/**
 * Add a short description for a binding.
 *
 * @param bindingName Name of the binding to add the description for.
 * @param shortDescription Description to use.
 */
inline void IO::AddShortDescription(const std::string& bindingName,
                                    const std::string& shortDescription)
{
#ifndef MLPACK_NO_STD_MUTEX
  std::lock_guard<std::mutex> lock(GetSingleton().docMutex);
#endif
  GetSingleton().docs[bindingName].shortDescription = shortDescription;
}

/**
 * Add a long description for a binding.
 *
 * @param bindingName Name of the binding to add the description for.
 * @param longDescription Function that returns the long description.
 */
inline void IO::AddLongDescription(
    const std::string& bindingName,
    const std::function<std::string()>& longDescription)
{
#ifndef MLPACK_NO_STD_MUTEX
  std::lock_guard<std::mutex> lock(GetSingleton().docMutex);
#endif
  GetSingleton().docs[bindingName].longDescription = longDescription;
}

/**
 * Add an example for a binding.
 *
 * @param bindingName Name of the binding to add the example for.
 * @param example Function that returns the example.
 */
inline void IO::AddExample(const std::string& bindingName,
                           const std::function<std::string()>& example)
{
#ifndef MLPACK_NO_STD_MUTEX
  std::lock_guard<std::mutex> lock(GetSingleton().docMutex);
#endif
  GetSingleton().docs[bindingName].example.push_back(std::move(example));
}

/**
 * Add a SeeAlso for a binding.
 *
 * @param bindingName Name of the binding to add the example for.
 * @param description Description of the SeeAlso.
 * @param link Link of the SeeAlso.
 */
inline void IO::AddSeeAlso(const std::string& bindingName,
                           const std::string& description,
                           const std::string& link)
{
#ifndef MLPACK_NO_STD_MUTEX
  std::lock_guard<std::mutex> lock(GetSingleton().docMutex);
#endif
  GetSingleton().docs[bindingName].seeAlso.push_back(
      std::make_pair(description, link));
}

// Returns the sole instance of this class.
inline IO& IO::GetSingleton()
{
  static IO singleton;
  return singleton;
}

// Returns the sole instance of the timers.
inline util::Timers& IO::GetTimers()
{
  return GetSingleton().timer;
}

/**
 * Return a new Params object initialized with all the parameters of the
 * binding `bindingName`.  This is intended to be called at the beginning of
 * the run of a binding.
 */
inline util::Params IO::Parameters(const std::string& bindingName)
{
  // We don't need a mutex here, because we are only randomly accessing elements
  // of the maps.
  std::map<char, std::string> resultAliases =
      GetSingleton().aliases[bindingName];
  // Merge in any persistent parameters (e.g. parameters in the "" binding map).
  std::map<char, std::string> persistentAliases = GetSingleton().aliases[""];
  resultAliases.insert(persistentAliases.begin(), persistentAliases.end());

  std::map<std::string, util::ParamData> resultParams =
      GetSingleton().parameters[bindingName];
  // Merge in any persistent parameters (e.g. parameters in the "" binding map).
  std::map<std::string, util::ParamData> persistentParams =
      GetSingleton().parameters[""];
  resultParams.insert(persistentParams.begin(), persistentParams.end());

  return util::Params(resultAliases, resultParams, GetSingleton().functionMap,
      bindingName, GetSingleton().docs[bindingName]);
}

} // namespace mlpack

#endif

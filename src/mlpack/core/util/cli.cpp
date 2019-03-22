/**
 * @file cli.cpp
 * @author Matthew Amidon
 *
 * Implementation of the CLI module for parsing parameters.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <list>
#include <boost/program_options.hpp>
#include <boost/scoped_ptr.hpp>
#include <iostream>

#include "cli.hpp"
#include "log.hpp"
#include "hyphenate_string.hpp"

#include "version.hpp"

#include <mlpack/core/data/load.hpp>
#include <mlpack/core/data/save.hpp>

using namespace mlpack;
using namespace mlpack::util;

// Fake ProgramDoc in case none is supplied.
static ProgramDoc emptyProgramDoc = ProgramDoc("", "", []() { return ""; },
    {});

/* Constructors, Destructors, Copy */
/* Make the constructor private, to preclude unauthorized instances */
CLI::CLI() : didParse(false), doc(&emptyProgramDoc)
{
  return;
}

// Private copy constructor; don't want copies floating around.
CLI::CLI(const CLI& /* other */) : didParse(false), doc(&emptyProgramDoc)
{
  return;
}

// Private copy operator; don't want copies floating around.
CLI& CLI::operator=(const CLI& /* other */) { return *this; }

void CLI::Add(ParamData&& data)
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
  util::PrefixedOutStream outstr(std::cerr,
        BASH_RED "[FATAL] " BASH_CLEAR, false, true /* fatal */);

  #undef BASH_RED
  #undef BASH_CLEAR

  // Define identifier and alias maps.
  std::map<std::string, util::ParamData>& parameters =
      GetSingleton().parameters;
  std::map<char, std::string>& aliases = GetSingleton().aliases;

  // If found in current map, print fatal error and terminate the program, but
  // only if the parameter is not consistent.
  if (parameters.count(data.name) && !data.persistent)
  {
    outstr << "Parameter --" << data.name << " (-" << data.alias << ") "
           << "is defined multiple times with the same identifiers."
           << std::endl;
  }
  if (data.alias != '\0' && aliases.count(data.alias) && !data.persistent)
  {
    outstr << "Parameter --" << data.name << " (-" << data.alias << ") "
           << "is defined multiple times with the same alias." << std::endl;
  }

  // Add the alias, if necessary.
  if (data.alias != '\0')
    GetSingleton().aliases[data.alias] = data.name;

  GetSingleton().parameters[data.name] = std::move(data);
}

/**
 * See if the specified flag was found while parsing.
 *
 * @param identifier The name of the parameter in question.
 */
bool CLI::HasParam(const std::string& key)
{
  std::string usedKey = key;
  const std::map<std::string, util::ParamData>& parameters =
      GetSingleton().parameters;

  if (!parameters.count(key))
  {
    // Check any aliases, but only after we are sure the actual option as given
    // does not exist.
    if (key.length() == 1 && GetSingleton().aliases.count(key[0]))
      usedKey = GetSingleton().aliases[key[0]];

    if (!parameters.count(usedKey))
    {
      Log::Fatal << "Parameter '--" << key << "' does not exist in this "
          << "program." << std::endl;
    }
  }
  const std::string& checkKey = usedKey;

  return (parameters.at(checkKey).wasPassed > 0);
}

// Returns the sole instance of this class.
CLI& CLI::GetSingleton()
{
  static CLI singleton;
  return singleton;
}

/**
 * Registers a ProgramDoc object, which contains documentation about the
 * program.
 *
 * @param doc Pointer to the ProgramDoc object.
 */
void CLI::RegisterProgramDoc(ProgramDoc* doc)
{
  // Only register the doc if it is not the dummy object we created at the
  // beginning of the file (as a default value in case this is never called).
  if (doc != &emptyProgramDoc)
    GetSingleton().doc = doc;
}

// Get the parameters that the CLI object knows about.
std::map<std::string, ParamData>& CLI::Parameters()
{
  return GetSingleton().parameters;
}

// Get the parameters that the CLI object knows about.
std::map<char, std::string>& CLI::Aliases()
{
  return GetSingleton().aliases;
}

// Get the program name as set by PROGRAM_INFO().
std::string CLI::ProgramName()
{
  return GetSingleton().doc->programName;
}

// Set a particular parameter as passed.
void CLI::SetPassed(const std::string& name)
{
  if (GetSingleton().parameters.count(name) == 0)
  {
    throw std::invalid_argument("CLI::SetPassed(): parameter " + name +
        " not known!");
  }

  // Set passed to true.
  GetSingleton().parameters[name].wasPassed = true;
}

// Store settings.
void CLI::StoreSettings(const std::string& name)
{
  // Take all of the parameters and put them in the map.  Clear anything old
  // first.
  std::get<0>(GetSingleton().storageMap[name]) = GetSingleton().parameters;
  std::get<1>(GetSingleton().storageMap[name]) = GetSingleton().aliases;
  std::get<2>(GetSingleton().storageMap[name]) = GetSingleton().functionMap;

  ClearSettings();
}

// Restore settings.
void CLI::RestoreSettings(const std::string& name, const bool fatal)
{
  if (GetSingleton().storageMap.count(name) == 0 && fatal)
  {
    throw std::invalid_argument("no settings stored under the name '" + name
        + "'");
  }
  else if (GetSingleton().storageMap.count(name) == 0 && !fatal)
  {
    // Nothing to do, just clear what's there.
    ClearSettings();
  }
  else
  {
    GetSingleton().parameters = std::get<0>(GetSingleton().storageMap[name]);
    GetSingleton().aliases = std::get<1>(GetSingleton().storageMap[name]);
    GetSingleton().functionMap = std::get<2>(GetSingleton().storageMap[name]);
  }
}

// Clear settings.
void CLI::ClearSettings()
{
  // Check for any parameters we need to keep.
  std::map<std::string, util::ParamData> persistent;
  std::map<char, std::string> persistentAliases;
  FunctionMapType persistentFunctions;

  // For the function mappings we have to preserve, we have to collect the
  // types.
  std::vector<std::string> persistentTypes;

  std::map<std::string, util::ParamData>::const_iterator it =
      GetSingleton().parameters.begin();
  while (it != GetSingleton().parameters.end())
  {
    // Is the parameter persistent?
    if (it->second.persistent)
    {
      persistent[it->first] = it->second; // Save the parameter.
      // Add to the list of types, if it hasn't already been added.
      if (std::find(persistentTypes.begin(), persistentTypes.end(),
          it->second.tname) == persistentTypes.end())
        persistentTypes.push_back(it->second.tname);
    }

    ++it;
  }

  // Now check if there are any persistent aliases.
  std::map<char, std::string>::const_iterator it2 =
      GetSingleton().aliases.begin();
  while (it2 != GetSingleton().aliases.end())
  {
    // Is this an alias to a persistent parameter?
    if (persistent.count(it2->second) > 0)
      persistentAliases[it2->first] = it2->second; // Save it.

    ++it2;
  }

  for (size_t i = 0; i < persistentTypes.size(); ++i)
  {
    // Add to persistent function map.
    persistentFunctions[persistentTypes[i]] =
        GetSingleton().functionMap[persistentTypes[i]];
  }

  // Save only the persistent parameters.
  GetSingleton().parameters = persistent;
  GetSingleton().aliases = persistentAliases;
  GetSingleton().functionMap = persistentFunctions;
}

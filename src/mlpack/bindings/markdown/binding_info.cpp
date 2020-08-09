/**
 * @file bindings/markdown/binding_info.cpp
 * @author Ryan Curtin
 *
 * Implementation of BindingInfo functions.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "binding_info.hpp"

using namespace std;

namespace mlpack {
namespace bindings {
namespace markdown {

util::ProgramName& BindingInfo::GetProgramName(const std::string& bindingName)
{
  if (GetSingleton().mapProgramName.count(bindingName) == 0)
  {
    throw std::invalid_argument("No program name associated with'" + bindingName
      + "'!");
  }

  return GetSingleton().mapProgramName.at(bindingName);
}

util::ShortDescription& BindingInfo::GetShortDescription(
    const std::string& bindingName)
{
  if (GetSingleton().mapShortDescription.count(bindingName) == 0)
  {
    throw std::invalid_argument("No short description associated with'"
      + bindingName + "'!");
  }

  return GetSingleton().mapShortDescription.at(bindingName);
}

util::LongDescription& BindingInfo::GetLongDescription(
    const std::string& bindingName)
{
  if (GetSingleton().mapLongDescription.count(bindingName) == 0)
  {
    throw std::invalid_argument("No long description associated with'"
      + bindingName + "'!");
  }

  return GetSingleton().mapLongDescription.at(bindingName);
}

std::vector<util::Example>& BindingInfo::GetExample(const std::string&
    bindingName)
{
  // Some bindings may not have examples.
  if (GetSingleton().mapExample.count(bindingName) == 0)
  {
    static std::vector<util::Example> empty;
    return empty;
  }

  return GetSingleton().mapExample.at(bindingName);
}

std::vector<util::SeeAlso>& BindingInfo::GetSeeAlso(const std::string&
    bindingName)
{
  if (GetSingleton().mapSeeAlso.count(bindingName) == 0)
  {
    throw std::invalid_argument("No see also associated with'" + bindingName
      + "'!");
  }

  return GetSingleton().mapSeeAlso.at(bindingName);
}

//! Register a ProgramName object with the given bindingName.
void BindingInfo::RegisterProgramName(const std::string& bindingName,
                                      const util::ProgramName& programName)
{
  GetSingleton().mapProgramName[bindingName] = programName;
}

//! Register a ShortDescription object with the given bindingName.
void BindingInfo::RegisterShortDescription(const std::string& bindingName,
                                           const util::ShortDescription&
                                                 shortDescription)
{
  GetSingleton().mapShortDescription[bindingName] = shortDescription;
}

//! Register a LongDescription object with the given bindingName.
void BindingInfo::RegisterLongDescription(const std::string& bindingName,
                                          const util::LongDescription&
                                                longDescription)
{
  GetSingleton().mapLongDescription[bindingName] = longDescription;
}

//! Register a Example object with the given bindingName.
void BindingInfo::RegisterExample(const std::string& bindingName,
                                  const util::Example& example)
{
  GetSingleton().mapExample[bindingName].push_back(example);
}

//! Register a SeeAlso object with the given bindingName.
void BindingInfo::RegisterSeeAlso(const std::string& bindingName,
                                  const util::SeeAlso& seeAlso)
{
  GetSingleton().mapSeeAlso[bindingName].push_back(seeAlso);
}

//! Get or modify the current language (don't set it to something invalid!).
std::string& BindingInfo::Language()
{
  return GetSingleton().language;
}

BindingInfo& BindingInfo::GetSingleton()
{
  static BindingInfo instance;
  return instance;
}

} // namespace markdown
} // namespace bindings
} // namespace mlpack

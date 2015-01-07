/**
 * @file save_restore_utility_impl.hpp
 * @author Neil Slagle
 *
 * The SaveRestoreUtility provides helper functions in saving and
 *   restoring models.  The current output file type is XML.
 *
 * This file is part of mlpack 1.0.12.
 *
 * mlpack is free software; you may redstribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef __MLPACK_CORE_UTIL_SAVE_RESTORE_UTILITY_IMPL_HPP
#define __MLPACK_CORE_UTIL_SAVE_RESTORE_UTILITY_IMPL_HPP

// In case it hasn't been included already.
#include "save_restore_utility.hpp"
#include "log.hpp"

namespace mlpack {
namespace util {

template<typename T>
T& SaveRestoreUtility::LoadParameter(T& t, const std::string& name)
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
    Log::Fatal << "LoadParameter(): node '" << name << "' not found.\n";
  }
  return t;
}

template<typename T>
std::vector<T>& SaveRestoreUtility::LoadParameter(std::vector<T>& v,
                                                  const std::string& name)
{
  std::map<std::string, std::string>::iterator it = parameters.find(name);
  if (it != parameters.end())
  {
    v.clear();
    std::string value = (*it).second;
    boost::char_separator<char> sep (",");
    boost::tokenizer<boost::char_separator<char> > tok (value, sep);
    std::list<std::list<double> > rows;
    for (boost::tokenizer<boost::char_separator<char> >::iterator
           tokIt = tok.begin();
         tokIt != tok.end();
         ++tokIt)
    {
      T t;
      std::istringstream iss (*tokIt);
      iss >> t;
      v.push_back(t);
    }
  }
  else
  {
    Log::Fatal << "LoadParameter(): node '" << name << "' not found.\n";
  }
  return v;
}

template<typename T>
void SaveRestoreUtility::SaveParameter(const T& t, const std::string& name)
{
  std::ostringstream output;
  // Manually increase precision to solve #313 for now, until we have a way to
  // store this as an actual binary number.
  output << std::setprecision(15) << t;
  parameters[name] = output.str();
}

template<typename T>
void SaveRestoreUtility::SaveParameter(const std::vector<T>& t,
                                       const std::string& name)
{
  std::ostringstream output;
  for (size_t index = 0; index < t.size(); ++index)
  {
    output << t[index] << ",";
  }
  std::string vectorAsStr = output.str();
  vectorAsStr.erase(vectorAsStr.length() - 1);
  parameters[name] = vectorAsStr;
}

}; // namespace util
}; // namespace mlpack

#endif

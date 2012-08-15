/**
 * @file cli_impl.hpp
 * @author Matthew Amidon
 *
 * Implementation of templated functions of the CLI class.
 *
 * This file is part of MLPACK 1.0.2.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef __MLPACK_CORE_IO_CLI_IMPL_HPP
#define __MLPACK_CORE_IO_CLI_IMPL_HPP

// In case it has not already been included.
#include "cli.hpp"

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
void CLI::Add(const std::string& path,
             const std::string& description,
             const std::string& alias,
             bool required)
{

  po::options_description& desc = CLI::GetSingleton().desc;
  // Must make use of boost syntax here.
  std::string progOptId = alias.length() ? path + "," + alias : path;

  // Add the alias, if necessary
  AddAlias(alias, path);

  // Add the option to boost program_options.
  desc.add_options()
    (progOptId.c_str(), po::value<T>(),  description.c_str());

  // Make sure the appropriate metadata is inserted into gmap.
  gmap_t& gmap = GetSingleton().globalValues;

  ParamData data;
  T tmp = T();

  data.desc = description;
  data.name = path;
  data.tname = TYPENAME(T);
  data.value = boost::any(tmp);
  data.wasPassed = false;

  gmap[path] = data;

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
T& CLI::GetParam(const std::string& identifier)
{
  // Used to ensure we have a valid value.
  T tmp = T();

  // Used to index into the globalValues map.
  std::string key = std::string(identifier);
  gmap_t& gmap = GetSingleton().globalValues;

  // Now check if we have an alias.
  amap_t& amap = GetSingleton().aliasValues;
  if (amap.count(key))
    key = amap[key];

  // What if we don't actually have any value?
  if (!gmap.count(key))
  {
    gmap[key] = ParamData();
    gmap[key].value = boost::any(tmp);
    *boost::any_cast<T>(&gmap[key].value) = tmp;
  }

  // What if we have meta-data, but no data?
  boost::any val = gmap[key].value;
  if (val.empty())
    gmap[key].value = boost::any(tmp);

  return *boost::any_cast<T>(&gmap[key].value);
}

}; // namespace mlpack

#endif

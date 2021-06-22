/**
 * @file core/util/param_checks_impl.hpp
 * @author Ryan Curtin
 *
 * Utility function implementation for checking arguments, and so forth.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_UTIL_PARAM_CHECKS_IMPL_HPP
#define MLPACK_CORE_UTIL_PARAM_CHECKS_IMPL_HPP

#include "param_checks.hpp"

namespace mlpack {
namespace util {

// Check that the arguments are given.
inline void RequireOnlyOnePassed(
    util::Params& params,
    const std::vector<std::string>& constraints,
    const bool fatal,
    const std::string& errorMessage,
    const bool allowNone)
{
  if (BINDING_IGNORE_CHECK(constraints))
    return;

  size_t set = 0;
  for (size_t i = 0; i < constraints.size(); ++i)
  {
    if (params.Has(constraints[i]))
      ++set;
  }

  util::PrefixedOutStream& stream = fatal ? Log::Fatal : Log::Warn;
  if (set > 1)
  {
    // Give different output depending on whether 2 or more parameters are
    // given.
    if (constraints.size() == 2)
    {
      stream << "Can only pass one of " << PRINT_PARAM_STRING(constraints[0])
          << " or " << PRINT_PARAM_STRING(constraints[1]);
    }
    else
    {
      stream << "Can only pass one of ";
      for (size_t i = 0 ; i < constraints.size() - 1; ++i)
        stream << PRINT_PARAM_STRING(constraints[i]) << ", ";
      stream << "or "
          << PRINT_PARAM_STRING(constraints[constraints.size() - 1]);
    }

    // Append a custom message.
    if (!errorMessage.empty())
      stream << "; " << errorMessage;
    stream << "!" << std::endl;
  }
  else if (set == 0 && !allowNone)
  {
    stream << (fatal ? "Must " : "Should ");

    // Give different output depending on whether 1, 2, or more parameters are
    // given.
    if (constraints.size() == 1)
    {
      stream << "specify " << PRINT_PARAM_STRING(constraints[0]);
    }
    else if (constraints.size() == 2)
    {
      stream << "specify one of " << PRINT_PARAM_STRING(constraints[0])
          << " or " << PRINT_PARAM_STRING(constraints[1]);
    }
    else
    {
      stream << "specify one of ";
      for (size_t i = 0; i < constraints.size() - 1; ++i)
        stream << PRINT_PARAM_STRING(constraints[i]) << ", ";
      stream << "or "
          << PRINT_PARAM_STRING(constraints[constraints.size() - 1]);
    }

    // Append a custom message.
    if (!errorMessage.empty())
      stream << "; " << errorMessage;
    stream << "!" << std::endl;
  }
}

inline void RequireAtLeastOnePassed(
    util::Params& params,
    const std::vector<std::string>& constraints,
    const bool fatal,
    const std::string& errorMessage)
{
  if (BINDING_IGNORE_CHECK(constraints))
    return;

  size_t set = 0;
  for (size_t i = 0; i < constraints.size(); ++i)
  {
    if (params.Has(constraints[i]))
      ++set;
  }

  if (set == 0)
  {
    util::PrefixedOutStream& stream = fatal ? Log::Fatal : Log::Warn;
    stream << (fatal ? "Must " : "Should ");
    if (constraints.size() == 1)
    {
      // This shouldn't happen... just use PARAM_*_REQ()...
      stream << "pass " << PRINT_PARAM_STRING(constraints[0]);
    }
    else if (constraints.size() == 2)
    {
      stream << "pass either " << PRINT_PARAM_STRING(constraints[0])
          << " or " << PRINT_PARAM_STRING(constraints[1]) << " or both";
    }
    else
    {
      stream << "pass one of ";
      for (size_t i = 0; i < constraints.size() - 1; ++i)
        stream << PRINT_PARAM_STRING(constraints[i]) << ", ";
      stream << "or "
          << PRINT_PARAM_STRING(constraints[constraints.size() - 1]);
    }

    // Append custom error message.
    if (!errorMessage.empty())
      stream << "; " << errorMessage << "!" << std::endl;
    else
      stream << "!" << std::endl;
  }
}

inline void RequireNoneOrAllPassed(
    util::Params& params,
    const std::vector<std::string>& constraints,
    const bool fatal,
    const std::string& errorMessage)
{
  if (BINDING_IGNORE_CHECK(constraints))
    return;

  size_t set = 0;
  for (size_t i = 0; i < constraints.size(); ++i)
  {
    if (params.Has(constraints[i]))
      ++set;
  }

  if (set != 0 && set < constraints.size())
  {
    util::PrefixedOutStream& stream = fatal ? Log::Fatal : Log::Warn;
    stream << (fatal ? "Must " : "Should ");
    if (constraints.size() == 2)
    {
      stream << "pass none or both of " << PRINT_PARAM_STRING(constraints[0])
          << " and " << PRINT_PARAM_STRING(constraints[1]);
    }
    else
    {
      // constraints.size() > 2.
      stream << "pass none or all of ";
      for (size_t i = 0; i < constraints.size() - 1; ++i)
        stream << PRINT_PARAM_STRING(constraints[i]) << ", ";
      stream << "and "
          << PRINT_PARAM_STRING(constraints[constraints.size() - 1]);
    }

    // Append custom error message.
    if (!errorMessage.empty())
      stream << "; " << errorMessage << "!" << std::endl;
    else
      stream << "!" << std::endl;
  }
}

template<typename T>
void RequireParamInSet(util::Params& params,
                       const std::string& name,
                       const std::vector<T>& set,
                       const bool fatal,
                       const std::string& errorMessage)
{
  if (BINDING_IGNORE_CHECK(name))
    return;

  if (std::find(set.begin(), set.end(), params.Get<T>(name)) == set.end())
  {
    // The item was not found in the set.
    util::PrefixedOutStream& stream = fatal ? Log::Fatal : Log::Warn;
    stream << "Invalid value of " << PRINT_PARAM_STRING(name) << " specified ("
        << PRINT_PARAM_VALUE(params.Get<T>(name), true) << "); ";
    if (!errorMessage.empty())
      stream << errorMessage << "; ";
    stream << "must be one of ";
    for (size_t i = 0; i < set.size() - 1; ++i)
      stream << PRINT_PARAM_VALUE(set[i], true) << ", ";
    stream << "or " << PRINT_PARAM_VALUE(set[set.size() - 1], true) << "!"
        << std::endl;
  }
}

template<typename T>
void RequireParamValue(util::Params& params,
                       const std::string& name,
                       const std::function<bool(T)>& conditional,
                       const bool fatal,
                       const std::string& errorMessage)
{
  if (BINDING_IGNORE_CHECK(name))
    return;

  // We need to make sure that the condition holds.
  bool condition = conditional(params.Get<T>(name));
  if (!condition)
  {
    // The condition failed.
    util::PrefixedOutStream& stream = fatal ? Log::Fatal : Log::Warn;
    stream << "Invalid value of " << PRINT_PARAM_STRING(name) << " specified ("
        << PRINT_PARAM_VALUE(params.Get<T>(name), false) << "); "
        << errorMessage << "!" << std::endl;
  }
}

inline void ReportIgnoredParam(
    util::Params& params,
    const std::vector<std::pair<std::string, bool>>& constraints,
    const std::string& paramName)
{
  if (BINDING_IGNORE_CHECK(paramName))
    return;

  // Determine whether or not the condition is true.
  bool condition = true;
  for (size_t i = 0; i < constraints.size(); ++i)
  {
    if (params.Has(constraints[i].first) != constraints[i].second)
    {
      condition = false;
      break;
    }
  }

  // If the condition is satisfied, then report that the parameter is ignored
  // (if the user passed it).
  if (condition && params.Has(paramName))
  {
    // The output will be different depending on whether there are 1, 2, or more
    // constraints.
    Log::Warn << PRINT_PARAM_STRING(paramName) << " ignored because ";
    if (constraints.size() == 1)
    {
      Log::Warn << PRINT_PARAM_STRING(constraints[0].first)
          << ((constraints[0].second) ? " is " : " is not ")
          << "specified!" << std::endl;
    }
    else if (constraints.size() == 2)
    {
      if (constraints[0].second == constraints[1].second)
      {
        Log::Warn << ((constraints[0].second) ? "both " : "neither ")
            << PRINT_PARAM_STRING(constraints[0].first)
            << ((constraints[0].second) ? "or " : "nor ")
            << PRINT_PARAM_STRING(constraints[1].first)
            << " are specified!" << std::endl;
      }
      else
      {
        Log::Warn << PRINT_PARAM_STRING(constraints[0].first)
            << ((constraints[0].second) ? " is " : " is not ")
            << "specified and "
            << ((constraints[1].second) ? " is " : " is not ")
            << "specified!" << std::endl;
      }
    }
    else
    {
      // List each constraint and whether or not it was or wasn't specified.
      for (size_t i = 0; i < constraints.size(); ++i)
      {
        Log::Warn << PRINT_PARAM_STRING(constraints[i].first)
            << ((constraints[i].second) ? " is " : " is not ")
            << ((i == constraints.size() - 1) ? "specified!"
                : "specified and ");
      }
      Log::Warn << std::endl;
    }
  }
}

inline void ReportIgnoredParam(util::Params& params,
                               const std::string& paramName,
                               const std::string& reason)
{
  // If the argument was passed, we need to print the reason.
  if (params.Has(paramName))
  {
    Log::Warn << PRINT_PARAM_STRING(paramName) << " ignored because "
        << reason << "!" << std::endl;
  }
}

} // namespace util
} // namespace mlpack

#endif

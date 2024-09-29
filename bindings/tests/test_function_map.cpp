/**
 * @file test_function_map.cpp
 * @author Ryan Curtin
 *
 * Implementation of the TestFunctionMap class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "test_function_map.hpp"

namespace mlpack {
namespace bindings {
namespace tests {

TestFunctionMap::TestFunctionMap()
{
  // Nothing to do.
}

TestFunctionMap& TestFunctionMap::GetSingleton()
{
  static TestFunctionMap singleton;
  return singleton;
}

void TestFunctionMap::RegisterFunction(
    const std::string& tname,
    const std::string& functionName,
    void (*func)(util::ParamData&, const void*, void*))
{
  GetSingleton().functionMap[tname][functionName] = func;
}

const TestFunctionMap::FunctionMapType& TestFunctionMap::FunctionMap()
{
  return GetSingleton().functionMap;
}

} // namespace tests
} // namespace bindings
} // namespace mlpack

/**
 * @file test_function_map.cpp
 * @author Ryan Curtin
 *
 * Implementation of the TestFunctionMap class.
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

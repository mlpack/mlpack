/**
 * @file bindings/python/print_py_wrapper.cpp
 * @author Nippun Sharma
 *
 * Implementation of a function to generate a .py file that
 * contains the wrapper for a method.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "print_wrapper_py.hpp"
// #include "get_methods_wrapper.hpp"
// #include "get_class_name_wrapper.hpp"
// #include "get_program_name.hpp"

#include <mlpack/core/util/io.hpp>

using namespace mlpack::util;
using namespace std;

namespace mlpack {
namespace bindings {
namespace python {

void PrintWrapperPY(const std::string& groupName,
                    const std::string& validMethods)
{
  cout << validMethods << endl;
}

} // python
} // bindings
} // mlpack

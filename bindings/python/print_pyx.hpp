/**
 * @file bindings/python/print_pyx.hpp
 * @author Ryan Curtin
 *
 * Given a list of ParamData structures, emit a .pyx file defining the Cython
 * binding.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_PYTHON_GENERATE_PYX_HPP
#define MLPACK_BINDINGS_PYTHON_GENERATE_PYX_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace bindings {
namespace python {

/**
 * Given a list of parameter definition and program documentation, print a
 * generated .pyx file to stdout.
 *
 * @param doc Documentation for the program.
 * @param mainFilename Filename of the main program (i.e.
 *      "/path/to/pca_main.cpp").
 * @param functionName Name of the function (i.e. "pca").
 */
void PrintPYX(const util::BindingDetails& doc,
              const std::string& mainFilename,
              const std::string& functionName);


} // namespace python
} // namespace bindings
} // namespace mlpack

#endif

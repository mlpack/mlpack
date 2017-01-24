/**
 * @file print_functions.hpp
 * @author Ryan Curtin
 *
 * Utility functions to print various parts of the program that we will need,
 * given a ParamData structure and possibly some other options.
 */
#ifndef MLPACK_BINDINGS_PYTHON_PRINT_FUNCTIONS_HPP
#define MLPACK_BINDINGS_PYTHON_PRINT_FUNCTIONS_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace bindings {
namespace python {

/**
 * Given parameter info, print the definition of the parameter.  You are
 * responsible for setting up the line---this does not handle indentation or
 * anything.
 */
void PrintDefinition(const util::ParamData& data);

/**
 * Given an option, print documentation for it.  You are responsible for setting
 * up the line---this does not handle indentation or anything.  This is meant to
 * produce a line of documentation for the docstring of the Python function,
 * describing a single parameter.
 *
 * The indent parameter should be passed to know how much to indent after a
 * newline, if needed.
 */
void PrintDocumentation(const util::ParamData& data,
                        const size_t indent);

/**
 * Given parameter info and the current number of spaces for indentation, print
 * the code to process the input.  This code assumes that data.input is true,
 * and should not be called when data.input is false.
 *
 * @param data Parameter data.
 * @param indent Number of spaces to indent.
 */
void PrintInputProcessing(const util::ParamData& data,
                          const size_t indent);

/**
 * Given parameter info and the current number of spaces for indentation, print
 * the code to process the output.  This code assumes that data.input is false,
 * and should not be called when data.input is true.
 *
 * @param data Parameter data.
 * @param indent Number of spaces to indent.
 */
void PrintOutputProcessing(const util::ParamData& data,
                           const size_t indent,
                           const bool onlyOutput);

} // namespace python
} // namespace bindings
} // namespace mlpack

#endif

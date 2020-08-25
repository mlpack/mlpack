/**
 * @file bindings/python/print_pyx.cpp
 * @author Ryan Curtin
 *
 * Implementation of function to generate a .pyx file given a list of parameters
 * for the function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "print_pyx.hpp"
#include <mlpack/core/util/io.hpp>
#include <mlpack/core/util/hyphenate_string.hpp>
#include <set>

using namespace mlpack::util;
using namespace std;

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
              const string& mainFilename,
              const string& functionName)
{
  // Restore parameters.
  IO::RestoreSettings(doc.programName);

  std::map<std::string, util::ParamData>& parameters = IO::Parameters();
  typedef std::map<std::string, util::ParamData>::iterator ParamIter;

  // Split into input and output parameters.  Take two passes on the input
  // parameters, so that we get the required ones first.
  vector<string> inputOptions, outputOptions;
  for (ParamIter it = parameters.begin(); it != parameters.end(); ++it)
  {
    util::ParamData& d = it->second;
    if (d.input && d.required)
    {
      // Ignore some parameters.
      if (d.name != "help" && d.name != "info" &&
          d.name != "version")
        inputOptions.push_back(it->first);
    }
    else if (!d.input)
    {
      outputOptions.push_back(it->first);
    }
  }

  for (ParamIter it = parameters.begin(); it != parameters.end(); ++it)
  {
    util::ParamData& d = it->second;
    if (d.input && !d.required &&
        d.name != "help" && d.name != "info" &&
        d.name != "version")
      inputOptions.push_back(it->first);
  }

  // First, we must generate the header comment.

  // Now import all the necessary packages.
  cout << "cimport arma" << endl;
  cout << "cimport arma_numpy" << endl;
  cout << "from io cimport IO" << endl;
  cout << "from io cimport SetParam, SetParamPtr, SetParamWithInfo, "
      << "GetParamPtr" << endl;
  cout << "from io cimport EnableVerbose, DisableVerbose, DisableBacktrace, "
      << "ResetTimers, EnableTimers" << endl;
  cout << "from matrix_utils import to_matrix, to_matrix_with_info" << endl;
  cout << "from serialization cimport SerializeIn, SerializeOut" << endl;
  cout << endl;
  cout << "import numpy as np" << endl;
  cout << "cimport numpy as np" << endl;
  cout << endl;
  cout << "import pandas as pd" << endl;
  cout << endl;
  cout << "from libcpp.string cimport string" << endl;
  cout << "from libcpp cimport bool as cbool" << endl;
  cout << "from libcpp.vector cimport vector" << endl;
  cout << endl;
  cout << "from cython.operator import dereference" << endl;
  cout << endl;

  // Import the program we will be using.
  cout << "cdef extern from \"<" << mainFilename << ">\" nogil:" << endl;
  cout << "  cdef void mlpackMain() nogil except +RuntimeError" << endl;
  cout << "  " << endl;
  // Print any class definitions we need to have.
  std::set<std::string> classes;
  for (ParamIter it = parameters.begin(); it != parameters.end(); ++it)
  {
    util::ParamData& d = it->second;
    if (classes.count(d.cppType) == 0)
    {
      const size_t indent = 2;
      IO::GetSingleton().functionMap[d.tname]["ImportDecl"](d, (void*) &indent,
          NULL);

      // Make sure we don't double-print the definition.
      classes.insert(d.cppType);
    }
  }

  cout << endl;

  // Print any extra class definitions we might need.
  for (ParamIter it = parameters.begin(); it != parameters.end(); ++it)
  {
    util::ParamData& d = it->second;
    if (d.input)
      IO::GetSingleton().functionMap[d.tname]["PrintClassDefn"](d, NULL, NULL);
  }

  // Print the definition.
  cout << "def " << functionName << "(";
  size_t indent = 4 /* 'def ' */ + functionName.size() + 1 /* '(' */;
  for (size_t i = 0; i < inputOptions.size(); ++i)
  {
    util::ParamData& d = parameters.at(inputOptions[i]);

    if (i != 0)
      cout << "," << endl << std::string(indent, ' ');

    IO::GetSingleton().functionMap[d.tname]["PrintDefn"](d, NULL, NULL);
  }

  // Print closing brace for function definition.
  cout << "):" << endl;

  // Print the comment describing the function and its parameters.
  cout << "  \"\"\"" << endl;
  cout << "  " << doc.programName << endl;
  cout << endl;

  // Print the description.
  cout << "  " << HyphenateString(doc.longDescription(), 2) << endl << endl;

  // Next print the examples.
  for (size_t j = 0; j < doc.example.size(); ++j)
  {
    cout << "  " << util::HyphenateString(doc.example[j](), 2) << endl << endl;
  }

  // Next, print information on the input options.
  cout << "  Input parameters:" << endl;
  cout << endl;
  for (size_t i = 0; i < inputOptions.size(); ++i)
  {
    util::ParamData& d = parameters.at(inputOptions[i]);

    cout << "  ";
    size_t indent = 4;
    IO::GetSingleton().functionMap[d.tname]["PrintDoc"](d, (void*) &indent,
        NULL);
    cout << endl;
  }
  cout << endl;
  cout << "  Output parameters:" << endl;
  cout << endl;
  for (size_t i = 0; i < outputOptions.size(); ++i)
  {
    util::ParamData& d = parameters.at(outputOptions[i]);

    cout << "  ";
    size_t indent = 4;
    IO::GetSingleton().functionMap[d.tname]["PrintDoc"](d, (void*) &indent,
        NULL);
    cout << endl;
  }
  cout << endl;
  cout << "A dict containing each of the named output parameters will be "
      << "returned." << endl;
  cout << "  \"\"\"" << endl;

  // Reset any timers and disable backtraces.
  cout << "  ResetTimers()" << endl;
  cout << "  EnableTimers()" << endl;
  cout << "  DisableBacktrace()" << endl;
  cout << "  DisableVerbose()" << endl;

  // Restore the parameters.
  cout << "  IO.RestoreSettings(\"" << doc.programName << "\")"
      << endl;

  // Determine whether or not we need to copy parameters.
  cout << "  if isinstance(copy_all_inputs, bool):" << endl;
  cout << "    if copy_all_inputs:" << endl;
  cout << "      SetParam[cbool](<const string> 'copy_all_inputs', "
      << "copy_all_inputs)" << endl;
  cout << "      IO.SetPassed(<const string> 'copy_all_inputs')" << endl;
  cout << "  else:" << endl;
  cout << "    raise TypeError(" <<"\"'copy_all_inputs\' must have type "
      << "\'bool'!\")" << endl;
  cout << endl;

  // Do any input processing.
  for (size_t i = 0; i < inputOptions.size(); ++i)
  {
    util::ParamData& d = parameters.at(inputOptions[i]);

    size_t indent = 2;
    IO::GetSingleton().functionMap[d.tname]["PrintInputProcessing"](d,
        (void*) &indent, NULL);
  }

  // Set all output options as passed.
  cout << "  # Mark all output options as passed." << endl;
  for (size_t i = 0; i < outputOptions.size(); ++i)
  {
    util::ParamData& d = parameters.at(outputOptions[i]);
    cout << "  IO.SetPassed(<const string> '" << d.name << "')" << endl;
  }

  // Call the method.
  cout << "  # Call the mlpack program." << endl;
  cout << "  mlpackMain()" << endl;

  // Do any output processing and return.
  cout << "  # Initialize result dictionary." << endl;
  cout << "  result = {}" << endl;
  cout << endl;

  for (size_t i = 0; i < outputOptions.size(); ++i)
  {
    util::ParamData& d = parameters.at(outputOptions[i]);

    std::tuple<size_t, bool> t = std::make_tuple(2, false);
    IO::GetSingleton().functionMap[d.tname]["PrintOutputProcessing"](d,
        (void*) &t, NULL);
  }

  // Clear the parameters.
  cout << endl;
  cout << "  IO.ClearSettings()" << endl;
  cout << endl;

  cout << "  return result" << endl;
}

} // namespace python
} // namespace bindings
} // namespace mlpack

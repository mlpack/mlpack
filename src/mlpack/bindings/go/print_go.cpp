/**
 * @file bindings/go/print_go.cpp
 * @author Yasmine Dumouchel
 *
 * Implementation of function to generate a .go file given a list of parameters
 * for the function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "print_go.hpp"
#include <mlpack/bindings/util/camel_case.hpp>
#include <mlpack/core/util/io.hpp>
#include <mlpack/core/util/hyphenate_string.hpp>
#include <set>

using namespace mlpack::util;
using namespace std;

namespace mlpack {
namespace bindings {
namespace go {

/**
 * Given a list of parameter definition and program documentation, print a
 * generated .go file to stdout.
 *
 * @param doc Documentation for the program.
 * @param functionName Name of the function (i.e. "pca").
 */
void PrintGo(const util::BindingDetails& doc,
             const std::string& functionName)
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
      if (d.name != "help" && d.name != "info" && d.name != "version")
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

  // First, we must generate the mlpack package name.
  cout << "package mlpack" << endl;
  cout << endl;

  // Now we must print the cgo's import libraries and files.
  cout << "/*" << endl;
  cout << "#cgo CFLAGS: -I./capi -Wall" << endl;
  cout << "#cgo LDFLAGS: -L. -lmlpack_go_" << functionName << endl;
  cout << "#include <capi/" << functionName << ".h>" << endl;
  cout << "#include <stdlib.h>" << endl;
  cout << "*/" << endl;
  cout << "import \"C\" " << endl;
  cout << endl;

  // Then we must print the import of the gonum package.
  for (ParamIter it = parameters.begin(); it != parameters.end(); ++it)
  {
    util::ParamData& d = it->second;
    if ((d.cppType).compare(0, 6, "arma::") == 0)
    {
      std::cout << "import \"gonum.org/v1/gonum/mat\" " << std::endl;
      break;
    }
  }
  cout << endl;
  std::string goFunctionName = util::CamelCase(functionName, false);

  // Print Go method configuration struct.
  cout << "type " << goFunctionName << "OptionalParam struct {"
      << std::endl;
  for (size_t i = 0; i < inputOptions.size(); ++i)
  {
    util::ParamData& d = parameters.at(inputOptions[i]);
    size_t indent = 4;
    IO::GetSingleton().functionMap[d.tname]["PrintMethodConfig"](d,
        (void*) &indent, NULL);
  }
  cout << "}" << endl;
  cout << endl;

  // Print Go method configurate struct initialization.
  cout << "func " << goFunctionName << "Options() *"
      << goFunctionName << "OptionalParam {"
      << endl;
  cout << "  " << "return &" << goFunctionName << "OptionalParam{" << endl;
  for (size_t i = 0; i < inputOptions.size(); ++i)
  {
    util::ParamData& d = parameters.at(inputOptions[i]);
    size_t indent = 4;
    IO::GetSingleton().functionMap[d.tname]["PrintMethodInit"](d,
        (void*) &indent, NULL);
  }
  cout << "  " << "}" << endl;
  cout << "}" << endl;
  cout << endl;

  // Print the comment describing the function and its parameters.
  cout << "/*" << endl;
  cout << "  " << HyphenateString(doc.longDescription(), 2) << endl << endl;

  // Print the examples.
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
    if (!d.required)
    {
      bool isLower = false;
      IO::GetSingleton().functionMap[d.tname]["PrintDoc"](d, (void*) &indent,
           &isLower);
    }
    else
    {
      bool isLower = true;
      IO::GetSingleton().functionMap[d.tname]["PrintDoc"](d, (void*) &indent,
          &isLower);
    }
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
    bool isLower = true;
    IO::GetSingleton().functionMap[d.tname]["PrintDoc"](d, (void*) &indent,
        &isLower);
    cout << endl;
  }
  cout << endl;
  cout << " */" << endl;

  // Print the function definition.
  cout << "func " << goFunctionName << "(";

  // Then we print the required input.
  size_t counter = 0;
  for (size_t i = 0; i < inputOptions.size(); ++i)
  {
    util::ParamData& d = parameters.at(inputOptions[i]);
    if (d.required)
    {
      if (i != 0)
        cout << ", ";

      IO::GetSingleton().functionMap[d.tname]["PrintDefnInput"](d, NULL, NULL);
      counter++;
    }
  }

  // Then we print the optional parameter struct input.
  if (counter == 0)
  {
    cout << "param *" << goFunctionName << "OptionalParam) (";
  }
  else
  {
    cout << ", param *" << goFunctionName << "OptionalParam) (";
  }

  // We must then print the output options.
  for (size_t i = 0; i < outputOptions.size(); ++i)
  {
    util::ParamData& d = parameters.at(outputOptions[i]);

    if (i != 0)
      cout << ", ";

    std::tuple<size_t, bool> t = std::make_tuple(2, false);
    IO::GetSingleton().functionMap[d.tname]["PrintDefnOutput"](d,
      (void*) &t, NULL);
  }

  // Print opening brace for function.
  cout << ") {" << endl;

  // Reset any timers and disable backtraces.
  cout << "  " << "resetTimers()" << endl;
  cout << "  " << "enableTimers()" << endl;
  cout << "  " << "disableBacktrace()" << endl;
  cout << "  " << "disableVerbose()" << endl;

  // Restore the parameters.
  cout << "  " << "restoreSettings(\"" << doc.programName << "\")" << endl;
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
  cout << "  " << "// Mark all output options as passed." << endl;
  for (size_t i = 0; i < outputOptions.size(); ++i)
  {
    util::ParamData& d = parameters.at(outputOptions[i]);
    cout << "  " << "setPassed(\"" << d.name << "\")" << endl;
  }
  cout << endl;

  // Call the method.
  cout << "  " << "// Call the mlpack program." << endl;
  cout << "  " << "C.mlpack" << goFunctionName << "()" << endl;
  cout << endl;

  // Do any output processing and return.
  cout << "  " << "// Initialize result variable and get output." << endl;

  for (size_t i = 0; i < outputOptions.size(); ++i)
  {
    util::ParamData& d = parameters.at(outputOptions[i]);

    IO::GetSingleton().functionMap[d.tname]["PrintOutputProcessing"](d,
        NULL, NULL);
  }

  // Clear the parameters.
  cout << endl;
  cout << "  " << "// Clear settings." << endl;
  cout << "  " << "clearSettings()" << endl;
  cout << endl;

  // Return output parameters.
  cout << "  " << "// Return output(s)." << endl;
  cout << "  " << "return ";
  for (size_t i = 0; i < outputOptions.size(); ++i)
  {
    if (i != 0)
      cout << ", ";

    const util::ParamData& d = parameters.at(outputOptions[i]);
    cout << util::CamelCase(d.name, true);
  }
  cout << endl;

  // Print closing bracket.
  cout << "}" << endl;
}

} // namespace go
} // namespace bindings
} // namespace mlpack

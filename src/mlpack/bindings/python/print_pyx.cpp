/**
 * @file generate_pyx.cpp
 * @author Ryan Curtin
 *
 * Implementation of function to generate a .pyx file given a list of parameters
 * for the function.
 */
#include "print_pyx.hpp"
#include "print_functions.hpp"
#include <mlpack/core/util/hyphenate_string.hpp>

using namespace mlpack::util;
using namespace std;

namespace mlpack {
namespace bindings {
namespace python {

/**
 * Given a list of parameter definition and program documentation, print a
 * generated .pyx file to stdout.
 *
 * @param parameters List of parameters the program will use (from CLI).
 * @param programInfo Documentation for the program.
 * @param mainFilename Filename of the main program (i.e.
 *      "/path/to/pca_main.cpp").
 * @param functionName Name of the function (i.e. "pca").
 */
void PrintPYX(const vector<ParamData>& parameters,
              const ProgramDoc& programInfo,
              const string& mainFilename,
              const string& functionName)
{
  // Split into input and output parameters.  Take two passes on the input
  // parameters, so that we get the required ones first.
  vector<size_t> inputOptions, outputOptions;
  for (size_t i = 0; i < parameters.size(); ++i)
  {
    if (parameters[i].input && parameters[i].required)
    {
      // Ignore some parameters.
      if (parameters[i].name != "help" && parameters[i].name != "info" &&
          parameters[i].name != "version")
        inputOptions.push_back(i);
    }
    else if (!parameters[i].input)
    {
      outputOptions.push_back(i);
    }
  }

  for (size_t i = 0; i < parameters.size(); ++i)
    if (parameters[i].input && !parameters[i].required &&
        parameters[i].name != "help" && parameters[i].name != "info" &&
        parameters[i].name != "version")
      inputOptions.push_back(i);

  // First, we must generate the header comment.

  // Now import all the necessary packages.
  cout << "cimport arma" << endl;
  cout << "cimport arma_numpy" << endl;
  cout << "from cli cimport CLI" << endl;
  cout << "from cli cimport SetParam" << endl;
  cout << endl;
  cout << "from libcpp.string cimport string" << endl;
  cout << "from libcpp cimport bool" << endl;
  cout << endl;
  cout << "from cython.operator import dereference" << endl;
  cout << endl;

  // Import the program we will be using.
  cout << "cdef extern from \"<" << mainFilename << ">\" nogil:" << endl;
  cout << "  cdef int mlpackMain() nogil" << endl;
  cout << endl;

  // Print the definition.
  cout << "def " << functionName << "(";
  size_t indent = 4 /* 'def ' */ + functionName.size() + 1 /* '(' */;
  for (size_t i = 0; i < inputOptions.size() - 1; ++i)
  {
    PrintDefinition(parameters[inputOptions[i]]);
    cout << "," << endl << std::string(indent, ' ');
  }
  // Print last option.
  if (inputOptions.size() >= 1)
    PrintDefinition(parameters[inputOptions[inputOptions.size() - 1]]);
  cout << "):" << endl;

  // Print the comment describing the function and its parameters.
  cout << "  \"\"\"" << endl;
  cout << "  " << programInfo.programName << endl;
  cout << endl;
  cout << "  " << HyphenateString(programInfo.documentation, 2) << endl;
  cout << endl << endl;
  cout << "  Parameters:" << endl;
  cout << endl;
  for (size_t i = 0; i < parameters.size(); ++i)
  {
    if (parameters[i].input)
    {
      cout << "    ";
      PrintDocumentation(parameters[i], 4);
      cout << endl;
    }
  }
  cout << "  \"\"\"" << endl;

  // Do any input processing.
  for (size_t i = 0; i < inputOptions.size(); ++i)
    PrintInputProcessing(parameters[inputOptions[i]], 2);

  // Set all output options as passed.
  cout << "  # Mark all output options as passed." << endl;
  for (size_t i = 0; i < outputOptions.size(); ++i)
    cout << "  CLI.SetPassed(<const string> '"
        << parameters[outputOptions[i]].name << "')" << endl;

  // Call the method.
  cout << "  # Call the mlpack program." << endl;
  cout << "  mlpackMain()" << endl;

  // Do any output processing and return.
  if (outputOptions.size() > 1)
  {
    cout << "  # Initialize result dictionary." << endl;
    cout << "  result = {}" << endl;
  }
  cout << endl;

  for (size_t i = 0; i < outputOptions.size(); ++i)
    PrintOutputProcessing(parameters[outputOptions[i]], 2,
        outputOptions.size() == 1);

  if (outputOptions.size() > 1)
  {
    cout << "  return result" << endl;
  }
}

} // namespace python
} // namespace bindings
} // namespace mlpack

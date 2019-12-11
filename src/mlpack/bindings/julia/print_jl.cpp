/**
 * @file print_jl.cpp
 * @author Ryan Curtin
 *
 * Implementation of utility PrintJL() function.
 */
#include "print_jl.hpp"
#include <mlpack/core/util/hyphenate_string.hpp>

using namespace mlpack;
using namespace std;

namespace mlpack {
namespace bindings {
namespace julia {

extern std::string programName;

/**
 * Print the code for a .jl binding for an mlpack program to stdout.
 */
void PrintJL(const util::ProgramDoc& programInfo,
             const string& functionName,
             const std::string& mlpackJuliaLibSuffix)
{
  // Restore parameters.
  CLI::RestoreSettings(programInfo.programName);

  const map<string, util::ParamData>& parameters = CLI::Parameters();
  typedef map<string, util::ParamData>::const_iterator ParamIter;

  // First, let's get a list of input and output options.  We'll take two passes
  // so that the required input options are the first in the list.
  vector<string> inputOptions, outputOptions;
  for (ParamIter it = parameters.begin(); it != parameters.end(); ++it)
  {
    const util::ParamData& d = it->second;
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
    const util::ParamData& d = it->second;
    if (d.input && !d.required &&
        d.name != "help" && d.name != "info" &&
        d.name != "version")
      inputOptions.push_back(it->first);
  }

  // First, define what we are exporting.
  cout << "export " << functionName << endl;
  cout << endl;

  // We need to include utility functions.
  cout << "using mlpack.util.cli" << endl;
  cout << endl;

  // Make sure the libraries we need are accessible.
  cout << "const " << functionName << "Library = joinpath(@__DIR__, "
      << "\"libmlpack_julia_" << functionName << mlpackJuliaLibSuffix << "\")"
      << endl;
  cout << endl;

  // Define mlpackMain() function to call.
  cout << "# Call the C binding of the mlpack " << functionName << " binding."
      << endl;
  cout << "function " << functionName << "_mlpackMain()" << endl;
  cout << "  success = ccall((:" << functionName << ", " << functionName
      << "Library), Bool, ())" << endl;
  cout << "  if !success" << endl;
  cout << "    # Throw an exception---false means there was a C++ exception."
      << endl;
  cout << "    throw(ErrorException(\"mlpack binding error; see output\"))"
      << endl;
  cout << "  end" << endl;
  cout << "end" << endl;
  cout << endl;

  // If we have any model types, we need to define functions to set and get
  // their values from the CLI object.  We do this with the PrintParamDefn()
  // function.  We'll gather all names of classes we've done this with, so that
  // we don't print any duplicates.  This should all be done inside of an
  // internal module.
  cout << "\" Internal module to hold utility functions. \"" << endl;
  cout << "module " << functionName << "_internal" << endl;
  cout << "  import .." << functionName << "Library" << endl;
  cout << endl;

  set<string> classNames;
  for (ParamIter it = parameters.begin(); it != parameters.end(); ++it)
  {
    const util::ParamData& d = it->second;
    if (classNames.count(d.cppType) == 0)
    {
      CLI::GetSingleton().functionMap[d.tname]["PrintParamDefn"](d, (void*)
          &functionName, NULL);

      // Avoid adding this definition again.
      classNames.insert(d.cppType);
    }
  }

  // End the module.
  cout << "end # module" << endl;
  cout << endl;

  // Print the documentation.
  cout << "\"\"\"" << endl;
  cout << "    " << functionName << "(";

  // Print a list of input arguments after the function name.
  bool defaults = false;
  for (size_t i = 0; i < inputOptions.size(); ++i)
  {
    const string& opt = inputOptions[i];
    const util::ParamData& d = parameters.at(opt);

    if (!defaults && !d.required)
    {
      // Open the bracket.
      cout << "; [";
      defaults = true;
    }
    else if (i > 0)
    {
      cout << ", ";
    }

    cout << d.name;
  }
  if (defaults)
    cout << "]";
  cout << ")" << endl;
  cout << endl;

  // Next print the description.
  cout << util::HyphenateString(programInfo.documentation(), 0) << endl;

  // Next, print information on the input options.
  cout << endl;
  cout << "# Arguments" << endl;
  cout << endl;

  for (size_t i = 0; i < inputOptions.size(); ++i)
  {
    const string& opt = inputOptions[i];
    const util::ParamData& d = parameters.at(opt);

    std::ostringstream oss;
    oss << " - ";

    CLI::GetSingleton().functionMap[d.tname]["PrintDoc"](d, NULL, (void*) &oss);

    cout << util::HyphenateString(oss.str(), 6) << endl;
  }

  cout << endl;
  cout << "# Return values" << endl;
  cout << endl;

  for (size_t i = 0; i < outputOptions.size(); ++i)
  {
    const string& opt = outputOptions[i];
    const util::ParamData& d = parameters.at(opt);

    std::ostringstream oss;
    oss << " - ";

    CLI::GetSingleton().functionMap[d.tname]["PrintDoc"](d, NULL, (void*) &oss);

    cout << util::HyphenateString(oss.str(), 6) << endl;
  }
  cout << endl;

  cout << "\"\"\"" << endl;

  // Print the signature.
  cout << "function " << functionName << "(";
  const size_t indent = 10 + functionName.size();

  // Print required input arguments as part of the function signature, followed
  // by non-required input arguments.
  defaults = false;
  for (size_t i = 0; i < inputOptions.size(); ++i)
  {
    const string& opt = inputOptions[i];
    const util::ParamData& d = parameters.at(opt);

    if (!defaults && !d.required)
    {
      cout << ";" << endl << string(indent, ' ');
      defaults = true;
    }
    else if (i > 0)
    {
      cout << "," << endl << string(indent, ' ');
    }

    CLI::GetSingleton().functionMap[d.tname]["PrintInputParam"](d, NULL,
        NULL);
  }

  // Print the 'points_are_rows' option.
  if (!defaults)
    cout << ";" << endl << string(indent, ' ');
  else
    cout << "," << endl << string(indent, ' ');
  cout << "points_are_rows::Bool = true)" << endl;

  // Force symbols to load.
  cout << "  # Force the symbols to load." << endl;
  cout << "  ccall((:loadSymbols, " << functionName << "Library), Nothing, ());"
      << endl;
  cout << endl;

  // Restore CLI settings.
  cout << "  CLIRestoreSettings(\"" << programName << "\")" << endl;
  cout << endl;

  // Handle each input argument's processing before calling mlpackMain().
  cout << "  # Process each input argument before calling mlpackMain()."
      << endl;
  for (const string& opt : inputOptions)
  {
    if (opt != "verbose")
    {
      const util::ParamData& d = parameters.at(opt);
      CLI::GetSingleton().functionMap[d.tname]["PrintInputProcessing"](d,
          &functionName, NULL);
    }
  }

  // Special handling for verbose output.
  cout << "  if verbose !== nothing && verbose === true" << endl;
  cout << "    CLIEnableVerbose()" << endl;
  cout << "  else" << endl;
  cout << "    CLIDisableVerbose()" << endl;
  cout << "  end" << endl;
  cout << endl;

  // Mark output parameters as passed.
  for (const string& opt : outputOptions)
  {
    const util::ParamData& d = parameters.at(opt);
    cout << "  CLISetPassed(\"" << d.name << "\")" << endl;
  }

  // Call the program.
  cout << "  # Call the program." << endl;
  cout << "  " << functionName << "_mlpackMain()" << endl;
  cout << endl;

  // Extract the results in order.
  cout << "  return ";
  string indentStr(9, ' ');
  for (size_t i = 0; i < outputOptions.size(); ++i)
  {
    const util::ParamData& d = parameters.at(outputOptions[i]);
    CLI::GetSingleton().functionMap[d.tname]["PrintOutputProcessing"](d,
        &functionName, NULL);

    // Print newlines if we are returning multiple output options.
    if (i + 1 < outputOptions.size())
      cout << "," << endl << indentStr;
  }

  cout << endl << "end" << endl;
}

} // namespace julia
} // namespace bindings
} // namespace mlpack

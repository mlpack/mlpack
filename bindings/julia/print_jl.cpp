/**
 * @file bindings/julia/print_jl.cpp
 * @author Ryan Curtin
 *
 * Implementation of utility PrintJL() function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "print_jl.hpp"
#include <mlpack/core/util/hyphenate_string.hpp>

#include <set>

using namespace mlpack::util;
using namespace std;

namespace mlpack {
namespace bindings {
namespace julia {

extern std::string programName;

/**
 * Print the code for a .jl binding for an mlpack program to stdout.
 */
void PrintJL(const string& bindingName,
             const string& functionName,
             const string& mlpackJuliaLibSuffix)
{
  Params p = IO::Parameters(bindingName);
  const BindingDetails& doc = p.Doc();

  map<string, ParamData>& parameters = p.Parameters();
  typedef map<string, ParamData>::iterator ParamIter;

  // First, let's get a list of input and output options.  We'll take two passes
  // so that the required input options are the first in the list.
  vector<string> inputOptions, outputOptions;
  for (ParamIter it = parameters.begin(); it != parameters.end(); ++it)
  {
    ParamData& d = it->second;
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
    ParamData& d = it->second;
    if (d.input && !d.required && d.name != "help" && d.name != "info" &&
        d.name != "version")
      inputOptions.push_back(it->first);
  }

  // First, define what we are exporting.
  cout << "export " << functionName << endl;
  cout << endl;

  // Now import any types we might need.
  set<string> classNames;
  for (ParamIter it = parameters.begin(); it != parameters.end(); ++it)
  {
    ParamData& d = it->second;
    if (classNames.count(d.cppType) == 0)
    {
      p.functionMap[d.tname]["PrintModelTypeImport"](d, NULL, NULL);

      // Avoid adding this import again.
      classNames.insert(d.cppType);
    }
  }
  cout << endl;

  // We need to include utility functions.
  cout << "using mlpack._Internal.params" << endl;
  cout << endl;

  // Make sure the libraries we need are accessible.
  cout << "const " << functionName << "Library = joinpath(@__DIR__, "
      << "\"libmlpack_julia_" << functionName << mlpackJuliaLibSuffix << "\")"
      << endl;
  cout << endl;

  // Define mlpackMain() function to call.
  cout << "# Call the C binding of the mlpack " << functionName << " binding."
      << endl;
  cout << "function call_" << bindingName << "(p, t)" << endl;
  cout << "  success = ccall((:mlpack_" << functionName << ", " << functionName
      << "Library), Bool, (Ptr{Nothing}, Ptr{Nothing}), p, t)" << endl;
  cout << "  if !success" << endl;
  cout << "    # Throw an exception---false means there was a C++ exception."
      << endl;
  cout << "    throw(ErrorException(\"mlpack binding error; see output\"))"
      << endl;
  cout << "  end" << endl;
  cout << "end" << endl;
  cout << endl;

  // If we have any model types, we need to define functions to set and get
  // their values from the IO object.  We do this with the PrintParamDefn()
  // function.  We'll gather all names of classes we've done this with, so that
  // we don't print any duplicates.  This should all be done inside of an
  // internal module.
  cout << "\" Internal module to hold utility functions. \"" << endl;
  cout << "module " << functionName << "_internal" << endl;
  cout << "  import .." << functionName << "Library" << endl;
  cout << endl;

  classNames.clear();
  for (ParamIter it = parameters.begin(); it != parameters.end(); ++it)
  {
    ParamData& d = it->second;
    if (classNames.count(d.cppType) == 0)
    {
      p.functionMap[d.tname]["PrintParamDefn"](d, (void*) &functionName, NULL);

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
    ParamData& d = parameters.at(opt);

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
  cout << HyphenateString(doc.longDescription(), 0) << endl << endl;

  // Next print the examples.
  for (size_t j = 0; j < doc.example.size(); ++j)
  {
    cout << util::HyphenateString(doc.example[j](), 0) << endl << endl;
  }

  // Next, print information on the input options.
  cout << "# Arguments" << endl;
  cout << endl;

  for (size_t i = 0; i < inputOptions.size(); ++i)
  {
    const string& opt = inputOptions[i];
    ParamData& d = parameters.at(opt);

    std::ostringstream oss;
    oss << " - ";

    p.functionMap[d.tname]["PrintDoc"](d, NULL, (void*) &oss);

    cout << HyphenateString(oss.str(), 6) << endl;
  }

  cout << endl;
  cout << "# Return values" << endl;
  cout << endl;

  for (size_t i = 0; i < outputOptions.size(); ++i)
  {
    const string& opt = outputOptions[i];
    ParamData& d = parameters.at(opt);

    std::ostringstream oss;
    oss << " - ";

    p.functionMap[d.tname]["PrintDoc"](d, NULL, (void*) &oss);

    cout << HyphenateString(oss.str(), 6) << endl;
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
    ParamData& d = parameters.at(opt);

    if (!defaults && !d.required)
    {
      cout << ";" << endl << string(indent, ' ');
      defaults = true;
    }
    else if (i > 0)
    {
      cout << "," << endl << string(indent, ' ');
    }

    p.functionMap[d.tname]["PrintInputParam"](d, NULL, NULL);
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

  // Create the set of model pointers.
  cout << "  # Create the set of model pointers to avoid setting multiple "
      << "finalizers." << endl;
  cout << "  modelPtrs = Set{Ptr{Nothing}}()" << endl;
  cout << endl;

  // Get an empty Params and Timers object.
  cout << "  p = GetParameters(\"" << bindingName << "\")" << endl;
  cout << "  t = Timers()" << endl;
  cout << endl;

  // Create a set where we will store the pointers associated with all memory
  // that Julia owns.  This is to prevent situations where we end up wrapping a
  // result object and telling Julia to own that too---in that case, the GC will
  // free the object twice!
  cout << "  juliaOwnedMemory = Set{Ptr{Nothing}}()" << endl;

  // Handle each input argument's processing before calling mlpackMain().
  cout << "  # Process each input argument before calling mlpackMain()."
      << endl;
  for (const string& opt : inputOptions)
  {
    if (opt != "verbose")
    {
      ParamData& d = parameters.at(opt);
      p.functionMap[d.tname]["PrintInputProcessing"](d, &functionName, NULL);
    }
  }

  // Special handling for verbose output.
  cout << "  if verbose !== nothing && verbose === true" << endl;
  cout << "    EnableVerbose()" << endl;
  cout << "  else" << endl;
  cout << "    DisableVerbose()" << endl;
  cout << "  end" << endl;
  cout << endl;

  // Mark output parameters as passed.
  for (const string& opt : outputOptions)
  {
    ParamData& d = parameters.at(opt);
    cout << "  SetPassed(p, \"" << d.name << "\")" << endl;
  }

  // Call the program.
  cout << "  # Call the program." << endl;
  cout << "  call_" << bindingName << "(p, t)" << endl;
  cout << endl;

  // Extract the results in order.
  cout << "  results = (";
  string indentStr(13, ' ');
  for (size_t i = 0; i < outputOptions.size(); ++i)
  {
    ParamData& d = parameters.at(outputOptions[i]);
    p.functionMap[d.tname]["PrintOutputProcessing"](d, &functionName, NULL);

    // Print newlines if we are returning multiple output options.
    if (i + 1 < outputOptions.size())
      cout << "," << endl << indentStr;
  }

  cout << ")" << endl << endl;

  // Clean up.
  cout << "  # We are responsible for cleaning up the `p` and `t` objects."
      << endl;
  cout << "  DeleteParameters(p)" << endl;
  cout << "  DeleteTimers(t)" << endl;
  cout << endl;
  cout << "  return results" << endl;
  cout << "end" << endl;
}

} // namespace julia
} // namespace bindings
} // namespace mlpack

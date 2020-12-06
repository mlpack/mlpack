/**
 * @file bindings/R/print_R.cpp
 * @author Yashwant Singh Parihar
 *
 * Implementation of utility PrintR() function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "print_R.hpp"
#include <mlpack/bindings/util/strip_type.hpp>
#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/hyphenate_string.hpp>

using namespace mlpack;
using namespace std;

namespace mlpack {
namespace bindings {
namespace r {


/**
 * Print the code for a .R binding for an mlpack program to stdout.
 *
 * @param doc Documentation for the binding.
 * @param functionName Name of the function (i.e. "pca").
 */
void PrintR(const util::BindingDetails& doc,
            const string& functionName)
{
  // Restore parameters.
  IO::RestoreSettings(doc.programName);

  map<string, util::ParamData>& parameters = IO::Parameters();
  typedef map<string, util::ParamData>::iterator ParamIter;

  // First, let's get a list of input and output options.  We'll take two passes
  // so that the required input options are the first in the list.
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

  // Print the documentation.
  // Print programName as @title.
  cout << "#' @title ";
  cout << util::HyphenateString(doc.programName, "#'   ") << endl;
  cout << "#'" << endl;

  // Next print the short description as @description.
  cout << "#' @description" << endl;
  cout << "#' ";
  cout << util::HyphenateString(doc.shortDescription, "#' ") << endl;

  // Next, print information on the input options.
  cout << "#'" << endl;

  for (size_t i = 0; i < inputOptions.size(); ++i)
  {
    const string& opt = inputOptions[i];
    util::ParamData& d = parameters.at(opt);

    bool out = false;
    IO::GetSingleton().functionMap[d.tname]["PrintDoc"](d, NULL, (void*) &out);

    cout << endl;
  }
  cout << "#'" << endl;

  // Next, print information on the output options.
  if (outputOptions.size() > 0)
    cout << "#' @return A list with several components:" << endl;

  for (size_t i = 0; i < outputOptions.size(); ++i)
  {
    const string& opt = outputOptions[i];
    util::ParamData& d = parameters.at(opt);

    bool out = true;
    IO::GetSingleton().functionMap[d.tname]["PrintDoc"](d, NULL, (void*) &out);

    cout << endl;
  }
  cout << "#'" << endl;

  // Next print the long description as @details.
  cout << "#' @details" << endl;
  cout << "#' ";
  cout << util::HyphenateString(doc.longDescription(), "#' ") << endl;
  cout << "#'" << endl;
  cout << "#' @author" << endl;
  cout << "#' mlpack developers" << endl;
  cout << "#'" << endl;

  // Next print the example as @examples.
  cout << "#' @export" << endl;
  if (doc.example.size() != 0)
    cout << "#' @examples" << endl;
  for (size_t j = 0; j < doc.example.size(); ++j)
  {
    const std::string str = doc.example[j]();
    size_t pos = 0;
    while (pos < str.length())
    {
      size_t splitpos = 0;
      // Find where example starts.
      splitpos = str.find("\n\\dontrun{", pos) - 1;
      // If no example left, then print all the comments that are left.
      if (splitpos == std::string::npos || splitpos > str.length())
      {
        splitpos = str.length();
        cout << util::HyphenateString(str.substr(pos, (splitpos - pos)),
            "#' # ");
        break;
      }
      if (splitpos != 0 && pos == 0)
        cout << "#' # ";
      // Print comments in the "example", if there is available.
      cout << util::HyphenateString(str.substr(pos, (splitpos - pos)),
              "#' # ", true);
      // Find where example ends.
      pos = str.find("\n}", pos) + 3;
      // Here length of example might be less 80, we must handle this carefully.
      // Print example in the "example".
      cout << util::HyphenateString(str.substr(splitpos, (pos - splitpos)),
              "#' ", true);
    }
    cout << endl;
  }

  // Print the definition.
  cout << functionName << " <- function(";
  size_t indent = functionName.size() + 13 /* <- function(*/;
  for (size_t i = 0; i < inputOptions.size(); ++i)
  {
    util::ParamData& d = parameters.at(inputOptions[i]);

    if (i != 0)
      cout << "," << endl << std::string(indent, ' ');

    IO::GetSingleton().functionMap[d.tname]["PrintInputParam"](d, NULL, NULL);
  }

  // Print closing brace for function definition.
  cout << ") {" << endl;

  // Restore IO settings.
  cout << "  # Restore IO settings." << endl;
  cout << "  IO_RestoreSettings(\"" << IO::ProgramName()
       << "\")" << endl;
  cout << endl;

  // Handle each input argument's processing before calling mlpackMain().
  cout << "  # Process each input argument before calling mlpackMain()."
       << endl;
  for (const string& opt : inputOptions)
  {
    if (opt != "verbose")
    {
      util::ParamData& d = parameters.at(opt);
      IO::GetSingleton().functionMap[d.tname]["PrintInputProcessing"](d,
          NULL, NULL);
    }
  }

  // Special handling for verbose output.
  cout << "  if (verbose) {" << endl;
  cout << "    IO_EnableVerbose()" << endl;
  cout << "  } else {" << endl;
  cout << "    IO_DisableVerbose()" << endl;
  cout << "  }" << endl;
  cout << endl;

  // Mark output parameters as passed.
  cout << "  # Mark all output options as passed." << endl;
  for (const string& opt : outputOptions)
  {
    util::ParamData& d = parameters.at(opt);
    cout << "  IO_SetPassed(\"" << d.name << "\")" << endl;
  }
  cout << endl;

  // Call the program.
  cout << "  # Call the program." << endl;
  cout << "  " << functionName << "_mlpackMain()" << endl << endl;

  // Add ModelType as attr to the model pointer.
  cout << "  # Add ModelType as attribute to the model pointer, if needed."
      << endl;
  for (size_t i = 0; i < outputOptions.size(); ++i)
  {
    util::ParamData& d = parameters.at(outputOptions[i]);
    IO::GetSingleton().functionMap[d.tname]["PrintSerializeUtil"](d,
        NULL, NULL);
  }
  cout << endl;

  // Extract the results in order.
  cout << "  # Extract the results in order." << endl;
  cout << "  out <- list(" << endl;
  string indentStr(4, ' ');
  for (size_t i = 0; i < outputOptions.size(); ++i)
  {
    if (i == 0)
       cout << indentStr;
    util::ParamData& d = parameters.at(outputOptions[i]);
    IO::GetSingleton().functionMap[d.tname]["PrintOutputProcessing"](d,
        NULL, NULL);
    // Print newlines if we are returning multiple output options.
    if (i + 1 < outputOptions.size())
      cout << "," << endl << indentStr;
  }
  cout << endl << "  )" << endl << endl;

  // Clear the parameters.
  cout << "  # Clear the parameters." << endl;
  cout << "  IO_ClearSettings()" << endl;
  cout << endl;
  cout << "  return(out)" << endl << "}" << endl;
}

} // namespace r
} // namespace bindings
} // namespace mlpack

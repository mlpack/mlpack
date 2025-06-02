/**
 * @file bindings/markdown/print_docs.cpp
 * @author Ryan Curtin
 *
 * Implementation of functions to print Markdown from documentation.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core/util/io.hpp>
#include <mlpack/core/util/binding_details.hpp>

#include "binding_info.hpp"
#include "print_docs.hpp"
#include "print_doc_functions.hpp"
#include "print_param_table.hpp"
#include "replace_all_copy.hpp"

// Make sure that this is defined.
#ifndef SRC_PREFIX
#define SRC_PREFIX "https://github.com/mlpack/mlpack/blob/master/src/"
#endif

// Relative path to the root of the documentation directory.
// (The assumption is that the documentation is built in doc/user/bindings/.)
#ifndef DOC_PREFIX
#define DOC_PREFIX "../../"
#endif

using namespace std;
using namespace mlpack;
using namespace mlpack::util;
using namespace mlpack::bindings;
using namespace mlpack::bindings::markdown;

void PrintHeaders(const string& bindingName,
                  const bool addWrapperDocs)
{
  // Get the name of the binding in the target language, and convert it to
  // lowercase (since the anchor link will be in lowercase).
  const string langBindingName = addWrapperDocs ? GetWrapperName(bindingName) :
          GetBindingName(bindingName);

  cout << "<li><a href=\"LINKROOTuser/bindings/" << BindingInfo::Language()
      << ".html#" << bindingName << "\">" << langBindingName << "</a></li>"
      << endl;
}

void PrintDocs(const string& bindingName,
               const vector<string>& validMethods,
               const bool addWrapperDocs)
{
  // Here we have to handle both wrapper and non wrapper bindings.

  // 'params' stores the Params object for non-wrapper bindings.
  Params params = IO::Parameters(bindingName);

  const BindingDetails& doc = params.Doc();

  // 'paramMethods' stores the Params object for all wrapper bindings.
  vector<Params> paramMethods;

  // adding elements to  'paramMethods'.
  for (size_t i = 0; i < validMethods.size(); ++i)
  {
    Params methodParams = IO::Parameters(bindingName + "_" + validMethods[i]);
    paramMethods.push_back(methodParams);
  }

  // First, for this section, print each of the names.
  // We have to check if docs are printed for wrappers or not.
  const string langBindingName = addWrapperDocs ? GetWrapperName(bindingName) :
      GetBindingName(bindingName);

  cout << "## " << langBindingName << endl;
  cout << "{: #" << bindingName << " }" << endl;
  cout << endl;

  // Next, print the logical name of the binding (that's known by
  // ProgramInfo).
  cout << "#### ";
  if (addWrapperDocs)
    cout << paramMethods[0].Doc().name << endl;
  else
    cout << doc.name << endl;
  cout << "{: #" << bindingName << "_descr }" << endl;
  cout << endl;

  // No need to print ProgramCall for wrappers (already included in example).
  if (!addWrapperDocs)
  {
    cout << ProgramCall(bindingName);
  }
  cout << endl;

  // Print short description.
  if (addWrapperDocs)
  {
    string desc = ReplaceAllCopy(paramMethods[0].Doc().longDescription(), "|",
        "\\|");
    cout << desc;
  }
  else
  {
    cout << doc.shortDescription << " ";
    cout << "[Detailed documentation](#" << bindingName
        << "_detailed-documentation)." << endl;
    cout << endl;
  }
  cout << endl;

  // Next, print the PROGRAM_INFO() documentation.
  if (!addWrapperDocs) // different structure for wrappers.
  {
    // We need to print the signature.

    // Now, iterate through each of the input options.
    cout << endl;
    cout << "### Input options" << endl;
    string inputInfo = PrintInputOptionInfo();
    if (inputInfo.size() > 0)
      cout << inputInfo << endl;
    cout << endl;

    cout << "| ***name*** | ***type*** | ***description*** | ***default*** |"
          << endl;
    cout << "|------------|------------|-------------------|---------------|"
        << endl;

    unordered_set<string> paramsSet;
    PrintParamTable(bindingName, params,
        {"name", "type", "description", "default"}, paramsSet,
        false, false, true, false);
    paramsSet.clear();

    cout << endl;

    // Determine if there are any output options, to see if we need
    // to print the header of the output options table.
    bool hasOutputOptions = false;
    map<string, ParamData>& parameters = params.Parameters();
    for (map<string, ParamData>::iterator it = parameters.begin();
        it != parameters.end(); ++it)
    {
      if (!it->second.input)
      {
        hasOutputOptions = true;
        break;
      }
    }

    if (hasOutputOptions)
    {
      // Next, iterate through the list of output options.
      cout << "### Output options" << endl;
      cout << endl;
      string outputInfo = PrintOutputOptionInfo();
      if (outputInfo.size() > 0)
        cout << outputInfo << endl;
      cout << endl;
      cout << "| ***name*** | ***type*** | ***description*** |" << endl;
      cout << "|------------|------------|-------------------|" << endl;
    }

    PrintParamTable(bindingName, params,
        {"name", "type", "description"}, paramsSet,
        false, false, false, true);
    paramsSet.clear();

    cout << endl;

    cout << "### Detailed documentation" << endl;
    cout << "{: #" << bindingName << "_detailed-documentation }" << endl;
    cout << endl;
    string desc = ReplaceAllCopy(doc.longDescription(), "|", "\\|");
    cout << desc << endl << endl;

    if (doc.example.size() > 0)
      cout << "### Example" << endl;
    for (size_t j = 0; j < doc.example.size(); ++j)
    {
      string eg = ReplaceAllCopy(doc.example[j](), "|", "\\|");
      cout << eg << endl << endl;
    }
    cout << "### See also" << endl;
    cout << endl;

    for (size_t j = 0; j < doc.seeAlso.size(); ++j)
    {
      cout << " - " << "[";
      // We need special processing if the user has specified a binding name
      // starting with @ (i.e., '@kfn' or similar).
      if (doc.seeAlso[j].first[0] == '@')
        cout << GetBindingName(doc.seeAlso[j].first.substr(1));
      else
        cout << doc.seeAlso[j].first;
      cout << "](";

      // We need special handling of source links.
      if (doc.seeAlso[j].second.substr(0, 4) == "@src")
      {
        cout << SRC_PREFIX << doc.seeAlso[j].second.substr(5);
      }
      else if (doc.seeAlso[j].second.substr(0, 4) == "@doc")
      {
        cout << DOC_PREFIX << doc.seeAlso[j].second.substr(5);
      }
      else if (doc.seeAlso[j].second[0] == '#')
      {
        cout << "#" << doc.seeAlso[j].second.substr(1);
      }
      else
      {
        cout << doc.seeAlso[j].second;
      }

      cout << ")" << endl;
    }
    cout << endl;
  }
  else // for wrappers.
  {
    cout << "### Parameters" << endl;
    cout << endl;

    cout << "| ***name*** | ***type*** | ***description*** | ***default*** |"
        << endl;
    cout << "|------------|------------|-------------------|---------------|"
        << endl;

    unordered_set<string> paramsSet; // to prevent duplicates.

    for (size_t j = 0; j < paramMethods.size(); ++j)
    {
      PrintParamTable(bindingName + "_" + validMethods[j],
          paramMethods[j], {"name", "type", "description",
          "default"}, paramsSet, true, false, true, false);
    }
    paramsSet.clear(); // for reusing.
    cout << endl;

    cout << "### Example" << endl;
    cout << endl;
    string example;

    for (size_t j = 0; j < paramMethods.size(); ++j)
    {
      for (size_t k = 0; k < paramMethods[j].Doc().example.size(); ++k)
      {
        string eg = ReplaceAllCopy(paramMethods[j].Doc().example[k](), "|",
            "\\|");
        example += eg + "\n";
      }
    }

    cout << "```" << BindingInfo::Language() << endl;
    // do not want the last "\n".
    cout << example.substr(0, example.size() - 1) << endl;
    cout << "```" << endl;
    cout << endl;

    cout << "### Methods" << endl;
    cout << endl;
    cout << "| **name** | **description** |" << endl;
    cout << "|----------|-----------------|" << endl;
    for (size_t j = 0; j < validMethods.size(); j++)
    {
      cout << "| " << GetMappedName(validMethods[j]) << " | ";
      cout << paramMethods[j].Doc().shortDescription;
      cout << " |" << endl;
    }
    cout << endl;

    // Print information for each method.
    for (size_t j = 0; j < validMethods.size(); j++)
    {
      cout << "### " << j + 1 << ". " << GetMappedName(validMethods[j])
          << endl;
      cout << endl;
      cout << paramMethods[j].Doc().shortDescription << endl;
      cout << endl;

      cout << "#### Input Parameters:" << endl;
      cout << endl;

      cout << "| **name** | **type** | **description** |" << endl;
      cout << "|----------|----------|-----------------|" << endl;

      PrintParamTable(bindingName + "_" + validMethods[j],
          paramMethods[j], {"name", "type", "description"},
          paramsSet, false, true, true, false);
      paramsSet.clear();
      cout << endl;

      cout << "#### Returns: " << endl;
      cout << endl;

      cout << "| **type** | **description** |" << endl;
      cout << "|----------|-----------------|" << endl;

      PrintParamTable(bindingName + "_" + validMethods[j],
          paramMethods[j], {"type", "description"},
          paramsSet, false, false, false, true);
      cout << endl;
    }
  }
}

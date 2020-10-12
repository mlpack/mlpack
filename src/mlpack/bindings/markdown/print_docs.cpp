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
#include "print_docs.hpp"

#include <mlpack/core/util/io.hpp>
#include <mlpack/core/util/binding_details.hpp>
#include "binding_info.hpp"
#include "print_doc_functions.hpp"

// Make sure that this is defined.
#ifndef DOXYGEN_PREFIX
#define DOXYGEN_PREFIX "https://mlpack.org/doc/mlpack-git/doxygen/"
#endif

using namespace std;
using namespace mlpack;
using namespace mlpack::util;
using namespace mlpack::bindings;
using namespace mlpack::bindings::markdown;

void PrintHeaders(const std::string& bindingName,
                  const std::vector<std::string>& languages)
{
  // We just want to print the name of the function and a link, as Markdown.
  // We have to mark it as having the right language with a div.
  for (size_t i = 0; i < languages.size(); ++i)
  {
    BindingInfo::Language() = languages[i];

    cout << " - [" << GetBindingName(bindingName) << "](#" << languages[i]
        << "_" << bindingName << "){: .language-link #" << languages[i] << " }"
        << endl;
  }
}

void PrintDocs(const std::string& bindingName,
               const vector<string>& languages)
{
  BindingDetails& doc = BindingInfo::GetBindingDetails(bindingName);

  IO::RestoreSettings(bindingName);

  // First, for this section, print each of the names.
  for (size_t i = 0; i < languages.size(); ++i)
  {
    BindingInfo::Language() = languages[i];

    cout << "<div class=\"language-title\" id=\"" << languages[i]
        << "\" markdown=\"1\">" << endl;
    cout << "## " << GetBindingName(bindingName) << endl;
    cout << "{: #" << languages[i] << "_" << bindingName << " }" << endl;
    cout << "</div>" << endl;
  }
  cout << endl;

  // Next, print the logical name of the binding (that's known by
  // ProgramInfo).
  cout << "#### " << doc.programName << endl;
  cout << endl;

  for (size_t i = 0; i < languages.size(); ++i)
  {
    BindingInfo::Language() = languages[i];

    cout << "<div class=\"language-decl\" id=\"" << languages[i]
        << "\" markdown=\"1\">" << endl;
    cout << ProgramCall(bindingName);
    cout << "</div>" << endl;
  }
  cout << endl;

  cout << doc.shortDescription << " ";
  for (size_t i = 0; i < languages.size(); ++i)
  {
    cout << "[Detailed documentation](#" << languages[i] << "_"
      << bindingName << "_detailed-documentation){: .language-detail-link #"
      << languages[i] << " }";
  }
  cout << "." << endl;

  // Next, print the PROGRAM_INFO() documentation for each language.
  for (size_t i = 0; i < languages.size(); ++i)
  {
    BindingInfo::Language() = languages[i];

    // This works with the Kramdown processor.
    cout << "<div class=\"language-section\" id=\"" << languages[i]
        << "\" markdown=\"1\">" << endl;

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
    map<string, ParamData>& parameters = IO::Parameters();
    for (map<string, ParamData>::iterator it = parameters.begin();
         it != parameters.end(); ++it)
    {
      if (!it->second.input)
        continue;

      // There are some special options that don't exist in some languages.
      if (languages[i] != "python" && it->second.name == "copy_all_inputs")
        continue;
      if (languages[i] != "cli" &&
          (it->second.name == "help" || it->second.name == "info" ||
           it->second.name == "version"))
        continue;

      // Print name, type, description, default.
      cout << "| ";
      // We need special processing if the language is go then the required
      // parameter will be lowerCamelCase.
      if (languages[i] == "go")
      {
        if (!it->second.required)
        {
          cout << ParamString(it->second.name) << " | ";
        }
        else
        {
          std::string name = ParamString(it->second.name);
          name[1] = std::tolower(name[1]);
          cout << name << " | ";
        }
      }
      else
      {
        cout << ParamString(it->second.name) << " | ";
      }
      cout << ParamType(it->second) << " | ";
      string desc = boost::replace_all_copy(it->second.desc, "|", "\\|");
      cout << desc; // just a string
      // Print whether or not it's a "special" language-only parameter.
      if (it->second.name == "copy_all_inputs" || it->second.name == "help" ||
          it->second.name == "info" || it->second.name == "version")
      {
        cout << "  <span class=\"special\">Only exists in "
            << PrintLanguage(languages[i]) << " binding.</span>";
      }
      cout << " | ";
      string def = PrintDefault(it->second.name);
      if (def.size() > 0)
        cout << "`" << def << "` |";
      else
        cout << " |";
      cout << endl;
    }
    cout << endl;

    // Determine if there are any output options, to see if we need
    // to print the header of the output options table.
    bool hasOutputOptions = false;
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
    for (map<string, ParamData>::iterator it = parameters.begin();
         it != parameters.end(); ++it)
    {
      if (it->second.input)
        continue;

      // Print name, type, description.
      cout << "| ";
      // We need special processing if the language is go then the output
      // parameter will be lowerCamelCase.
      if (languages[i] == "go")
      {
        std::string name = ParamString(it->second.name);
        name[1] = std::tolower(name[1]);
        cout << name << " | ";
      }
      else
      {
        cout << ParamString(it->second.name) << " | ";
      }
      cout << ParamType(it->second) << " | ";
      cout << it->second.desc;
      // Print whether or not it's a "special" language-only parameter.
      if (it->second.name == "copy_all_inputs" || it->second.name == "help" ||
          it->second.name == "info" || it->second.name == "version")
      {
        cout << "  <span class=\"special\">Only exists in "
            << PrintLanguage(languages[i]) << " binding.</span>";
      }
      cout << " |";
      cout << endl;
    }
    cout << endl;

    cout << "### Detailed documentation" << endl;
    cout << "{: #" << languages[i] << "_" << bindingName
        << "_detailed-documentation }" << endl;
    cout << endl;
    string desc = boost::replace_all_copy(doc.longDescription(),
                                          "|", "\\|");
    cout << desc << endl << endl;

    if (doc.example.size() > 0)
      cout << "### Example" << endl;
    for (size_t j = 0; j < doc.example.size(); ++j)
    {
      string eg = boost::replace_all_copy(doc.example[j](),
                                          "|", "\\|");
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

      // We need special handling of Doxygen information.
      if (doc.seeAlso[j].second.substr(0, 8) == "@doxygen")
      {
        cout << DOXYGEN_PREFIX << doc.seeAlso[j].second.substr(9);
      }
      else if (doc.seeAlso[j].second[0] == '#')
      {
        cout << "#" << languages[i] << "_"
            << doc.seeAlso[j].second.substr(1);
      }
      else
      {
        cout << doc.seeAlso[j].second;
      }

      cout << ")" << endl;
    }
    cout << endl;

    cout << "</div>" << endl;
    cout << endl;
  }

  IO::ClearSettings();
}

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

#include <boost/algorithm/string/replace.hpp>

#include "binding_info.hpp"
#include "print_docs.hpp"
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

void PrintHeaders(const string& bindingName,
                  const vector<string>& languages,
                  const vector<bool>& addWrapperDocs)
{
  // We just want to print the name of the function and a link, as Markdown.
  // We have to mark it as having the right language with a div.
  for (size_t i = 0; i < languages.size(); ++i)
  {
    BindingInfo::Language() = languages[i];

    // Get the name of the binding in the target language, and convert it to
    // lowercase (since the anchor link will be in lowercase).
    const string langBindingName = 
        addWrapperDocs[i] ? GetWrapperName(bindingName) :
            GetBindingName(bindingName);

    string anchorName = GetBindingName(bindingName);
    transform(anchorName.begin(), anchorName.end(), anchorName.begin(),
        [](unsigned char c) { return tolower(c); });
    // Strip '()' from the end if needed.
    if (anchorName.substr(anchorName.size() - 2, 2) == "()")
      anchorName = anchorName.substr(0, anchorName.size() - 2);

    cout << " - [" << langBindingName << "](#" << languages[i]
        << "_" << anchorName << "){: .language-link #" << languages[i] << " }"
        << endl;
  }
}

void PrintDocs(const string& bindingName,
               const vector<string>& languages,
               const vector<string>& validMethods,
               const vector<bool>& addWrapperDocs)
{
  // Here we have to handle both wrapper and non wrapper bindings.

  // 'params' stores the Params object for non-wrapper bindings.
  Params params = IO::Parameters(bindingName);
  const BindingDetails& doc = params.Doc();

  // 'paramMethods' stores the Params object for all wrapper bindings.
  map<string, Params> paramMethods;

  // adding elements to  'paramMethods'.
  for (size_t i=0; i<validMethods.size(); ++i)
  {
    Params methodParams = IO::Parameters(bindingName + "_" + validMethods[i]);
    paramMethods[validMethods[i]] = methodParams;
  }

  // First, for this section, print each of the names.
  for (size_t i = 0; i < languages.size(); ++i)
  {
    BindingInfo::Language() = languages[i];

    cout << "<div class=\"language-title\" id=\"" << languages[i]
        << "\" markdown=\"1\">" << endl;

    // here we have to check if docs are printed for wrappers or not.
    const string langBindingName = 
        addWrapperDocs[i] ? GetWrapperName(bindingName) :
            GetBindingName(bindingName);

    cout << "## " << langBindingName << endl;
    cout << "{: #" << languages[i] << "_" << bindingName << " }" << endl;
    cout << "</div>" << endl;
  }
  cout << endl;

  // Next, print the logical name of the binding (that's known by
  // ProgramInfo).
  cout << "#### " << doc.name << endl;
  cout << endl;

  for (size_t i = 0; i < languages.size(); ++i)
  {
    BindingInfo::Language() = languages[i];

    cout << "<div class=\"language-decl\" id=\"" << languages[i]
        << "\" markdown=\"1\">" << endl;
    // no need to print ProgramCall for wrappers (already included in example).
    if(!addWrapperDocs[i])
      cout << ProgramCall(bindingName);
    cout << "</div>" << endl;
  }
  cout << endl;

  for (size_t i = 0; i < languages.size(); ++i)
  {
    // for wrapper docs, there is a single long description, that is present
    // in a single _main.cpp file.
    if(addWrapperDocs[i])
    {
      string desc = boost::replace_all_copy(
          paramMethods[validMethods[0]].Doc().longDescription(), "|", "\\|");
      cout << desc << "\n";
    }
    else
    {
      // for non-wrappers we just print the short description.
      cout << doc.shortDescription << " ";
    }
    if(addWrapperDocs[i])
      cout << "[](#";
    else
      cout << "[Detailed documentation](#";

    cout  << languages[i] << "_"
      << bindingName << "_detailed-documentation){: .language-detail-link #"
      << languages[i] << " }";
    cout << "." << endl;
  }

  // Next, print the PROGRAM_INFO() documentation for each language.
  for (size_t i = 0; i < languages.size(); ++i)
  {
    if(!addWrapperDocs[i]) // different structure for wrappers.
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
      map<string, ParamData>& parameters = params.Parameters();
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
            cout << ParamString(bindingName, it->second.name) << " | ";
          }
          else
          {
            string name = ParamString(bindingName, it->second.name);
            name[1] = tolower(name[1]);
            cout << name << " | ";
          }
        }
        else
        {
          cout << ParamString(bindingName, it->second.name) << " | ";
        }
        cout << ParamType(params, it->second) << " | ";
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
        string def = PrintDefault(bindingName, it->second.name);
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
          string name = ParamString(bindingName, it->second.name);
          name[1] = tolower(name[1]);
          cout << name << " | ";
        }
        else
        {
          cout << ParamString(bindingName, it->second.name) << " | ";
        }
        cout << ParamType(params, it->second) << " | ";
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
    else // for wrappers.
    {
      BindingInfo::Language() = languages[i];
      cout << "<div class=\"language-section\" id=\"" << languages[i]
          << "\" markdown=\"1\">" << endl;
      cout << endl;

      cout << "### Parameters" << endl;
      cout << "| ***name*** | ***type*** | ***description*** | ***default*** |"
          << endl;
      cout << "|------------|------------|-------------------|---------------|"
          << endl;

      unordered_set<string> paramsSet; // to prevent duplicates.
      for(auto mainItr=paramMethods.begin(); mainItr!=paramMethods.end();
          ++mainItr)
      {
        map<string, ParamData> parameters = mainItr->second.Parameters();

        for (map<string, ParamData>::iterator it = parameters.begin();
            it != parameters.end(); ++it)
        {
          if(it->second.name == "help" || it->second.name == "version" ||
              it->second.name == "version" || it->second.name == "info")
            continue;

          bool isSerial;
          mainItr->second.functionMap[it->second.tname]["IsSerializable"](
              it->second, NULL, (void*)& isSerial);

          bool isHyperParam = false;
          size_t foundArma = it->second.cppType.find("arma");
          if(it->second.input && foundArma == string::npos &&
              !isSerial)
            isHyperParam = true;

          if(!isHyperParam) continue;

          if (paramsSet.find(it->second.name) != paramsSet.end()) continue;
          else paramsSet.insert(it->second.name);

          // Print name, type, description.
          cout << "| ";
          // We need special processing if the language is go then the output
          // parameter will be lowerCamelCase.
          if (languages[i] == "go")
          {
            string name = ParamString(bindingName + "_" + mainItr->first,
                it->second.name);
            name[1] = tolower(name[1]);
            cout << name << " | ";
          }
          else
          {
            cout << ParamString(bindingName + "_" + mainItr->first,
                it->second.name) << " | ";
          }
          cout << ParamType(params, it->second) << " | ";
          cout << it->second.desc;
          cout << " |";
          string def = PrintDefault(bindingName + "_" + mainItr->first,
              it->second.name);
          if (def.size() > 0)
            cout << "`" << def << "` |";
          else
            cout << " |";
          cout << endl;
        }
      }
      cout << endl;

      cout << "### Example" << endl;
      string example;
      for(auto itr=paramMethods.begin(); itr!=paramMethods.end();
          ++itr)
      {
        for(size_t j = 0; j < itr->second.Doc().example.size();
            ++j)
        {
          string eg = boost::replace_all_copy(itr->second.Doc().example[j](),
              "|", "\\|");
          example += eg + "\n";
        }
      }
      cout << "```" << languages[i] << "\n";
      cout << example << "\n";
      cout << "```" << "\n";

      cout << "### Methods" << endl;
      cout << "| **name** | **description** |" << endl;
      cout << "|----------|-----------------|" << endl;
      for(size_t i=0; i<validMethods.size(); i++)
      {
        cout << "| " << validMethods[i] << " | ";
        cout << paramMethods[validMethods[i]].Doc().shortDescription;
        cout << " |" << endl;
      }

      // Print information for each method.
      for(size_t i=0; i<validMethods.size(); i++)
      {
        cout << "### " << i+1 << ". " << validMethods[i] << "\n";
        cout << paramMethods[validMethods[i]].Doc().shortDescription << "\n";
        map<string, ParamData> parameters = paramMethods[validMethods[i]].Parameters();
        cout << "#### Input Parameters:" << "\n";
        cout << "| **name** | **type** | **description** |" << "\n";
        cout << "|----------|----------|-----------------|" << "\n";
        for (map<string, ParamData>::iterator it = parameters.begin();
            it != parameters.end(); ++it)
        {
          if(!it->second.input) continue;

          // skip if it is not arma type.
          size_t foundArma = it->second.cppType.find("arma");
          if(foundArma == string::npos) continue;

          // Print name, type, description.
          cout << "| ";
          // We need special processing if the language is go then the output
          // parameter will be lowerCamelCase.
          if (languages[i] == "go")
          {
            string name = ParamString(bindingName + "_" + validMethods[i],
                it->second.name);
            name[1] = tolower(name[1]);
            cout << name << " | ";
          }
          else
          {
            cout << ParamString(bindingName + "_" + validMethods[i],
                it->second.name) << " | ";
          }
          cout << ParamType(params, it->second) << " | ";
          cout << it->second.desc;
          cout << " |";
          cout << endl;
        }

        cout << endl;
        cout << "#### Returns: " << "\n";
        cout << "| **type** | **description** |" << "\n";
        cout << "|----------|-----------------|" << "\n";

        for (map<string, ParamData>::iterator it = parameters.begin();
            it != parameters.end(); ++it)
        {
          if(it->second.input) continue; // we are printing output options.

          bool isSerial;
          paramMethods[validMethods[i]].functionMap[it->second.tname]["IsSerializable"](
              it->second, NULL, (void*)& isSerial);

          // Print type, description.
          cout << "| ";
          cout << ParamType(params, it->second) << " | ";
          cout << it->second.desc;
          cout << " |";
          string def = PrintDefault(bindingName + "_" + validMethods[i],
              it->second.name);
          if (def.size() > 0)
            cout << "`" << def << "` |";
          else
            cout << " |";
          cout << endl;
        }
      }
    }
  }
}

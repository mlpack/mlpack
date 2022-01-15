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


void KMP(std::string pat,std::string txt,std::vector<int> &pos){
	int M = pat.size();int N = txt.size();int lps[M];
	int len = 0; lps[0] = 0; int i = 1;
	while (i < M) {
		if (pat[i] == pat[len]) {len++; lps[i] = len; i++;}
		else{
			if (len != 0) len = lps[len - 1];
			else lps[i]=0 , i++;
		}
	}
	i=0;int j=0;
	while (i < N) {
		if (pat[j] == txt[i]) j++,i++;
		if (j == M) { pos.push_back(i-j); j = lps[j - 1];}
		else if (i < N && pat[j] != txt[i]) {
			if (j != 0) j = lps[j - 1];
			else i = i + 1;
		}
	}
}

std::string replace_all_copy(std::string txt,std::string from,std::string to){
    std::vector<int>pos;
    KMP(from,txt,pos);
    int N=txt.size();int M=from.size();std::string new_txt;std::vector<int>::iterator itr=pos.begin();
    for(int i=0;i<N;i++){
        if(itr!=pos.end() && i==(*itr)){ new_txt+=to; i+=M; itr++;}
        new_txt+=txt[i];
    }
    return new_txt;
}

void PrintHeaders(const std::string& bindingName,
                  const std::vector<std::string>& languages)
{
  // We just want to print the name of the function and a link, as Markdown.
  // We have to mark it as having the right language with a div.
  for (size_t i = 0; i < languages.size(); ++i)
  {
    BindingInfo::Language() = languages[i];

    // Get the name of the binding in the target language, and convert it to
    // lowercase (since the anchor link will be in lowercase).
    const std::string langBindingName = GetBindingName(bindingName);
    std::string anchorName = langBindingName;
    std::transform(anchorName.begin(), anchorName.end(), anchorName.begin(),
        [](unsigned char c) { return std::tolower(c); });
    // Strip '()' from the end if needed.
    if (anchorName.substr(anchorName.size() - 2, 2) == "()")
      anchorName = anchorName.substr(0, anchorName.size() - 2);

    cout << " - [" << langBindingName << "](#" << languages[i]
        << "_" << anchorName << "){: .language-link #" << languages[i] << " }"
        << endl;
  }
}

void PrintDocs(const std::string& bindingName,
               const vector<string>& languages)
{
  Params params = IO::Parameters(bindingName);
  const BindingDetails& doc = params.Doc();

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
  cout << "#### " << doc.name << endl;
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
          std::string name = ParamString(bindingName, it->second.name);
          name[1] = std::tolower(name[1]);
          cout << name << " | ";
        }
      }
      else
      {
        cout << ParamString(bindingName, it->second.name) << " | ";
      }
      cout << ParamType(params, it->second) << " | ";
      string desc = replace_all_copy(it->second.desc, "|", "\\|");
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
        std::string name = ParamString(bindingName, it->second.name);
        name[1] = std::tolower(name[1]);
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
    string desc = replace_all_copy(doc.longDescription(),
                                          "|", "\\|");
    cout << desc << endl << endl;

    if (doc.example.size() > 0)
      cout << "### Example" << endl;
    for (size_t j = 0; j < doc.example.size(); ++j)
    {
      string eg = replace_all_copy(doc.example[j](),
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
}

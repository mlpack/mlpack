#include <mlpack/core/util/io.hpp>
#include <mlpack/core/util/binding_details.hpp>

#include "binding_info.hpp"
#include "print_param_table.hpp"
#include "print_doc_functions.hpp"

#include <boost/algorithm/string/replace.hpp>

using namespace std;
using namespace mlpack;
using namespace mlpack::util;
using namespace mlpack::bindings;
using namespace mlpack::bindings::markdown;

void PrintParamTable(const string& bindingName,
                     const string& language,
                     Params& params,
                     const set<string>& headers,
                     unordered_set<string>& paramsSet,
                     const bool onlyHyperParams,
                     const bool onlyMatrixParams,
                     const bool onlyInputParams,
                     const bool onlyOutputParams)
{
  map<string, ParamData>& parameters = params.Parameters();
  for (map<string, ParamData>::iterator it = parameters.begin();
      it != parameters.end(); ++it)
  {
    bool printCondition = true;
    if (onlyOutputParams) printCondition &= !it->second.input;
    if (onlyInputParams) printCondition &= it->second.input;
    if (onlyHyperParams)
    {
      bool isSerial;
      params.functionMap[it->second.tname]["IsSerializable"](
          it->second, NULL, (void*)& isSerial);

      bool isHyperParam = false;
      size_t foundArma = it->second.cppType.find("arma");
      if(it->second.input && foundArma == string::npos &&
          !isSerial)
        isHyperParam = true;
      printCondition &= isHyperParam;
    }
    if (onlyMatrixParams)
    {
      size_t foundArma = it->second.cppType.find("arma");
      printCondition &= (foundArma != string::npos);
    }

    if (!printCondition)
      continue;

    // There are some special options that don't exist in some languages.
    if (language != "python" && it->second.name == "copy_all_inputs")
      continue;
    if (language != "cli" &&
        (it->second.name == "help" || it->second.name == "info" ||
        it->second.name == "version"))
      continue;

    if (paramsSet.find(it->second.name) != paramsSet.end()) continue; // if already printed.
    paramsSet.insert(it->second.name); // insert if not already there.

    // Print name, type, description, default.
    cout << "| ";
    // We need special processing if the language is go then the required
    // parameter will be lowerCamelCase.
    if (headers.find("name") != headers.end())
    {
      if (language == "go")
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
    }

    if (headers.find("type") != headers.end())
      cout << ParamType(params, it->second) << " | ";

    if (headers.find("description") != headers.end())
    {
      string desc = boost::replace_all_copy(it->second.desc, "|", "\\|");
      cout << desc; // just a string
      // Print whether or not it's a "special" language-only parameter.
      if (it->second.name == "copy_all_inputs" || it->second.name == "help" ||
          it->second.name == "info" || it->second.name == "version")
      {
        cout << "  <span class=\"special\">Only exists in "
            << PrintLanguage(language) << " binding.</span>";
      }
      cout << " | ";
    }

    if (headers.find("default") != headers.end())
    {
      string def = PrintDefault(bindingName, it->second.name);
      if (def.size() > 0)
        cout << "`" << def << "` |";
      else
        cout << " |";
    }
    cout << endl;
  }
}

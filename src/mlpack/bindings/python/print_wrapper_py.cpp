/**
 * @file bindings/python/print_py_wrapper.cpp
 * @author Nippun Sharma
 *
 * Implementation of a function to generate a .py file that
 * contains the wrapper for a method.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "print_wrapper_py.hpp"
#include "get_methods_wrapper.hpp"
#include "get_class_name_wrapper.hpp"
#include "get_program_name.hpp"
#include "get_arma_type.hpp"
#include "get_valid_name.hpp"
#include <mlpack/core/util/io.hpp>

using namespace mlpack::util;
using namespace std;

namespace mlpack {
namespace bindings {
namespace python {

void PrintWrapperPY(const std::string& category,
                    const std::string& groupName,
                    const std::string& validMethods)
{
  map<string, Params> params;
  set<string> serializable; // all serializable parameters.
  map<string, bool> hyperParams; // if a parameter is hyperparameter.
  map<string, bool> isBool; // if a parameter is a bool.
  vector<string> methods = GetMethods(validMethods);
  int indent = 0; // keeps track of indentation at each point.

  for(auto i: methods)
  {
    params[i] = IO::Parameters(groupName + "_" + i);
  }

  if(category == "regression")
    cout << "class BaseEstimator:\n  pass\n\n";
  else if(category == "classification")
    cout << "class BaseEstimator:\n  pass\n\nclass ClassifierMixin:\n  pass\n\n";

  string importString = "";

  for(auto i: methods)
  {
    importString += "from mlpack." + groupName + "_" + i + " ";
		importString += "import " + groupName + "_" + i + "\n";
  }

  if(category == "regression")
  {
    importString += "import warnings\nwarnings.filterwarnings(\"default\"";
    importString += ", category=ImportWarning)\n";
    importString += "try:\n  from sklearn.base import BaseEstimator\nexcept:\n";
    importString += "  warnings.warn(\"scikit learn not present\", category";
    importString += "=ImportWarning)\n\n";
  }
  else if(category == "classification")
  {
    importString += "import warnings\nwarnings.filterwarnings(\"default\"";
    importString += ", category=ImportWarning)\n";
    importString += "try:\n  from sklearn.base import BaseEstimator, ";
    importString += "ClassifierMixin\nexcept:\n";
    importString += "  warnings.warn(\"scikit learn not present\", category";
    importString += "=ImportWarning)\n\n";
  }

  cout << importString << endl;

  for(auto i=params.begin(); i!=params.end(); i++)
  {
    map<string, ParamData> methodParams = i->second.Parameters();
    for(auto itr=methodParams.begin(); itr!=methodParams.end(); itr++)
    {
      bool isSerial;
      i->second.functionMap[itr->second.tname]["IsSerializable"](
          itr->second, NULL, (void*)& isSerial);
      if(isSerial)
      	serializable.insert(itr->second.cppType);

      bool isHyperParam = false;
      size_t foundArma = itr->second.cppType.find("arma");
      if(itr->second.input && foundArma == string::npos &&
          !isSerial)
        isHyperParam = true;
      hyperParams[itr->first] = isHyperParam;

      if(itr->second.cppType == "bool")
        isBool[itr->first] = true;
      else
        isBool[itr->first] = false;
    }
  }

  string className = GetClassName(groupName);

  // print class.
  if(category == "regression")
    cout << "class " << className << "(BaseEstimator)" << ":" << endl;
  else if(category == "classification")
  {
    cout << "class " << className << "(BaseEstimator, ClassifierMixin)" <<
        ":" << endl;
  }
  else
    cout << "class " << className << ":" << endl;

  indent += 2;
	cout << string(indent, ' ') << "def __init__(self," << endl;
  for(auto itr=hyperParams.begin(); itr!=hyperParams.end(); itr++)
  {
    // indent + 13, here 13 -> def __init__(
    if(itr->second && isBool[itr->first])
    {
      cout << string(indent+13, ' ') << GetValidName(itr->first)
          << " = False," << endl;
    }
    else if(itr->second && !isBool[itr->first])
    {
      cout << string(indent+13, ' ') << GetValidName(itr->first)
          << " = None," << endl;
    }
  }
  cout << string(indent+12, ' ') << "):" << endl;
  cout << endl;

  indent += 2;
  cout << string(indent, ' ') << "# serializable attributes." << endl;
  if(serializable.size() == 0)
    cout << string(indent, ' ') << "# None" << endl;

  for(auto itr=serializable.begin(); itr!=serializable.end(); itr++)
  {
    cout << string(indent, ' ');
    cout << "self._" << *itr << " = None" << endl;
  }
  cout << endl;
  cout << string(indent, ' ') << "# hyper-parameters." << endl;
  for(auto itr=hyperParams.begin(); itr!=hyperParams.end(); itr++)
  {
    if(itr->second)
    {
      string validName = GetValidName(itr->first);
      cout << string(indent, ' ');
      cout << "self." << validName << " = " << validName << endl;
    }
  }
  cout << endl;

  indent -= 2;

  // print all method definitions.
  for(auto methodName: methods)
  {
    cout << string(indent, ' ') << "def " << methodName << "(self," << endl;
    int addIndent = 4 + methodName.size() + 1;
    map<string, ParamData> methodParams = params[methodName].Parameters();
    vector<string> inputParamNames;
    map<string, string> mapToScikitNames; // "train" = ..., "labels" = ...
    map<string, string> invMapToScikitNames; // "X" = ..., "y" = ...
    int numMatrixInputs = 0;
    int numVectorInputs = 0;

    if(category == "regression" || category == "classification")
    {
      for(auto itr=methodParams.begin(); itr!=methodParams.end(); itr++)
      {
        if(numMatrixInputs > 1)
          Log::Fatal << "More than one matrix input parameters for " <<
              methodName << " method!" << endl;
        if(numVectorInputs > 1)
          Log::Fatal << "More than one vector input parameters for " <<
              methodName << " method!" << endl;

        if(itr->second.input)
        {
          if(itr->second.cppType == "arma::mat" ||
             itr->second.cppType ==
             "std::tuple<mlpack::data::DatasetInfo, arma::mat>" ||
             itr->second.cppType == "arma::Mat<size_t>")
          {
            numMatrixInputs++;
            mapToScikitNames[itr->first] = "X";
            invMapToScikitNames["X"] = itr->first;
          }
          else if(itr->second.cppType == "arma::vec" ||
                  itr->second.cppType == "arma::rowvec" ||
                  itr->second.cppType == "arma::Row<size_t>" ||
                  itr->second.cppType == "arma::Col<size_t>")
          {
            numVectorInputs++;
            mapToScikitNames[itr->first] = "y";
            invMapToScikitNames["y"] = itr->first;
          }
        }
      }
    }

    if(category == "classification" || category == "regression")
    {
      bool hasX = false;
      bool hasy = false;
      // print X, y in order.
      if(invMapToScikitNames.find("X") != invMapToScikitNames.end())
      {
        cout << string(indent + addIndent, ' ');
        cout << "X = None," << endl;
        hasX = true;
      }
      if(invMapToScikitNames.find("y") != invMapToScikitNames.end())
      {
        cout << string(indent + addIndent, ' ');
        cout << "y = None," << endl;
        hasy = true;
      }
      if(hasX)
      {
        cout << string(indent + addIndent, ' ');
        cout << GetValidName(invMapToScikitNames["X"]) << " = None," << endl;
      }
      if(hasy)
      {
        cout << string(indent + addIndent, ' ');
        cout << GetValidName(invMapToScikitNames["y"]) << " = None," << endl;
      }

      cout << string(indent + addIndent - 1, ' ');
      cout << "):" << endl;
      cout << endl;

      // implement a basic logic here to transfer values of X and y
      // to original names that can be used later on.
      string logicString = "";
      indent += 2;

      if(hasX)
      {
        string realName = GetValidName(invMapToScikitNames["X"]);
        logicString +=
        string(indent, ' ') + "if X is not None and " + realName +
            " is None:\n";
        logicString +=
        string(indent, ' ') + "  " + realName + " = X\n";
        logicString +=
        string(indent, ' ') + "elif X is not None and " + realName +
            " is not None:\n";
        logicString +=
        string(indent, ' ') + "  raise ValueError(\"" + realName +
            " and X both cannot be not None!!\")\n\n";
      }

      if(hasy)
      {
        string realName = GetValidName(invMapToScikitNames["y"]);
        logicString +=
        string(indent, ' ') + "if y is not None and " + realName +
            " is None:\n";
        logicString +=
        string(indent, ' ') + "  " + realName + " = y\n";
        logicString +=
        string(indent, ' ') + "elif y is not None and " + realName +
            " is not None:\n";
        logicString +=
        string(indent, ' ') + "  raise ValueError(\"" + realName +
            " and y both cannot be not None!!\")\n\n";
      }
      cout << logicString << endl;
    }
    else
    {
      // print input parameters.
      for(auto itr=methodParams.begin(); itr!=methodParams.end(); itr++)
      {
        string validName = GetValidName(itr->first);

        if(itr->second.input && serializable.find(itr->second.cppType) ==
            serializable.end() && !hyperParams[itr->first])
        {
          cout << string(indent + addIndent, ' ');
          cout << validName << " = None," << endl;
        }
      }
      cout << string(indent + addIndent - 1, ' ');
      cout << "):" << endl;
      cout << endl;
      indent += 2;
    }

    // print definition.
    cout << string(indent, ' ') << "out = " << groupName + "_" + methodName +
        "(";
    addIndent = 6 + groupName.size() + 1 + methodName.size() + 1;
    int count = 0; // just for reference.
    for(auto itr=methodParams.begin(); itr!=methodParams.end(); itr++)
    {
      if(itr->second.input)
      {
        string validName = GetValidName(itr->first);

        if(count != 0)
          cout << string(indent + addIndent, ' ');
        cout << validName << " = ";

        if(serializable.find(itr->second.cppType) != serializable.end())
          cout << "self._" << itr->second.cppType << "," << endl;
        else if(hyperParams[itr->first])
          cout << "self." << validName << "," << endl;
        else
          cout << validName << "," << endl;

        count++;
      }
    }
    cout << string(indent + addIndent - 1, ' ');
    cout << ")" << endl;
    cout << endl;

    // print output parameters.
    string returnString = string(indent, ' ') + "return ";
    bool outputsOnlySerial = true;
    for(auto itr=methodParams.begin(); itr!=methodParams.end(); itr++)
    {
      if(!itr->second.input)
      {
        if(serializable.find(itr->second.cppType) != serializable.end())
        {
          cout << string(indent, ' ');
          cout << "self._" << itr->second.cppType << " = out[\"" <<
              itr->first << "\"]" << endl;
        }
        else
        {
          outputsOnlySerial = false;
          returnString += "out[\"" + itr->first + "\"], ";
        }
      }
    }
    cout << endl;
    // return somethings.
    if(outputsOnlySerial)
      cout << returnString + "self" << endl;
    else
      cout << returnString.substr(0, returnString.size() - 2) << endl;

    cout << endl;
    indent -= 2;
  }
}

} // python
} // bindings
} // mlpack

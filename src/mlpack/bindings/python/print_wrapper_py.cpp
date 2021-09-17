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
#include "get_arma_type.hpp"
#include "wrapper_functions.hpp"
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

  // all serializable parameters.
  set<string> serializable;

  // if a parameter is hyperparameter.
  map<string, bool> hyperParams;

  // if a parameter is a bool.
  map<string, bool> isBool;

  vector<string> methods = GetMethods(validMethods);

  // keeps track of indentation at each point.
  int indent = 0;

  for(string i: methods)
  {
    params[i] = IO::Parameters(groupName + "_" + i);
  }

  if(category == "regression")
  {
    cout << "class BaseEstimator:" << endl;
    cout << "  pass" << endl;
    cout << endl;
  }
  else if(category == "classification")
  {
    cout << "class BaseEstimator:" << endl;
    cout << "  pass" << endl;
    cout << endl;
    cout << "class ClassifierMixin:" << endl;
    cout << "  pass" << endl;
    cout << endl;
  }

  // Import different mlpack programs that are to be wrapped.
  for(int i=0; i<methods.size(); i++)
  {
    cout << "from mlpack." << groupName << "_" << methods[i] << " ";
    cout << "import " << groupName << "_" << methods[i] << endl;
  }

  // Try importing scikit-learn, only for classification and
  // regression.
  if(category == "regression")
  {
    cout << "try:" << endl;
    cout << "  from sklearn.base import BaseEstimator" << endl;
  }
  else if(category == "classification")
  {
    cout << "try: " << endl;
    cout << "  from sklearn.base import BaseEstimator, ClassifierMixin";
    cout << endl;
  }

  if(category == "regression" || category == "classification")
  {
    cout << "except:" << endl;
    cout << "  pass" << endl;
    cout << endl;
  }

  // Check and store if every parameter is serializable,
  // a hyperparameter and boolean.
  for(map<string, Params>::iterator i=params.begin();
      i!=params.end(); i++)
  {
    map<string, ParamData> methodParams = i->second.Parameters();

    for(map<string, ParamData>::iterator itr=methodParams.begin();
        itr!=methodParams.end(); itr++)
    {
      // Checking for serializability.
      bool isSerial;
      i->second.functionMap[itr->second.tname]["IsSerializable"](
          itr->second, NULL, (void*)& isSerial);
      if(isSerial)
        serializable.insert(itr->second.cppType);

      // Checking for hyperparameter.
      bool isHyperParam = false;
      size_t foundArma = itr->second.cppType.find("arma");
      if(itr->second.input && foundArma == string::npos && !isSerial)
        isHyperParam = true;
      hyperParams[itr->first] = isHyperParam;

      // Checking for boolean.
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

  // print the __init__ method.
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

  // storing given arguments in attributes.
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
  for(string methodName: methods)
  {
    // print method name.
    cout << string(indent, ' ') << "def " << GetMappedName(methodName);
    cout << "(self, " << endl;
    int addIndent = 4 + GetMappedName(methodName).size() + 1;
    map<string, ParamData> methodParams = params[methodName].Parameters();

    vector<string> inputParamNames;

    // maps matrix parameters to X and vector parameters to y.
    map<string, string> mapToScikitNames;
    // maps X to matrix parameter name and y to vector parameter name.
    map<string, string> invMapToScikitNames;

    int numMatrixInputs = 0;
    int numVectorInputs = 0;

    if(category == "regression" || category == "classification")
    {
      for(map<string, ParamData>::iterator itr=methodParams.begin();
          itr!=methodParams.end(); itr++)
      {
        // Throw error if there are more than one matrix input params,
        // or more than one vector input params.
        if(numMatrixInputs > 1)
          Log::Fatal << "More than one matrix input parameters for " <<
              methodName << "(" << GetMappedName(methodName) << ") method!" <<
              endl;
        if(numVectorInputs > 1)
          Log::Fatal << "More than one vector input parameters for " <<
              methodName << "(" << GetMappedName(methodName) << ") method!" <<
              endl;

        if(itr->second.input)
        {
          // If this is a matrix parameter.
          if(itr->second.cppType == "arma::mat" ||
             itr->second.cppType ==
             "std::tuple<mlpack::data::DatasetInfo, arma::mat>" ||
             itr->second.cppType == "arma::Mat<size_t>")
          {
            numMatrixInputs++;
            mapToScikitNames[itr->first] = "X";
            invMapToScikitNames["X"] = itr->first;
          }
          // If this is a vector parameter.
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

    // Now we print the matrix and vector params.
    if(category == "classification" || category == "regression")
    {
      bool hasX = false;
      bool hasy = false;

      // First print X, if it is present.
      if(invMapToScikitNames.find("X") != invMapToScikitNames.end())
      {
        cout << string(indent + addIndent, ' ');
        cout << "X = None," << endl;
        hasX = true;
      }

      // Now print y, if it is present.
      if(invMapToScikitNames.find("y") != invMapToScikitNames.end())
      {
        cout << string(indent + addIndent, ' ');
        cout << "y = None," << endl;
        hasy = true;
      }

      // Now print actual names, to make it work for
      // both mapped and actual names.

      // Now print actual name for X.
      if(hasX)
      {
        cout << string(indent + addIndent, ' ');
        cout << GetValidName(invMapToScikitNames["X"]) << " = None," << endl;
      }

      // Now print actual name for y.
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
        
        cout << string(indent, ' ') << "if X is not None and ";
        cout << realName << " is None:" << endl;

        cout << string(indent, ' ') << "  " << realName << " = X" << endl;

        cout << string(indent, ' ') << "elif X is not None and " << realName;
        cout << " is not None:" << endl;

        cout << string(indent, ' ') << "  raise ValueError(\"" << realName;
        cout << " and X both cannot be not None!\")" << endl;
        cout << endl;
      }

      if(hasy)
      {
        string realName = GetValidName(invMapToScikitNames["y"]);

        cout << string(indent, ' ') << "if y is not None and ";
        cout << realName << " is None:" << endl;

        cout << string(indent, ' ') << "  " << realName << " = y" << endl;

        cout << string(indent, ' ') << "elif y is not None and " << realName;
        cout << " is not None:" << endl;

        cout << string(indent, ' ') << "  raise ValueError(\"" << realName;
        cout << " and y both cannot be not None!\")" << endl;
        cout << endl;
      }
    }
    else
    {
      // print input parameters.
      for(map<string, ParamData>::iterator itr=methodParams.begin();
          itr!=methodParams.end(); itr++)
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

    // first pass through the parameters and print all required parameters.
    for(map<string, ParamData>::iterator itr=methodParams.begin();
        itr!=methodParams.end(); itr++)
    {
      if(itr->second.input && itr->second.required)
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

    // Now print all non-required parameters.
    for(map<string, ParamData>::iterator itr=methodParams.begin();
        itr!=methodParams.end(); itr++)
    {
      if(itr->second.input && !itr->second.required)
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

    for(map<string, ParamData>::iterator itr=methodParams.begin();
        itr!=methodParams.end(); itr++)
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

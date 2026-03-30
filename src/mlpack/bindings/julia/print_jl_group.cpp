/**
 * @file bindings/julia/print_jl_group.cpp
 * @author Ryan Curtin
 *
 * Implementation of utility PrintJLGroup() function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "print_jl_group.hpp"
#include <mlpack/core/util/hyphenate_string.hpp>
#include <mlpack/bindings/util/strip_type.hpp>
#include <mlpack/bindings/util/validate_methods.hpp>
#include <mlpack/bindings/util/wrapper_utilities.hpp>

using namespace mlpack::util;
using namespace mlpack::bindings::util;
using namespace std;

using mlpack::util::ParamData;

namespace mlpack {
namespace bindings {
namespace julia {

/**
 * Print the code for a .jl binding for an mlpack program to stdout.
 */
void PrintJLGroup(const string& /* category */,
                  const string& groupName,
                  const string& validGroupMethods)
{
  map<string, Params> params;

  // Get the list of valid member methods that we need to print.
  vector<string> methods = GetMethods(validGroupMethods);

  // This contains the name of the _train binding.
  string trainMethod;
  // This contains the only serializable parameter.
  ParamData* modelType = nullptr;
  // This contains all of the hyperparameters from the _train binding.
  vector<ParamData*> hyperparams;

  // Process each method, extracting its parameters and populating the maps
  // above.
  ExtractGroupData(groupName, methods, params, trainMethod, modelType,
      hyperparams);

  // Check validity of methods:
  //   "train" should have two required parameters of matrix type, and output
  //   only one model parameter.
  //
  //   "predict"/"classify"/"probabilities" should have two required parameters,
  //   one of matrix type and one of model type, and output only one matrix
  //   parameter.
  //
  //   All methods should have only one output parameter.
  ValidateMethods(methods, params, "PrintJLGroup()");

  const string modelName = StripType(modelType->cppType);

  // Import the modules that actually contain each binding.
  for (size_t i = 0; i < methods.size(); ++i)
  {
    cout << "import ." << groupName << "_" << methods[i] << endl;
  }
  cout << endl;

  // Define the struct that holds the model.
  cout << "mutable struct " << modelName << endl;
  cout << "  # Hyperparameters for controlling model training behavior."
      << endl;
  for (size_t i = 0; i < hyperparams.size(); ++i)
  {
    // This will print definitions like:
    //   iterations::Int
    const ParamData& d = *hyperparams[i];
    cout << "  ";
    params[trainMethod].functionMap[d.tname]["PrintMemberDefn"]((ParamData&) d,
        NULL, NULL);
    cout << endl;
  }
  cout << endl;

  cout << "  # Holds the pointer to the model, which is passed to mlpack."
      << endl;
  cout << "  ptr::Ptr{Nothing}" << endl;
  cout << endl;

  // Now we need the user-facing constructor, which can accept constructors as
  // keyword arguments.
  cout << "  " << modelName << "( ;" << endl;
  for (size_t i = 0; i < hyperparams.size(); ++i)
  {
    // This will print definitions like:
    //    iterations::Union{Int,Missing}
    const ParamData& d = *hyperparams[i];
    cout << "    ";
    params[trainMethod].functionMap[d.tname]["PrintMemberDefn"]((ParamData&) d,
        NULL, NULL);
    cout << " = ";
    string defaultParam;
    params[trainMethod].functionMap[d.tname]["DefaultParam"]((ParamData&) d,
        NULL, (void*) &defaultParam);
    cout << defaultParam;
    if (i != hyperparams.size() - 1)
      cout << "," << endl;
    else
      cout << ") =" << endl;
  }
  cout << "    new(";
  for (size_t i = 0; i < hyperparams.size(); ++i)
    cout << hyperparams[i]->name << ", ";

  // We have to set ptr as the last element.
  cout << "Ptr{Nothing}())" << endl;

  // Close the struct definition.
  cout << "end" << endl << endl;

  // Now print functions to call each of the individual bindings.
  for (size_t i = 0; i < methods.size(); ++i)
  {
    /**
     * Instead of using the mlpack-specific binding function names, where
     * applicable we use more Julia-specific names.  These are the names used by
     * packages that implement the ScikitLearnInterface.jl package (and the
     * MLJModelInterface.jl package).
     */
    string useName = methods[i];
    if (useName == "train")
      useName = "fit!";
    else if (useName == "classify")
      useName = "predict";
    else if (useName == "probabilities")
      useName = "predict_proba";

    string indent(useName.size() + 1, ' ');
    cout << useName << "(in_model::" << modelName;

    // Print required input parameters.  (We already validated them earlier.)
    Params& ps = const_cast<Params&>(params.at(methods[i]));
    for (auto& it : ps.Parameters())
    {
      ParamData& d = it.second;
      if (d.required && d.input)
      {
        cout << "," << endl << indent;
        ps.functionMap[d.tname]["PrintMemberDefn"]((ParamData&) d, NULL, NULL);
      }
    }

    // Now print non-required input parameters.
    bool anyNonRequiredPrinted = false;
    for (auto& it : ps.Parameters())
    {
      ParamData& d = it.second;
      if (!d.required && d.input)
      {
        if (!anyNonRequiredPrinted)
        {
          // To split between required (non-keyword) and non-required (keyword)
          // arguments, we split the list with a ;.
          cout << ";" << endl << indent;
          anyNonRequiredPrinted = true;
        }
        else
        {
          cout << "," << endl << indent;
        }

        ps.functionMap[d.tname]["PrintMemberDefn"]((ParamData&) d,
            NULL, NULL);
        cout << "=";
        std::string defaultParam;
        ps.functionMap[d.tname]["DefaultParam"]((ParamData&) d,
            NULL, (void*) &defaultParam);
        cout << defaultParam;
      }
    }

    cout << ")" << endl;

    // Print the function call.
    cout << "  return " << groupName << "_" << methods[i] << "." << groupName
        << "_" << methods[i] << "(in_model.ptr";

    // Print required input parameters.
    for (auto& it : ps.Parameters())
    {
      ParamData& d = it.second;
      if (d.required && d.input)
        cout << ", " << d.name;
    }

    // Print non-required input parameters.
    for (auto& it : ps.Parameters())
    {
      ParamData& d = it.second;
      if (!d.required && d.input)
        cout << ", " << d.name << "=" << d.name;
    }

    cout << ")" << endl;
    cout << "end" << endl << endl;
  }
}

} // namespace julia
} // namespace bindings
} // namespace mlpack

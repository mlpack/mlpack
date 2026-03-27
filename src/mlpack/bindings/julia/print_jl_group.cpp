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
#include <mlpack/bindings/util/wrapper_utilities.hpp>

#include <set>

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

  const string modelName = StripType(modelType->cppType);

  // Import the modules that actually contain each binding.
  for (size_t i = 0; i < methods.size(); ++i)
  {
    cout << "import ." << methods[i] << endl;
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
  cout << "  " << modelType << "( ;" << endl;
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
  cout << "    new(" << endl;
  for (size_t i = 0; i < hyperparams.size(); ++i)
    cout << hyperparams[i]->name << ", ";

  // We have to set ptr as the last element.
  cout << "Ptr{Nothing}())" << endl;

  // Close the struct definition.
  cout << "end" << endl;
}

} // namespace julia
} // namespace bindings
} // namespace mlpack

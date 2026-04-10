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
  cout << "  function " << modelName << "( ;" << endl;
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
      cout << ")" << endl;
  }
  cout << "    res = new(";
  for (size_t i = 0; i < hyperparams.size(); ++i)
    cout << hyperparams[i]->name << ", ";

  // We have to set ptr as the last element.
  cout << "Ptr{Nothing}())" << endl;
  // Set the finalizer.
  cout << "    # Set the finalizer so that memory is freed when the object "
      << "is deleted." << endl;
  cout << "    finalizer(x -> _Internal." << groupName << "_" << trainMethod
      << "_internal.Delete" << modelName << "(x.ptr), res)" << endl;
  cout << "    return res" << endl;
  cout << "  end" << endl;

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

    string indent(useName.size() + 10, ' ');
    cout << "function " << useName << "(in_model::" << modelName;

    // Print required input parameters.  (We already validated them earlier.)
    Params& ps = const_cast<Params&>(params.at(methods[i]));
    for (auto& it : ps.Parameters())
    {
      ParamData& d = it.second;
      if (d.required && d.input)
      {
        bool serializable;
        ps.functionMap[d.tname]["IsSerializable"](d, NULL,
            (void*) &serializable);
        if (serializable)
          continue; // Don't print serializable parameters---we already did.

        cout << "," << endl << indent;
        ps.functionMap[d.tname]["PrintMemberDefn"]((ParamData&) d, NULL, NULL);
      }
    }

    // Now print non-required input parameters, but only for non-training
    // bindings.
    bool anyNonRequiredPrinted = false;
    if (methods[i] != "train")
    {
      for (auto& it : ps.Parameters())
      {
        ParamData& d = it.second;
        if (!d.required && d.input)
        {
          if (!anyNonRequiredPrinted)
          {
            // To split between required (non-keyword) and non-required
            // (keyword) arguments, we split the list with a ;.
            cout << ";" << endl << indent;
            anyNonRequiredPrinted = true;
          }
          else
          {
            cout << "," << endl << indent;
          }

          ps.functionMap[d.tname]["PrintMemberDefn"]((ParamData&) d,
              NULL, NULL);
          cout << " = ";
          std::string defaultParam;
          ps.functionMap[d.tname]["DefaultParam"]((ParamData&) d,
              NULL, (void*) &defaultParam);
          cout << defaultParam;
        }
      }
    }

    cout << ")" << endl;

    // Print the function call.
    if (methods[i] == "train")
    {
      cout << "  # Delete an existing model." << endl;
      cout << "  if in_model.ptr != C_NULL" << endl;
      cout << "    _Internal." << groupName << "_" << methods[i]
          << "_internal.Delete" << modelName << "(in_model.ptr)" << endl;
      cout << "  end" << endl;
      cout << endl;

      cout << "  in_model.ptr = _Internal." << groupName << "_" << methods[i]
          << "(";
    }
    else
    {
      cout << "  return _Internal." << groupName << "_" << methods[i] << "(";
    }

    // Print required input parameters.
    bool printedAny = false;
    for (auto& it : ps.Parameters())
    {
      ParamData& d = it.second;
      if (d.required && d.input)
      {
        if (printedAny)
          cout << ", ";

        bool serializable;
        ps.functionMap[d.tname]["IsSerializable"](d, NULL,
            (void*) &serializable);
        if (serializable)
          cout << "in_model.ptr";
        else
          cout << d.name;

        printedAny = true;
      }
    }

    // Print non-required input parameters.
    for (auto& it : ps.Parameters())
    {
      ParamData& d = it.second;
      if (!d.required && d.input)
      {
        // Don't print any hyperparameters.
        if (methods[i] == "train")
          continue;

        cout << ", " << d.name << "=" << d.name;
      }
    }
    cout << ")" << endl;
    // Make sure the function doesn't return anything if it's fit!().
    if (methods[i] == "train")
    {
      cout << endl << "  nothing" << endl;
    }

    cout << "end" << endl << endl;
  }

  // Now print serialization shims, like:
  //
  // function Serialization.serialize(s::Serialization.AbstractSerializer,
  //                                  model::LARS)
  //   Serialization.writetag(s.io, Serialization.OBJECT_TAG)
  //   Serialization.serialize(s, LARS)
  //   write(s.io, model.lambda1)
  //   ...
  //   _Internal.lars_train_internal.serializeLARS(s.io, model.ptr)
  // end
  cout << "function Serialization.serialize("
      << "s::Serialization.AbstractSerializer, model::" << modelName << ")"
      << endl;
  cout << "  Serialization.writetag(s.io, Serialization.OBJECT_TAG)" << endl;
  cout << "  Serialization.serialize(s, " << modelName << ")" << endl;
  for (ParamData* p : hyperparams)
    cout << "  write(s.io, model." << p->name << ")" << endl;

  cout << "  hasModel = (model.ptr != C_NULL)" << endl;
  cout << "  write(s.io, hasModel)" << endl;
  cout << "  if hasModel == true" << endl;
  cout << "    _Internal." << groupName << "_" << trainMethod << "_internal."
      << "serialize" << modelName << "(s.io, model.ptr)" << endl;
  cout << "  end" << endl;
  cout << "end" << endl << endl;

  // function Serialization.deserialize(s::Serialization.AbstractSerializer,
  //                                    ::Type{LARS})
  //   model = LARS()
  //   model.lambda1 = read(s.io, Float64)
  //   ...
  //   model.ptr = _Internal.lars_train_internal.deserializeLARS(s.io)
  //   return model
  // end
  cout << "function Serialization.deserialize("
      << "s::Serialization.AbstractSerializer, ::Type{" << modelName << "})"
      << endl;
  cout << "  model = " << modelName << "()" << endl;
  for (ParamData* p : hyperparams)
  {
    string juliaType;
    params.at(trainMethod).functionMap[p->tname]["GetPrintableType"](*p,
        NULL, (void*) &juliaType);
    cout << "  model." << p->name << " = read(s.io, " << juliaType << ")"
        << endl;
  }

  cout << "  hasModel = read(s.io, Bool)" << endl;
  cout << "  if hasModel == true" << endl;
  cout << "    model.ptr = _Internal." << groupName << "_" << trainMethod
      << "_internal.deserialize" << modelName << "(s.io)" << endl;
  cout << "  else" << endl;
  cout << "    model.ptr = Ptr{Nothing}(0)" << endl;
  cout << "  end" << endl;
  cout << "  return model" << endl;
  cout << "end" << endl;
}

} // namespace julia
} // namespace bindings
} // namespace mlpack

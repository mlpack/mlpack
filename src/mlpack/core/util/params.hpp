/**
 * @file params.hpp
 * @author Ryan Curtin
 *
 * The Params class stores parameter settings for an individual binding.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_UTIL_PARAMS_HPP
#define MLPACK_CORE_UTIL_PARAMS_HPP

//#include "forward.hpp"
#include "param_data.hpp"
#include "binding_details.hpp"
#include <map>

namespace mlpack {
namespace util {

/**
 * The Params class holds all information about the parameters passed to a
 * specific binding.
 */
class Params
{
 public:
  // Convenience typedef for function maps.
  using FunctionMapType = std::map<std::string, std::map<std::string,
      void (*)(ParamData&, const void*, void*)>>;

  /**
   * Create a new Params class.  In general this should only be called via
   * `IO::Parameters()`.
   */
  Params(const std::map<char, std::string>& aliases,
         const std::map<std::string, ParamData>& parameters,
         FunctionMapType& functionMap,
         const std::string& bindingName,
         const BindingDetails& doc);

  /**
   * Empty constructor. For wrapping in bindings.
   */
  Params();

  /**
   * Return `true` if the specified parameter was given.
   *
   * @param identifier The name of the parameter in question.
   */
  bool Has(const std::string& identifier) const;

  /**
   * Get the value of type T found for the parameter specified by `identifier`.
   * You can set the value using this reference safely.
   *
   * @param identifier The name of the parameter in question.
   */
  template<typename T>
  T& Get(const std::string& identifier);

  /**
   * Cast the given parameter of the given type to a short, printable
   * `std::string`, for use in status messages.  The message returned here
   * should be only a handful of characters, and certainly no longer than one
   * line.
   *
   * @param identifier The name of the parameter in question.
   */
  template<typename T>
  std::string GetPrintable(const std::string& identifier);

  /**
   * Get the raw value of the parameter before any processing that Get() might
   * normally do.  So, e.g., for command-line programs, this does not
   * perform any data loading or manipulation like Get() does.  So if you
   * want to access a matrix or model (or similar) parameter before it is
   * loaded, this is the method to use.
   *
   * @param identifier The name of the parameter in question.
   */
  template<typename T>
  T& GetRaw(const std::string& identifier);

  /**
   * Given two (matrix) parameters, ensure that the first is an in-place copy of
   * the second.  This will generally do nothing (as the bindings already do
   * this automatically), except for command-line bindings, where we need to
   * ensure that the output filename is the same as the input filename.
   *
   * @param outputParamName Name of output (matrix) parameter.
   * @param inputParamName Name of input (matrix) parameter.
   */
  // TODO: it would be really nice to remove this!  It's only used by MeanShift
  // and KMeans bindings.
  void MakeInPlaceCopy(const std::string& outputParamName,
                       const std::string& inputParamName);

  //! Get the map of parameters.
  std::map<std::string, ParamData>& Parameters() { return parameters; }
  //! Get the map of aliases.
  std::map<char, std::string>& Aliases() { return aliases; }

  //! Get the binding name.
  const std::string& BindingName() const { return bindingName; }

  //! Get the binding details.
  const BindingDetails& Doc() const { return doc; }

  /**
   * Set the particular parameter as passed.
   *
   * @param identifier The name of the parameter to set as passed.
   */
  void SetPassed(const std::string& identifier);

  /**
   * Check all input matrices for NaN and inf values, and throw an exception if
   * any are found.
   */
  void CheckInputMatrices();

 private:
  //! Convenience map from alias values to names.
  std::map<char, std::string> aliases;
  //! Map of parameters.
  std::map<std::string, ParamData> parameters;

 public:
  //! Map for functions and types.
  //! Note: this was originally created as a way to avoid virtual inheritance.
  //! However, the design would be much cleaner if we simply used virtual
  //! inheritance for different option types.
  FunctionMapType functionMap;

 private:
  //! Holds the name of the binding.
  std::string bindingName;

  //! Holds the BindingDetails object.
  BindingDetails doc;

  //! Utility function, used by CheckInputMatrices().
  template<typename T>
  void CheckInputMatrix(const T& matrix, const std::string& identifier);
};

} // namespace util
} // namespace mlpack

// Implementation intentionally not included.

#endif

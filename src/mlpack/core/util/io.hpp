/**
 * @file core/util/io.hpp
 * @author Matthew Amidon
 *
 * This file implements the IO subsystem which is a global singleton
 * intended to handle parameter passing for different binding types.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_UTIL_IO_HPP
#define MLPACK_CORE_UTIL_IO_HPP

#include <iostream>
#include <list>
#include <map>
#include <string>

#include <mlpack/prereqs.hpp>

#include "timers.hpp"
#include "binding_details.hpp"
#include "version.hpp"

#include "param_data.hpp"
#include "params.hpp"
#include "params_impl.hpp"

#include <mlpack/core/data/load.hpp>
#include <mlpack/core/data/save.hpp>

// TODO: this entire set of code is related to the bindings and maybe should go
// into src/mlpack/bindings/util/.
namespace mlpack {

// TODO: completely go through this documentation and clean it up
/**
 * @brief Parses the command line for parameters and holds user-specified
 *     parameters.
 *
 * The IO class is a subsystem by which parameters for machine learning methods
 * can be specified and accessed.  In conjunction with the macros PARAM_DOUBLE,
 * PARAM_INT, PARAM_STRING, PARAM_FLAG, and others, this class aims to make user
 * configurability of mlpack methods very easy.  There are only three methods in
 * IO that a user should need:  IO::ParseCommandLine(), IO::GetParam(), and
 * IO::HasParam() (in addition to the PARAM_*() macros).
 *
 * @section addparam Adding parameters to a program
 *
 * @code
 * $ ./executable --bar=5
 * @endcode
 *
 * @note The = is optional; a space can also be used.
 *
 * A parameter is specified by using one of the following macros (this is not a
 * complete list; see core/io/io.hpp):
 *
 *  - PARAM_FLAG(ID, DESC, ALIAS)
 *  - PARAM_DOUBLE(ID, DESC, ALIAS, DEF)
 *  - PARAM_INT(ID, DESC, ALIAS, DEF)
 *  - PARAM_STRING(ID, DESC, ALIAS, DEF)
 *
 * @param ID Name of the parameter.
 * @param DESC Short description of the parameter (one/two sentences).
 * @param ALIAS An alias for the parameter.
 * @param DEF Default value of the parameter.
 *
 * The flag (boolean) type automatically defaults to false; it is specified
 * merely as a flag on the command line (no '=true' is required).
 *
 * Here is an example of a few parameters being defined; this is for the KNN
 * binding (methods/neighbor_search/knn_main.cpp):
 *
 * @code
 * PARAM_STRING_REQ("reference_file", "File containing the reference dataset.",
 *     "r");
 * PARAM_STRING_REQ("distances_file", "File to output distances into.", "d");
 * PARAM_STRING_REQ("neighbors_file", "File to output neighbors into.", "n");
 * PARAM_INT_REQ("k", "Number of furthest neighbors to find.", "k");
 * PARAM_STRING("query_file", "File containing query points (optional).", "q",
 *     "");
 * PARAM_INT("leaf_size", "Leaf size for tree building.", "l", 20);
 * PARAM_FLAG("naive", "If true, O(n^2) naive mode is used for computation.",
 *     "N");
 * PARAM_FLAG("single_mode", "If true, single-tree search is used (as opposed "
 *     "to dual-tree search.", "s");
 * @endcode
 *
 * More documentation is available on the PARAM_*() macros in the documentation
 * for core/io/io.hpp.
 *
 * @section programinfo Documenting the program itself
 *
 * In addition to allowing documentation for each individual parameter and
 * module, the BINDING_NAME() macro provides support for documenting the
 * programName, BINDING_SHORT_DESC() macro provides support for documenting the
 * shortDescription, BINDING_LONG_DESC() macro provides support for documenting
 * the longDescription, the BINDING_EXAMPLE() macro provides support for
 * documenting the example and the BINDING_SEE_ALSO() macro provides support for
 * documenting the seeAlso. There should only be one instance of the
 * BINDING_NAME(), BINDING_SHORT_DESC() and BINDING_LONG_DESC() macros and there
 * can be multiple instance of BINDING_EXAMPLE() and BINDING_SEE_ALSO() macro.
 * Below is an example:
 *
 * @code
 * BINDING_NAME("Maximum Variance Unfolding");
 * BINDING_SHORT_DESC("An implementation of Maximum Variance Unfolding");
 * BINDING_LONG_DESC( "This program performs maximum "
 *    "variance unfolding on the given dataset, writing a lower-dimensional "
 *    "unfolded dataset to the given output file.");
 * BINDING_EXAMPLE("mvu", "input", "dataset", "new_dim", 5, "output", "output");
 * BINDING_SEE_ALSO("Perceptron", "#perceptron");
 * @endcode
 *
 * This description should be verbose, and explain to a non-expert user what the
 * program does and how to use it.  If relevant, paper citations should be
 * included.
 *
 * @section parseio Parsing the command line with IO
 *
 * To have IO parse the command line at the beginning of code execution, only a
 * call to ParseCommandLine() is necessary:
 *
 * @code
 * int main(int argc, char** argv)
 * {
 *   IO::ParseCommandLine(argc, argv);
 *
 *   ...
 * }
 * @endcode
 *
 * IO provides --help and --info options which give nicely formatted
 * documentation of each option; the documentation is generated from the DESC
 * arguments in the PARAM_*() macros.
 *
 * @section getparam Getting parameters with IO
 *
 * When the parameters have been defined, the next important thing is how to
 * access them.  For this, the HasParam() and GetParam() methods are
 * used.  For instance, to see if the user passed the flag (boolean) "naive":
 *
 * @code
 * if (IO::HasParam("naive"))
 * {
 *   Log::Info << "Naive has been passed!" << std::endl;
 * }
 * @endcode
 *
 * To get the value of a parameter, such as a string, use GetParam:
 *
 * @code
 * const std::string filename = IO::GetParam<std::string>("filename");
 * @endcode
 *
 * @note
 * Options should only be defined in files which define `main()` (that is, main
 * bindings).  If options are defined elsewhere, they may be spuriously
 * included into other bindings and confuse users.  Similarly, if your
 * binding has options which you did not define, it is probably because the
 * option is defined somewhere else and included in your binding.
 *
 * @bug
 * The __COUNTER__ variable is used in most cases to guarantee a unique global
 * identifier for options declared using the PARAM_*() macros.  However, not all
 * compilers have this support--most notably, gcc < 4.3.  In that case, the
 * __LINE__ macro is used as an attempt to get a unique global identifier, but
 * collisions are still possible, and they produce bizarre error messages. See
 * https://github.com/mlpack/mlpack/issues/100 for more information.
 */
class IO
{
 public:
  /**
   * Adds a parameter to the hierarchy; use the PARAM_*() macros instead of this
   * (i.e. PARAM_INT()).
   *
   * @param bindingName Name of the binding that this parameter is associated
   *      with.
   * @param d Utility structure holding parameter data.
   */
  static void AddParameter(const std::string& bindingName, util::ParamData&& d);

  /**
   * Add a function to the function map.
   *
   * @param type Type that this function should be called for.
   * @param name Name of the function.
   * @param func Function to call.
   */
  static void AddFunction(const std::string& type,
                          const std::string& name,
                          void (*func)(util::ParamData&, const void*, void*));

  /**
   * Add a user-friendly name for a binding.
   *
   * @param bindingName Name of the binding to add the user-friendly name for.
   * @param name User-friendly name.
   */
  static void AddBindingName(const std::string& bindingName,
                             const std::string& name);

  /**
   * Add a short description for a binding.
   *
   * @param bindingName Name of the binding to add the description for.
   * @param shortDescription Description to use.
   */
  static void AddShortDescription(const std::string& bindingName,
                                  const std::string& shortDescription);

  /**
   * Add a long description for a binding.
   *
   * @param bindingName Name of the binding to add the description for.
   * @param longDescription Function that returns the long description.
   */
  static void AddLongDescription(
      const std::string& bindingName,
      const std::function<std::string()>& longDescription);

  /**
   * Add an example for a binding.
   *
   * @param bindingName Name of the binding to add the example for.
   * @param example Function that returns the example.
   */
  static void AddExample(const std::string& bindingName,
                         const std::function<std::string()>& example);

  /**
   * Add a SeeAlso for a binding.
   *
   * @param bindingName Name of the binding to add the example for.
   * @param description Description of the SeeAlso.
   * @param link Link of the SeeAlso.
   */
  static void AddSeeAlso(const std::string& bindingName,
                         const std::string& description,
                         const std::string& link);

  /**
   * Return a new Params object initialized with all the parameters of the
   * binding `bindingName`.  This is intended to be called at the beginning of
   * the run of a binding.
   */
  static util::Params Parameters(const std::string& bindingName);

  /**
   * Retrieve the singleton.  As an end user, if you are just using the IO
   * object, you should not need to use this function---the other static
   * functions should be sufficient.
   *
   * In this case, the singleton is used to store data for the static methods,
   * as there is no point in defining static methods only to have users call
   * private instance methods.
   *
   * @return The singleton instance for use in the static methods.
   */
  static IO& GetSingleton();

  /**
   * Retrieve the global Timers object.
   */
  static util::Timers& GetTimers();

 private:
#ifndef MLPACK_NO_STD_MUTEX
  //! Ensure only one thread can call Add() at a time to modify the map.
  std::mutex mapMutex;
#endif
  //! Map from alias values to names, for each binding name.
  std::map<std::string, std::map<char, std::string>> aliases;
  //! Map of parameters, for each binding name.
  std::map<std::string, std::map<std::string, util::ParamData>> parameters;
  //! Map of functions.  Note that this is not specific to a binding, so we only
  //! have one.
  using FunctionMapType = std::map<std::string, std::map<std::string,
      void (*)(util::ParamData&, const void*, void*)>>;
  FunctionMapType functionMap;

#ifndef MLPACK_NO_STD_MUTEX
  //! Ensure only one thread can modify the docs map at a time.
  std::mutex docMutex;
#endif
  //! Map of binding details.
  std::map<std::string, util::BindingDetails> docs;

  //! Holds the timer objects.
  util::Timers timer;

  //! So that Timer::Start() and Timer::Stop() can access the timer variable.
  friend class Timer;

  /**
   * Make the constructor private, to preclude unauthorized instances.
   */
  IO();

  //! Private copy constructor; we don't want copies floating around.
  IO(const IO& other);
  //! Private copy operator; we don't want copies floating around.
  IO& operator=(const IO& other);
};

} // namespace mlpack

// This file must be included after IO is declared and fully defined.
#include "program_doc.hpp"

// Include the implementation.
#include "io_impl.hpp"

// Now include the implementation of the timers.
#include "timers_impl.hpp"

#endif

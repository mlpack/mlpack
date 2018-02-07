/**
 * @file cli.hpp
 * @author Matthew Amidon
 *
 * This file implements the CLI subsystem which is intended to replace FX.
 * This can be used more or less regardless of context.  In the future,
 * it might be expanded to include file I/O.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_UTIL_CLI_HPP
#define MLPACK_CORE_UTIL_CLI_HPP

#include <list>
#include <iostream>
#include <map>
#include <string>

#include <boost/any.hpp>

#include <mlpack/prereqs.hpp>

#include "timers.hpp"
#include "program_doc.hpp"
#include "version.hpp"

#include "param_data.hpp"

namespace mlpack {
namespace util {

// Externally defined in option.hpp, this class holds information about the
// program being run.
class ProgramDoc;

} // namespace util

/**
 * @brief Parses the command line for parameters and holds user-specified
 *     parameters.
 *
 * The CLI class is a subsystem by which parameters for machine learning methods
 * can be specified and accessed.  In conjunction with the macros PARAM_DOUBLE,
 * PARAM_INT, PARAM_STRING, PARAM_FLAG, and others, this class aims to make user
 * configurability of mlpack methods very easy.  There are only three methods in
 * CLI that a user should need:  CLI::ParseCommandLine(), CLI::GetParam(), and
 * CLI::HasParam() (in addition to the PARAM_*() macros).
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
 * complete list; see core/io/cli.hpp):
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
 * executable (methods/neighbor_search/knn_main.cpp):
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
 * for core/io/cli.hpp.
 *
 * @section programinfo Documenting the program itself
 *
 * In addition to allowing documentation for each individual parameter and
 * module, the PROGRAM_INFO() macro provides support for documenting the program
 * itself.  There should only be one instance of the PROGRAM_INFO() macro.
 * Below is an example:
 *
 * @code
 * PROGRAM_INFO("Maximum Variance Unfolding", "This program performs maximum "
 *    "variance unfolding on the given dataset, writing a lower-dimensional "
 *    "unfolded dataset to the given output file.");
 * @endcode
 *
 * This description should be verbose, and explain to a non-expert user what the
 * program does and how to use it.  If relevant, paper citations should be
 * included.
 *
 * @section parsecli Parsing the command line with CLI
 *
 * To have CLI parse the command line at the beginning of code execution, only a
 * call to ParseCommandLine() is necessary:
 *
 * @code
 * int main(int argc, char** argv)
 * {
 *   CLI::ParseCommandLine(argc, argv);
 *
 *   ...
 * }
 * @endcode
 *
 * CLI provides --help and --info options which give nicely formatted
 * documentation of each option; the documentation is generated from the DESC
 * arguments in the PARAM_*() macros.
 *
 * @section getparam Getting parameters with CLI
 *
 * When the parameters have been defined, the next important thing is how to
 * access them.  For this, the HasParam() and GetParam() methods are
 * used.  For instance, to see if the user passed the flag (boolean) "naive":
 *
 * @code
 * if (CLI::HasParam("naive"))
 * {
 *   Log::Info << "Naive has been passed!" << std::endl;
 * }
 * @endcode
 *
 * To get the value of a parameter, such as a string, use GetParam:
 *
 * @code
 * const std::string filename = CLI::GetParam<std::string>("filename");
 * @endcode
 *
 * @note
 * Options should only be defined in files which define `main()` (that is, main
 * executables).  If options are defined elsewhere, they may be spuriously
 * included into other executables and confuse users.  Similarly, if your
 * executable has options which you did not define, it is probably because the
 * option is defined somewhere else and included in your executable.
 *
 * @bug
 * The __COUNTER__ variable is used in most cases to guarantee a unique global
 * identifier for options declared using the PARAM_*() macros.  However, not all
 * compilers have this support--most notably, gcc < 4.3.  In that case, the
 * __LINE__ macro is used as an attempt to get a unique global identifier, but
 * collisions are still possible, and they produce bizarre error messages. See
 * https://github.com/mlpack/mlpack/issues/100 for more information.
 */
class CLI
{
 public:
  /**
   * Adds a parameter to the hierarchy; use the PARAM_*() macros instead of this
   * (i.e. PARAM_INT()).
   *
   * @param d Utility structure holding parameter data.
   */
  static void Add(util::ParamData&& d);

  /**
   * See if the specified flag was found while parsing.
   *
   * @param identifier The name of the parameter in question.
   */
  static bool HasParam(const std::string& identifier);

  /**
   * Get the value of type T found while parsing.  You can set the value using
   * this reference safely.
   *
   * @param identifier The name of the parameter in question.
   */
  template<typename T>
  static T& GetParam(const std::string& identifier);

  /**
   * Cast the given parameter of the given type to a short, printable
   * std::string, for use in status messages.  Ideally the message returned here
   * should be only a handful of characters, and certainly no longer than one
   * line.
   *
   * @param identifier The name of the parameter in question.
   */
  template<typename T>
  static std::string GetPrintableParam(const std::string& identifier);

  /**
   * Get the raw value of the parameter before any processing that GetParam()
   * might normally do.  So, e.g., for command-line programs, this does not
   * perform any data loading or manipulation like GetParam() does.  So if you
   * want to access a matrix or model (or similar) parameter before it is
   * loaded, this is the method to use.
   *
   * @param identifier The name of the parameter in question.
   */
  template<typename T>
  static T& GetRawParam(const std::string& identifier);

  /**
   * Retrieve the singleton.  As an end user, if you are just using the CLI
   * object, you should not need to use this function---the other static
   * functions should be sufficient.
   *
   * In this case, the singleton is used to store data for the static methods,
   * as there is no point in defining static methods only to have users call
   * private instance methods.
   *
   * @return The singleton instance for use in the static methods.
   */
  static CLI& GetSingleton();

  /**
   * Registers a ProgramDoc object, which contains documentation about the
   * program.  If this method has been called before (that is, if two
   * ProgramDocs are instantiated in the program), a fatal error will occur.
   *
   * @param doc Pointer to the ProgramDoc object.
   */
  static void RegisterProgramDoc(util::ProgramDoc* doc);

  //! Return a modifiable list of parameters that CLI knows about.
  static std::map<std::string, util::ParamData>& Parameters();
  //! Return a modifiable list of aliases that CLI knows about.
  static std::map<char, std::string>& Aliases();

  //! Get the program name as set by the PROGRAM_INFO() macro.
  static std::string ProgramName();

  /**
   * Mark a particular parameter as passed.
   *
   * @param name Name of the parameter.
   */
  static void SetPassed(const std::string& name);

  /**
   * Take all parameters and function mappings and store them, under the given
   * name.  This can later be restored with RestoreSettings().  If settings have
   * already been saved under the given name, they will be overwritten.  This
   * also clears the current parameters and function map.
   *
   * @param name Name of settings to save.
   */
  static void StoreSettings(const std::string& name);

  /**
   * Restore all of the parameters and function mappings of the given name, if
   * they exist.  A std::invalid_argument exception will be thrown if fatal is
   * true and no settings with the given name have been stored (with
   * StoreSettings()).
   *
   * @param name Name of settings to restore.
   * @param fatal Whether to throw an exception on an unknown name.
   */
  static void RestoreSettings(const std::string& name, const bool fatal = true);

  /**
   * Clear all of the settings, removing all parameters and function mappings.
   */
  static void ClearSettings();

 private:
  //! Convenience map from alias values to names.
  std::map<char, std::string> aliases;
  //! Map of parameters.
  std::map<std::string, util::ParamData> parameters;

 public:
  //! Map for functions and types.
  //! Use as functionMap["typename"]["functionName"].
  typedef std::map<std::string, std::map<std::string,
      void (*)(const util::ParamData&, const void*, void*)>> FunctionMapType;
  FunctionMapType functionMap;

 private:
  //! Storage map for parameters.
  std::map<std::string, std::tuple<std::map<std::string, util::ParamData>,
      std::map<char, std::string>, FunctionMapType>> storageMap;

 public:
  //! True, if CLI was used to parse command line options.
  bool didParse;

  //! Holds the name of the program for --version.  This is the true program
  //! name (argv[0]) not what is given in ProgramDoc.
  std::string programName;

  //! Holds the timer objects.
  Timers timer;

  //! So that Timer::Start() and Timer::Stop() can access the timer variable.
  friend class Timer;

  //! Pointer to the ProgramDoc object.
  util::ProgramDoc* doc;

 private:
  /**
   * Make the constructor private, to preclude unauthorized instances.
   */
  CLI();

  //! Private copy constructor; we don't want copies floating around.
  CLI(const CLI& other);
  //! Private copy operator; we don't want copies floating around.
  CLI& operator=(const CLI& other);
};

} // namespace mlpack

// Include the actual definitions of templated methods
#include "cli_impl.hpp"

#endif

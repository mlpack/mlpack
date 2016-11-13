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
#include <boost/program_options.hpp>

#include "timers.hpp"
#include "cli_deleter.hpp" // To make sure we can delete the singleton.
#include "version.hpp"
#include "param.hpp"

#include <mlpack/prereqs.hpp>

/**
 * The TYPENAME macro is used internally to convert a type into a string.
 */
#define TYPENAME(x) (std::string(typeid(x).name()))

namespace po = boost::program_options;

namespace mlpack {

namespace util {

// Externally defined in option.hpp, this class holds information about the
// program being run.
class ProgramDoc;

/**
 * Utility struct to return the type that boost::program_options should accept
 * for a given input type.  In general, there is no change from the input type.
 */
template<typename T>
struct ParameterType
{
  typedef T type;
};

/**
 * For matrix types, boost::program_options will accept a std::string, not an
 * arma::mat (since it is not clear how to specify a matrix on the
 * command-line).
 */
template<typename eT>
struct ParameterType<arma::Mat<eT>>
{
  typedef std::string type;
};

/**
 * This structure holds all of the information about a single parameter,
 * including its value (which is set when ParseCommandLine() is called).  It
 * does not hold any information about whether or not it was passed---that is
 * handled elsewhere.  A ParamData struct is only useful in order to get
 * "static" information about a parameter.
 */
struct ParamData
{
  //! Name of this parameter.  This is the name used for HasParam() and
  //! GetParam().
  std::string name;
  //! Description of this parameter, if any.
  std::string desc;
  //! Type information of this parameter.  Note that this is TYPENAME() of the
  //! user-visible parameter type, not whatever is given by ParameterType<>.
  std::string tname;
  //! Alias for this parameter.
  char alias;
  //! True if the wasPassed value should not be ignored.
  bool isFlag;
  //! True if this is a matrix that should not be transposed.  Ignored if the
  //! parameter is not a matrix.
  bool noTranspose;
  //! True if this option is required.
  bool required;
  //! True if this option is an input option (otherwise, it is output).
  bool input;
  //! If this is an input parameter that needs extra loading, this indicates
  //! whether or not it has been loaded.
  bool loaded;
  //! The actual value that is held, as passed from the user (so the type could
  //! be different than the type of the parameter).
  boost::any value;
  //! The value that the user interacts with, if the type is different than the
  //! type of the parameter.  This is used to store matrices, for instance,
  //! because 'value' must hold the string name that the user passed.
  boost::any mappedValue;
  //! The name of the parameter, as seen by boost::program_options.
  std::string boostName;
};

/**
 * If needed, map the parameter name to the name that is used by boost.  This
 * is generally the same as the name, but for matrices it may be different.
 */
template<typename T>
std::string MapParameterName(
    const std::string& identifier,
    const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0);

/**
 * If needed, map 'trueValue' to the right type and return it.  This is called
 * from GetParam().
 */
template<typename T>
T& HandleParameter(
    typename util::ParameterType<T>::type& value,
    util::ParamData& d,
    const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0);

//! This must be overloaded for matrices.
template<typename T>
std::string MapParameterName(
    const std::string& identifier,
    const typename boost::enable_if<arma::is_arma_type<T>>::type* = 0);

//! This must be overloaded for matrices.
template<typename T>
T& HandleParameter(
    typename util::ParameterType<T>::type& value,
    util::ParamData& d,
    const typename boost::enable_if<arma::is_arma_type<T>>::type* = 0);

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
   * @param identifier The name of the parameter.
   * @param description Short string description of the parameter.
   * @param alias An alias for the parameter, defaults to '\0' (no alias).
   * @param required Indicates if parameter must be set on command line.
   * @param input If true, the parameter is an input (not output) parameter.
   * @param noTranspose If the parameter is a matrix and this is true, then the
   *      matrix will not be transposed on loading.
   */
  template<class T>
  static void Add(const T& defaultValue,
                  const std::string& identifier,
                  const std::string& description,
                  const char alias = '\0',
                  const bool required = false,
                  const bool input = true,
                  const bool noTranspose = false);

  /**
   * Destroy the CLI object.  This resets the pointer to the singleton, so in
   * case someone tries to access it after destruction, a new one will be made
   * (the program will not fail).
   */
  static void Destroy();

  /**
   * Grab the value of type T found while parsing.  You can set the value using
   * this reference safely.
   *
   * @param identifier The name of the parameter in question.
   */
  template<typename T>
  static T& GetParam(const std::string& identifier);

  /**
   * Get the unmapped (i.e. what the user specifies on the command-line) value
   * of type ParameterType<T>::value found while parsing.  You cans et the value
   * using this reference safely.  You should not need to use this function
   * unless you are doing something tricky (like getting the filename a user
   * specified for a matrix parameter or something).
   *
   * @param identifier The name of the parameter in question.
   */
  template<typename T>
  static typename util::ParameterType<T>::type& GetUnmappedParam(
      const std::string& identifier);

  /**
   * Retrieve the singleton.
   *
   * Not exposed to the outside, so as to spare users some ungainly
   * x.GetSingleton().foo() syntax.
   *
   * In this case, the singleton is used to store data for the static methods,
   * as there is no point in defining static methods only to have users call
   * private instance methods.
   *
   * @return The singleton instance for use in the static methods.
   */
  static CLI& GetSingleton();

  /**
   * See if the specified flag was found while parsing.
   *
   * @param identifier The name of the parameter in question.
   */
  static bool HasParam(const std::string& identifier);

  /**
   * Hyphenate a string or split it onto multiple 80-character lines, with some
   * amount of padding on each line.  This is ued for option output.
   *
   * @param str String to hyphenate (splits are on ' ').
   * @param padding Amount of padding on the left for each new line.
   */
  static std::string HyphenateString(const std::string& str, int padding);

  /**
   * Parses the commandline for arguments.
   *
   * @param argc The number of arguments on the commandline.
   * @param argv The array of arguments as strings.
   */
  static void ParseCommandLine(int argc, char** argv);

  /**
   * Print out the help info for the given parameter (or all parameters if no
   * argument is specified).
   */
  static void PrintHelp(const std::string& param = "");

  /**
   * Registers a ProgramDoc object, which contains documentation about the
   * program.  If this method has been called before (that is, if two
   * ProgramDocs are instantiated in the program), a fatal error will occur.
   *
   * @param doc Pointer to the ProgramDoc object.
   */
  static void RegisterProgramDoc(util::ProgramDoc* doc);

  /**
   * Destructor.
   */
  ~CLI();

 private:
  //! The documentation and names of options.
  po::options_description desc;

  //! Values of the options given by user.
  po::variables_map vmap;

  //! Convenience list of output options.
  std::list<std::string> outputOptions;
  //! Convenience list of required options.
  std::list<std::string> requiredOptions;
  //! Convenience map from alias values to names.
  std::map<char, std::string> aliases;

  //! Map of parameters.
  std::map<std::string, util::ParamData> parameters;

  //! The singleton itself.
  static CLI* singleton;

  //! True, if CLI was used to parse command line options.
  bool didParse;

  //! Holds the name of the program for --version.  This is the true program
  //! name (argv[0]) not what is given in ProgramDoc.
  std::string programName;

  //! Holds the timer objects.
  Timers timer;

  //! So that Timer::Start() and Timer::Stop() can access the timer variable.
  friend class Timer;

 public:
  //! Pointer to the ProgramDoc object.
  util::ProgramDoc *doc;

 private:
  /**
   * Maps a given alias to a given parameter.
   *
   * @param alias The name of the alias to be mapped.
   * @param original The name of the parameter to be mapped.
   */
  static void AddAlias(const std::string& alias, const std::string& original);

  /**
   * Returns an alias, if given the name of the original.
   *
   * @param value The value in a key:value pair where the key
   * is an alias.
   * @return The alias associated with value.
   */
  static char AliasReverseLookup(const std::string& value);

  /**
   * Make the constructor private, to preclude unauthorized instances.
   */
  CLI();

  //! Private copy constructor; we don't want copies floating around.
  CLI(const CLI& other);

  //! Metaprogramming structure for vector detection.
  template<typename T>
  struct IsStdVector { const static bool value = false; };

  //! Metaprogramming structure for vector detection.
  template<typename T, typename A>
  struct IsStdVector<std::vector<T, A>> { const static bool value = true; };

  /**
   * Add an option if it is not a vector type.  This is a utility function used
   * by CLI::Add.
   *
   * @tparam Type of parameter.
   * @param optId Name of parameter.
   * @param descr Description.
   */
  template<typename T>
  void AddOption(
      const char* optId,
      const char* descr,
      const typename boost::disable_if<IsStdVector<T>>::type* /* junk */ = 0);

  /**
   * Add an option if it is a vector type.  This is a utility function used by
   * CLI::Add.
   *
   * @tparam Type of parameter.
   * @param optId Name of parameter.
   * @param descr Description.
   */
  template<typename T>
  void AddOption(
      const char* optId,
      const char* descr,
      const typename boost::enable_if<IsStdVector<T>>::type* /* junk */ = 0);
};

} // namespace mlpack

// Include the actual definitions of templated methods
#include "cli_impl.hpp"

#endif

/**
 * @file cli.hpp
 * @author Matthew Amidon
 *
 * This file implements the CLI subsystem which is intended to replace FX.
 * This can be used more or less regardless of context.  In the future,
 * it might be expanded to include file I/O.
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

} // namespace util

/**
 * Aids in the extensibility of CLI by focusing potential
 * changes into one structure.
 */
struct ParamData
{
  //! Name of this parameter.
  std::string name;
  //! Description of this parameter, if any.
  std::string desc;
  //! Type information of this parameter.
  std::string tname;
  //! The actual value of this parameter.
  boost::any value;
  //! True if this parameter was passed in via command line or file.
  bool wasPassed;
  //! True if the wasPassed value should not be ignored.
  bool isFlag;
};

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
   * (i.e. PARAM_INT()). Uses char* and not std::string since the vast majority
   * of use cases will be literal strings.
   *
   * @param identifier The name of the parameter.
   * @param description Short string description of the parameter.
   * @param alias An alias for the parameter, defaults to "" which is no alias.
   *    ("").
   * @param required Indicates if parameter must be set on command line.
   * @param input If true, the parameter is an input (not output) parameter.
   */
  static void Add(const std::string& path,
                  const std::string& description,
                  const std::string& alias = "",
                  const bool required = false,
                  const bool input = true);

  /**
   * Adds a parameter to the hierarchy; use the PARAM_*() macros instead of this
   * (i.e. PARAM_INT()). Uses char* and not std::string since the vast majority
   * of use cases will be literal strings.  If the argument requires a
   * parameter, you must specify a type.
   *
   * @param identifier The name of the parameter.
   * @param description Short string description of the parameter.
   * @param alias An alias for the parameter, defaults to "" which is no alias.
   * @param required Indicates if parameter must be set on command line.
   * @param input If true, the parameter is an input (not output) parameter.
   */
  template<class T>
  static void Add(const std::string& identifier,
                  const std::string& description,
                  const std::string& alias = "",
                  const bool required = false,
                  const bool input = true);

  /**
   * Adds a flag parameter to the hierarchy; use PARAM_FLAG() instead of this.
   *
   * @param identifier The name of the paramater.
   * @param description Short string description of the parameter.
   * @param alias An alias for the parameter, defaults to "" which is no alias.
   */
  static void AddFlag(const std::string& identifier,
                      const std::string& description,
                      const std::string& alias = "");

  /**
   * Parses the parameters for 'help' and 'info'.
   * If found, will print out the appropriate information and kill the program.
   */
  static void DefaultMessages();

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
   * Get the description of the specified node.
   *
   * @param identifier Name of the node in question.
   * @return Description of the node in question.
   */
  static std::string GetDescription(const std::string& identifier);

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
   * Removes duplicate flags.
   *
   * @param bpo The basic_program_options to remove duplicate flags from.
   */
  static void RemoveDuplicateFlags(po::basic_parsed_options<char>& bpo);

  /**
   * Print out the current hierarchy.
   */
  static void Print();

  /**
   * Print out the help info of the hierarchy.
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

  //! Identifier names of required options.
  std::list<std::string> requiredOptions;

  //! Pathnames of input options.
  std::list<std::string> inputOptions;
  //! Pathnames of output options.
  std::list<std::string> outputOptions;

  //! Map of global values.
  typedef std::map<std::string, ParamData> gmap_t;
  gmap_t globalValues;

  //! Map for aliases, from alias to actual name.
  typedef std::map<std::string, std::string> amap_t;
  amap_t aliasValues;

  //! The singleton itself.
  static CLI* singleton;

  //! True, if CLI was used to parse command line options.
  bool didParse;

  //! Hold the name of the program for --version.
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
  static std::string AliasReverseLookup(const std::string& value);

  /**
   * Checks that all required parameters have been specified on the command
   * line.  If any have not been specified, an error message is printed and the
   * program is terminated.
   */
  static void RequiredOptions();

  /**
   * Parses the values given on the command line, overriding any default values.
   */
  static void UpdateGmap();

  /**
   * Make the constructor private, to preclude unauthorized instances.
   */
  CLI();

  /**
   * Initialize desc with a particular name.
   *
   * @param optionsName Name of the module, as far as boost is concerned.
   */
  CLI(const std::string& optionsName);

  //! Private copy constructor; we don't want copies floating around.
  CLI(const CLI& other);
};

} // namespace mlpack

// Include the actual definitions of templated methods
#include "cli_impl.hpp"

#endif

/***
 * @file io.h
 * @author Matthew Amidon 
 * 
 * This file implements the IO subsystem which is intended to replace FX.
 * This can be used more or less regardless of context.  In the future, 
 * it might be expanded to include file I/O.
 */

#ifndef MLPACK_IO_IO_H
#define MLPACK_IO_IO_H

#include <boost/program_options.hpp>
#include <boost/any.hpp>
#include <iostream>
#include <map>
#include <string>
#include <list>

#include "optionshierarchy.h"
#include "printing.h"

/* These defines facilitate the registering of command line options.  Use the
 * macro which specifies the type of the option you want to add.
 *
 * @param ID Name of the parameter.
 * @param DESC Quick description of the parameter.
 * @param PARENT Parent module of the parameter.
 *
 * The parameter will then be specified with --PARENT/ID=value.
 */
#define PARAM_BOOL(ID, DESC, PARENT) PARAM(bool, ID, DESC, PARENT, false, false)
#define PARAM_INT(ID, DESC, PARENT) PARAM(int, ID, DESC, PARENT, 0, false)
#define PARAM_FLOAT(ID, DESC, PARENT) PARAM(float, ID, DESC, PARENT, 0.0f \
  , false)
#define PARAM_STRING(ID, DESC, PARENT) PARAM(std::string, ID, DESC, PARENT, \
    "", false)
#define PARAM_VECTOR(T, ID, DESC, PARENT) PARAM(std::vector<T>, ID, DESC, \
    PARENT, std::vector<T>(), false)

#define PARAM_BOOL_REQ(ID, DESC, PARENT) PARAM(bool, ID, DESC, PARENT, false, \
  true)
#define PARAM_INT_REQ(ID, DESC, PARENT) PARAM(int, ID, DESC, PARENT, 0, true)
#define PARAM_FLOAT_REQ(ID, DESC, PARENT) PARAM(float, ID, DESC, PARENT, 0.0f, \
  true)
#define PARAM_STRING_REQ(ID, DESC, PARENT) PARAM(std::string, ID, DESC, \
    PARENT, "", true);
#define PARAM_VECTOR_REQ(T, ID, DESC, PARENT) PARAM(std::vector<T>, ID, DESC, \
    PARENT, std::vector<T>(), true);

/// These are ugly, but necessary utility functions we must use to generate a
/// unique identifier inside of the PARAM() module.
#define JOIN(x, y) JOIN_AGAIN(x, y)
#define JOIN_AGAIN(x, y) x ## y

/***
 * Define an input parameter.  Don't use this function, use the other ones above
 * that call it.  Note that we are using the CURRENT_PARAM_NUM macro for naming
 * these actual parameters, which is a bit of an ugly hack... but this is the
 * preprocessor, after all.  We don't have much choice other than ugliness.
 * 
 * @param T type of parameter
 * @param ID name of parameter (for --help, pass "help")
 * @param DESC description of parameter (string)
 * @param PARENT name of parent module (string)
 * @param DEF default value of the parameter
 * @param REQ whether or not parameter is required (bool)
 */
#define PARAM(T, ID, DESC, PARENT, DEF, REQ) static mlpack::Option<T> \
    JOIN(io_option_dummy_object_, __COUNTER__) \
    (false, DEF, ID, DESC, PARENT, REQ);

#define PARAM_MODULE(ID, DESC) static mlpack::Option<int> \
    JOIN(io_option_module_dummy_object_, __COUNTER__) (true, 0, ID, DESC, NULL);

#define PARAM_COMPLEX_TYPE(T, ID, DESC, PARENT) static mlpack::Option<T> \
    JOIN(io_option_dummy_object_, __COUNTER__) (false, ID, DESC, PARENT, false);



namespace po = boost::program_options;

namespace mlpack {

class IO {
  public:
   /*
   * Adds a parameter to the heirarchy. Use char* and not 
   * std::string since the vast majority of use cases will 
   * be literal strings.
   * 
   * 
   *  @param identifier The name of the parameter.
   * @param description Short string description of the parameter.
   * @param parent Full pathname of a parent module, default is root node.
   * @param required Indicates if parameter must be set on command line.
   */
   static void Add(const char* identifier, 
                   const char* description, 
                   const char* parent=NULL, 
                   bool required = false);
    
   /*
   * Adds a parameter to the heirarchy. Use char* and not 
   * std::string since the vast majority of use cases will 
   * be literal strings.
   * If the argument requires a parameter, you must specify a type
   * 
   * @param identifier The name of the parameter.
   * @param description Short string description of the parameter.
   * @param parent Full pathname of a parent module, default is root node.
   * @param required Indicates if parameter must be set on command line.
   */
   template<class T>
   static void Add(const char* identifier, 
                   const char* description, 
                   const char* parent, 
                   bool required=false); 

   /*
   * Registers a complex type for use globally, 
   * without referencing the commandline 
   *
   * @param identifier The name of the parameter.
   * @param description Short string description of the parameter.
   * @param parent Full pathname of a parent module, default is root node.
   * @param required Indicates if parameter must be set on command line.
   */
   template<class T>
   static void AddComplexType(const char* identifier, 
                              const char* description, 
                              const char* parent);
    
   /* 
   * See if the specified flag was found while parsing. 
   * 
   * @param identifier The name of the parameter in question. 
   */
   static bool CheckValue(const char* identifier);
      
   /*
   * Grab the value of type T found while parsing.  
   * Should use checkValue first.
   *
   * @param argc The number of arguments on the commandline.
   * @param argv The array of arguments as strings 
   */
   template<typename T>
   static T& GetValue(const char* identifier); 
   
   /*
   * Grab the description of the specified node.
   * 
   * @param identifier Name of the node in question.
   * @return Description of the node in question. 
   */
   static std::string GetDescription(const char* identifier);

   /*
   * Parses the commandline for arguments.
   *
   * @param argc The number of arguments on the commandline.
   * @param argv The array of arguments as strings 
   */
   static void ParseCommandLine(int argc, char** argv);
    
   /*
   * Parses a stream for arguments
   *
   * @param stream The stream to be parsed.
   */
   static void ParseStream(std::istream& stream);
      
   /* Prints out the current heirachy */
   static void Print();

// We only use PrefixedOutStream if the program is compiled with debug symbols.
#ifdef DEBUG
//Prints debug output with the appropriate tag.
   static io::PrefixedOutStream Debug;
#else
//Dumps debug output into the bit nether regions.
   static io::NullOutStream Debug;
#endif
//Prints output with their respective tags of [INFO], [WARN], and [FATAL]
   static io::PrefixedOutStream Info;
   static io::PrefixedOutStream Warn;
   static io::PrefixedOutStream Fatal;

   /*
   * Initializes a timer, available like a normal value specified on 
   * the command line.  Timers are of type timval
   *
   * @param timerName The name of the timer in question.
   */
   static void StartTimer(const char* timerName);
     
   /* 
   * Halts the timer, and replaces it's value with 
   * the delta time from it's start 
   *
   * @param timerName The name of the timer in question.
   */
   static void StopTimer(const char* timerName);

   //Destructor
   ~IO();
  private:
   //The documentation and names of options
   po::options_description desc;
    
   //Values of the options given by user
   po::variables_map vmap;
    
   //Pathnames of required options
   std::list<std::string> requiredOptions;
    
   //Map of global values, stored here instead of in OptionsHierarchy
   //For ease of implementation
   std::map<std::string, boost::any> globalValues;
      
   //Store a relative index of path names
   io::OptionsHierarchy hierarchy;
    
   //Sanity checks strings before sending them to optionshierarchy
   //Returns the pathname placed in the hierarchy
   std::string ManageHierarchy(const char* id, 
                               const char* parent, 
                               std::string& tname, 
                               const char* description = "");
      
   /* Cleans up input pathnames, rendering strings such as /foo/bar
      and foo/bar equivalent inputs */
   static std::string SanitizeString(const char* str);
    
   //The singleton, obviously
   static IO* singleton;
    
   /* 
   * Not exposed to the outside, so as to spare users some ungainly
   * x.GetSingleton().foo() syntax.
   * In this case, the singleton is used to store data for the static methods, 
   * as there is no point in defining static methods only to have users call 
   * private instance methods 
   *
   * Returns the singleton instance for use in the static methods 
   */
   static IO& GetSingleton();
    
   /* Make the constructor private, to preclude unauthorized instances */
   IO();
   
   /* 
   * Initialize desc with a particular name.
   * 
   * @param optionsName Name of the module, as far as boost is concerned.
   */ 
   IO(std::string& optionsName);

   //Private copy constructor; don't want copies floating around.
   IO(const IO& other);
};


//Include the actual definitions of templated methods 
#include "io_impl.h"
};

#endif

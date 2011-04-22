#ifndef IO_H
#define IO_H

#include <boost/program_options.hpp>
#include <boost/any.hpp>
#include <iostream>
#include <map>
#include <string>
#include <list>
#include <boost/scoped_ptr.hpp>

#include "optionshierarchy.h"
#include "printing.h"

/* These defines facilitate the registering of command line options */
#define PARAM(T, ID, DESC, PARENT) static mlpack::Option<T> ID = \
    mlpack::Option<T>(false, #ID, DESC, #PARENT);
#define PARAM_BOOL(ID, DESC, PARENT) static mlpack::Option<int> ID = \
    mlpack::Option<int>(true, #ID, DESC, #PARENT);
#define PARAM_MODULE(ID, DESC) static mlpack::Option<int> ID = \
    mlpack::Option<int>(true, #ID, DESC, NULL);
#define PARAM_CUSTOM(T, ID, DESC) static mlpack::Option<T> ID = \
    mlpack::Option<T>(false, #ID, DESC, NULL);
#define PARAM_COMPLEX_TYPE(T, ID, DESC, PARENT) static mlpack::Option<T> ID = \
    mlpack::Option<T>(#ID, DESC, #PARENT);

#define TIMER(ID, DESC, PARENT) PARAM_COMPLEX_TYPE(timeval, ID, DESC, PARENT);
#define PARAM_STRING(ID, DESC, PARENT) PARAM(std::string, ID, DESC, PARENT);
#define PARAM_INT(ID, DESC, PARENT) PARAM(int, ID, DESC, PARENT);
#define PARAM_VECTOR(T, ID, DESC, PARENT) PARAM(std::vector<T>, ID, DESC, PARENT);

namespace po = boost::program_options;

namespace mlpack {
class IO {
  public:
   /* Adds a parameter to the heirarchy.  Incidentally, we are using 
      char* and not std::string since the vast majority of use cases will 
      be literal strings*/
   static void Add(const char* identifier, 
                   const char* description, 
                   const char* parent=NULL, 
                   bool required = false);
    
   /* If the argument requires a parameter, you must specify a type */
      
   template<class T>
   static void Add(const char* identifier, 
                   const char* description, 
                   const char* parent, 
                   bool required=false); 

   /*Registers a complex type for use globally, 
     without referencing the commandline */
   template<class T>
   static void AddComplexType(const char* identifier, 
                              const char* description, 
                              const char* parent);
    
   /* See if the specified flag was found while parsing.  
      Non-zero return value indicates success.*/
   static int CheckValue(const char* identifier);
      
   /* Grab the value of type T found while parsing.  
      Should use checkValue first.*/
   template<typename T>
   static T& GetValue(const char* identifier); 
      
   /* The proper commandline parse method */
   static void ParseCommandLine(int argc, char** argv);
    
   /* Parses a stream for options & values */
   static void ParseStream(std::istream& stream);
      
   /* Prints out the current heirachy */
   static void Print();
   static ostream& Out;  
   static ostream& DebugOut;

   static const char* endl;
   /* Prints a fatal error message */
   static void PrintFatal(const char* msg);
   /* Prints a notification */
   static void PrintNotify(const char* msg);
   /* Prints a warning */
   static void PrintWarn(const char* msg);
   /* Prints a message, only when in debug mode */
   static void PrintDebug(const char* msg);
   /* Prints all valid values that it can */
   static void PrintData();
     
   /* Initializes a timer, available like a normal value specified on 
      the command line.  Timers are of type timval*/
   static void StartTimer(const char* timerName);
     
   /* Halts the timer, and replaces it's value with 
      the delta time from it's start */
   static void StopTimer(const char* timerName);

   //Destructor
   ~IO();
  private:
   /* Private member variables & methods */
    
   /* Hierarchy, everything is in a global namespace.  
      That said, this namespace will require qualified names.
      These names will take the form of node/child/child2 */
      
    
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
    
      
   /* Not exposed to the outside, so as to spare users some ungainly
      x.GetSingleton().foo() syntax.
      In this case, the singleton is used to store data for the static methods, 
      as there is no point in defining static methods only to have users call 
      private instance methods */
    
   /* Returns the singleton instance for use in the static methods */
   static IO& GetSingleton();
    
   /* Make the constructor private, to preclude unauthorized instances */
   IO();
   //Initialize desc with a particular name
   IO(std::string& optionsName);
   IO(const IO& other);
};


//Include the actual definitions of templated methods 
#include "io_impl.h"
};

#endif

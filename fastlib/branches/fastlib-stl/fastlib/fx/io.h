#ifndef IO_H
#define IO_H

#include <boost/program_options.hpp>
#include <boost/any.hpp>
#include <iostream>
#include <map>
#include <string>
#include <list>

#include "optionshierarchy.h"
#include "printing.h"

/* These defines facilitate the registering of command line options */
#define PARAM(T, ID, DESC, PARENT) static mlpack::Option<T> ID = mlpack::Option<T>(false, #ID, DESC, #PARENT);
#define PARAM_BOOL(ID, DESC, PARENT) static mlpack::Option<int> ID = mlpack::Option<int>(true, #ID, DESC, #PARENT);
#define PARAM_MODULE(ID, DESC) static mlpack::Option<int> ID = mlpack::Option<int>(true, #ID, DESC, NULL);
#define PARAM_CUSTOM(T, ID, DESC) static mlpack::Option<T> ID = mlpack::Option<T>(false, #ID, DESC, NULL);

#define PARAM_TIMER(ID, DESC, PARENT) PARAM(timeval, ID, DESC, PARENT)
#define PARAM_STRING(ID, DESC, PARENT) PARAM(std::string, ID, DESC, PARENT)
#define PARAM_INT(ID, DESC, PARENT) PARAM(int, ID, DESC, PARENT)
#define PARAM_VECTOR(T, ID, DESC, PARENT) PARAM(std::vector<T>, ID, DESC, PARENT)

namespace po = boost::program_options;

namespace mlpack {
  class IO {
    public:
      /* Adds a parameter to the heirarchy.  Incidentally, we are using 
        char* and not std::string since the vast majority of use cases will 
        be literal strings*/
      static void add(const char* identifier, const char* description, 
                                  const char* parent=NULL, bool required = false);
    
      static void add(const char* identifier, const char* description, 
                                  const char* parent, bool required, bool out);
    
      /* If the argument requires a parameter, you must specify a type */
      template<class T>
      static void add(const char* identifier, const char* description, 
                                  const char* parent = NULL, bool required = false) {
        add<T>(identifier, description, parent, required, false);
      }
      
      template<class T>
      static void add(const char* identifier, const char* description, 
                                  const char* parent, bool required, bool out) {
        //Use singleton for state, wrap this up a parallel data structure 
        po::options_description& desc = getSingleton().desc;
        std::string tmp = TYPENAME(T);
                                    
        //Generate the full path string, and place the node in the hierarchy
        std::string path = getSingleton().manageHierarchy(identifier, parent, tmp, description);
                                    
        //Add the option to boost program_options
        desc.add_options()
          (path.c_str(), po::value<T>(), description);
        
        //If the option is required, add it to the required options list
        if(required)
          getSingleton().requiredOptions.push_front(path);
        return;
      }

    
      /* See if the specified flag was found while parsing.  Non-zero return value indicates success.*/
      static int checkValue(const char* identifier);
      
      /* Grab the value of type T found while parsing.  Should use checkValue first.*/
      template<typename T>
      static T& getValue(const char* identifier) {
        //Used to ensure we have a valid value
        T tmp;
        //Used to index into the globalValues map
        std::string key = std::string(identifier);
        std::map<std::string, boost::any>& gmap = getSingleton().globalValues;
        
        if(checkValue(identifier) && !gmap.count(key)) //If we have the option, set it's value
          gmap[key] = boost::any(getSingleton().vmap[identifier].as<T>());
      
        //We may have whatever is on the commandline, but what if
        //The programmer has made modifications?
        if(!gmap.count(key)) //The programmer hasn't done anything, lets register it then
          gmap[key] = boost::any(tmp);
        
        
        return *boost::any_cast<T>(&gmap[key]);
      }
      
      /* The proper commandline parse method */
      static void parseCommandLine(int argc, char** argv);
    
      /* Parses a stream for options & values */
      static void parseStream(std::istream& stream);
      
      /* Prints out the current heirachy */
      static void print();
      
      /* Prints a fatal error message */
      static void printFatal(const char* msg);
      /* Prints a notification */
      static void printNotify(const char* msg);
      /* Prints a warning */
      static void printWarn(const char* msg);
      /* Prints all valid values that it can */
      static void printData();
      
      /* Initializes a timer, available like a normal value specified on the command line.  
          Timers are of type timval, as defined in sys/time.h*/
      static void startTimer(const char* timerName);
      
      /* Halts the timer, and replaces it's value with the delta time from it's start */
      static void stopTimer(const char* timerName);
      
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
      OptionsHierarchy hierarchy;
    
      //Sanity checks strings before sending them to optionshierarchy
      //Returns the pathname placed in the hierarchy
      std::string manageHierarchy(const char* id, const char* parent, std::string& tname, const char* description = "");
      
      //Cleans up input pathnames, rendering strings such as /foo/bar and foo/bar equivalent inputs
      static std::string sanitizeString(const char* str);
    
      //The singleton, obviously
      static IO* singleton;
    
      
      /* Not exposed to the outside, so as to spare users some ungainly
        x.getSingleton().foo() syntax 
        In this case, the singleton is used to store data for the static methods, 
        as there is no point in defining static methods only to have users call 
        private instance methods */
    
      /* Returns the singleton instance for use in the static methods */
      static IO& getSingleton();
    
      /* Make the constructor private, to preclude unauthorized instances */
      IO();
      //Initialize desc with a particular name
      IO(std::string& optionsName);
      IO(const IO& other);
      ~IO();			
  };
  
    /*
  This class is used to facilitate easy addition of options to the program. 
  */
  template<typename N>
  class Option {
    //Base add an option
   public:
    Option(bool ignoreTemplate, const char* identifier, const char* description, 
      const char* parent=NULL, bool required=false) {
        if(ignoreTemplate)
          IO::add(identifier, description, parent, required);
        else
          IO::add<N>(identifier, description, parent, required);
      }
  };
};

#endif

#include "io.h"

#include <list>
#include <boost/program_options.hpp>
#include <boost/any.hpp>
#include <boost/scoped_ptr.hpp>
#include <iostream>
#include <string>
#include <sys/time.h>
#include <execinfo.h>

#include "printing.h"
#include "nulloutstream.h"

#define BASH_RED "\033[0;31m"
#define BASH_GREEN "\033[0;32m"
#define BASH_YELLOW "\033[0;33m"
#define BASH_CYAN "\033[0;36m"
#define BASH_CLEAR "\033[0m"

using namespace mlpack;
using namespace mlpack::io;

IO* IO::singleton = NULL;

#ifdef DEBUG
PrefixedOutStream IO::Debug = PrefixedOutStream(std::cout, 
    BASH_CYAN "[DEBUG] " BASH_CLEAR);
#else
NullOutStream IO::Debug = NullOutStream();
#endif
PrefixedOutStream IO::Info = PrefixedOutStream(std::cout,
    BASH_GREEN "[INFO ] " BASH_CLEAR);
PrefixedOutStream IO::Warn = PrefixedOutStream(std::cout,
    BASH_YELLOW "[WARN ] " BASH_CLEAR);
PrefixedOutStream IO::Fatal = PrefixedOutStream(std::cerr,
    BASH_RED "[FATAL] " BASH_CLEAR);

/* For clarity, we will alias boost's namespace */
namespace po = boost::program_options;


/* Constructors, Destructors, Copy */
/* Make the constructor private, to preclude unauthorized instances */
IO::IO() : desc("Allowed Options") , hierarchy("Allowed Options") {
  
  return;
}

 /* 
   * Initialize desc with a particular name.
   * 
   * @param optionsName Name of the module, as far as boost is concerned.
   */ 
IO::IO(std::string& optionsName) : 
    desc(optionsName.c_str()), hierarchy(optionsName.c_str()) {
  return;
}

//Private copy constructor; don't want copies floating around.
IO::IO(const IO& other) : desc(other.desc){
  return;
}

IO::~IO() {
  return;
}

/* Methods */

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
void IO::Add(const char* identifier, 
             const char* description, 
             const char* parent, 
             bool required) {

  po::options_description& desc = IO::GetSingleton().desc;

  //Generate the full pathname and insert the node into the hierarchy
  std::string tmp = TYPENAME(bool);
  std::string path = 
    IO::GetSingleton().ManageHierarchy(identifier, parent, tmp, description);

  //Add the option to boost program_options
  desc.add_options()
    (path.c_str(), description);

  //If the option is required, add it to the required options list
  if (required)
    GetSingleton().requiredOptions.push_front(path);
  
  return;
}

 /* 
   * See if the specified flag was found while parsing. 
   * 
   * @param identifier The name of the parameter in question. 
   */
bool IO::CheckValue(const char* identifier) {
  std::string key = std::string(identifier);

  int isInVmap = GetSingleton().vmap.count(key);
  int isInGmap = GetSingleton().globalValues.count(key);

  //Return true if we have a defined value for identifier
  return (isInVmap || isInGmap); 
}

/*
   * Grab the description of the specified node.
   * 
   * @param identifier Name of the node in question.
   * @return Description of the node in question. 
   */
std::string IO::GetDescription(const char* identifier) {
  std::string tmp = std::string(identifier);
  OptionsHierarchy* h = GetSingleton().hierarchy.FindNode(tmp);
  
  if (h == NULL)
    return std::string("");

  OptionsData d = h->GetNodeData();
  return d.desc;
}

//Returns the sole instance of this class
IO& IO::GetSingleton() {
  if (singleton == NULL) {
    singleton = new IO();
    Add<std::string>("info", "get help info on a specific module", NULL, false);
  }
  return *singleton;
}	

/* 
 * Properly formats strings such that there aren't too few or too many '/'s.
 * 
 * @param id The name of the parameter, eg bar in foo/bar.
 * @param parent The full name of the parameter's parent, 
 *   eg foo/bar in foo/bar/buzz.
 * @param tname String identifier of the parameter's type.
 * @param description String description of the parameter.
 */
std::string IO::ManageHierarchy(const char* id, 
                                const char* parent, 
                                std::string& tname, 
                                const char* description) {

  std::string path(id), desc(description);
  
  path = SanitizeString(parent)+id;
  
  //Add the sanity checked string to the hierarchy
  if (desc.length() == 0)
    hierarchy.AppendNode(path, tname);
  else
    hierarchy.AppendNode(path, tname, desc);
  return path;
}

/*
   * Parses the commandline for arguments.
   *
   * @param argc The number of arguments on the commandline.
   * @param argv The array of arguments as strings 
   */
void IO::ParseCommandLine(int argc, char** line) {
  po::variables_map& vmap = GetSingleton().vmap;
  po::options_description& desc = GetSingleton().desc;
  std::list<std::string> rOpt = GetSingleton().requiredOptions;
  std::map<std::string, boost::any> gmap = GetSingleton().globalValues;
  //Parse the command line, place the options & values into vmap
  try{ 
    po::store(po::parse_command_line(argc, line, desc), vmap);
  }catch(std::exception& ex) {
    IO::Fatal << ex.what() << std::endl;
  }
  //Flush the buffer, make sure changes are propogated to vmap
  po::notify(vmap);	
 
  //Iterate through Gmap, and overwrite default values with anything found on 
  //command line. 
  std::map<std::string, boost::any>::iterator i;
  for (i = gmap.begin(); i != gmap.end(); i++) {
    po::variable_value tmp = vmap[i->first];
    if (!tmp.empty()) //We need to overwrite gmap.
      gmap[i->first] = tmp.value();
  }
  
  //Default help message
  if (CheckValue("help")) {
    GetSingleton().hierarchy.PrintAllHelp(); 
    exit(0);  //The user doesn't want to run the program, he wants help. 
  }
  else if (CheckValue("info")) {
    std::string str = GetValue<std::string>("info");
    //the info node should always be there.
    GetSingleton().hierarchy.FindNode(str)->PrintNodeHelp();
    exit(0);
  }

  //Now, warn the user if they missed any required options
  std::list<std::string>::iterator iter;
  for (iter = rOpt.begin(); iter != rOpt.end(); iter++)
    if (!CheckValue((*iter).c_str()))// If a required option isn't there...
      IO::Fatal << "Required option --" << iter->c_str() << " is undefined..."
          << std::endl;
}

 /*
   * Parses a stream for arguments
   *
   * @param stream The stream to be parsed.
   */
void IO::ParseStream(std::istream& stream) {
  IO::Debug << "Compiled with debug checks." << std::endl;

  po::variables_map& vmap = GetSingleton().vmap;
  po::options_description& desc = GetSingleton().desc;
  std::list<std::string> rOpt = GetSingleton().requiredOptions;
  
  //Parse the stream, place options & values into vmap
  try{
  po::store(po::parse_config_file(stream, desc), vmap);
  }catch(std::exception& ex) {
    IO::Fatal << ex.what() << std::endl;
  }
  //Flush the buffer, make s ure changes are propgated to vmap
  po::notify(vmap);
  
  //Now, warn the user if they missed any required options
  std::list<std::string>::iterator iter;
  for (iter = rOpt.begin(); iter != rOpt.end(); iter++)
    if (!CheckValue((*iter).c_str())) //If a required option isn't there...
      IO::Fatal << "Required option --" << iter->c_str() << " is undefined..."
          << std::endl;
}

/* Prints out the current heirachy */
void IO::Print() {
  IO::GetSingleton().hierarchy.PrintAll();
}

/* Cleans up input pathnames, rendering strings such as /foo/bar
      and foo/bar/ equivalent inputs */
std::string IO::SanitizeString(const char* str) {
  if (str != NULL) {
    std::string p(str);
    //Lets sanity check string, remove superfluous '/' prefixes
    if (p.find_first_of("/") == 0)
      p = p.substr(1,p.length()-1);
    //Add necessary '/' suffixes to parent
    if (p.find_last_of("/") != p.length()-1)
      p = p+"/";
    return p;
  }

  return std::string("");
}

/*
 * Initializes a timer, available like a normal value specified on 
 * the command line.  Timers are of type timval
 *
 * @param timerName The name of the timer in question.
 */
void IO::StartTimer(const char* timerName) {
  //Don't want to actually document the timer, the user can do that if he wants
  timeval tmp;
  
  tmp.tv_sec = 0;
  tmp.tv_usec = 0;
  
  gettimeofday(&tmp, NULL);
  GetValue<timeval>(timerName) = tmp;
}
      
 /* 
   * Halts the timer, and replaces it's value with 
   * the delta time from it's start 
   *
   * @param timerName The name of the timer in question.
   */
void IO::StopTimer(const char* timerName) {
  timeval delta, b, &a = GetValue<timeval>(timerName);  
  gettimeofday(&b, NULL);
  
  //Calculate the delta time
  timersub(&b, &a, &delta);
  a = delta; 
}


PARAM_MODULE("help", "default help info");

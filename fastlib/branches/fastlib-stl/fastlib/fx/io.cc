#include "io.h"

#include <list>
#include <boost/program_options.hpp>
#include <boost/any.hpp>
#include <iostream>
#include <string>
#include <sys/time.h>

#define BASH_RED "\033[0;31m"
#define BASH_GREEN "\033[0;32m"
#define BASH_YELLOW "\033[1:33m"
#define BASH_CLEAR "\033[0m"

using namespace mlpack;

/* For clarity, we will alias boost's namespace */
namespace po = boost::program_options;

/* Declare the singleton variable's initial value, NULL */
IO* IO::singleton = 0;

/* Constructors, Destructors, Copy */
IO::IO() : desc("Allowed Options") , hierarchy("Allowed Options") {
  return;
}

IO::IO(std::string& optionsName) : desc(optionsName.c_str()), hierarchy(optionsName.c_str()) {
  return;
}

IO::IO(const IO::IO& other) : desc(other.desc){
  return;
}

IO::~IO() {
  return;
}

/* Methods */
void IO::add(const char* identifier, const char* description, const char* parent, bool required) {
  add(identifier, description, parent, required, false); //Allow the programmer to use fewer parameters...
}

void IO::add(const char* identifier, const char* description, const char* parent, bool required, bool out) {
  po::options_description& desc = IO::getSingleton().desc;
  //Generate the full pathname and insert the node into the hierarchy
  std::string path = IO::getSingleton().manageHierarchy(identifier, parent, description);
  //Add the option to boost program_options
  desc.add_options()
    (path.c_str(), description);
  //If the option is required, add it to the required options list
  if(required)
    getSingleton().requiredOptions.push_front(path);
  
  return;
}

//Returns true if the specified value has been flagged by the user
int IO::checkValue(const char* identifier) {
  return getSingleton().vmap.count(identifier);
}
  
//Returns the sole instance of this class
IO& IO::getSingleton() {
  if(!singleton) {
    singleton = new IO();
    //Add the default rules.
    add("help", "default help info", NULL, false);
    add<std::string>("info", "default submodule info option", NULL, false);
  }
  return *singleton;
}	

//Generates a full pathname, and places it in the hierarchy
std::string IO::manageHierarchy(const char* id, const char* parent, const char* description) {
  std::string path(id), desc(description);
  
  path = sanitizeString(parent)+id;
  
  //Add the sanity checked string to the hierarchy
  if(desc.length() == 0)
    hierarchy.appendNode(path);
  else
    hierarchy.appendNode(path, desc);
  return path;
}

//Parses the command line for options
void IO::parseCommandLine(int argc, char** line) {
  po::variables_map& vmap = getSingleton().vmap;
  po::options_description& desc = getSingleton().desc;
  std::list<std::string> rOpt = getSingleton().requiredOptions;
  
  //Parse the command line, place the options & values into vmap
  try{ 
    po::store(po::parse_command_line(argc, line, desc), vmap);
  }catch(std::exception& ex) {
    printFatal(ex.what());
  }
  //Flush the buffer, make sure changes are propogated to vmap
  po::notify(vmap);	
  
  //Now, warn the user if they missed any required options
  for(std::list<std::string>::iterator iter = rOpt.begin(); iter != rOpt.end(); iter++)
    if(!checkValue((*iter).c_str())) {//If a required option isn't there...
      printWarn("Required option --");
      printWarn((*iter).c_str());
      printWarn(" is undefined...");
    }
  
  //Default help message
  if(checkValue("help"))
    print();
  else if(checkValue("info")) {
    std::string str = getValue<std::string>("info");
    getSingleton().hierarchy.print(str);
  }
}


void IO::parseStream(std::istream& stream) {
  po::variables_map& vmap = getSingleton().vmap;
  po::options_description& desc = getSingleton().desc;
  std::list<std::string> rOpt = getSingleton().requiredOptions;
  
  //Parse the stream, place options & values into vmap
  try{
  po::store(po::parse_config_file(stream, desc), vmap);
  }catch(std::exception& ex) {
    printFatal(ex.what());
  }
  //Flush the buffer, make s ure changes are propgated to vmap
  po::notify(vmap);
  
  //Now, warn the user if they missed any required options
  for(std::list<std::string>::iterator iter = rOpt.begin(); iter != rOpt.end(); iter++)
    if(!checkValue((*iter).c_str())) {//If a required option isn't there...
      printWarn("Required option --");
      printWarn((*iter).c_str());
      printWarn(" is undefined...");
    }
}

//Prints the current state, right now just for debugging purposes
void IO::print() {
  IO::getSingleton().hierarchy.print();
}

//Prints an error message
void IO::printFatal(const char* msg) {
  cout << BASH_RED << "[FATAL] " << BASH_CLEAR << msg << endl;
}

//Prints a notification
void IO::printNotify(const char* msg) {
  cout << BASH_YELLOW << "[NOTIFY] " << BASH_CLEAR << msg << endl;
}

void IO::printWarn(const char* msg) {
  cout << BASH_YELLOW << "[WARN]" << BASH_YELLOW << msg << endl;
}

/* Initializes a timer, available like a normal value specified on the command line.  
    Timers are of type timval, as defined in sys/time.h*/
void IO::startTimer(const char* timerName) {
  std::string key = std::string(timerName);
  
  //We don't want to actually document the timer, the user can do that if he wants to.
  timeval tmp;
  
  tmp.tv_sec = 0;
  tmp.tv_usec = 0;
  
  gettimeofday(&tmp, NULL);
  getValue<timeval>(timerName) = tmp;
}
      
/* Halts the timer, and replaces it's value with the delta time from it's start */
void IO::stopTimer(const char* timerName) {
  std::string key = std::string(timerName);
 
  //We don't want to actually document the timer, the user can do that if he wants to.
  if(!getSingleton().globalValues.count(key))
    return;
  
  timeval delta, b, &a = getValue<timeval>(timerName);  
  gettimeofday(&b, NULL);
  
  //Calculate the delta time
  timersub(&b, &a, &delta);
  a = delta; 
}

//Sanitizes strings, rendering input such as /foo/bar and foo/bar equal
std::string IO::sanitizeString(const char* str) {
  if(str != NULL) {
    std::string p(str);
    //Lets sanity check string, remove superfluous '/' prefixes
    if(p.find_first_of("/") == 0)
      p = p.substr(1,p.length()-1);
    //Add necessary '/' suffixes to parent
    if(p.find_last_of("/") != p.length()-1)
      p = p+"/";
    return p;
  }

  return std::string("");
}


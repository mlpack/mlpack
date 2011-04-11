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


#define BASH_RED "\033[0;31m"
#define BASH_GREEN "\033[0;32m"
#define BASH_YELLOW "\033[1;33m"
#define BASH_CLEAR "\033[0m"

using namespace mlpack;

IO* IO::singleton;

/* For clarity, we will alias boost's namespace */
namespace po = boost::program_options;


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
void IO::Add(const char* identifier, const char* description, const char* parent, bool required) {
  po::options_description& desc = IO::GetSingleton().desc;
  //Generate the full pathname and insert the node into the hierarchy
  std::string tmp = TYPENAME(bool);
  std::string path = IO::GetSingleton().ManageHierarchy(identifier, parent, tmp, description);
  //Add the option to boost program_options
  desc.add_options()
    (path.c_str(), description);
  //If the option is required, add it to the required options list
  if(required)
    GetSingleton().requiredOptions.push_front(path);
  
  return;
}

//Returns true if the specified value has been flagged by the user
int IO::CheckValue(const char* identifier) {
  return GetSingleton().vmap.count(identifier);
}
  
//Returns the sole instance of this class
IO& IO::GetSingleton() {
  if(singleton == NULL)
    singleton = new IO();
  return *singleton;
}	

//Generates a full pathname, and places it in the hierarchy
std::string IO::ManageHierarchy(const char* id, const char* parent, std::string& tname, const char* description) {
  std::string path(id), desc(description);
  
  path = SanitizeString(parent)+id;
  
  //Add the sanity checked string to the hierarchy
  if(desc.length() == 0)
    hierarchy.AppendNode(path, tname);
  else
    hierarchy.AppendNode(path, tname, desc);
  return path;
}

//Parses the command line for options
void IO::ParseCommandLine(int argc, char** line) {
  po::variables_map& vmap = GetSingleton().vmap;
  po::options_description& desc = GetSingleton().desc;
  std::list<std::string> rOpt = GetSingleton().requiredOptions;
  
  //Parse the command line, place the options & values into vmap
  try{ 
    po::store(po::parse_command_line(argc, line, desc), vmap);
  }catch(std::exception& ex) {
    PrintFatal(ex.what());
  }
  //Flush the buffer, make sure changes are propogated to vmap
  po::notify(vmap);	
  
  //Now, warn the user if they missed any required options
  for(std::list<std::string>::iterator iter = rOpt.begin(); iter != rOpt.end(); iter++)
    if(!CheckValue((*iter).c_str())) {//If a required option isn't there...
      PrintWarn("Required option --");
      PrintWarn((*iter).c_str());
      PrintWarn(" is undefined...");
    }
  
  //Default help message
  if(CheckValue("help")) {
    Print();
    exit(0);  //The user doesn't want to run the program, he wants to learn how to run it. 
  }
  else if(CheckValue("info")) {
    std::string str = GetValue<std::string>("info");
    GetSingleton().hierarchy.PrintAll();
    exit(0);
  }
}


void IO::ParseStream(std::istream& stream) {
  po::variables_map& vmap = GetSingleton().vmap;
  po::options_description& desc = GetSingleton().desc;
  std::list<std::string> rOpt = GetSingleton().requiredOptions;
  
  //Parse the stream, place options & values into vmap
  try{
  po::store(po::parse_config_file(stream, desc), vmap);
  }catch(std::exception& ex) {
    PrintFatal(ex.what());
  }
  //Flush the buffer, make s ure changes are propgated to vmap
  po::notify(vmap);
  
  //Now, warn the user if they missed any required options
  for(std::list<std::string>::iterator iter = rOpt.begin(); iter != rOpt.end(); iter++)
    if(!CheckValue((*iter).c_str())) {//If a required option isn't there...
      PrintWarn("Required option --");
      PrintWarn((*iter).c_str());
      PrintWarn(" is undefined...");
    }
}

//Prints the current state, right now just for debugging purposes
void IO::Print() {
  IO::GetSingleton().hierarchy.PrintAll();
}

//Prints an error message
void IO::PrintFatal(const char* msg) {
  cout << BASH_RED << "[FATAL] " << BASH_CLEAR << msg << endl;
}

//Prints a notification
void IO::PrintNotify(const char* msg) {
  cout << BASH_GREEN << "[NOTIFY] " << BASH_CLEAR << msg << endl;
}

void IO::PrintWarn(const char* msg) {
  cout << BASH_YELLOW << "[WARN] " << BASH_CLEAR << msg << endl;
}

/* Print whatever data we can */
void IO::PrintData() {
}


/* Initializes a timer, available like a normal value specified on the command line.  
    Timers are of type timval, as defined in sys/time.h*/
void IO::StartTimer(const char* timerName) {
  //We don't want to actually document the timer, the user can do that if he wants to.
  timeval tmp;
  
  tmp.tv_sec = 0;
  tmp.tv_usec = 0;
  
  gettimeofday(&tmp, NULL);
  GetValue<timeval>(timerName) = tmp;
}
      
/* Halts the timer, and replaces it's value with the delta time from it's start */
void IO::StopTimer(const char* timerName) {
  timeval delta, b, &a = GetValue<timeval>(timerName);  
  gettimeofday(&b, NULL);
  
  //Calculate the delta time
  timersub(&b, &a, &delta);
  a = delta; 
}

//Sanitizes strings, rendering input such as /foo/bar and foo/bar equal
std::string IO::SanitizeString(const char* str) {
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

PARAM_MODULE(help, "default help info");
PARAM_CUSTOM(std::string, info, "default submodule info option");


#include "io.h"

#include <list>
#include <boost/program_options.hpp>
#include <boost/any.hpp>
#include <boost/scoped_ptr.hpp>
#include <iostream>
#include <string>
#include <execinfo.h>

#ifndef _WIN32
  #include <sys/time.h> //linux
#else
  #include <winsock.h> //timeval on windows
  #include <windows.h> //GetSystemTimeAsFileTime on windows
//gettimeofday has no equivalent will need to write extra code for that.
  #if defined(_MSC_VER) || defined(_MSC_EXTENSIONS)
    #define DELTA_EPOCH_IN_MICROSECS 11644473600000000Ui64
  #else
    #define DELTA_EPOCH_IN_MICROSECS 11644473600000000ULL
  #endif
#endif //_WIN32

#include "option.h"

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
    BASH_GREEN "[INFO ] " BASH_CLEAR, true /* unless --verbose */, false);
PrefixedOutStream IO::Warn = PrefixedOutStream(std::cout,
    BASH_YELLOW "[WARN ] " BASH_CLEAR, false, false);
PrefixedOutStream IO::Fatal = PrefixedOutStream(std::cerr,
    BASH_RED "[FATAL] " BASH_CLEAR, false, true /* fatal */);
std::ostream& IO::cout = std::cout;

/* For clarity, we will alias boost's namespace */
namespace po = boost::program_options;

// Fake ProgramDoc in case none is supplied.
static ProgramDoc empty_program_doc = ProgramDoc("", "", "");

/* Constructors, Destructors, Copy */
/* Make the constructor private, to preclude unauthorized instances */
IO::IO() : desc("Allowed Options") , hierarchy("Allowed Options"),
    did_parse(false), doc(&empty_program_doc) {
  return;
}

/*
 * Initialize desc with a particular name.
 *
 * @param optionsName Name of the module, as far as boost is concerned.
 */
IO::IO(std::string& optionsName) :
    desc(optionsName.c_str()), hierarchy(optionsName.c_str()),
    did_parse(false), doc(&empty_program_doc) {
  return;
}

// Private copy constructor; don't want copies floating around.
IO::IO(const IO& other) : desc(other.desc),
    did_parse(false), doc(&empty_program_doc) {
  return;
}

IO::~IO() {
  // Terminate the program timer.
  StopTimer("total_time");

  // Did the user ask for verbose output?  If so we need to print everything.
  // But only if the user did not ask for help or info.
  if (GetParam<bool>("verbose")) {
    Info << "Execution parameters:" << std::endl;
    hierarchy.PrintLeaves();

    Info << "Program timers:" << std::endl;
    hierarchy.PrintTimers();
  }

  // Notify the user if we are debugging, but only if we actually parsed the
  // options.  This way this output doesn't show up inexplicably for someone who
  // may not have wanted it there (i.e. in Boost unit tests).
  if (did_parse)
    Debug << "Compiled with debugging symbols." << std::endl;

  return;
}

/* Methods */

/*
 * Adds a parameter to the hierarchy. Use char* and not
 * std::string since the vast majority of use cases will
 * be literal strings.
 *
 * @param identifier The name of the parameter.
 * @param description Short string description of the parameter.
 * @param parent Full pathname of a parent module, default is root node.
 * @param required Indicates if parameter must be set on command line.
 */
void IO::Add(const char* identifier,
             const char* description,
             const char* parent,
             bool required) {

  po::options_description& desc = IO::GetSingleton().desc;

  // Generate the full pathname and insert the node into the hierarchy.
  std::string tmp = TYPENAME(bool);
  std::string path =
    IO::GetSingleton().ManageHierarchy(identifier, parent, tmp, description);

  // Add the option to boost::program_options.
  desc.add_options()
    (path.c_str(), description);

  // If the option is required, add it to the required options list.
  if (required)
    GetSingleton().requiredOptions.push_front(path);

  return;
}


/*
 * @brief Adds a flag paramater to IO.
 */

void IO::AddFlag(const char* identifier,
                 const char* description,
                 const char* parent) {
  po::options_description& desc = IO::GetSingleton().desc;

  //Generate the full pathname and insert node into the hierarchy
  std::string tname = TYPENAME(bool);
  std::string path =
    IO::GetSingleton().ManageHierarchy(identifier, parent, tname, description);

  //Add the option to boost program_options
  desc.add_options()
    (path.c_str(), po::value<bool>()->implicit_value(true), description);
}

void IO::Assert(bool condition) {
  AssertMessage(condition, "Assert failed.");
}

void IO::AssertMessage(bool condition, const char* message) {
  if(!condition) {
    IO::Debug << message << std::endl;
    exit(1);
  }
}

/*
 * See if the specified flag was found while parsing.
 *
 * @param identifier The name of the parameter in question.
 */
bool IO::HasParam(const char* identifier) {
  std::string key = std::string(identifier);
  
  //Does the parameter exist at all?
  int isInVmap = GetSingleton().vmap.count(key);
  int isInGmap = GetSingleton().globalValues.count(key);

  //Lets check if the parameter is boolean, if it is we just want to see 
  //If it was passed at program initiation.
  OptionsHierarchy* node = IO::GetSingleton().hierarchy.FindNode(key);
  if(node) {//Sanity check
    OptionsData data = node->GetNodeData();
    if(data.tname == std::string(TYPENAME(bool))) //Actually check if its bool
      return IO::GetParam<bool>(identifier);
  }

  //Return true if we have a defined value for identifier
  return (isInVmap || isInGmap);
}


/*
 * Searches for unqualified parameters, when one is found prepend the default
 * module path onto it.
 *
 * @param argc The number of parameters
 * @param argv 2D array of the parameter strings themselves
 * @return some valid modified strings
 */
std::vector<std::string> IO::InsertDefaultModule(int argc, char** argv) {
  std::vector<std::string> ret;
  std::string path = GetSingleton().doc->defaultModule;
  path = SanitizeString(path.c_str());

  for(int i = 1; i < argc; i++) {//First parameter is just the program name.
    std::string str = argv[i];

    //Are we lacking any qualifiers?
    if(str.find('/') == std::string::npos &&
       str.compare("--help") != 0 &&
       str.compare("--info") != 0)
      str = "--"+path+str.substr(2,str.length());

    ret.push_back(str);
  }

  return ret;
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

std::vector<std::string> IO::GetFolder(const char* folder) {
  std::string str = folder;
  return GetSingleton().hierarchy.GetRelativePaths(str);
}

//Returns the sole instance of this class
IO& IO::GetSingleton() {
  if (singleton == NULL) {
    singleton = new IO();
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
 * @param desc String description of the parameter.
 */
std::string IO::ManageHierarchy(const char* id,
                                const char* parent,
                                std::string& tname,
                                const char* desc) {

  std::string path(id);

  path = SanitizeString(parent) + id;

  AddToHierarchy(path, tname, desc);

  return path;
}

/***
 * Add a parameter to the hierarchy.  We assume the string has already been
 * sanity-checked.
 *
 * @param path Full pathname of the parameter (parent/parameter).
 * @param tname String identifier of the parameter's type (TYPENAME(T)).
 * @param desc String description of the parameter (optional).
 */
void IO::AddToHierarchy(std::string& path, std::string& tname,
                        const char* desc) {
  // Make sure we don't overwrite any data.
  if (hierarchy.FindNode(path) != NULL)
    return;

  // Add the sanity checked string to the hierarchy
  std::string d(desc);
  if (d.length() == 0)
    hierarchy.AppendNode(path, tname);
  else
    hierarchy.AppendNode(path, tname, d);
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

  // Insert the default module where appropriate
  std::vector<std::string> in = InsertDefaultModule(argc, line);

  // Parse the command line, place the options & values into vmap
  try {
    po::store(po::parse_command_line(argc, line, desc), vmap);
  } catch(std::exception& ex) {
    IO::Fatal << ex.what() << std::endl;
  }
  // Flush the buffer, make sure changes are propagated to vmap
  po::notify(vmap);
  UpdateGmap();
  DefaultMessages();
  RequiredOptions();

  StartTimer("total_time");
}

/*
 * Parses a stream for arguments
 *
 * @param stream The stream to be parsed.
 */
void IO::ParseStream(std::istream& stream) {
  po::variables_map& vmap = GetSingleton().vmap;
  po::options_description& desc = GetSingleton().desc;

  // Parse the stream, place options & values into vmap
  try {
    po::store(po::parse_config_file(stream, desc), vmap);
  } catch (std::exception& ex) {
    IO::Fatal << ex.what() << std::endl;
  }
  // Flush the buffer, make sure changes are propagated to vmap
  po::notify(vmap);

  UpdateGmap();
  DefaultMessages();
  RequiredOptions();

  StartTimer("total_time");
}

/*
 * Parses the values given on the command line,
 * overriding any default values.
 */
void IO::UpdateGmap() {
  std::map<std::string, boost::any>& gmap = GetSingleton().globalValues;
  po::variables_map& vmap = GetSingleton().vmap;

  //Iterate through Gmap, and overwrite default values with anything found on
  //command line.
  std::map<std::string, boost::any>::iterator i;
  for (i = gmap.begin(); i != gmap.end(); i++) {
    po::variable_value tmp = vmap[i->first];
    if (!tmp.empty()) //We need to overwrite gmap.
      gmap[i->first] = tmp.value();
  }
}

/**
 * Registers a ProgramDoc object, which contains documentation about the
 * program.
 *
 * @param doc Pointer to the ProgramDoc object.
 */
void IO::RegisterProgramDoc(ProgramDoc* doc) {
  // Only register the doc if it is not the dummy object we created at the
  // beginning of the file (as a default value in case this is never called).
  if (doc != &empty_program_doc)
    GetSingleton().doc = doc;
}

/***
 * Destroy the IO object.  This resets the pointer to the singleton, so in case
 * someone tries to access it after destruction, a new one will be made (the
 * program will not fail).
 */
void IO::Destroy() {
  if (singleton != NULL) {
    delete singleton;
    singleton = NULL; // Reset pointer.
  }
}

/*
 * Parses the parameters for 'help' and 'info'
 * If found, will print out the appropriate information
 * and kill the program.
 */
void IO::DefaultMessages() {
  // Default help message
  if (GetParam<bool>("help")) {
    // A little snippet about the program itself, if we have it.
    if (GetSingleton().doc != &empty_program_doc) {
      cout << GetSingleton().doc->programName << std::endl << std::endl;
      cout << "  " << OptionsHierarchy::HyphenateString(
        GetSingleton().doc->documentation, 2) << std::endl << std::endl;
    }

    GetSingleton().hierarchy.PrintAllHelp();
    exit(0); // The user doesn't want to run the program, he wants help.
  }
  if (HasParam("info")) {
    std::string str = GetParam<std::string>("info");
    // The info node should always be there, but the user may not have specified
    // anything.
    if (str != "") {
      OptionsHierarchy* node = GetSingleton().hierarchy.FindNode(str);
      if(node != NULL)
        node->PrintNodeHelp();
      else
        IO::Fatal << "Invalid parameter: " << str << std::endl;
      exit(0);
    }
  }
  if (GetParam<bool>("verbose"))
    Info.ignoreInput = false;

  // Notify the user if we are debugging.  This is not done in the constructor
  // because the output streams may not be set up yet.  We also don't want this
  // message twice if the user just asked for help or information.
  Debug << "Compiled with debugging symbols." << std::endl;
}

/*
 * Checks that all parameters specified as required
 * have been specified on the command line.
 * If they havent, prints an error message and kills the
 * program.
 */
void IO::RequiredOptions() {
  po::variables_map& vmap = GetSingleton().vmap;
  std::list<std::string> rOpt = GetSingleton().requiredOptions;

  //Now, warn the user if they missed any required options
  std::list<std::string>::iterator iter;
  for (iter = rOpt.begin(); iter != rOpt.end(); iter++) {
  std::string str = *iter;
  if (!vmap.count(str)) // If a required option isn't there...
      IO::Fatal << "Required option --" << iter->c_str() << " is undefined."
          << std::endl;
  }
}

/* Prints out the current hierachy */
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
  // Don't want to actually document the timer, the user can do that if he wants
  timeval tmp;

  tmp.tv_sec = 0;
  tmp.tv_usec = 0;

  // Also add it to the hierarchy for printing at the end of execution.
  // We don't have any documentation, so omit it.  Since program execution
  // already started, a user couldn't get to the documentation anyway.
  std::string name(timerName);
  std::string tname = TYPENAME(timeval);
  IO::GetSingleton().AddToHierarchy(name, tname);

#ifndef _WIN32
  gettimeofday(&tmp, NULL);
#else
  FileTimeToTimeVal(&tmp);
#endif


  GetParam<timeval>(timerName) = tmp;
}

/*
 * Halts the timer, and replaces it's value with
 * the delta time from it's start
 *
 * @param timerName The name of the timer in question.
 */
void IO::StopTimer(const char* timerName) {
  timeval delta, b, &a = GetParam<timeval>(timerName);

#ifndef _WIN32
  gettimeofday(&b, NULL);
#else
  FileTimeToTimeVal(&b);
#endif
  //Calculate the delta time
  timersub(&b, &a, &delta);
  a = delta;
}

#ifdef _WIN32
void IO::FileTimeToTimeVal(timeval* tv) {
  FILETIME ftime;
  uint64_t ptime = 0;

  //Acquire the filetime
  GetSystemTimeAsFileTime(&ftime);


  //Now convert FILETIME to timeval
  //FILETIME records time in 100 nsec intervals gettimeofday in microsecs
  //FILETIME starts it's epoch January 1 1601; not January 1 1970
  ptime |= ftime.dwHighDateTime;
  ptime = ptime << 32;
  ptime |= ftime.dwLowDateTime;
  ptime /= 10;  //Convert from 100 nsec INTERVALS to microseconds
  ptime -= DELTA_EPOC_IN_MICROSECONDS; //Subtract difference from 1970 & 1601

  //Convert from microseconds to seconds
  tv.tv_sec = (long)(ptime/1000000UL);
  tv.tv_usec = (long)(ptime%1000000UL);
}
#endif //Win32

// Add help parameter.
PARAM_FLAG("help", "Default help info.", "");
PARAM_STRING("info", "Get help on a specific module or option.", "", "");
PARAM_FLAG("verbose", "Display informational messages and the full list of "
    "parameters and timers at the end of execution.", "");

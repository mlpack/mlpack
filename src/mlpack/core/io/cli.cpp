/**
 * @file cli.cpp
 * @author Matthew Amidon
 *
 * Implementation of the CLI module for parsing parameters.
 */
#include <list>
#include <boost/program_options.hpp>
#include <boost/any.hpp>
#include <boost/scoped_ptr.hpp>
#include <iostream>
#include <string>
#include <execinfo.h>

#ifndef _WIN32
  #include <sys/time.h> // For Linux.
#else
  #include <winsock.h> // timeval on Windows.
  #include <windows.h> // GetSystemTimeAsFileTime() on Windows.
// gettimeofday() has no equivalent; we will need to write extra code for that.
  #if defined(_MSC_VER) || defined(_MSC_EXTENSCLINS)
    #define DELTA_EPOCH_IN_MICROSECS 11644473600000000Ui64
  #else
    #define DELTA_EPOCH_IN_MICROSECS 11644473600000000ULL
  #endif
#endif // _WIN32

#include "cli.hpp"
#include "log.hpp"
#include "../utilities/timers.hpp"

#include "option.hpp"

using namespace mlpack;
using namespace mlpack::io;

CLI* CLI::singleton = NULL;

/* For clarity, we will alias boost's namespace. */
namespace po = boost::program_options;

// Fake ProgramDoc in case none is supplied.
static ProgramDoc empty_program_doc = ProgramDoc("", "", "");

/* Constructors, Destructors, Copy */
/* Make the constructor private, to preclude unauthorized instances */
CLI::CLI() : desc("Allowed Options") , did_parse(false), 
  doc(&empty_program_doc)
{
  return;
}

/**
 * Initialize desc with a particular name.
 *
 * @param optionsName Name of the module, as far as boost is concerned.
 */
CLI::CLI(std::string& optionsName) :
    desc(optionsName.c_str()), did_parse(false), doc(&empty_program_doc)
{
  return;
}

// Private copy constructor; don't want copies floating around.
CLI::CLI(const CLI& other) : desc(other.desc),
    did_parse(false), doc(&empty_program_doc)
{
  return;
}

CLI::~CLI()
{
  // Terminate the program timer.
  Timers::StopTimer("total_time");

  // Did the user ask for verbose output?  If so we need to print everything.
  // But only if the user did not ask for help or info.
  if (GetParam<bool>("verbose"))
  {
    Log::Info << "Execution parameters:" << std::endl;
    //hierarchy.PrintLeaves(), todo replace functionality;

    Log::Info << "Program timers:" << std::endl;
    std::map<std::string, timeval> times = Timers::GetAllTimers();
    std::map<std::string, timeval>::iterator iter;
    for (iter = times.begin(); iter != times.end(); iter++)
    {
      Log::Info << "  " << iter->first << ": ";
      Timers::PrintTimer(iter->first.c_str());
    }
  }

  // Notify the user if we are debugging, but only if we actually parsed the
  // options.  This way this output doesn't show up inexplicably for someone who
  // may not have wanted it there (i.e. in Boost unit tests).
  if (did_parse)
    Log::Debug << "Compiled with debugging symbols." << std::endl;

  return;
}

/* Methods */

/**
 * Adds a parameter to the hierarchy. Use char* and not std::string since the
 * vast majority of use cases will be literal strings.
 *
 * @param identifier The name of the parameter.
 * @param description Short string description of the parameter.
 * @param parent Full pathname of a parent module, default is root node.
 * @param required Indicates if parameter must be set on command line.
 */
void CLI::Add(const char* identifier,
             const char* description,
             const char* parent,
             bool required)
{
  po::options_description& desc = CLI::GetSingleton().desc;

  // Generate the full pathname and insert the node into the hierarchy.
  std::string tmp = TYPENAME(bool);
  std::string path = identifier;

  // Add the option to boost::program_options.
  desc.add_options()
    (path.c_str(), description);

  // If the option is required, add it to the required options list.
  if (required)
    GetSingleton().requiredOptions.push_front(path);

  return;
}

/*
 * @brief Adds a flag parameter to CLI.
 */
void CLI::AddFlag(const char* identifier,
                 const char* description,
                 const char* parent)
{
  po::options_description& desc = CLI::GetSingleton().desc;

  //Generate the full pathname and insert node into the hierarchy
  std::string path = identifier;

  // Add the option to boost::program_options.
  desc.add_options()
    (path.c_str(), po::value<bool>()->implicit_value(true), description);
}

/**
 * See if the specified flag was found while parsing.
 *
 * @param identifier The name of the parameter in question.
 */
bool CLI::HasParam(const char* identifier)
{
  po::variables_map vmap = GetSingleton().vmap;
  gmap_t gmap = GetSingleton().globalValues;
  std::string key = std::string(identifier);

  //Does the parameter exist at all?
  //int isInVmap = vmap.count(key);
  int isInGmap = gmap.count(key);

  // Check if the parameter is boolean; if it is, we just want to see if it was
  // passed at program initiation.
  // TODO, reimpliment flag logic.
  
  // Return true if we have a defined value for identifier.
  return isInGmap;
}

/**
 * Grab the description of the specified node.
 *
 * @param identifier Name of the node in question.
 * @return Description of the node in question.
 */
std::string CLI::GetDescription(const char* identifier)
{
  gmap_t gmap = GetSingleton().globalValues;
  std::string name = std::string(identifier);

  if(gmap.count(name))
    return gmap[name].desc;
  else
    return "";
    
}

// Returns the sole instance of this class.
CLI& CLI::GetSingleton()
{
  if (singleton == NULL)
    singleton = new CLI();

  return *singleton;
}

/**
 * Parses the commandline for arguments.
 *
 * @param argc The number of arguments on the commandline.
 * @param argv The array of arguments as strings
 */
void CLI::ParseCommandLine(int argc, char** line)
{
  po::variables_map& vmap = GetSingleton().vmap;
  po::options_description& desc = GetSingleton().desc;

  // Parse the command line, place the options & values into vmap
  try
  {
    po::store(po::parse_command_line(argc, line, desc), vmap);
  }
  catch (std::exception& ex)
  {
    Log::Fatal << ex.what() << std::endl;
  }

  // Flush the buffer, make sure changes are propagated to vmap
  po::notify(vmap);
  UpdateGmap();
  DefaultMessages();
  RequiredOptions();

  Timers::StartTimer("total_time");
}

/**
 * Parses a stream for arguments
 *
 * @param stream The stream to be parsed.
 */
void CLI::ParseStream(std::istream& stream)
{
  po::variables_map& vmap = GetSingleton().vmap;
  po::options_description& desc = GetSingleton().desc;

  // Parse the stream; place options & values into vmap.
  try
  {
    po::store(po::parse_config_file(stream, desc), vmap);
  }
  catch (std::exception& ex)
  {
    Log::Fatal << ex.what() << std::endl;
  }

  // Flush the buffer; make sure changes are propagated to vmap.
  po::notify(vmap);

  UpdateGmap();
  DefaultMessages();
  RequiredOptions();

  Timers::StartTimer("total_time");
}

/**
 * Parses the values given on the command line, overriding any default values.
 */
void CLI::UpdateGmap()
{
  gmap_t& gmap = GetSingleton().globalValues;
  po::variables_map& vmap = GetSingleton().vmap;

  // Iterate through Gmap, and overwrite default values with anything found on
  // command line.
  gmap_t::iterator i;
  for (i = gmap.begin(); i != gmap.end(); i++)
  {
    po::variable_value tmp = vmap[i->first];
    if (!tmp.empty()){ // We need to overwrite gmap.
      ParamData param;
      param.value = tmp.value();
      gmap[i->first] = param;
    }
  }
}

/**
 * Registers a ProgramDoc object, which contains documentation about the
 * program.
 *
 * @param doc Pointer to the ProgramDoc object.
 */
void CLI::RegisterProgramDoc(ProgramDoc* doc)
{
  // Only register the doc if it is not the dummy object we created at the
  // beginning of the file (as a default value in case this is never called).
  if (doc != &empty_program_doc)
    GetSingleton().doc = doc;
}

/**
 * Destroy the CLI object.  This resets the pointer to the singleton, so in case
 * someone tries to access it after destruction, a new one will be made (the
 * program will not fail).
 */
void CLI::Destroy()
{
  if (singleton != NULL)
  {
    delete singleton;
    singleton = NULL; // Reset pointer.
  }
}

/**
 * Parses the parameters for 'help' and 'info'
 * If found, will print out the appropriate information
 * and kill the program.
 */
void CLI::DefaultMessages()
{
  // Default help message
  if (GetParam<bool>("help"))
  {
    // A little snippet about the program itself, if we have it.
    if (GetSingleton().doc != &empty_program_doc)
    {
      std::cout << GetSingleton().doc->programName << std::endl << std::endl;
    }
    Log::Fatal << "TODO: reimpliment printing functionality" << std::endl;
    exit(0); // The user doesn't want to run the program, he wants help.
  }

  if (HasParam("info"))
  {
    std::string str = GetParam<std::string>("info");
    // The info node should always be there, but the user may not have specified
    // anything.
    if (str != "")
    {
      Log::Fatal << "TODO: Reimpliment printing functionality" << std::endl;
     /* //OptionsHierarchy* node = GetSingleton().hierarchy.FindNode(str);
      if(node != NULL)
        //node->PrintNodeHelp();
      else
        Log::Fatal << "Invalid parameter: " << str << std::endl;*/
      exit(0);
    }
  }

  if (GetParam<bool>("verbose"))
    Log::Info.ignoreInput = false;

  // Notify the user if we are debugging.  This is not done in the constructor
  // because the output streams may not be set up yet.  We also don't want this
  // message twice if the user just asked for help or information.
  Log::Debug << "Compiled with debugging symbols." << std::endl;
}

/**
 * Checks that all parameters specified as required have been specified on the
 * command line.  If they havent, prints an error message and kills the program.
 */
void CLI::RequiredOptions()
{
  po::variables_map& vmap = GetSingleton().vmap;
  std::list<std::string> rOpt = GetSingleton().requiredOptions;

  //Now, warn the user if they missed any required options
  std::list<std::string>::iterator iter;
  for (iter = rOpt.begin(); iter != rOpt.end(); iter++) {
    std::string str = *iter;
    if (!vmap.count(str))
    { // If a required option isn't there...
      Timers::StopTimer("total_time"); // Execution stop here, pretty much.
      Log::Fatal << "Required option --" << str.c_str() << " is undefined."
          << std::endl;
    }
  }
}

/* Prints out the current hierarchy. */
void CLI::Print()
{
  Log::Fatal << "TODO: Reimpliment printing functionality" << std::endl;
}

// Add help parameter.
PARAM_FLAG("help", "Default help info.", "");
PARAM_STRING("info", "Get help on a specific module or option.", "", "");
PARAM_FLAG("verbose", "Display informational messages and the full list of "
    "parameters and timers at the end of execution.", "");

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

#include "cli.hpp"
#include "log.hpp"

#include "option.hpp"

using namespace mlpack;
using namespace mlpack::util;

CLI* CLI::singleton = NULL;

/* For clarity, we will alias boost's namespace. */
namespace po = boost::program_options;

// Fake ProgramDoc in case none is supplied.
static ProgramDoc emptyProgramDoc = ProgramDoc("", "");

/* Constructors, Destructors, Copy */
/* Make the constructor private, to preclude unauthorized instances */
CLI::CLI() : desc("Allowed Options") , didParse(false), doc(&emptyProgramDoc)
{
  return;
}

/**
 * Initialize desc with a particular name.
 *
 * @param optionsName Name of the module, as far as boost is concerned.
 */
CLI::CLI(const std::string& optionsName) :
    desc(optionsName), didParse(false), doc(&emptyProgramDoc)
{
  return;
}

// Private copy constructor; don't want copies floating around.
CLI::CLI(const CLI& other) : desc(other.desc),
    didParse(false), doc(&emptyProgramDoc)
{
  return;
}

CLI::~CLI()
{
  // Terminate the program timers.
  std::map<std::string, std::chrono::microseconds>::iterator it;
  for (it = timer.GetAllTimers().begin(); it != timer.GetAllTimers().end();
       ++it)
  {
    std::string i = (*it).first;
    if (timer.GetState(i) == 1)
      Timer::Stop(i);
  }

  // Did the user ask for verbose output?  If so we need to print everything.
  // But only if the user did not ask for help or info.
  if (HasParam("verbose") && !HasParam("help") && !HasParam("info"))
  {
    Log::Info << std::endl << "Execution parameters:" << std::endl;
    Print();

    Log::Info << "Program timers:" << std::endl;
    std::map<std::string, std::chrono::microseconds>::iterator it;
    for (it = timer.GetAllTimers().begin(); it != timer.GetAllTimers().end();
        ++it)
    {
      std::string i = (*it).first;
      Log::Info << "  " << i << ": ";
      timer.PrintTimer((*it).first);
    }
  }

  // Notify the user if we are debugging, but only if we actually parsed the
  // options.  This way this output doesn't show up inexplicably for someone who
  // may not have wanted it there (i.e. in Boost unit tests).
  if (didParse)
    Log::Debug << "Compiled with debugging symbols." << std::endl;

  return;
}

/**
 * Adds a parameter to the hierarchy. Use char* and not std::string since the
 * vast majority of use cases will be literal strings.
 *
 * @param identifier The name of the parameter.
 * @param description Short string description of the parameter.
 * @param alias An alias for the parameter.
 * @param required Indicates if parameter must be set on command line.
 */
void CLI::Add(const std::string& identifier,
              const std::string& description,
              const std::string& alias,
              bool required)
{
  po::options_description& desc = CLI::GetSingleton().desc;

  // Must make use of boost option name syntax.
  std::string progOptId = alias.length() ? identifier + "," + alias : identifier;

  // Deal with a required alias.
  AddAlias(alias, identifier);

  // Add the option to boost::program_options.
  desc.add_options()(progOptId.c_str(), description.c_str());

  // Make sure the description, etc. ends up in gmap.
  gmap_t& gmap = GetSingleton().globalValues;

  ParamData data;
  data.desc = description;
  data.tname = "";
  data.name = identifier;
  data.isFlag = false;
  data.wasPassed = false;

  gmap[identifier] = data;

  // If the option is required, add it to the required options list.
  if (required)
    GetSingleton().requiredOptions.push_front(identifier);

  return;
}

/*
 * Adds an alias mapping for a given parameter.
 *
 * @param alias The alias we will use for the parameter.
 * @param original The name of the actual parameter we will be mapping to.
 */
void CLI::AddAlias(const std::string& alias, const std::string& original)
{
  // Conduct the mapping.
  if (alias.length())
  {
    amap_t& amap = GetSingleton().aliasValues;
    amap[alias] = original;
  }
}

/*
 * @brief Adds a flag parameter to CLI.
 */
void CLI::AddFlag(const std::string& identifier,
                  const std::string& description,
                  const std::string& alias)
{
  // Reuse functionality from Add().
  Add(identifier, description, alias, false);

  // Insert the proper metadata into gmap.
  gmap_t& gmap = GetSingleton().globalValues;

  ParamData data;
  data.desc = description;
  data.tname = TYPENAME(bool);
  data.name = std::string(identifier);
  data.isFlag = true;
  data.wasPassed = false;

  gmap[data.name] = data;
}

std::string CLI::AliasReverseLookup(const std::string& value)
{
  amap_t& amap = GetSingleton().aliasValues;
  amap_t::iterator iter;
  for (iter = amap.begin(); iter != amap.end(); ++iter)
    if (iter->second == value) // Found our match.
      return iter->first;

  return ""; // Nothing found.
}

/**
 * Parses the parameters for 'help' and 'info' If found, will print out the
 * appropriate information and kill the program.
 */
void CLI::DefaultMessages()
{
  // --version is prioritized over --help.
  if (HasParam("version"))
  {
    std::cout << GetSingleton().programName << ": part of "
        << util::GetVersion() << std::endl;
    exit(0);
  }

  // Default help message.
  if (HasParam("help"))
  {
    Log::Info.ignoreInput = false;
    PrintHelp();
    exit(0); // The user doesn't want to run the program, he wants help.
  }

  if (HasParam("info"))
  {
    Log::Info.ignoreInput = false;
    std::string str = GetParam<std::string>("info");

    // The info node should always be there, but the user may not have specified
    // anything.
    if (str != "")
    {
      PrintHelp(str);
      exit(0);
    }

    // Otherwise just print the generalized help.
    PrintHelp();
    exit(0);
  }

  if (HasParam("verbose"))
  {
    // Give [INFO ] output.
    Log::Info.ignoreInput = false;
  }

  // Notify the user if we are debugging.  This is not done in the constructor
  // because the output streams may not be set up yet.  We also don't want this
  // message twice if the user just asked for help or information.
  Log::Debug << "Compiled with debugging symbols." << std::endl;
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
 * See if the specified flag was found while parsing.
 *
 * @param identifier The name of the parameter in question.
 */
bool CLI::HasParam(const std::string& key)
{
  return GetParam<bool>(key);
}

/**
 * GetParam<bool>() is equivalent to HasParam().
 */
template<>
bool& CLI::GetParam<bool>(const std::string& key)
{
  std::string used_key = key;
  po::variables_map vmap = GetSingleton().vmap;
  gmap_t& gmap = GetSingleton().globalValues;

  // Take any possible alias into account.
  amap_t& amap = GetSingleton().aliasValues;
  if (amap.count(key))
    used_key = amap[key];

  // Does the parameter exist at all?
  int isInGmap = gmap.count(used_key);

  // Check if the parameter is boolean; if it is, we just want to see if it was
  // passed.
  if (isInGmap)
    return gmap[used_key].wasPassed;

  // The parameter was not passed in; terminate the program.
  Log::Fatal << "Parameter '--" << key << "' does not exist in this program."
      << std::endl;

  // These lines will never be reached, but must be here to make the compiler
  // happy.
  bool* trash = new bool;
  *trash = false;
  return *trash;
}

/**
 * Hyphenate a string or split it onto multiple 80-character lines, with some
 * amount of padding on each line.  This is used for option output.
 *
 * @param str String to hyphenate (splits are on ' ').
 * @param padding Amount of padding on the left for each new line.
 */
std::string CLI::HyphenateString(const std::string& str, int padding)
{
  size_t margin = 80 - padding;
  if (str.length() < margin)
    return str;
  std::string out("");
  unsigned int pos = 0;
  // First try to look as far as possible.
  while (pos < str.length())
  {
    size_t splitpos;
    // Check that we don't have a newline first.
    splitpos = str.find('\n', pos);
    if (splitpos == std::string::npos || splitpos > (pos + margin))
    {
      // We did not find a newline.
      if (str.length() - pos < margin)
      {
        splitpos = str.length(); // The rest fits on one line.
      }
      else
      {
        splitpos = str.rfind(' ', margin + pos); // Find nearest space.
        if (splitpos <= pos || splitpos == std::string::npos) // Not found.
          splitpos = pos + margin;
      }
    }
    out += str.substr(pos, (splitpos - pos));
    if (splitpos < str.length())
    {
      out += '\n';
      out += std::string(padding, ' ');
    }

    pos = splitpos;
    if (str[pos] == ' ' || str[pos] == '\n')
      pos++;
  }
  return out;
}

/**
 * Grab the description of the specified node.
 *
 * @param identifier Name of the node in question.
 * @return Description of the node in question.
 */
std::string CLI::GetDescription(const std::string& identifier)
{
  gmap_t& gmap = GetSingleton().globalValues;
  std::string name = std::string(identifier);

  //Take any possible alias into account
  amap_t& amap = GetSingleton().aliasValues;
  if (amap.count(name))
    name = amap[name];


  if (gmap.count(name))
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
  Timer::Start("total_time");

  GetSingleton().programName = std::string(line[0]);
  po::variables_map& vmap = GetSingleton().vmap;
  po::options_description& desc = GetSingleton().desc;

  // Parse the command line, place the options & values into vmap
  try
  {
    // Get the basic_parsed_options
    po::basic_parsed_options<char> bpo(
      po::parse_command_line(argc, line, desc));

    // Look for any duplicate parameters, removing duplicate flags
    RemoveDuplicateFlags(bpo);

    // Record the basic_parsed_options
    po::store(bpo, vmap);
  }
  catch (std::exception& ex)
  {
    Log::Fatal << "Caught exception from parsing command line:\t";
    Log::Fatal << ex.what() << std::endl;
  }

  // Flush the buffer, make sure changes are propagated to vmap
  po::notify(vmap);
  UpdateGmap();
  DefaultMessages();
  RequiredOptions();
}

/*
 * Removes duplicate flags.
 *
 * @param bpo The basic_program_options to remove duplicate flags from.
 */
void CLI::RemoveDuplicateFlags(po::basic_parsed_options<char>& bpo)
{
  // Iterate over all the program_options, looking for duplicate parameters
  for (unsigned int i = 0; i < bpo.options.size(); i++)
  {
    for (unsigned int j = i + 1; j < bpo.options.size(); j++)
    {
      if (bpo.options[i].string_key == bpo.options[j].string_key)
      {
        // If a duplicate is found, check to see if either one has a value
        if (bpo.options[i].value.size() == 0 &&
            bpo.options[j].value.size() == 0)
        {
          // If neither has a value, consider it a duplicate flag and remove the
          // duplicate. It's important to not break out of this loop because
          // there might be another duplicate later on in the vector.
          bpo.options.erase(bpo.options.begin()+j);
        }
        else
        {
          // If one or both has a value, produce an error and politely
          // terminate. We pull the name from the original_tokens, rather than
          // from the string_key, because the string_key is the parameter after
          // aliases have been expanded.
          Log::Fatal << "\"" << bpo.options[j].original_tokens[0] << "\""
              << " is defined multiple times." << std::endl;
        }
      }
    }
  }
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

  Timer::Start("total_time");
}

/* Prints out the current hierarchy. */
void CLI::Print()
{
  gmap_t& gmap = GetSingleton().globalValues;
  gmap_t::iterator iter;

  // Print out all the values.
  for (iter = gmap.begin(); iter != gmap.end(); ++iter)
  {
    std::string key = iter->first;

    Log::Info << "  " << key << ": ";

    // Now, figure out what type it is, and print it.
    // We can handle strings, ints, bools, floats, doubles.
    ParamData data = iter->second;
    if (data.tname == TYPENAME(std::string))
    {
      std::string value = GetParam<std::string>(key);
      if (value == "")
        Log::Info << "\"\"";
      Log::Info << value;
    }
    else if (data.tname == TYPENAME(int))
    {
      int value = GetParam<int>(key);
      Log::Info << value;
    }
    else if (data.tname == TYPENAME(bool))
    {
      bool value = HasParam(key);
      Log::Info << (value ? "true" : "false");
    }
    else if (data.tname == TYPENAME(float))
    {
      float value = GetParam<float>(key);
      Log::Info << value;
    }
    else if (data.tname == TYPENAME(double))
    {
      double value = GetParam<double>(key);
      Log::Info << value;
    }
    else
    {
      // We don't know how to print this, or it's a timeval which is printed
      // later.
      Log::Info << "(Unknown data type - " << data.tname << ")";
    }

    Log::Info << std::endl;
  }
  Log::Info << std::endl;
}

/* Prints the descriptions of the current hierarchy. */
void CLI::PrintHelp(const std::string& param)
{
  std::string used_param = param;
  gmap_t& gmap = GetSingleton().globalValues;
  amap_t& amap = GetSingleton().aliasValues;
  gmap_t::iterator iter;
  ProgramDoc docs = *GetSingleton().doc;

  // If we pass a single param, alias it if necessary.
  if (used_param != "" && amap.count(used_param))
    used_param = amap[used_param];

  // Do we only want to print out one value?
  if (used_param != "" && gmap.count(used_param))
  {
    ParamData data = gmap[used_param];
    std::string alias = AliasReverseLookup(used_param);
    alias = alias.length() ? " (-" + alias + ")" : alias;

    // Figure out the name of the type.
    std::string type = "";
    if (data.tname == TYPENAME(std::string))
      type = " [string]";
    else if (data.tname == TYPENAME(int))
      type = " [int]";
    else if (data.tname == TYPENAME(bool))
      type = ""; // Nothing to pass for a flag.
    else if (data.tname == TYPENAME(float))
      type = " [float]";
    else if (data.tname == TYPENAME(double))
      type = " [double]";

    // Now, print the descriptions.
    std::string fullDesc = "  --" + used_param + alias + type + "  ";

    if (fullDesc.length() <= 32) // It all fits on one line.
      std::cout << fullDesc << std::string(32 - fullDesc.length(), ' ');
    else // We need multiple lines.
      std::cout << fullDesc << std::endl << std::string(32, ' ');

    std::cout << HyphenateString(data.desc, 32) << std::endl;
    return;
  }
  else if (used_param != "")
  {
    // User passed a single variable, but it doesn't exist.
    std::cerr << "Parameter --" << used_param << " does not exist."
        << std::endl;
    exit(1); // Nothing left to do.
  }

  // Print out the descriptions.
  if (docs.programName != "")
  {
    std::cout << docs.programName << std::endl << std::endl;
    std::cout << "  " << HyphenateString(docs.documentation, 2) << std::endl
        << std::endl;
  }
  else
    std::cout << "[undocumented program]" << std::endl << std::endl;

  for (size_t pass = 0; pass < 2; ++pass)
  {
    bool printedHeader = false;

    // Print out the descriptions of everything else.
    for (iter = gmap.begin(); iter != gmap.end(); ++iter)
    {
      std::string key = iter->first;
      ParamData data = iter->second;
      std::string desc = data.desc;
      std::string alias = AliasReverseLookup(key);
      alias = alias.length() ? " (-" + alias + ")" : alias;

      // Is the option required or not?
      bool required = false;
      std::list<std::string>::iterator iter;
      std::list<std::string>& rOpt = GetSingleton().requiredOptions;
      for (iter = rOpt.begin(); iter != rOpt.end(); ++iter)
        if ((*iter) == key)
          required = true;

      if ((pass == 0) && !required)
        continue; // Don't print this one.
      if ((pass == 1) && required)
        continue; // Don't print this one.

      if (!printedHeader)
      {
        printedHeader = true;
        if (pass == 0)
          std::cout << "Required options:" << std::endl << std::endl;
        else
          std::cout << "Options: " << std::endl << std::endl;
      }

      if (pass == 1) // Append default value to description.
      {
        desc += "  Default value ";
        std::stringstream tmp;

        if (data.tname == TYPENAME(std::string))
          tmp << "'" << boost::any_cast<std::string>(data.value) << "'.";
        else if (data.tname == TYPENAME(int))
          tmp << boost::any_cast<int>(data.value) << '.';
        else if (data.tname == TYPENAME(bool))
          desc = data.desc; // No extra output for that.
        else if (data.tname == TYPENAME(float))
          tmp << boost::any_cast<float>(data.value) << '.';
        else if (data.tname == TYPENAME(double))
          tmp << boost::any_cast<double>(data.value) << '.';

        desc += tmp.str();
      }

      // Figure out the name of the type.
      std::string type = "";
      if (data.tname == TYPENAME(std::string))
        type = " [string]";
      else if (data.tname == TYPENAME(int))
        type = " [int]";
      else if (data.tname == TYPENAME(bool))
        type = ""; // Nothing to pass for a flag.
      else if (data.tname == TYPENAME(float))
        type = " [float]";
      else if (data.tname == TYPENAME(double))
        type = " [double]";

      // Now, print the descriptions.
      std::string fullDesc = "  --" + key + alias + type + "  ";

      if (fullDesc.length() <= 32) // It all fits on one line.
        std::cout << fullDesc << std::string(32 - fullDesc.length(), ' ');
      else // We need multiple lines.
        std::cout << fullDesc << std::endl << std::string(32, ' ');

      std::cout << HyphenateString(desc, 32) << std::endl;
    }

    std::cout << std::endl;

  }

  // Helpful information at the bottom of the help output, to point the user to
  // citations and better documentation (if necessary).  See ticket #201.
  std::cout << HyphenateString("For further information, including relevant "
      "papers, citations, and theory, consult the documentation found at "
      "http://www.mlpack.org or included with your distribution of mlpack.", 0)
      << std::endl;
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
  if (doc != &emptyProgramDoc)
    GetSingleton().doc = doc;
}

/**
 * Checks that all parameters specified as required have been specified on the
 * command line.  If they havent, prints an error message and kills the program.
 */
void CLI::RequiredOptions()
{
  po::variables_map& vmap = GetSingleton().vmap;
  std::list<std::string> rOpt = GetSingleton().requiredOptions;

  // Now, warn the user if they missed any required options.
  std::list<std::string>::iterator iter;
  for (iter = rOpt.begin(); iter != rOpt.end(); ++iter)
  {
    std::string str = *iter;
    if (!vmap.count(str))
    { // If a required option isn't there...
      Log::Fatal << "Required option --" << str << " is undefined."
          << std::endl;
    }
  }
}

/**
 * Parses the values given on the command line, overriding any default values.
 */
void CLI::UpdateGmap()
{
  gmap_t& gmap = GetSingleton().globalValues;
  po::variables_map& vmap = GetSingleton().vmap;

  // Iterate through vmap, and overwrite default values with anything found on
  // command line.
  po::variables_map::iterator i;
  for (i = vmap.begin(); i != vmap.end(); ++i)
  {
    ParamData param;
    if (gmap.count(i->first)) // We need to preserve certain data
      param = gmap[i->first];

    param.value = vmap[i->first].value();
    param.wasPassed = true;
    gmap[i->first] = param;
  }
}

// Add help parameter.
PARAM_FLAG("help", "Default help info.", "h");
PARAM_STRING("info", "Get help on a specific module or option.", "", "");
PARAM_FLAG("verbose", "Display informational messages and the full list of "
    "parameters and timers at the end of execution.", "v");
PARAM_FLAG("version", "Display the version of mlpack.", "V");

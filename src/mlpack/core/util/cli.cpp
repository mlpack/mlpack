/**
 * @file cli.cpp
 * @author Matthew Amidon
 *
 * Implementation of the CLI module for parsing parameters.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <list>
#include <boost/program_options.hpp>
#include <boost/scoped_ptr.hpp>
#include <iostream>

#include "cli.hpp"
#include "log.hpp"

#include <mlpack/core/data/load.hpp>
#include <mlpack/core/data/save.hpp>

using namespace mlpack;
using namespace mlpack::util;

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

// Private copy constructor; don't want copies floating around.
CLI::CLI(const CLI& other) : desc(other.desc),
    didParse(false), doc(&emptyProgramDoc)
{
  return;
}

CLI::~CLI()
{
  // We only need to print output parameters if we're not printing any help or
  // anything like that.
  if (!HasParam("help") && !HasParam("info"))
  {
    // Save any output matrices.
    std::list<std::string>::const_iterator it = outputOptions.begin();
    while (it != outputOptions.end())
    {
      ParamData& d = parameters[*it];

      // It seems there is not a better way to do this, unfortunately.
      if (d.tname == TYPENAME(arma::mat))
      {
        arma::mat& output = *boost::any_cast<arma::mat>(&d.mappedValue);
        const std::string& filename = *boost::any_cast<std::string>(&d.value);

        if (output.n_elem > 0 && filename != "")
          data::Save(filename, output, false, !d.noTranspose);
      }
      else if (d.tname == TYPENAME(arma::Mat<size_t>))
      {
        arma::Mat<size_t>& output =
            *boost::any_cast<arma::Mat<size_t>>(&d.mappedValue);
        const std::string& filename = *boost::any_cast<std::string>(&d.value);

        if (output.n_elem > 0 && filename != "")
          data::Save(filename, output, false, !d.noTranspose);
      }

      ++it;
    }

    // Now print any output options.
    it = outputOptions.begin();
    while (it != outputOptions.end())
    {
      ParamData& d = parameters[*it];

      if (d.tname != TYPENAME(arma::mat) &&
          d.tname != TYPENAME(arma::Mat<size_t>))
      {
        // Don't print any output options with "_file" in the name.
        if (d.name.substr(d.name.length() - 5, 5) == "_file")
        {
          ++it;
          continue;
        }

        // Now, we must print it, so figure out what the type is.
        if (d.tname == TYPENAME(std::string))
        {
          std::string value = GetParam<std::string>(d.name);
          std::cout << d.name << ": " << value << std::endl;
        }
        else if (d.tname == TYPENAME(int))
        {
          int value = GetParam<int>(d.name);
          std::cout << d.name << ": " << value << std::endl;
        }
        else if (d.tname == TYPENAME(double))
        {
          double value = GetParam<double>(d.name);
          std::cout << d.name << ": " << value << std::endl;
        }
        else
        {
          std::cout << d.name << ": unknown data type" << std::endl;
        }
      }

      ++it;
    }
  }

  // Terminate the program timers.
  std::map<std::string, std::chrono::microseconds>::iterator it2;
  for (it2 = timer.GetAllTimers().begin(); it2 != timer.GetAllTimers().end();
       ++it2)
  {
    std::string i = (*it2).first;
    if (timer.GetState(i) == 1)
      Timer::Stop(i);
  }

  // Did the user ask for verbose output?  If so we need to print everything.
  // But only if the user did not ask for help or info.
  if (HasParam("verbose") && !HasParam("help") && !HasParam("info"))
  {
    Log::Info << std::endl << "Execution parameters:" << std::endl;

    std::map<std::string, ParamData>::iterator iter = parameters.begin();

    // Print out all the values.
    while (iter != parameters.end())
    {
      std::string key = iter->second.boostName;

      Log::Info << "  " << key << ": ";

      // Now, figure out what type it is, and print it.
      // We can handle strings, ints, bools, floats, doubles.
      util::ParamData& data = iter->second;
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
      else if (data.tname == TYPENAME(arma::mat) ||
               data.tname == TYPENAME(arma::Mat<size_t>))
      {
        // For matrix parameters, print the name of the file.
        std::string value = *boost::any_cast<std::string>(&data.value);
        if (value == "")
          Log::Info << "\"\"";
        Log::Info << value;
      }
      else
      {
        // We don't know how to print this.
        Log::Info << "(Unknown data type - " << data.tname << ")";
      }

      Log::Info << std::endl;
      ++iter;
    }
    Log::Info << std::endl;

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
  // may not have wanted it there, such as in Boost unit tests.
  if (didParse)
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
  std::string usedKey = key;
  po::variables_map vmap = GetSingleton().vmap;
  std::map<std::string, util::ParamData> parameters = GetSingleton().parameters;

  if (!parameters.count(key))
  {
    // Check any aliases, but only after we are sure the actual option as given
    // does not exist.
    if (key.length() == 1 && GetSingleton().aliases.count(key[0]))
      usedKey = GetSingleton().aliases[key[0]];

    if (!parameters.count(usedKey))
      Log::Fatal << "Parameter '--" << key << "' does not exist in this "
          << "program." << std::endl;
  }

  return (vmap.count(parameters[usedKey].boostName) > 0);
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

  // Parse the command line, place the options & values into vmap.
  try
  {
    // Get the basic_parsed_options.
    po::basic_parsed_options<char> bpo(
        po::parse_command_line(argc, line, desc));

    // Iterate over all the program_options, looking for duplicate parameters.
    // If we find any, remove the duplicates.  Note that vector options can have
    // duplicates so we check for those with max_tokens().
    for (unsigned int i = 0; i < bpo.options.size(); i++)
    {
      for (unsigned int j = i + 1; j < bpo.options.size(); j++)
      {
        if ((bpo.options[i].string_key == bpo.options[j].string_key) &&
            (desc.find(bpo.options[i].string_key, false).
                semantic()->max_tokens() <= 1))
        {
          // If a duplicate is found, check to see if either one has a value.
          if (bpo.options[i].value.size() == 0 &&
              bpo.options[j].value.size() == 0)
          {
            // If neither has a value, consider it a duplicate flag and remove
            // the duplicate. It's important to not break out of this loop
            // because there might be another duplicate later on in the vector.
            bpo.options.erase(bpo.options.begin() + j);
            --j;
          }
          else
          {
            // If one or both has a value, produce an error and politely
            // terminate. We pull the name from the original_tokens, rather than
            // from the string_key, because the string_key is the parameter
            // after aliases have been expanded.
            Log::Fatal << "\"" << bpo.options[j].original_tokens[0] << "\""
                << " is defined multiple times." << std::endl;
          }
        }
      }
    }

    // Record the basic_parsed_options
    po::store(bpo, vmap);
  }
  catch (std::exception& ex)
  {
    Log::Fatal << "Caught exception from parsing command line:\t";
    Log::Fatal << ex.what() << std::endl;
  }

  // Flush the buffer, make sure changes are propagated to vmap.
  po::notify(vmap);

  // If the user specified any of the default options (--help, --version, or
  // --info), handle those.

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

  // Now, warn the user if they missed any required options.
  std::list<std::string>& rOpt = GetSingleton().requiredOptions;
  std::list<std::string>::iterator iter;
  std::map<std::string, util::ParamData>& parameters =
      GetSingleton().parameters;
  for (iter = rOpt.begin(); iter != rOpt.end(); ++iter)
  {
    const std::string& boostName = parameters[*iter].boostName;
    if (!vmap.count(parameters[*iter].boostName))
      Log::Fatal << "Required option --" << boostName << " is undefined."
          << std::endl;
  }

  // Iterate through vmap, and overwrite default values with anything found on
  // command line.
  po::variables_map::iterator i;
  for (i = vmap.begin(); i != vmap.end(); ++i)
  {
    // There is not a possibility of an unknown option, since
    // boost::program_options will throw an exception.  Because some names may
    // be mapped, we have to look through each ParamData object and get its
    // boost name.
    std::string identifier;
    std::map<std::string, util::ParamData>::const_iterator it =
        parameters.begin();
    while (it != parameters.end())
    {
      if (it->second.boostName == i->first)
      {
        identifier = it->first;
        break;
      }

      ++it;
    }

    util::ParamData& param = parameters[identifier];
    if (param.isFlag)
      param.value = boost::any(true);
    else
      param.value = vmap[i->first].value();
  }
}

/* Prints the descriptions of the current hierarchy. */
void CLI::PrintHelp(const std::string& param)
{
  std::string usedParam = param;
  std::map<std::string, util::ParamData>& parameters =
      GetSingleton().parameters;
  std::map<char, std::string>& aliases = GetSingleton().aliases;

  std::map<std::string, util::ParamData>::iterator iter;
  ProgramDoc docs = *GetSingleton().doc;

  // If we pass a single param, alias it if necessary.
  if (usedParam.length() == 1 && aliases.count(usedParam[0]))
    usedParam = aliases[usedParam[0]];

  // Do we only want to print out one value?
  if (usedParam != "" && parameters.count(usedParam))
  {
    util::ParamData& data = parameters[usedParam];
    std::string alias = (data.alias != '\0') ? " (-"
        + std::string(1, data.alias) + ")" : "";

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
    else if (data.tname == TYPENAME(arma::mat) ||
             data.tname == TYPENAME(arma::Mat<size_t>))
      type = " [string]"; // Since we take strings for matrices.

    // Now, print the descriptions.
    std::string fullDesc = "  --" + usedParam + alias + type + "  ";

    if (fullDesc.length() <= 32) // It all fits on one line.
      std::cout << fullDesc << std::string(32 - fullDesc.length(), ' ');
    else // We need multiple lines.
      std::cout << fullDesc << std::endl << std::string(32, ' ');

    std::cout << HyphenateString(data.desc, 32) << std::endl;
    return;
  }
  else if (usedParam != "")
  {
    // User passed a single variable, but it doesn't exist.
    std::cerr << "Parameter --" << usedParam << " does not exist."
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

  for (size_t pass = 0; pass < 3; ++pass)
  {
    bool printedHeader = false;

    // Print out the descriptions of everything else.
    for (iter = parameters.begin(); iter != parameters.end(); ++iter)
    {
      util::ParamData& data = iter->second;
      std::string key = data.boostName;
      std::string desc = data.desc;
      std::string alias = (iter->second.alias != '\0') ?
          std::string(1, iter->second.alias) : "";
      alias = alias.length() ? " (-" + alias + ")" : alias;

      // Filter un-printed options.
      if ((pass == 0) && !(data.required && data.input)) // Required input.
        continue;
      if ((pass == 1) && !(!data.required && data.input)) // Optional input.
        continue;
      if ((pass == 2) && data.input) // Output options only (always optional).
        continue;

      // For reverse compatibility: this can be removed when these options are
      // gone in mlpack 3.0.0.  We don't want to print the deprecated options.
      if (data.name == "inputFile")
        continue;

      if (!printedHeader)
      {
        printedHeader = true;
        if (pass == 0)
          std::cout << "Required input options:" << std::endl << std::endl;
        else if (pass == 1)
          std::cout << "Optional input options: " << std::endl << std::endl;
        else if (pass == 2)
          std::cout << "Optional output options: " << std::endl << std::endl;
      }

      if (pass >= 1) // Append default value to description.
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
        else if (data.tname == TYPENAME(arma::mat) ||
                 data.tname == TYPENAME(arma::Mat<size_t>))
          tmp << "'" << boost::any_cast<std::string>(data.value) << "'.";

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
      else if (data.tname == TYPENAME(arma::mat) ||
               data.tname == TYPENAME(arma::Mat<size_t>))
        type = " [string]";

      // Now, print the descriptions.
      std::string fullDesc = "  --" + key + alias + type + "  ";

      if (fullDesc.length() <= 32) // It all fits on one line.
        std::cout << fullDesc << std::string(32 - fullDesc.length(), ' ');
      else // We need multiple lines.
        std::cout << fullDesc << std::endl << std::string(32, ' ');

      std::cout << HyphenateString(desc, 32) << std::endl;
    }

    if (printedHeader)
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

// Add default parameters that are included in every program.
PARAM_FLAG("help", "Default help info.", "h");
PARAM_STRING_IN("info", "Get help on a specific module or option.", "", "");
PARAM_FLAG("verbose", "Display informational messages and the full list of "
    "parameters and timers at the end of execution.", "v");
PARAM_FLAG("version", "Display the version of mlpack.", "V");

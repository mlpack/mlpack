/**
 * @file bindings/cli/parse_command_line.hpp
 * @author Ryan Curtin
 * @author Matthew Amidon
 *
 * Parse the command line options.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_CLI_PARSE_COMMAND_LINE_HPP
#define MLPACK_BINDINGS_CLI_PARSE_COMMAND_LINE_HPP

#include <mlpack/core.hpp>
#include "print_help.hpp"

#include "third_party/CLI/CLI11.hpp"

namespace mlpack {
namespace bindings {
namespace cli {

// Add default parameters that are included in every program.
PARAM_FLAG("help", "Default help info.", "h");
PARAM_STRING_IN("info", "Print help on a specific option.", "", "");
PARAM_FLAG("verbose", "Display informational messages and the full list of "
    "parameters and timers at the end of execution.", "v");
PARAM_FLAG("version", "Display the version of mlpack.", "V");

/**
 * Parse the command line, setting all of the options inside of the CLI object
 * to their appropriate given values.
 */
void ParseCommandLine(int argc, char** argv)
{
  // First, we need to build the CLI11 variables for parsing.
  CLI::App app;
  app.set_help_flag();

  // Go through list of options in order to add them.
  std::map<std::string, util::ParamData>& parameters = IO::Parameters();
  using ItType = std::map<std::string, util::ParamData>::iterator;

  for (ItType it = parameters.begin(); it != parameters.end(); ++it)
  {
    // Add the parameter to desc.
    util::ParamData& d = it->second;
    IO::GetSingleton().functionMap[d.tname]["AddToCLI11"](d, NULL, (void*)
        &app);
  }

  // Mark that we did parsing.
  IO::GetSingleton().didParse = true;

  // Parse the command line, then place the values in the right place.
  try
  {
    try
    {
      app.parse(argc, argv);
    }
    catch (const CLI::ArgumentMismatch& err)
    {
      Log::Fatal << "An option is defined multiple times: "
                 << app.exit(err) << std::endl;
    }
    catch (const CLI::OptionNotFound& onf)
    {
      Log::Fatal << "Required option --" << app.exit(onf) << "!" << std::endl;
    }
    catch (const CLI::ParseError& pe)
    {
      Log::Fatal << app.exit(pe) << std::endl;
    }
  }
  catch (std::exception& ex)
  {
    Log::Fatal << "Caught exception from parsing command line: "
      << ex.what() << std::endl;
  }

  // If the user specified any of the default options (--help, --version, or
  // --info), handle those.

  // --version is prioritized over --help.
  if (IO::HasParam("version"))
  {
    std::cout << IO::GetSingleton().ProgramName() << ": part of "
        << util::GetVersion() << "." << std::endl;
    exit(0); // Don't do anything else.
  }

  // Default help message.
  if (IO::HasParam("help"))
  {
    Log::Info.ignoreInput = false;
    PrintHelp();
    exit(0); // The user doesn't want to run the program, he wants help.
  }

  // Info on a specific parameter.
  if (IO::HasParam("info"))
  {
    Log::Info.ignoreInput = false;
    std::string str = IO::GetParam<std::string>("info");

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

  // Print whether or not we have debugging symbols.  This won't show anything
  // if we have not compiled in debugging mode.
  Log::Debug << "Compiled with debugging symbols." << std::endl;

  if (IO::HasParam("verbose"))
  {
    // Give [INFO ] output.
    Log::Info.ignoreInput = false;
  }

  // Now, issue an error if we forgot any required options.
  for (std::map<std::string, util::ParamData>::const_iterator iter =
       parameters.begin(); iter != parameters.end(); ++iter)
  {
    util::ParamData d = iter->second;
    if (d.required)
    {
      // CLI11 expects the parameter name to have "--" prepended.
      std::string cliName;
      IO::GetSingleton().functionMap[d.tname]["MapParameterName"](d, NULL,
          (void*) &cliName);
      cliName = "--" + cliName;

      if (!app.count(cliName))
      {
        Log::Fatal << "Required option " << cliName << " is undefined."
            << std::endl;
      }
    }
  }
}

} // namespace cli
} // namespace bindings
} // namespace mlpack

#endif

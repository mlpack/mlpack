/**
 * @file end_program.hpp
 * @author Ryan Curtin
 * @author Matthew Amidon
 *
 * Terminate the program; handle --verbose option; print output parameters.
 */
#ifndef MLPACK_BINDINGS_CLI_END_PROGRAM_HPP
#define MLPACK_BINDINGS_CLI_END_PROGRAM_HPP

#include <mlpack/core/util/cli.hpp>

namespace mlpack {
namespace bindings {
namespace cli {

/**
 * Handle command-line program termination.  If --help or --info was passed, we
 * won't make it here, so we don't have to write any contingencies for that.
 */
inline void EndProgram()
{
  // Stop the CLI timers.
  CLI::StopTimers();

  // Print any output.
  const std::map<std::string, util::ParamData>& parameters = CLI::Parameters();
  std::map<std::string, util::ParamData>::const_iterator it =
      parameters.begin();
  while (it != parameters.end())
  {
    const util::ParamData& d = it->second;
    if (!d.input)
      CLI::GetSingleton().functionMap[d.tname]["OutputParam"](d, NULL, NULL);

    ++it;
  }

  if (CLI::HasParam("verbose"))
  {
    Log::Info << std::endl << "Execution parameters:" << std::endl;

    // Print out all the values.
    it = parameters.begin();
    while (it != parameters.end())
    {
      // Now, figure out what type it is, and print it.
      // We can handle strings, ints, bools, floats, doubles.
      const util::ParamData& data = it->second;
      std::string boostName;
      CLI::GetSingleton().functionMap[data.tname]["MapParameterName"](data,
          NULL, (void*) &boostName);
      Log::Info << "  " << boostName << ": ";

      std::string printableParam;
      CLI::GetSingleton().functionMap[data.tname]["GetPrintableParam"](data,
          NULL, (void*) &printableParam);
      Log::Info << printableParam << std::endl;

      ++it;
    }

    Log::Info << "Program timers:" << std::endl;
    std::map<std::string, std::chrono::microseconds>::iterator it2;
    for (it2 = CLI::GetSingleton().timer.GetAllTimers().begin();
         it2 != CLI::GetSingleton().timer.GetAllTimers().end(); ++it2)
    {
      std::string i = (*it2).first;
      Log::Info << "  " << i << ": ";
      CLI::GetSingleton().timer.PrintTimer((*it2).first);
    }
  }
}

} // namespace cli
} // namespace bindings
} // namespace mlpack

#endif

/**
 * @file end_program.hpp
 * @author Ryan Curtin
 * @author Matthew Amidon
 *
 * Terminate the program; handle --verbose option; print output parameters.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
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
  CLI::GetSingleton().timer.StopAllTimers();

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
      // We can handle strings, ints, bools, doubles.
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
    for (auto it2 : CLI::GetSingleton().timer.GetAllTimers())
    {
      Log::Info << "  " << it2.first << ": ";
      CLI::GetSingleton().timer.PrintTimer(it2.first);
    }
  }

  // Lastly clean up any memory.  If we are holding any pointers, then we "own"
  // them.  But we may hold the same pointer twice, so we have to be careful to
  // not delete it multiple times.
  std::unordered_map<void*, const util::ParamData*> memoryAddresses;
  it = parameters.begin();
  while (it != parameters.end())
  {
    const util::ParamData& data = it->second;

    void* result;
    CLI::GetSingleton().functionMap[data.tname]["GetAllocatedMemory"](data,
        NULL, (void*) &result);
    if (result != NULL && memoryAddresses.count(result) == 0)
      memoryAddresses[result] = &data;

    ++it;
  }

  // Now we have all the unique addresses that need to be deleted.
  std::unordered_map<void*, const util::ParamData*>::const_iterator it2;
  it2 = memoryAddresses.begin();
  while (it2 != memoryAddresses.end())
  {
    const util::ParamData& data = *(it2->second);

    CLI::GetSingleton().functionMap[data.tname]["DeleteAllocatedMemory"](data,
        NULL, NULL);

    ++it2;
  }
}

} // namespace cli
} // namespace bindings
} // namespace mlpack

#endif

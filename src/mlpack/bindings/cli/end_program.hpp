/**
 * @file bindings/cli/end_program.hpp
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

#include <mlpack/core/util/io.hpp>

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
  IO::GetSingleton().timer.StopAllTimers();

  // Print any output.
  std::map<std::string, util::ParamData>& parameters = IO::Parameters();
  for (auto& it : parameters)
  {
    util::ParamData& d = it.second;
    if (!d.input)
      IO::GetSingleton().functionMap[d.tname]["OutputParam"](d, NULL, NULL);
  }

  if (IO::HasParam("verbose"))
  {
    Log::Info << std::endl << "Execution parameters:" << std::endl;

    // Print out all the values.
    for (auto& it : parameters)
    {
      // Now, figure out what type it is, and print it.
      // We can handle strings, ints, bools, doubles.
      util::ParamData& data = it.second;
      std::string cliName;
      IO::GetSingleton().functionMap[data.tname]["MapParameterName"](data,
          NULL, (void*) &cliName);
      Log::Info << "  " << cliName << ": ";

      std::string printableParam;
      IO::GetSingleton().functionMap[data.tname]["GetPrintableParam"](data,
          NULL, (void*) &printableParam);
      Log::Info << printableParam << std::endl;
    }

    Log::Info << "Program timers:" << std::endl;
    for (auto& it2 : IO::GetSingleton().timer.GetAllTimers())
    {
      Log::Info << "  " << it2.first << ": ";
      IO::GetSingleton().timer.PrintTimer(it2.first);
    }
  }

  // Lastly clean up any memory.  If we are holding any pointers, then we "own"
  // them.  But we may hold the same pointer twice, so we have to be careful to
  // not delete it multiple times.
  std::unordered_map<void*, util::ParamData*> memoryAddresses;
  std::cout << "deleting used memory\n";
  for (auto& it : parameters)
  {
    util::ParamData& data = it.second;

    void* result;
    IO::GetSingleton().functionMap[data.tname]["GetAllocatedMemory"](data,
        NULL, (void*) &result);
    if (result != NULL)
      std::cout << "got memory address " << result << " for param " << data.name
<< "\n";
    if (result != NULL && memoryAddresses.count(result) == 0)
    {
      std::cout << "pushed to results!\n";
      memoryAddresses[result] = &data;
    }
  }

  // Now we have all the unique addresses that need to be deleted.
  std::unordered_map<void*, util::ParamData*>::const_iterator it2;
  it2 = memoryAddresses.begin();
  while (it2 != memoryAddresses.end())
  {
    util::ParamData& data = *(it2->second);
    std::cout << "delete allocated memory, param " << data.name << "\n";

    IO::GetSingleton().functionMap[data.tname]["DeleteAllocatedMemory"](data,
        NULL, NULL);

    ++it2;
  }
}

} // namespace cli
} // namespace bindings
} // namespace mlpack

#endif

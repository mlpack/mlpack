/**
 * @file clean_memory.cpp
 * @author Ryan Curtin
 *
 * Delete any pointers held by the CLI object.
 */
#include "clean_memory.hpp"

#include <mlpack/core.hpp>

namespace mlpack {
namespace bindings {
namespace tests {

/**
 * Delete any pointers held by the CLI object.
 */
void CleanMemory()
{
  // If we are holding any pointers, then we "own" them.  But we may hold the
  // same pointer twice, so we have to be careful to not delete it multiple
  // times.
  std::unordered_map<void*, const util::ParamData*> memoryAddresses;
  auto it = CLI::Parameters().begin();
  while (it != CLI::Parameters().end())
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

} // namespace tests
} // namespace bindings
} // namespace mlpack

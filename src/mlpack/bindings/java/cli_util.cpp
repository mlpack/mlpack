#include "cli_util.hpp"

namespace mlpack {
namespace util {

void SetPassed(const char* name)
{
  CLI::SetPassed(name);
}

void RestoreSettings(const char* name)
{
  CLI::RestoreSettings(name);
}

}
}

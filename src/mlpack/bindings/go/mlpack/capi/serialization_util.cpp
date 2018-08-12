#include "serialization_util.hpp"

namespace mlpack {
namespace bindings {
namespace go {

extern "C" const char *MLPACK_SerializeOut(const char *ptr, const char *name)
{
  std::string val = SerializeOut(ptr, name);
  char *cstr = const_cast<char*>(val.c_str());
  return cstr;
}

extern "C" void MLPACK_SerializeIn(const char *ptr, const char *str, const char *name)
{
  SerializeIn(ptr, str, name);
}

} // namespace go
} // namespace bindings
} // namespace mlpack

#include "cli_util.h"
#include "cli_util.hpp"
#include <mlpack/core/util/cli.hpp>

namespace mlpack {

extern "C" void MLPACK_ResetTimers()
{
  CLI::GetSingleton().timer.Reset();
}

extern "C" void MLPACK_EnableTimers()
{
  Timer::EnableTiming();
}

extern "C" void MLPACK_DisableBacktrace()
{
  Log::Fatal.backtrace = false;
}

extern "C" void MLPACK_EnableVerbose()
{
  Log::Info.ignoreInput = false;
}

extern "C" void MLPACK_DisableVerbose()
{
  Log::Info.ignoreInput = true;
}

extern "C" void MLPACK_ClearSettings()
{
  CLI::ClearSettings();
}

extern "C" void MLPACK_SetPassed(const char *name)
{
  CLI::SetPassed(name);
}

extern "C" void MLPACK_RestoreSettings(const char *name)
{
  CLI::RestoreSettings(name);
}

extern "C" void MLPACK_SetParamDouble(const char *identifier, double value)
{
  util::SetParam(identifier, value);
}

extern "C" void MLPACK_SetParamInt(const char *identifier, int value)
{
  util::SetParam(identifier, value);
}

extern "C" void MLPACK_SetParamFloat(const char *identifier, float value)
{
  util::SetParam(identifier, value);
}

extern "C" void MLPACK_SetParamBool(const char *identifier, bool value)
{
  util::SetParam(identifier, value);
}

extern "C" void MLPACK_SetParamString(const char *identifier, const char *value)
{
  std::string val;
  val.assign(value);
  util::SetParam(identifier, val);
}

extern "C" void MLPACK_SetParamPtr(const char *identifier, const double *ptr, const bool copy)
{
  util::SetParamPtr(identifier, ptr, copy);
}


extern "C" bool MLPACK_HasParam(const char *identifier)
{
  return CLI::HasParam(identifier);
}

extern "C" char *MLPACK_GetParamString(const char *identifier)
{
  std::string val = CLI::GetParam<std::string>(identifier);
  char *cstr = const_cast<char*>(val.c_str());
  return cstr;
}

extern "C" double MLPACK_GetParamDouble(const char *identifier)
{
  double val = CLI::GetParam<double>(identifier);
  return val;
}

extern "C" int MLPACK_GetParamInt(const char *identifier)
{
  int val = CLI::GetParam<int>(identifier);
  return val;
}

extern "C" bool MLPACK_GetParamBool(const char *identifier)
{
  bool val = CLI::GetParam<bool>(identifier);
  return val;
}

extern "C" void *MLPACK_GetVecIntPtr(const char *identifier)
{
  // std::vector<int> vec = CLI::GetParam<std::vector<int>>(identifier);
  // return vec.get_allocator();
}

extern "C" void *MLPACK_GetVecStringPtr(const char *identifier)
{
  // std::vector<std::string> vec = CLI::GetParam<std::vector<std::string>>(identifier);
  // return vec.get_allocator();
}

extern "C" int MLPACK_VecIntSize(const char *identifier)
{
  std::vector<int> output = CLI::GetParam<std::vector<int>>(identifier);
  return output.size();
}

extern "C" int MLPACK_VecStringSize(const char *identifier)
{
  std::vector<std::string> output = CLI::GetParam<std::vector<std::string>>(identifier);
  return output.size();
}
} // namespace mlpack

/**
 * @file julia_util.cpp
 * @author Ryan Curtin
 *
 * Implementations of Julia binding functionality.
 */
#include "julia_util.h"
#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/cli.hpp>

using namespace mlpack;

extern "C"
{

/**
 * Call CLI::RestoreSettings() for a given program name.
 */
void CLI_RestoreSettings(const char* programName)
{
  CLI::RestoreSettings(programName);
}

/**
 * Call CLI::SetParam<int>().
 */
void CLI_SetParamInt(const char* paramName, int paramValue)
{
  CLI::GetParam<int>(paramName) = paramValue;
  CLI::SetPassed(paramName);
}

/**
 * Call CLI::SetParam<double>().
 */
void CLI_SetParamDouble(const char* paramName, double paramValue)
{
  CLI::GetParam<double>(paramName) = paramValue;
  CLI::SetPassed(paramName);
}

/**
 * Call CLI::SetParam<std::string>().
 */
void CLI_SetParamString(const char* paramName, const char* paramValue)
{
  CLI::GetParam<std::string>(paramName) = paramValue;
  CLI::SetPassed(paramName);
}

/**
 * Call CLI::SetParam<bool>().
 */
void CLI_SetParamBool(const char* paramName, bool paramValue)
{
  CLI::GetParam<bool>(paramName) = paramValue;
  CLI::SetPassed(paramName);
}

/**
 * Call CLI::SetParam<arma::mat>().
 */
void CLI_SetParamMat(const char* paramName,
                     double* memptr,
                     const size_t rows,
                     const size_t cols,
                     const bool pointsAsRows)
{
  // Create the matrix as an alias.
  arma::mat m(memptr, rows, cols, false, true);
  CLI::GetParam<arma::mat>(paramName) = pointsAsRows ? m.t() : std::move(m);
  CLI::SetPassed(paramName);
}

/**
 * Call CLI::SetParam<arma::Mat<size_t>>().
 */
void CLI_SetParamUmat(const char* paramName,
                      size_t* memptr,
                      const size_t rows,
                      const size_t cols,
                      const bool pointsAsRows)
{
  // Create the matrix as an alias.
  arma::Mat<size_t> m(memptr, rows, cols, false, true);
  CLI::GetParam<arma::Mat<size_t>>(paramName) = pointsAsRows ? m.t() :
      std::move(m);
  CLI::SetPassed(paramName);
}

/**
 * Get the number of rows in a matrix parameter.
 */
size_t CLI_GetParamMatRows(const char* paramName)
{
  return CLI::GetParam<arma::mat>(paramName).n_rows;
}

/**
 * Get the number of columns in a matrix parameter.
 */
size_t CLI_GetParamMatCols(const char* paramName)
{
  return CLI::GetParam<arma::mat>(paramName).n_cols;
}

/**
 * Get the memory pointer for a matrix parameter.
 * Note that this will assume that whatever is calling will take ownership of
 * the memory!
 */
double* CLI_GetParamMat(const char* paramName)
{
  arma::access::rw(CLI::GetParam<arma::mat>(paramName).mem_state) = 1;
  return CLI::GetParam<arma::mat>(paramName).memptr();
}

/**
 * Enable verbose output.
 */
void CLI_EnableVerbose()
{
  Log::Info.ignoreInput = false;
}

/**
 * Disable verbose output.
 */
void CLI_DisableVerbose()
{
  Log::Info.ignoreInput = true;
}

/**
 * Reset the state of all timers.
 */
void CLI_ResetTimers()
{
  CLI::GetSingleton().timer.Reset();
}

/**
 * Set an argument as passed to the CLI object.
 */
void CLI_SetPassed(const char* paramName)
{
  CLI::SetPassed(paramName);
}

}

#include "julia_pca.h"

#define BINDING_TYPE BINDING_TYPE_PYX
#include <mlpack/methods/perceptron/perceptron_main.cpp>

static void perceptron_mlpackMain()
{
  mlpackMain();
}

extern "C"
{

void perceptron()
{
  perceptron_mlpackMain();
}

void loadSymbols()
{
  // Do nothing.
}

void* CLI_GetParamPerceptronModelPtr(const char* paramName)
{
  return (void*) CLI::GetParam<PerceptronModel*>(paramName);
}

void CLI_SetParamPerceptronModelPtr(const char* paramName, void* ptr)
{
  CLI::GetParam<PerceptronModel*>(paramName) = (PerceptronModel*) ptr;
  CLI::SetPassed(paramName);
}

}

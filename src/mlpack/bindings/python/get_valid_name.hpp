#ifndef MLPACK_BINDINGS_PYTHON_GET_VALID_NAME_HPP
#define MLPACK_BINDINGS_PYTHON_GET_VALID_NAME_HPP

#include <string>
using namespace std;

namespace mlpack {
namespace bindings {
namespace python {

inline string GetValidName(const string& paramName)
{
  string correctParamName = paramName;

  if(paramName == "lambda") correctParamName = "lambda_";
  else if(paramName == "input") correctParamName = "input_";
  else correctParamName = paramName;

  return correctParamName;
}

} // python
} // bindings
} // mlpack

#endif

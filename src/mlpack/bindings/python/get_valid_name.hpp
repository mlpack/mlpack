#ifndef MLPACK_BINDINGS_PYTHON_GET_VALID_NAME_HPP
#define MLPACK_BINDINGS_PYTHON_GET_VALID_NAME_HPP

#include <string>
using namespace std;

namespace mlpack {
namespace bindings {
namespace python {

string GetValidName(const string& paramName)
{
  if(paramName == "lambda") return "lambda_";
  else if(paramName == "input") return "input_";
  else return paramName;
}

} // python
} // bindings
} // mlpack

#endif

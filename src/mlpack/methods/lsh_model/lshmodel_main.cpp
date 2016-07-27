#include <mlpack/core.hpp>

#include "lshmodel.hpp"
using namespace mlpack;

PROGRAM_INFO("LSH Model (TODO: Complete this)", "");

PARAM_STRING_IN("reference_file", "File containing the dataset", "r", "");
PARAM_STRING_OUT("output_model_file", "File to save trained LSH model to", "m");

int main(int argc, char* argv[])
{
  std::cout << "Hello!" << std::endl;
  return 0;
}

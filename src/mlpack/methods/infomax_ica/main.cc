/**
 * @file main.cc
 * @author Chip Mappus
 *
 * main for using infomax ICA method.
 */
#include <mlpack/core.h>
#include "infomax_ica.h"

PARAM_STRING_REQ("data", "The name of the file containing mixture data.",
    "info");

using namespace mlpack;
using namespace infomax_ica;

int main(int argc, char *argv[]) {
  CLI::ParseCommandLine(argc, argv);

  const char *data_file_name = CLI::GetParam<std::string>("info/data").c_str();
  double lambda = CLI::GetParam<double>("info/lambda");
  int B = CLI::GetParam<int>("info/B");
  double epsilon = CLI::GetParam<double>("info/epsilon");

  arma::mat dataset;
  dataset.load(data_file_name);

  InfomaxICA ica(lambda, B, epsilon);
  ica.applyICA(dataset);

  arma::mat west;
  ica.getUnmixing(west);
  //ica->displayMatrix(west);
}

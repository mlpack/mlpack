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

int main(int argc, char *argv[]) {
  IO::ParseCommandLine(argc, argv);

  const char *data_file_name = IO::GetParam<std::string>("info/data").c_str();
  double lambda = IO::GetParam<double>("info/lambda");
  int B = IO::GetParam<int>("info/B");
  double epsilon = IO::GetParam<double>("info/epsilon");

  arma::mat dataset;
  data::Load(data_file_name, dataset);

  InfomaxICA ica(lambda, B, epsilon);
  ica.applyICA(dataset);

  arma::mat west;
  ica.getUnmixing(west);
  //ica->displayMatrix(west);
}

/**
 * @author Michael Fox
 * @file gmm_convert_main.cpp
 *
 * This program converts an older GMM XML file to the new format.
 */
#include <mlpack/core.hpp>
#include "gmm.hpp"

using namespace mlpack;
using namespace mlpack::gmm;
using namespace mlpack::util;
using namespace std;

PROGRAM_INFO("Gaussian Mixture Model (GMM) file converter",
    "This program takes a fitted GMM XML file from older MLPACK versions (1.0.9"
    " and older) and converts it to the current format.");

PARAM_STRING_REQ("input_file", "File containing the fitted model.", "i");
PARAM_STRING("output_file", "The file to write the model to (as XML).", "o",
    "gmm.xml");

int main(int argc, char* argv[])
{
  CLI::ParseCommandLine(argc, argv);
  string inputFile = CLI::GetParam<string>("input_file");
  SaveRestoreUtility load;

  if (!load.ReadFile(inputFile))
    Log::Fatal << " Could not read file '" << inputFile << "'!\n";

  size_t gaussians, dimensionality;
  load.LoadParameter(gaussians, "gaussians");
  load.LoadParameter(dimensionality, "dimensionality");
  GMM<> gmm(gaussians, dimensionality);

  load.LoadParameter(gmm.Weights(), "weights");

  // We need to do a little error checking here.
  if (gmm.Weights().n_elem != gmm.Gaussians())
  {
    Log::Fatal << "GMM::Load('" << inputFile << "'): file reports "
        << gmm.Gaussians() << " gaussians but weights vector only contains "
        << gmm.Weights().n_elem << " elements!" << endl;
  }

  for (size_t i = 0; i < gaussians; ++i)
  {
    stringstream o;
    arma::mat covariance;
    o << i;
    string meanName = "mean" + o.str();
    string covName = "covariance" + o.str();

    load.LoadParameter(gmm.Component(i).Mean(), meanName);
    load.LoadParameter(covariance, covName);
    gmm.Component(i).Covariance(std::move(covariance));
  }

  gmm.Save(CLI::GetParam<string>("output_file"));
}

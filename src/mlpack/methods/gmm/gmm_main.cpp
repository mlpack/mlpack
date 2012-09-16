/**
 * @author Parikshit Ram (pram@cc.gatech.edu)
 * @file gmm_main.cpp
 *
 * This program trains a mixture of Gaussians on a given data matrix.
 *
 * This file is part of MLPACK 1.0.3.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */
#include "gmm.hpp"

PROGRAM_INFO("Gaussian Mixture Model (GMM) Training",
    "This program takes a parametric estimate of a Gaussian mixture model (GMM)"
    " using the EM algorithm to find the maximum likelihood estimate.  The "
    "model is saved to an XML file, which contains information about each "
    "Gaussian.");

PARAM_STRING_REQ("input_file", "File containing the data on which the model "
    "will be fit.", "i");
PARAM_INT("gaussians", "Number of Gaussians in the GMM.", "g", 1);
PARAM_STRING("output_file", "The file to write the trained GMM parameters into "
    "(as XML).", "o", "gmm.xml");
PARAM_INT("seed", "Random seed.  If 0, 'std::time(NULL)' is used.", "s", 0);

using namespace mlpack;
using namespace mlpack::gmm;
using namespace mlpack::utilities;

int main(int argc, char* argv[])
{
  CLI::ParseCommandLine(argc, argv);

  // Check parameters and load data.
  if (CLI::GetParam<int>("seed") != 0)
    math::RandomSeed((size_t) CLI::GetParam<int>("seed"));
  else
    math::RandomSeed((size_t) std::time(NULL));

  arma::mat dataPoints;
  data::Load(CLI::GetParam<std::string>("input_file").c_str(), dataPoints,
      true);

  int gaussians = CLI::GetParam<int>("gaussians");
  if (gaussians <= 0)
  {
    Log::Fatal << "Invalid number of Gaussians (" << gaussians << "); must "
        "be greater than or equal to 1." << std::endl;
  }

  // Calculate mixture of Gaussians.
  GMM<> gmm(size_t(gaussians), dataPoints.n_rows);

  ////// Computing the parameters of the model using the EM algorithm //////
  Timer::Start("em");
  gmm.Estimate(dataPoints);
  Timer::Stop("em");

  ////// OUTPUT RESULTS //////
  mlpack::utilities::SaveRestoreUtility save;
  save.SaveParameter(gmm.Gaussians(), "gaussians");
  save.SaveParameter(gmm.Dimensionality(), "dimensionality");
  save.SaveParameter(trans(gmm.Weights()), "weights");
  for (size_t i = 0; i < gmm.Gaussians(); ++i)
  {
    // Generate names for the XML nodes.
    std::stringstream o;
    o << i;
    std::string meanName = "mean" + o.str();
    std::string covName = "covariance" + o.str();

    // Now save them.
    save.SaveParameter(trans(gmm.Means()[0]), meanName);
    save.SaveParameter(gmm.Covariances()[0], covName);
  }

  save.WriteFile(CLI::GetParam<std::string>("output_file").c_str());
}

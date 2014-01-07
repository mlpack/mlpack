/**
 * @file hmm_loglik_main.cpp
 * @author Ryan Curtin
 *
 * Compute the log-likelihood of a given sequence for a given HMM.
 *
 * This file is part of MLPACK 1.0.8.
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
#include <mlpack/core.hpp>

#include "hmm.hpp"
#include "hmm_util.hpp"

#include <mlpack/methods/gmm/gmm.hpp>

PROGRAM_INFO("Hidden Markov Model (HMM) Sequence Log-Likelihood", "This "
    "utility takes an already-trained HMM (--model_file) and evaluates the "
    "log-likelihood of a given sequence of observations (--input_file).  The "
    "computed log-likelihood is given directly to stdout.");

PARAM_STRING_REQ("input_file", "File containing observations,", "i");
PARAM_STRING_REQ("model_file", "File containing HMM (XML).", "m");

using namespace mlpack;
using namespace mlpack::hmm;
using namespace mlpack::distribution;
using namespace mlpack::util;
using namespace mlpack::gmm;
using namespace arma;
using namespace std;

int main(int argc, char** argv)
{
  // Parse command line options.
  CLI::ParseCommandLine(argc, argv);

  // Load observations.
  const string inputFile = CLI::GetParam<string>("input_file");
  const string modelFile = CLI::GetParam<string>("model_file");

  mat dataSeq;
  data::Load(inputFile, dataSeq, true);

  // Load model, but first we have to determine its type.
  SaveRestoreUtility sr;
  sr.ReadFile(modelFile);
  string type;
  sr.LoadParameter(type, "hmm_type");

  double loglik = 0;
  if (type == "discrete")
  {
    HMM<DiscreteDistribution> hmm(1, DiscreteDistribution(1));

    LoadHMM(hmm, sr);

    // Verify only one row in observations.
    if (dataSeq.n_cols == 1)
      dataSeq = trans(dataSeq);

    if (dataSeq.n_rows > 1)
      Log::Fatal << "Only one-dimensional discrete observations allowed for "
          << "discrete HMMs!" << endl;

    loglik = hmm.LogLikelihood(dataSeq);
  }
  else if (type == "gaussian")
  {
    HMM<GaussianDistribution> hmm(1, GaussianDistribution(1));

    LoadHMM(hmm, sr);

    // Verify correct dimensionality.
    if (dataSeq.n_rows != hmm.Emission()[0].Mean().n_elem)
      Log::Fatal << "Observation dimensionality (" << dataSeq.n_rows << ") "
          << "does not match HMM Gaussian dimensionality ("
          << hmm.Emission()[0].Mean().n_elem << ")!" << endl;

    loglik = hmm.LogLikelihood(dataSeq);
  }
  else if (type == "gmm")
  {
    HMM<GMM<> > hmm(1, GMM<>(1, 1));

    LoadHMM(hmm, sr);

    // Verify correct dimensionality.
    if (dataSeq.n_rows != hmm.Emission()[0].Dimensionality())
      Log::Fatal << "Observation dimensionality (" << dataSeq.n_rows << ") "
          << "does not match HMM Gaussian dimensionality ("
          << hmm.Emission()[0].Dimensionality() << ")!" << endl;

    loglik = hmm.LogLikelihood(dataSeq);
  }
  else
  {
    Log::Fatal << "Unknown HMM type '" << type << "' in file '" << modelFile
        << "'!" << endl;
  }

  cout << loglik << endl;
}

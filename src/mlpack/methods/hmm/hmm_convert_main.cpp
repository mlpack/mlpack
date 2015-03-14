/**
 * @file hmm_convert_main.cpp
 * @author Ryan Curtin
 * @author Michael Fox
 *
 * Convert an HMM (XML) file from older MLPACK versions to current format.
 */
#include <mlpack/core.hpp>

#include "hmm.hpp"
#include "hmm_util.hpp"

#include <mlpack/methods/gmm/gmm.hpp>


PROGRAM_INFO("Hidden Markov Model (HMM) File Converter", "This utility takes "
    "an already-trained HMM (--model_file) and converts it to the new format "
    "(--output_file).");

PARAM_STRING_REQ("model_file", "File containing HMM (XML).", "m");
PARAM_STRING("output_file", "File to save HMM (XML) to.", "o", "output.xml");

using namespace mlpack;
using namespace mlpack::hmm;
using namespace mlpack::distribution;
using namespace mlpack::util;
using namespace mlpack::gmm;
using namespace mlpack::math;
using namespace arma;
using namespace std;

int main(int argc, char** argv)
{
  // Parse command line options.
  CLI::ParseCommandLine(argc, argv);

  // Load model
  const string modelFile = CLI::GetParam<string>("model_file");

  // Load model, but first we have to determine its type.
  SaveRestoreUtility sr, sr2;
  sr.ReadFile(modelFile);
  string emissionType;
  sr.LoadParameter(emissionType, "hmm_type");

  mat observations;
  Col<size_t> sequence;
  if (emissionType == "discrete")
  {
    HMM<DiscreteDistribution> hmm(1, DiscreteDistribution(1));
    ConvertHMM(hmm, sr);
		hmm.Save(sr2);
  }
  else if (emissionType == "gaussian")
  {
    HMM<GaussianDistribution> hmm(1, GaussianDistribution(1));
    ConvertHMM(hmm, sr);
		hmm.Save(sr2);
  }
  else if (emissionType == "gmm")
  {
    HMM<GMM<> > hmm(1, GMM<>(1, 1));
    ConvertHMM(hmm, sr);
		hmm.Save(sr2);
  }
  else
  {
    Log::Fatal << "Unknown HMM type '" << emissionType << "' in file '" << modelFile
        << "'!" << endl;
  }

  // Save the converted model.
  const string outputFile = CLI::GetParam<string>("output_file");
  sr2.WriteFile(outputFile);

	return 0;
}

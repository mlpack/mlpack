/**
 * @file nmf_main.cpp
 * @author Mohan Rajendran
 *
 * Main executable to run NMF.
 */
#include <mlpack/core.hpp>

#include "nmf.hpp"

using namespace mlpack;
using namespace mlpack::nmf;
using namespace std;

// Document program.
PROGRAM_INFO("Non-negative Matrix Factorization", "This program performs the "
    "non-negative matrix factorization on the given vector. It will store the "
    "calculated factors in the reference matrix arguments supplied.");

// Parameters for program.
PARAM_STRING_REQ("input_file", "Input matrix to perform NMF on.", "i");
PARAM_STRING_REQ("W_output_file", "File to save the calculated W matrix to.",
    "w");
PARAM_STRING_REQ("H_output_file", "File to save the calculated H matrix to.",
    "h");
PARAM_INT_REQ("rank", "Rank of the factorization.", "r");
PARAM_INT("max_iterations", "Number of iterations before NMF terminates", 
    "m", 10000);
PARAM_DOUBLE("max_residue", "The maximum root mean square allowed below which "
    "the program termiates", "e", 1e-10);

int main(int argc, char** argv)
{
  // Parse commandline.
  CLI::ParseCommandLine(argc, argv);

  // Load input dataset.
  string inputFile = CLI::GetParam<string>("input_file");
  arma::mat V;
  data::Load(inputFile.c_str(), V);
  arma::mat W;
  arma::mat H;

  // Find out the rank of the factorization.
  size_t r = CLI::GetParam<int>("rank");
  if (r<1)
  {
    Log::Fatal << "The rank of the factorization cannot be less than 1. "
          << std::endl;
  }
  
  size_t maxiterations = CLI::GetParam<int>("max_iterations");
  double maxresidue = CLI::GetParam<double>("max_residue");

  // Perform NMF. 
  NMF<> nmf(maxiterations,maxresidue);
  Log::Info << "Performing NMF on the given matrix..." << endl;
  nmf.Apply(V,W,H,r);

  // Save results
  string outputFile = CLI::GetParam<string>("W_output_file");
  data::Save(outputFile, W);
  outputFile = CLI::GetParam<string>("H_output_file");
  data::Save(outputFile, H);
}

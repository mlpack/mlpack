/**
 * @file radical_main.cpp
 * @author Nishant Mehta
 *
 * Executable for RADICAL
 */

#include <mlpack/core.hpp>
#include <armadillo>
#include "radical.hpp"

using namespace std;
using namespace arma;

PROGRAM_INFO("RADICAL", "An implementation of RADICAL, a method for independent "
	     "component analysis (ICA)");

PARAM_STRING_REQ("X", "Input dataset filename for ICA", "");
PARAM_STRING_REQ("Y", "Independent components filename", "");
PARAM_STRING_REQ("W", "Unmixing matrix filename", "");

PARAM_DOUBLE("noise_std_dev", "Standard deviation of Gaussian noise", "",
	     0.175);
PARAM_INT("n_replicates", "Number of Gaussian-perturbed replicates to use "
	  "(per point) in Radical2D", "",
	  30);
PARAM_INT("n_angles", "Number of angles to consider in brute-force search "
	  "during Radical2D", "",
	  150);
PARAM_INT("n_sweeps", "Number of sweeps (each sweep calls Radical2D once for "
	  "each pair of dimensions", "",
	  0);


int main(int argc, char* argv[]) {

  // Handle parameters
  CLI::ParseCommandLine(argc, argv);
  
  // load the data
  const std::string matXFilename = CLI::GetParam<std::string>("X");
  mat matX;
  data::Load(matXFilename, matX);
  
  
  // load parameters
  double noiseStdDev = CLI::GetParam<double>("noise_std_dev");
  u32 nReplicates = CLI::GetParam<int>("n_replicates");
  u32 nAngles = CLI::GetParam<int>("n_angles");
  u32 nSweeps = CLI::GetParam<int>("n_sweeps");
  if(nSweeps == 0) {
    nSweeps = matX.n_rows - 1;
  }
  
  // run RADICAL
  mlpack::radical::Radical rad(noiseStdDev, nReplicates, nAngles, nSweeps);
  mat matY;
  mat matW;
  rad.DoRadical(matX, matY, matW);
  
  // save results
  const std::string matYFilename = CLI::GetParam<std::string>("Y");
  data::Save(matYFilename, matY);
  
  const std::string matWFilename = CLI::GetParam<std::string>("W");
  data::Save(matWFilename, matW);
  
  
  /*
  // compute and print objective  
  mat matYT = trans(matY);
  double valEst = 0;
  for(u32 i = 0; i < matYT.n_cols; i++) {
    vec Yi = vec(matYT.col(i));
    valEst += rad.Vasicek(Yi);
  }
  printf("objective(estimate) = %f\n", valEst);
  */
  
  
  
}

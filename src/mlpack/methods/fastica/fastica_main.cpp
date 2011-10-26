/**
 * @file fastica_main.cc
 *
 * Demonstrates usage of fastica.h
 *
 * @see fastica.h
 *
 * @author Nishant Mehta
 */
#include "fastica.hpp"

/**
 * Here are usage instructions for this implementation of FastICA. Default values are given
 * to the right in parentheses.
 *
 * Parameters specific to this driver:
 *
 *   @param data = data file with each row being one sample (REQUIRED PARAMETER)
 *   @param ic_filename = independent components results filename (ic.dat)
 *   @param unmixing_filename = unmixing matrix results filename (unmixing.dat)
 *
 * Parameters specific to fastica.h (should be preceded by 'fastica/' on the command line):
 *
 *   @param seed = (long) seed to the random number generator (clock() + time(0))
 *   @param approach = {deflation, symmetric} (deflation)
 *   @param nonlinearity = {logcosh, gauss, kurtosis, skew} (logcosh)
 *   @param num_of_IC = integer constant for number of independent components to find (dimensionality of data)
 *   @param fine_tune = {true, false} (false)
 *   @param a1 = numeric constant for logcosh nonlinearity (1)
 *   @param a2 = numeric constant for gauss nonlinearity (1)
 *   @param mu = numeric constant for fine-tuning Newton-Raphson method (1)
 *   @param stabilization = {true, false} (false)
 *   @param epsilon = threshold for convergence (0.0001)
 *   @param max_num_iterations = maximum number of fixed-point iterations
 *   @param max_fine_tune = maximum number of fine-tuning iterations
 *   @param percent_cut = number in [0,1] indicating percent data to use in stabilization updates (1)
 *
 * Example use:
 *
 * @code
 * ./fastica --data=X_t.dat --ic_filename=ic.dat --unmixing_filename=W.dat
 * --fastica/approach=symmetric --fastica/nonlinearity=gauss
 * --fastica/stabilization=true --fastica/epsilon=0.0000001 --percent_cut=0.5
 * @endcode
 *
 * Note: Compile with verbose mode to display convergence-related values
 */

PARAM_STRING_REQ("input_file", "File containing input data.", "fastica");
PARAM_STRING("ic_file", "File containing IC data.", "fastica", "ic.dat");
PARAM_STRING("unmixing_file", "File containing unmixing data.", "fastica", "unmixing.dat");

int main(int argc, char *argv[]) {
  arma::mat X;
  using namespace mlpack;
  using namespace fastica;

  CLI::ParseCommandLine(argc, argv);
  const char* data = CLI::GetParam<std::string>("fastica/input_file").c_str();
  X.load(data);

  const char* ic_filename = CLI::GetParam<std::string>("fastica/ic_file").c_str();
  const char* unmixing_filename =
    CLI::GetParam<std::string>("fastica/unmixing_file").c_str();

  FastICA fastica;

  int success_status = false;
  if(fastica.Init(X) == true) {
    arma::mat W, Y;
    if(fastica.DoFastICA(W, Y) == true) {
      Y.save(ic_filename);
      arma::mat Z = trans(W);
      Z.save(unmixing_filename);
      success_status = true;
      mlpack::Log::Debug << W << std::endl;
    }
  }


  if(success_status == false) {
    mlpack::Log::Debug << "FAILED!" << std::endl;
  }

  return success_status;
}

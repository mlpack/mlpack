/** @file sparse_coding_main.cpp
 *  @author Nishant Mehta
 *
 *  Executable for Sparse Coding
 */

#include <mlpack/core.hpp>
#include "sparse_coding.hpp"

PROGRAM_INFO("Sparse Coding", "An implementation of l1-norm and l1+l2-norm "
	     "regularized Sparse Coding with Dictionary Learning");

PARAM_DOUBLE_REQ("lambda1", "sparse coding l1-norm regularization parameter.", "l");
PARAM_DOUBLE("lambda2", "sparse coding l2-norm regularization parameter.", "", 0);

PARAM_INT_REQ("n_atoms", "number of atoms in dictionary.", "k");

PARAM_INT_REQ("n_iterations", "number of iterations for sparse coding.", "");

PARAM_STRING_REQ("data", "path to the input data.", "");
PARAM_STRING("initial_dictionary", "Filename for initial dictionary.", "", "");
PARAM_STRING("results_dir", "Directory for results.", "", "");



using namespace arma;
using namespace std;
using namespace mlpack;
using namespace mlpack::sparse_coding;

int main(int argc, char* argv[]) {
  CLI::ParseCommandLine(argc, argv);
  
  std::srand(time(NULL));
  
  double lambda1 = CLI::GetParam<double>("lambda1");
  double lambda2 = CLI::GetParam<double>("lambda2");
  
  // if using fx-run, one could just leave results_dir blank
  const char* resultsDir = CLI::GetParam<string>("results_dir").c_str();
  
  const char* dataFullpath = CLI::GetParam<string>("data").c_str();

  const char* initialDictionaryFullpath = 
    CLI::GetParam<string>("initial_dictionary").c_str();
  
  size_t nIterations = CLI::GetParam<int>("n_iterations");
  
  size_t nAtoms = CLI::GetParam<int>("n_atoms");
  
  mat matX;
  matX.load(dataFullpath);
  
  u32 nPoints = matX.n_cols;
  printf("Loaded %d points in %d dimensions\n", nPoints, matX.n_rows);

  // normalize each point since these are images
  for(u32 i = 0; i < nPoints; i++) {
    matX.col(i) /= norm(matX.col(i), 2);
  }
  
  // run Sparse Coding
  SparseCoding sc(matX, nAtoms, lambda1, lambda2);
  
  if(strlen(initialDictionaryFullpath) == 0) {
    sc.DataDependentRandomInitDictionary();
  }
  else {
    mat matInitialD;
    matInitialD.load(initialDictionaryFullpath);
    if(matInitialD.n_cols != nAtoms) {
      Log::Fatal << "The specified initial dictionary to load has " 
		 << matInitialD.n_cols << " atoms, but the learned dictionary "
		 << "was specified to have " << nAtoms << " atoms!\n";
      return EXIT_FAILURE;
    }
    if(matInitialD.n_rows != matX.n_rows) {
      Log::Fatal << "The specified initial dictionary to load has "
		 << matInitialD.n_rows << " dimensions, but the specified data "
		 << "has " << matX.n_rows << " dimensions!\n";
      return EXIT_FAILURE;
    }
    sc.SetDictionary(matInitialD);
  }
  
  
  Timer::Start("sparse_coding");
  sc.DoSparseCoding(nIterations);
  Timer::Stop("sparse_coding"); 
  
  mat learnedD = sc.MatD();
  mat learnedZ =  sc.MatZ();

  if(strlen(resultsDir) == 0) {
    data::Save("D.csv", learnedD);
    data::Save("Z.csv", learnedZ);
  }
  else {
    char* dataFullpath = (char*) malloc(320 * sizeof(char));

    sprintf(dataFullpath, "%s/D.csv", resultsDir);
    data::Save(dataFullpath, learnedD);
    
    sprintf(dataFullpath, "%s/Z.csv", resultsDir);
    data::Save(dataFullpath, learnedZ);
    
    free(dataFullpath);
  }
}

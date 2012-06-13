/**
 * @file sparse_coding_main.cpp
 * @author Nishant Mehta
 *
 * Executable for Sparse Coding.
 */
#include <mlpack/core.hpp>
#include "sparse_coding.hpp"

PROGRAM_INFO("Sparse Coding", "An implementation of l1-norm and l1+l2-norm "
    "regularized Sparse Coding with Dictionary Learning");

PARAM_DOUBLE_REQ("lambda1", "Sparse coding l1-norm regularization parameter.",
    "l");
PARAM_DOUBLE("lambda2", "Sparse coding l2-norm regularization parameter.", "L",
    0);

PARAM_INT_REQ("n_atoms", "Number of atoms in the dictionary.", "k");

PARAM_INT_REQ("n_iterations", "Number of iterations for sparse coding.", "n");

PARAM_STRING_REQ("input_file", "Filename of the input data.", "i");
PARAM_STRING("initial_dictionary", "Filename for optional initial dictionary.",     "d", "");
PARAM_STRING("results_dir", "Directory to store results in.", "r", "");
PARAM_INT("seed", "Random seed.  If 0, 'std::time(NULL) is used.", "s", 0);

using namespace arma;
using namespace std;
using namespace mlpack;
using namespace mlpack::math;
using namespace mlpack::sparse_coding;

int main(int argc, char* argv[])
{
  CLI::ParseCommandLine(argc, argv);

  if (CLI::GetParam<int>("seed") != 0)
    RandomSeed((size_t) CLI::GetParam<int>("seed"));
  else
    RandomSeed((size_t) std::time(NULL));

  double lambda1 = CLI::GetParam<double>("lambda1");
  double lambda2 = CLI::GetParam<double>("lambda2");

  const char* resultsDir = CLI::GetParam<string>("results_dir").c_str();
  const char* dataFullpath = CLI::GetParam<string>("data").c_str();
  const char* initialDictionaryFullpath =
      CLI::GetParam<string>("initial_dictionary").c_str();

  size_t nIterations = CLI::GetParam<int>("n_iterations");

  size_t nAtoms = CLI::GetParam<int>("n_atoms");

  mat matX;
  data::Load(dataFullpath, matX);

  const size_t nPoints = matX.n_cols;
  Log::Info << "Loaded " << nPoints << " points in " << matX.n_rows <<
      " dimensions." << endl;

  // Normalize each point since these are images.
  for (size_t i = 0; i < nPoints; ++i)
    matX.col(i) /= norm(matX.col(i), 2);

  // Run the sparse coding algorithm.
  SparseCoding<> sc(matX, nAtoms, lambda1, lambda2);

  if (strlen(initialDictionaryFullpath) != 0)
  {
    mat matInitialD;
    data::Load(initialDictionaryFullpath, matInitialD);

    if (matInitialD.n_cols != nAtoms)
    {
      Log::Fatal << "The specified initial dictionary to load has "
          << matInitialD.n_cols << " atoms, but the learned dictionary "
          << "was specified to have " << nAtoms << " atoms!" << endl;
    }

    if (matInitialD.n_rows != matX.n_rows)
    {
      Log::Fatal << "The specified initial dictionary to load has "
          << matInitialD.n_rows << " dimensions, but the specified data "
          << "has " << matX.n_rows << " dimensions!" << endl;
    }

    sc.Dictionary() = matInitialD;
  }

  Timer::Start("sparse_coding");
  sc.Encode(nIterations);
  Timer::Stop("sparse_coding");

  mat learnedD = sc.Dictionary();
  mat learnedZ = sc.Codes();

  if (strlen(resultsDir) == 0)
  {
    data::Save("D.csv", learnedD);
    data::Save("Z.csv", learnedZ);
  }
  else
  {
    stringstream datapath;
    datapath << resultsDir << "/D.csv";

    data::Save(datapath.str(), learnedD);

    datapath.clear();
    datapath << resultsDir << "/Z.csv";

    data::Save(datapath.str(), learnedZ);
  }
}

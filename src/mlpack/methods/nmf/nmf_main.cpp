/**
 * @file nmf_main.cpp
 * @author Mohan Rajendran
 *
 * Main executable to run NMF.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/cli.hpp>
#include <mlpack/core/util/mlpack_main.hpp>

#include <mlpack/methods/amf/amf.hpp>
#include <mlpack/methods/amf/init_rules/random_init.hpp>
#include <mlpack/methods/amf/init_rules/given_init.hpp>
#include <mlpack/methods/amf/init_rules/merge_init.hpp>
#include <mlpack/methods/amf/update_rules/nmf_mult_dist.hpp>
#include <mlpack/methods/amf/update_rules/nmf_mult_div.hpp>
#include <mlpack/methods/amf/update_rules/nmf_als.hpp>
#include <mlpack/methods/amf/termination_policies/simple_residue_termination.hpp>

using namespace mlpack;
using namespace mlpack::amf;
using namespace mlpack::util;
using namespace std;

// Document program.
PROGRAM_INFO("Non-negative Matrix Factorization",
    // Short description.
    "An implementation of non-negative matrix factorization.  This can be used "
    "to decompose an input dataset into two low-rank non-negative components.",
    // Long description.
    "This program performs non-negative matrix factorization on the given "
    "dataset, storing the resulting decomposed matrices in the specified "
    "files.  For an input dataset V, NMF decomposes V into two matrices W "
    "and H such that "
    "\n\n"
    "V = W * H"
    "\n\n"
    "where all elements in W and H are non-negative.  If V is of size (n x m),"
    " then W will be of size (n x r) and H will be of size (r x m), where r is "
    "the rank of the factorization (specified by the " +
    PRINT_PARAM_STRING("rank") + " parameter)."
    "\n\n"
    "Optionally, the desired update rules for each NMF iteration can be chosen "
    "from the following list:"
    "\n\n"
    " - multdist: multiplicative distance-based update rules (Lee and Seung "
    "1999)\n"
    " - multdiv: multiplicative divergence-based update rules (Lee and Seung "
    "1999)\n"
    " - als: alternating least squares update rules (Paatero and Tapper 1994)"
    "\n\n"
    "The maximum number of iterations is specified with " +
    PRINT_PARAM_STRING("max_iterations") + ", and the minimum residue "
    "required for algorithm termination is specified with the " +
    PRINT_PARAM_STRING("min_residue") + " parameter."
    "\n\n"
    "For example, to run NMF on the input matrix " + PRINT_DATASET("V") + " "
    "using the 'multdist' update rules with a rank-10 decomposition and "
    "storing the decomposed matrices into " + PRINT_DATASET("W") + " and " +
    PRINT_DATASET("H") + ", the following command could be used: "
    "\n\n" +
    PRINT_CALL("nmf", "input", "V", "w", "W", "h", "H", "rank", 10,
        "update_rules", "multdist"),
    SEE_ALSO("@cf", "#cf"),
    SEE_ALSO("Alternating matrix factorization tutorial",
        "@doxygen/amftutorial.html"),
    SEE_ALSO("Non-negative matrix factorization on Wikipedia",
        "https://en.wikipedia.org/wiki/Non-negative_matrix_factorization"),
    SEE_ALSO("Algorithms for non-negative matrix factorization (pdf)",
        "http://papers.nips.cc/paper/1861-algorithms-for-non-negative-matrix-"
        "factorization.pdf"),
    SEE_ALSO("mlpack::amf::AMF C++ class documentation",
        "@doxygen/classmlpack_1_1amf_1_1AMF.html"));

// Parameters for program.
PARAM_MATRIX_IN_REQ("input", "Input dataset to perform NMF on.", "i");
PARAM_MATRIX_OUT("w", "Matrix to save the calculated W to.", "W");
PARAM_MATRIX_OUT("h", "Matrix to save the calculated H to.", "H");
PARAM_INT_IN_REQ("rank", "Rank of the factorization.", "r");

PARAM_INT_IN("max_iterations", "Number of iterations before NMF terminates (0 "
    "runs until convergence.", "m", 10000);
PARAM_INT_IN("seed", "Random seed.  If 0, 'std::time(NULL)' is used.", "s", 0);
PARAM_DOUBLE_IN("min_residue", "The minimum root mean square residue allowed "
    "for each iteration, below which the program terminates.", "e", 1e-5);

PARAM_STRING_IN("update_rules", "Update rules for each iteration; ( multdist | "
    "multdiv | als ).", "u", "multdist");

PARAM_MATRIX_IN("initial_w", "Initial W matrix.", "p");
PARAM_MATRIX_IN("initial_h", "Initial H matrix.", "q");

void LoadInitialWH(const bool bindingTransposed, arma::mat& w, arma::mat& h)
{
  // Note that these datasets will typically be transposed on load, since we are
  // likely receiving it from a row-major language, but we get it in a
  // column-major form.  Therefore, we're actually decomposing V^T = W^T * H^T.
  // Effectively this means we are solving, for the user, V = H*W.  Therefore,
  // we actually have to switch what we are saving, so we will save the W we get
  // from amf.Apply() as H, and vice versa.
  if (bindingTransposed)
  {
    w = CLI::GetParam<arma::mat>("initial_h");
    h = CLI::GetParam<arma::mat>("initial_w");
  }
  else
  {
    h = CLI::GetParam<arma::mat>("initial_h");
    w = CLI::GetParam<arma::mat>("initial_w");
  }
}

void SaveWH(const bool bindingTransposed, arma::mat&& w, arma::mat&& h)
{
  // The same transposition applies when saving.
  if (bindingTransposed)
  {
    CLI::GetParam<arma::mat>("w") = std::move(h);
    CLI::GetParam<arma::mat>("h") = std::move(w);
  }
  else
  {
    CLI::GetParam<arma::mat>("h") = std::move(h);
    CLI::GetParam<arma::mat>("w") = std::move(w);
  }
}

template<typename UpdateRuleType>
void ApplyFactorization(const arma::mat& V,
                        const size_t r,
                        arma::mat& W,
                        arma::mat& H)
{
  const size_t maxIterations = CLI::GetParam<int>("max_iterations");
  const double minResidue = CLI::GetParam<double>("min_residue");

  SimpleResidueTermination srt(minResidue, maxIterations);

  // Load input dataset.  We know if the data is transposed based on the
  // BINDING_MATRIX_TRANSPOSED macro, which will be 'true' or 'false'.
  arma::mat initialW, initialH;
  LoadInitialWH(BINDING_MATRIX_TRANSPOSED, initialW, initialH);
  if (CLI::HasParam("initial_w") && CLI::HasParam("initial_h"))
  {
    // Initialize W and H with given matrices
    GivenInitialization ginit = GivenInitialization(initialW, initialH);
    AMF<SimpleResidueTermination,
        GivenInitialization,
        UpdateRuleType> amf(srt, ginit);
    amf.Apply(V, r, W, H);
  }
  else if (CLI::HasParam("initial_w"))
  {
    // Merge GivenInitialization and RandomInitialization rules
    // to initialize W with the given matrix, and H with random noise
    GivenInitialization ginit = GivenInitialization(initialW);
    RandomInitialization rinit = RandomInitialization();
    MergeInitialization<GivenInitialization, RandomInitialization> minit =
        MergeInitialization<GivenInitialization, RandomInitialization>
        (ginit, rinit);
    AMF<SimpleResidueTermination,
        MergeInitialization<GivenInitialization, RandomInitialization>,
        UpdateRuleType> amf(srt, minit);
    amf.Apply(V, r, W, H);
  }
  else if (CLI::HasParam("initial_h"))
  {
    // Merge GivenInitialization and RandomInitialization rules
    // to initialize H with the given matrix, and W with random noise
    GivenInitialization ginit = GivenInitialization(initialH, false);
    RandomInitialization rinit = RandomInitialization();
    MergeInitialization<RandomInitialization, GivenInitialization> minit =
        MergeInitialization<RandomInitialization, GivenInitialization>
        (rinit, ginit);
    AMF<SimpleResidueTermination,
        MergeInitialization<RandomInitialization, GivenInitialization>,
        UpdateRuleType> amf(srt, minit);
    amf.Apply(V, r, W, H);
  }
  else
  {
    // Use random initialization
    AMF<SimpleResidueTermination,
        RandomInitialization,
        UpdateRuleType> amf(srt);
    amf.Apply(V, r, W, H);
  }
}

static void mlpackMain()
{
  // Initialize random seed.
  if (CLI::GetParam<int>("seed") != 0)
    math::RandomSeed((size_t) CLI::GetParam<int>("seed"));
  else
    math::RandomSeed((size_t) std::time(NULL));

  // Gather parameters.
  const size_t r = CLI::GetParam<int>("rank");
  const string updateRules = CLI::GetParam<string>("update_rules");

  // Validate parameters.
  RequireParamValue<int>("rank", [](int x) { return x > 0; }, true,
      "the rank of the factorization must be greater than 0");
  RequireParamInSet<string>("update_rules", { "multdist", "multdiv", "als" },
      true, "unknown update rules");
  RequireParamValue<int>("max_iterations", [](int x) { return x >= 0; },
      true, "max_iterations must be non-negative");

  RequireAtLeastOnePassed({ "h", "w" }, false, "no output will be saved");

  arma::mat V = std::move(CLI::GetParam<arma::mat>("input"));

  arma::mat W;
  arma::mat H;

  // Perform NMF with the specified update rules.
  if (updateRules == "multdist")
  {
    Log::Info << "Performing NMF with multiplicative distance-based update "
        << "rules." << std::endl;
    ApplyFactorization<NMFMultiplicativeDistanceUpdate>(V, r, W, H);
  }
  else if (updateRules == "multdiv")
  {
    Log::Info << "Performing NMF with multiplicative divergence-based update "
        << "rules." << std::endl;
    ApplyFactorization<NMFMultiplicativeDivergenceUpdate>(V, r, W, H);
  }
  else if (updateRules == "als")
  {
    Log::Info << "Performing NMF with alternating least squared update rules."
        << std::endl;
    ApplyFactorization<NMFALSUpdate>(V, r, W, H);
  }

  // Save results.  Remember from our discussion in the comments earlier that we
  // may need to switch the names of the outputs.
  SaveWH(BINDING_MATRIX_TRANSPOSED, std::move(W), std::move(H));
}

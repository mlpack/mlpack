#include <mlpack/core.h>
#include "exact_max_ip.h"
#include "approx_max_ip.h"
#include "check_nn_utils.h"

#include <string>
#include <armadillo>

using namespace mlpack;
using namespace std;

PROGRAM_INFO("Approx-maximum Inner Product", "This program "
  	     "returns the approx-maximum inner product for a "
  	     "given query over a set of points (references).", 
 	     "approx_maxip");

PARAM_STRING_REQ("r", "The reference set", "");
PARAM_STRING_REQ("q", "The set of queries", "");
PARAM_STRING("rank_file", "The file containing the ranks.",
	     "", "");
PARAM_FLAG("donaive", "The flag to trigger the naive "
	   "computation", "");
PARAM_FLAG("dofastapprox", "The flag to trigger the "
	   "tree-based rank-approximate search algorithm",
	   "");

/**
 * This function checks if the neighbors computed 
 * by two different methods is the same.
 */
void compute_error(arma::mat* rdata, arma::mat* qdata,
		   arma::Mat<size_t>* indices);

void compute_error(arma::Mat<size_t>* indices, size_t rdata_size);


int main (int argc, char *argv[]) {

  srand( time(NULL) );

  CLI::ParseCommandLine(argc, argv);

  arma::mat rdata, qdata;
  string rfile = CLI::GetParam<string>("r");
  string qfile = CLI::GetParam<string>("q");

  Log::Info << "Loading files..." << endl;
  if (!data::Load(rfile.c_str(), rdata))
    Log::Fatal << "Reference file "<< rfile << " not found." << endl;

  if (!data::Load(qfile.c_str(), qdata)) 
    Log::Fatal << "Query file " << qfile << " not found." << endl;

  Log::Info << "File loaded..." << endl;
  
  Log::Info << "R(" << rdata.n_rows << ", " << rdata.n_cols 
	   << "), Q(" << qdata.n_rows << ", " << qdata.n_cols 
	   << ")" << endl;


  arma::Mat<size_t> nac, apc;
  arma::mat din, dia;

  // size_t knns = CLI::GetParam<int>("approx_maxip/knns");

  double naive_comp;
  double approx_comp;


  // Naive computation
  if (CLI::HasParam("donaive")) {
    MaxIP naive;
    Log::Info << "Brute computation" << endl;
    Log::Info << "Initializing...." << endl;

    CLI::StartTimer("naive_init");
    naive.InitNaive(qdata, rdata);
    CLI::StopTimer("naive_init");
    Log::Info << "Initialized." << endl;

    Log::Info << "Computing Max IP....." << endl;
    naive_comp = naive.ComputeNaive(&nac, &din);
    Log::Info << "Max IP Computed." << endl;

    if (CLI::GetParam<string>("rank_file") != "") {
      string rank_file = CLI::GetParam<string>("rank_file");
      check_nn_utils::compute_error(rank_file,  rdata.n_cols, &nac);
    } else {
      check_nn_utils::compute_error(&rdata, &qdata, &nac);
    }
  }


  // Approximate computation
  if (CLI::HasParam("dofastapprox")) {
    ApproxMaxIP fast_approx;
    Log::Info << "Tree-based approximate computation" << endl;
    Log::Info << "Initializing...." << endl;

    CLI::StartTimer("fast_initapprox");
    fast_approx.InitApprox(qdata, rdata);
    CLI::StopTimer("fast_initapprox");
    Log::Info << "Initialized." << endl;

    Log::Info << "Computing tree-based Approximate Max IP....." << endl;
    approx_comp = fast_approx.ComputeApprox(&apc, &dia);
    Log::Info << "Tree-based Max IP Computed." << endl;


    Log::Warn << "Speed of fast-approx over naive: "
	      << rdata.n_cols  << " / " << (float) approx_comp << " = "
	      <<(float) (rdata.n_cols / approx_comp) << endl;


    if (CLI::GetParam<string>("rank_file") != "") {
      string rank_file = CLI::GetParam<string>("rank_file");
      double epsilon = CLI::GetParam<double>("approx_maxip/epsilon");
      double alpha = CLI::GetParam<double>("approx_maxip/alpha");

      check_nn_utils::check_rank_bound(rank_file, rdata.n_cols,
				       epsilon, alpha, &apc);

      check_nn_utils::compute_error(rank_file,  rdata.n_cols, &apc);
    } else {
      check_nn_utils::compute_error(&rdata, &qdata, &apc);
    }
  }
}

#include <mlpack/core.h>
#include "exact_max_ip.h"
#include "check_nn_utils.h"

#include <string>
#include <armadillo>

using namespace mlpack;
using namespace std;

PROGRAM_INFO("Maximum Inner Product", "This program "
 	     "returns the maximum inner product for a "
 	     "given query over a set of points (references).", 
	     "maxip");

// PARAM_MODULE("maxip_main", "Parameters for the main "
// 	     "file to compute the maximum inner product "
// 	     "for a given query over a given set of "
// 	     "references.");

PARAM_STRING_REQ("r", "The reference set", "");
PARAM_STRING_REQ("q", "The set of queries", "");


PARAM_STRING("rank_file", "The file containing the ranks.",
	     "", "");


PARAM_FLAG("donaive", "The flag to trigger the naive "
	   "computation", "");
PARAM_FLAG("dofastexact", "The flag to trigger the tree-based"
	   " search algorithm", "");

PARAM_FLAG("check_results", "The flag to trigger the "
 	   "checking of the output", "");

// PARAM_STRING("maxip_file", "The file where the output "
// 	     "will be written into", "", "results.txt");


int main (int argc, char *argv[]) {

  CLI::ParseCommandLine(argc, argv);

  arma::mat rdata, qdata;
  string rfile = CLI::GetParam<string>("r");
  string qfile = CLI::GetParam<string>("q");

  Log::Info << "Loading files..." << endl;
  if (rdata.load(rfile.c_str()) == false)
    Log::Fatal << "Reference file "<< rfile << " not found." << endl;

  if (qdata.load(qfile.c_str()) == false) 
    Log::Fatal << "Query file " << qfile << " not found." << endl;

  Log::Info << "File loaded..." << endl;
  
  rdata = arma::trans(rdata);
  qdata = arma::trans(qdata);

  Log::Info << "R(" << rdata.n_rows << ", " << rdata.n_cols 
	   << "), Q(" << qdata.n_rows << ", " << qdata.n_cols 
	   << ")" << endl;


  arma::Mat<size_t> nac, exc;
  arma::mat din, die;

  double naive_comp, fast_comp;

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

//     if (CLI::HasParam("print_results")) {
//       FILE *fp=fopen(CLI::GetParam<string>("maxip_file").c_str(), "w");
//       if (fp == NULL)
// 	Log::Fatal << "Error while opening " 
// 		  << CLI::GetParam<string>("maxip_file")
// 		  << endl;

//       for(size_t i = 0 ; i < nac.n_elem / knns ; i++) {
//         fprintf(fp, "%zu", i);
//         for(size_t j = 0; j < knns; j++)
//           fprintf(fp, ", %zu, %lg", 
//                   nac(i*knns+j), din(i*knns+j));
//         fprintf(fp, "\n");
//       }
//       fclose(fp);
//     }  
  }

  // Exact computation
  if (CLI::HasParam("dofastexact")) {
    MaxIP fast_exact;
    Log::Info << "Tree-based computation" << endl;
    Log::Info << "Initializing...." << endl;

    CLI::StartTimer("fast_init");
    fast_exact.Init(qdata, rdata);
    CLI::StopTimer("fast_init");
    Log::Info << "Initialized." << endl;

    Log::Info << "Computing tree-based Max IP....." << endl;
    fast_comp = fast_exact.ComputeNeighbors(&exc, &die);
    Log::Info << "Tree-based Max IP Computed." << endl;

    Log::Warn << "Speed of fast-exact over naive: "
	      << rdata.n_cols  << " / " << (float) fast_comp << " = "
	      <<(float) (rdata.n_cols / fast_comp) << endl;

    //     if (CLI::HasParam("print_results")) {
    //       FILE *fp=fopen(CLI::GetParam<string>("maxip_file").c_str(), "w");
    //       if (fp == NULL)
    // 	Log::Fatal << "Error while opening " 
    // 		  << CLI::GetParam<string>("maxip_file") 
    // 		  << endl;

    //       for(size_t i = 0 ; i < exc.n_elem / knns ; i++) {
    //         fprintf(fp, "%zu", i);
    //         for(size_t j = 0; j < knns; j++)
    //           fprintf(fp, ", %zu, %lg", 
    //                   exc(i*knns+j), die(i*knns+j));
    //         fprintf(fp, "\n");
    //       }
    //       fclose(fp);
    //     }
  }

  if (CLI::HasParam("check_results")) {
    if (CLI::HasParam("donaive") && CLI::HasParam("dofastexact")) {
      check_nn_utils::count_mismatched_neighbors(nac, din, exc, die);
      Log::Warn << "Speed of fast-exact over naive: "
		<< naive_comp << " / " << (float) fast_comp << " = "
		<<(float) (naive_comp / fast_comp) << endl;
    } else if (CLI::HasParam("dofastexact")) {
      if (CLI::GetParam<string>("rank_file") != "") {
	string rank_file = CLI::GetParam<string>("rank_file");
	check_nn_utils::compute_error(rank_file,  rdata.n_cols, &exc);
      } else {
	check_nn_utils::compute_error(&rdata, &qdata, &exc);
      }
    } else if (CLI::HasParam("donaive")) {
      if (CLI::GetParam<string>("rank_file") != "") {
	string rank_file = CLI::GetParam<string>("rank_file");
	check_nn_utils::compute_error(rank_file,  rdata.n_cols, &nac);
      } else {
	check_nn_utils::compute_error(&rdata, &qdata, &nac);
      }
    }
  }
}

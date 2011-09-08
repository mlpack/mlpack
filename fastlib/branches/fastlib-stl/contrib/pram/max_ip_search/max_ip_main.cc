#include <armadillo>
#include <string>
#include "exact_max_ip.h"

// PROGRAM_INFO("Maximum Inner Product", "This program "
// 	     "returns the maximum inner product for a "
// 	     "given query over a set of points (references).");

PARAM_MODULE("maxip_main", "Parameters for the main "
	     "file to compute the maximum inner product "
	     "for a given query over a given set of "
	     "references.");

PARAM_STRING_REQ("r", "The reference set", "maxip_main");
PARAM_STRING_REQ("q", "The set of queries", "maxip_main");

PARAM_FLAG("donaive", "The flag to trigger the naive "
	   "computation", "maxip_main");
PARAM_FLAG("dofastexact", "The flag to trigger the tree-based"
	   " search algorithm", "maxip_main");
PARAM_FLAG("dofastapprox", "The flag to trigger the "
	   "tree-based rank-approximate search algorithm",
	   "maxip_main");

PARAM_STRING("result_file", "The file where the output "
	     "will be written into", "maxip_main",
	     "results.txt");
PARAM_FLAG("print_results", "The flag to trigger the "
	   "printing of the output", "maxip_main");

/**
 * This function checks if the neighbors computed 
 * by two different methods is the same.
 */
// void compare_neighbors(arma::Col<index_t>*, arma::vec*, 
//                        arma::Col<index_t>*, arma::vec*);

// void count_mismatched_neighbors(arma::Col<index_t>*, arma::vec*, 
// 				arma::Col<index_t>*, arma::vec*);

int main (int argc, char *argv[]) {

  IO::ParseCommandLine(argc, argv);

  arma::mat rdata, qdata;

  IO::Info << "Loading files..." << std::endl;
  data::Load(IO::GetParam<char *>("r"), rdata);
  data::Load(IO::GetParam<char *>("q"), qdata);
  IO::Info << "File loaded..." << std::endl;
  
  IO::Info << "R(" << rdata.n_rows << ", " << rdata.n_cols 
	   << "), Q(" << qdata.n_rows << ", " << qdata.n_cols 
	   << ")" << std::endl;


  arma::Col<index_t> nac, exc;
  arma::vec din, die;

  index_t knns = IO::GetParam<int>("maxip/knns");

  // Naive computation
  if (IO::HasParam("donaive")) {
    MaxIP naive;
    IO::Info << "Brute computation" << std::endl;
    IO::Info << "Initializing...." << std::endl;

    IO::StartTimer("naive_init");
    naive.InitNaive(qdata, rdata);
    IO::StopTimer("naive_init");
    IO::Info << "Initialized." << std::endl;

    IO::Info << "Computing Max IP....." << std::endl;
    IO::StartTimer("naive_search");
    naive.ComputeNaive(&nac, &din);
    IO::StopTimer("naive_search");
    IO::Info << "Max IP Computed." << std::endl;

    if (IO::HasParam("print_results")) {
      FILE *fp=fopen(IO::GetParam<char *>("result_file"), "w");
      if (fp == NULL)
	IO::Fatal << "Error while opening " 
		  << IO::GetParam<char *>("result_file") 
		  << "..." << strerror(errno);

      for(index_t i = 0 ; i < nac.n_elem / knns ; i++) {
        fprintf(fp, "%"LI"d", i);
        for(index_t j = 0; j < knns; j++)
          fprintf(fp, ",%"LI"d,%lg", 
                  nac(i*knns+j), din(i*knns+j));
        fprintf(fp, "\n");
      }
      fclose(fp);
    }
  }

  // Exact computation
  if (IO::HasParam("dofastexact")) {
    MaxIP fast_exact;
    IO::Info << "Tree-based computation" << std::endl;
    IO::Info << "Initializing...." << std::endl;

    IO::StartTimer("fast_init");
    fast_exact.Init(qdata, rdata);
    IO::StopTimer("fast_init");
    IO::Info << "Initialized." << std::endl;

    IO::Info << "Computing tree-based Max IP....." << std::endl;
    IO::StartTimer("fast_search");
    fast_exact.ComputeNeighbors(&exc, &die);
    IO::StopTimer("fast_search");
    IO::Info << "Tree-based Max IP Computed." << std::endl;

    if (IO::HasParam("print_results")) {
      FILE *fp=fopen(IO::GetParam<char *>("result_file"), "w");
      if (fp == NULL)
	IO::Fatal << "Error while opening " 
		  << IO::GetParam<char *>("result_file") 
		  << "..." << strerror(errno);

      for(index_t i = 0 ; i < exc.n_elem / knns ; i++) {
        fprintf(fp, "%"LI"d", i);
        for(index_t j = 0; j < knns; j++)
          fprintf(fp, ",%"LI"d,%lg", 
                  exc(i*knns+j), die(i*knns+j));
        fprintf(fp, "\n");
      }
      fclose(fp);
    }
  }

  //   compare_neighbors(&neighbor_indices, &dist_sq, &exc, &die);
  //  count_mismatched_neighbors(&exc, &die, &apc, &dia);
}

// void compare_neighbors(arma::Col<index_t> *a, 
//                        arma::vec *da,
//                        arma::Col<index_t> *b, 
//                        arma::vec *db) {
  
//   IO::Info << "Comparing results for %"LI"d queries", a->size());
//   DEBUG_SAME_SIZE(a->size(), b->size());
//   index_t *x = a->begin();
//   index_t *y = a->end();
//   index_t *z = b->begin();

//   for(index_t i = 0; x != y; x++, z++, i++) {
//     DEBUG_WARN_MSG_IF(*x != *z || (*da)[i] != (*db)[i], 
//                       "point %"LI"d brute: %"LI"d:%lf fast: %"LI"d:%lf",
//                       i, *z, (*db)[i], *x, (*da)[i]);
//   }
// }

// void count_mismatched_neighbors(arma::Col<index_t> *a, 
// 				arma::vec *da,
// 				arma::Col<index_t> *b, 
// 				arma::vec *db) {

//   IO::Info << "Comparing results for %"LI"d queries", a->size());
//   DEBUG_SAME_SIZE(a->size(), b->size());
//   index_t *x = a->begin();
//   index_t *y = a->end();
//   index_t *z = b->begin();
//   index_t count_mismatched = 0;

//   for(index_t i = 0; x != y; x++, z++, i++) {
//     if (*x != *z || (*da)[i] != (*db)[i]) {
//       ++count_mismatched;
//     }
//   }
//   IO::Info << "%"LI"d/%"LI"d errors", count_mismatched, a->size());
// }

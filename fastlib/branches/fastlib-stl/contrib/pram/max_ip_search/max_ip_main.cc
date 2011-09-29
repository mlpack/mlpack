#include <fastlib/fastlib.h>
#include "exact_max_ip.h"

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

PARAM_FLAG("donaive", "The flag to trigger the naive "
	   "computation", "");
PARAM_FLAG("dofastexact", "The flag to trigger the tree-based"
	   " search algorithm", "");
PARAM_FLAG("dofastapprox", "The flag to trigger the "
	   "tree-based rank-approximate search algorithm",
	   "");
PARAM_FLAG("print_results", "The flag to trigger the "
	   "printing of the output", "");

PARAM_STRING("maxip_file", "The file where the output "
	     "will be written into", "", "results.txt");



/**
 * This function checks if the neighbors computed 
 * by two different methods is the same.
 */
void count_mismatched_neighbors(arma::Col<size_t>, arma::vec, 
 				arma::Col<size_t>, arma::vec);


int main (int argc, char *argv[]) {

  IO::ParseCommandLine(argc, argv);

  arma::mat rdata, qdata;
  string rfile = IO::GetParam<string>("r");
  string qfile = IO::GetParam<string>("q");

  IO::Info << "Loading files..." << endl;
  if (!data::Load(rfile.c_str(), rdata))
    IO::Fatal << "Reference file "<< rfile << " not found." << endl;

  if (!data::Load(qfile.c_str(), qdata)) 
    IO::Fatal << "Query file " << qfile << " not found." << endl;

  IO::Info << "File loaded..." << endl;
  
  IO::Info << "R(" << rdata.n_rows << ", " << rdata.n_cols 
	   << "), Q(" << qdata.n_rows << ", " << qdata.n_cols 
	   << ")" << endl;


  arma::Col<size_t> nac, exc;
  arma::vec din, die;

  size_t knns = IO::GetParam<int>("maxip/knns");

  double naive_comp, fast_comp;


  // exit(0);
  // Naive computation
  if (IO::HasParam("donaive")) {
    MaxIP naive;
    IO::Info << "Brute computation" << endl;
    IO::Info << "Initializing...." << endl;

    IO::StartTimer("naive_init");
    naive.InitNaive(qdata, rdata);
    IO::StopTimer("naive_init");
    IO::Info << "Initialized." << endl;

    IO::Info << "Computing Max IP....." << endl;
    IO::StartTimer("naive_search");
    naive_comp = naive.ComputeNaive(&nac, &din);
    IO::StopTimer("naive_search");
    IO::Info << "Max IP Computed." << endl;

    if (IO::HasParam("print_results")) {
      FILE *fp=fopen(IO::GetParam<string>("maxip_file").c_str(), "w");
      if (fp == NULL)
	IO::Fatal << "Error while opening " 
		  << IO::GetParam<string>("maxip_file")
		  << endl;
// 		  << "..." << strerror(errno);

      for(size_t i = 0 ; i < nac.n_elem / knns ; i++) {
        fprintf(fp, "%zu", i);
        for(size_t j = 0; j < knns; j++)
          fprintf(fp, ", %zu, %lg", 
                  nac(i*knns+j), din(i*knns+j));
        fprintf(fp, "\n");
      }
      fclose(fp);
    }  

    //IO::Info << "Comparing results for " << nac.n_elem << " queries." << endl;
  }

  //IO::Info << "Comparing results for " << nac.n_elem << " queries." << endl;

  // Exact computation
  if (IO::HasParam("dofastexact")) {
    MaxIP fast_exact;
    IO::Info << "Tree-based computation" << endl;
    IO::Info << "Initializing...." << endl;

    IO::StartTimer("fast_init");
    fast_exact.Init(qdata, rdata);
    IO::StopTimer("fast_init");
    IO::Info << "Initialized." << endl;

    IO::Info << "Computing tree-based Max IP....." << endl;
    IO::StartTimer("fast_search");
    fast_comp = fast_exact.ComputeNeighbors(&exc, &die);
    IO::StopTimer("fast_search");
    IO::Info << "Tree-based Max IP Computed." << endl;

    if (IO::HasParam("print_results")) {
      FILE *fp=fopen(IO::GetParam<string>("maxip_file").c_str(), "w");
      if (fp == NULL)
	IO::Fatal << "Error while opening " 
		  << IO::GetParam<string>("maxip_file") 
		  << endl;
// 		  << "..." << strerror(errno);

      for(size_t i = 0 ; i < exc.n_elem / knns ; i++) {
        fprintf(fp, "%zu", i);
        for(size_t j = 0; j < knns; j++)
          fprintf(fp, ", %zu, %lg", 
                  exc(i*knns+j), die(i*knns+j));
        fprintf(fp, "\n");
      }
      fclose(fp);
    }
  }

  //   compare_neighbors(&neighbor_indices, &dist_sq, &exc, &die);

  if (IO::HasParam("donaive") && IO::HasParam("dofastexact")) {
    count_mismatched_neighbors(nac, din, exc, die);
    IO::Warn << "Speed of fast-exact over naive: "
	     << naive_comp << " / " << (float) fast_comp << " = "
	     <<(float) (naive_comp / fast_comp) << endl;
  }

}

// void compare_neighbors(arma::Col<size_t> *a, 
//                        arma::vec *da,
//                        arma::Col<size_t> *b, 
//                        arma::vec *db) {
  
//   IO::Info << "Comparing results for %zud queries", a->size());
//   DEBUG_SAME_SIZE(a->size(), b->size());
//   size_t *x = a->begin();
//   size_t *y = a->end();
//   size_t *z = b->begin();

//   for(size_t i = 0; x != y; x++, z++, i++) {
//     DEBUG_WARN_MSG_IF(*x != *z || (*da)[i] != (*db)[i], 
//                       "point %zud brute: %zud:%lf fast: %zud:%lf",
//                       i, *z, (*db)[i], *x, (*da)[i]);
//   }
// }

void count_mismatched_neighbors(arma::Col<size_t> a, 
 				arma::vec da,
 				arma::Col<size_t> b, 
 				arma::vec db) {
  IO::Warn << "Comparing results for " << a.n_elem << " queries." << endl;
  assert(a.n_elem == b.n_elem);
  size_t count_mismatched = 0;

//   IO::Warn << "Mismatches: " << endl;
  for(size_t i = 0; i < a.n_elem;  i++) {
    if (da(i) != db(i)) {
      ++count_mismatched;
      IO::Warn << da(i) - db(i) << endl;
    }
  }

  IO::Warn << count_mismatched << " / " << a.n_elem
	    << " errors." << endl;

}

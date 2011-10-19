#include <mlpack/core.h>
#include "exact_max_ip.h"
#include "approx_max_ip.h"

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
void count_mismatched_neighbors(arma::Mat<size_t>, arma::mat, 
 				arma::Mat<size_t>, arma::mat);

void compute_error(arma::mat rdata, arma::mat qdata,
		   arma::Mat<size_t> indices, arma::mat values);

void compute_error(arma::Mat<size_t> indices, arma::mat values,
		   size_t rdata_size);


int main (int argc, char *argv[]) {

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


  arma::Mat<size_t> nac, exc, apc;
  arma::mat din, die, dia;

  size_t knns = CLI::GetParam<int>("maxip/knns");

  double naive_comp, fast_comp, approx_comp;


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



  if (CLI::HasParam("donaive") && CLI::HasParam("dofastexact")) {
    count_mismatched_neighbors(nac, din, exc, die);
    Log::Warn << "Speed of fast-exact over naive: "
	      << naive_comp << " / " << (float) fast_comp << " = "
	      <<(float) (naive_comp / fast_comp) << endl;
  } else if (CLI::HasParam("dofastexact")) {
    Log::Warn << "Speed of fast-exact over naive: "
	      << rdata.n_cols  << " / " << (float) fast_comp << " = "
	      <<(float) (rdata.n_cols / fast_comp) << endl;


    if (CLI::HasParam("rank_file"))
      compute_error(&exc, &die, rdata.n_cols);
    else
      compute_error(&rdata, &qdata, &exc, &die);
  }

  if (CLI::HasParam("dofastapprox")) {
    Log::Warn << "Speed of fast-approx over naive: "
	      << rdata.n_cols  << " / " << (float) approx_comp << " = "
	      <<(float) (rdata.n_cols / approx_comp) << endl;

    if (CLI::HasParam("rank_file"))
      compute_error(&apc, &dia, rdata.n_cols);
    else
      compute_error(&rdata, &qdata, &apc, &dia);
  }

}

// void compare_neighbors(arma::Col<size_t> *a, 
//                        arma::vec *da,
//                        arma::Col<size_t> *b, 
//                        arma::vec *db) {
  
//   Log::Info << "Comparing results for %zud queries", a->size());
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

void count_mismatched_neighbors(arma::Mat<size_t> a, 
 				arma::mat da,
 				arma::Mat<size_t> b, 
 				arma::mat db) {
  Log::Warn << "Comparing results for " << a.n_cols << " queries." << endl;
  assert(a.n_rows == b.n_rows);
  assert(a.n_cols == b.n_cols);
  size_t count_mismatched = 0;

//   Log::Warn << "Mismatches: " << endl;
  for(size_t i = 0; i < a.n_cols;  i++) {
    for (size_t j = 0; j < a.n_rows; j++) {
      if (da(j, i) != db(j, i)) {
	++count_mismatched;
	// Log::Warn << da(i) - db(i) << endl;
      }
    }
  }
    
  Log::Warn << count_mismatched << " / " << a.n_elem
	    << " errors." << endl;

}


// Compute the median rank of the collection
// and the mean precision for all queries.
void compute_error(arma::mat* rdata, arma::mat* qdata,
		   arma::Mat<size_t>* indices, arma::mat* values) {

  assert(indices->n_cols == qdata->n_cols);
  assert(indices->n_rows == values->n_rows);

  arma::Col<size_t> all_ranks_list;
  size_t all_correct = 0;

  size_t k = indices->n_rows;

  all_ranks_list.set_size(qdata->n_cols * indices->n_rows);


  double perc_done = 10.0;
  double done_sky = 1.0;
        
  // do it with a loop over the queries.
  for (size_t i = 0; i < qdata->n_cols; i++) {

    // obtaining the ips
    arma::vec ip_q = arma::trans(arma::trans(qdata->col(i)) 
				 * (*rdata));
      
    assert(ip_q.n_elem == rdata->n_cols);

    // obtaining the ranks
    arma::uvec srt_ind = arma::sort_index(ip_q, sort_type = 1);

    for (size_t j = 0; j < k; j++) {

      size_t rank = srt_ind(indices(j, i)) + 1;
      all_ranks_list( i * k + j ) = rank;
      if (rank < k + 1)
	all_correct++;

    }

    double pdone = i * 100 / qdata.n_cols;

    if (pdone >= done_sky * perc_done) {
      if (done_sky > 1) {
	printf("\b\b\b=%zu%%", (size_t) pdone); fflush(NULL); 
      } else {
	printf("=%zu%%", (size_t) pdone); fflush(NULL);
      }
      done_sky++;
    }

  } // query-loop

  double pdone = 100;

  if (pdone >= done_sky * perc_done) {
    if (done_sky > 1) {
      printf("\b\b\b=%zu%%", (size_t) pdone); fflush(NULL); 
    } else {
      printf("=%zu%%", (size_t) pdone); fflush(NULL);
    }
    done_sky++;
  }
  printf("\n");fflush(NULL);

  Log::Info << "Errors Computed!" << endl;


  double avg_precision = (double) all_correct 
    / (double) (k * qdata->n_cols);

  size_t median_rank = arma::median(all_ranks_list);


  Log::Info << "Avg. Precision@" << k << ": " 
	    << precision << endl 
	    << "Median Rank@" << k << ": "
	    << median_rank << endl;

}


void compute_error(arma::Mat<size_t>* indices, arma::mat* values,
		   size_t rdata_size) {

  assert(indices->n_cols == values->n_cols);
  assert(indices->n_rows == values->n_rows);

  arma::Col<size_t> all_ranks_list;
  size_t all_correct = 0;

  size_t k = indices->n_rows;

  all_ranks_list.set_size(indices->n_cols * indices->n_rows);


  double perc_done = 10.0;
  double done_sky = 1.0;
        
  FILE *rank_file = fopen(CLI::GetParam<string>("rank_file").c_str(), "r");

  // do it with a loop over the queries.
  for (size_t i = 0; i < indices->n_cols; i++) {

    // obtaining the rank list
    arma::Col<size_t> srt_ind;
    srt_ind.set_size(rdata_size);

    if (rank_file != NULL) {
      char *line = NULL;
      size_t len = 0;
      getline(&line, &len, rank_file);

      char *pch = strtok(line, ",\n");
      size_t rank_index = 0;

      while(pch != NULL) {
	srt_ind(rank_index++) = atoi(pch);
	pch = strtok(NULL, ",\n");
      }

      free(line);
      free(pch);
      assert(rank_index == rdata_size);
    }

    for (size_t j = 0; j < k; j++) {

      size_t rank = srt_ind(indices(j, i)) + 1;
      all_ranks_list( i * k + j ) = rank;
      if (rank < k + 1)
	all_correct++;

    }

    double pdone = i * 100 / qdata.n_cols;

    if (pdone >= done_sky * perc_done) {
      if (done_sky > 1) {
	printf("\b\b\b=%zu%%", (size_t) pdone); fflush(NULL); 
      } else {
	printf("=%zu%%", (size_t) pdone); fflush(NULL);
      }
      done_sky++;
    }

  } // query-loop


  fclose(rank_file);

  double pdone = 100;

  if (pdone >= done_sky * perc_done) {
    if (done_sky > 1) {
      printf("\b\b\b=%zu%%", (size_t) pdone); fflush(NULL); 
    } else {
      printf("=%zu%%", (size_t) pdone); fflush(NULL);
    }
    done_sky++;
  }
  printf("\n");fflush(NULL);

  Log::Info << "Errors Computed!" << endl;


  double avg_precision = (double) all_correct 
    / (double) (k * qdata->n_cols);

  size_t median_rank = arma::median(all_ranks_list);


  Log::Info << "Avg. Precision@" << k << ": " 
	    << precision << endl 
	    << "Median Rank@" << k << ": "
	    << median_rank << endl;

}

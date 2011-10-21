#include <mlpack/core.h>

#include <string>
#include <armadillo>

using namespace mlpack;
using namespace std;

/**
 * These function checks if the neighbors computed 
 * by two different methods is the same.
 */

namespace check_nn_utils {

  void invert_index(arma::uvec srt_ind, arma::uvec* inv_ind) {

    inv_ind->set_size(srt_ind.n_elem);

    for (size_t i = 0; i < srt_ind.n_elem; i++)
      (*inv_ind)(srt_ind[i]) = i+1;

    return;
  }

  void count_mismatched_neighbors(arma::Mat<size_t> a, 
				  arma::mat da,
				  arma::Mat<size_t> b, 
				  arma::mat db) {
    Log::Warn << "Comparing results for " 
	      << a.n_cols << " queries." << endl;
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
		     arma::Mat<size_t>* indices) {

    assert(indices->n_cols == qdata->n_cols);

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
      arma::uvec srt_ind = arma::sort_index(ip_q, 1);
      arma::uvec* rank_ind = new arma::uvec();
      invert_index(srt_ind, rank_ind);

      for (size_t j = 0; j < k; j++) {

	assert((*indices)(j, i) != -1);

	if ((*indices)(j, i) != -1) {

	  size_t rank = (*rank_ind)((*indices)(j, i));
	  all_ranks_list( i * k + j ) = rank;
	  if (rank < k + 1)
	    all_correct++;
	} else {
	  // if no result found, just penalize the worst rank
	  all_ranks_list( i * k + j ) = rdata->n_cols + 1;
	}

      }

      double pdone = i * 100 / qdata->n_cols;

      if (pdone >= done_sky * perc_done) {
	if (done_sky > 1) {
	  printf("\b\b\b=%zu%%", (size_t) pdone); fflush(NULL); 
	} else {
	  printf("=%zu%%", (size_t) pdone); fflush(NULL);
	}
	done_sky++;
      }

      srt_ind.reset();
      rank_ind->reset();
      delete(rank_ind);

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
	      << avg_precision << endl 
	      << "Median Rank@" << k << ": "
	      << median_rank << endl;

  }


  void compute_error(string rank_file, size_t rdata_size,
		     arma::Mat<size_t>* indices) {

    arma::Col<size_t> all_ranks_list;
    size_t all_correct = 0;

    size_t k = indices->n_rows;

    all_ranks_list.set_size(indices->n_cols * indices->n_rows);


    double perc_done = 10.0;
    double done_sky = 1.0;
        
    FILE *rank_fp = fopen(rank_file.c_str(), "r");

    // do it with a loop over the queries.
    for (size_t i = 0; i < indices->n_cols; i++) {

      // obtaining the rank list
      arma::Col<size_t> rank_ind;
      rank_ind.set_size(rdata_size);

      if (rank_fp != NULL) {
	char *line = NULL;
	size_t len = 0;
	getline(&line, &len, rank_fp);

	char *pch = strtok(line, ",\n");
	size_t rank_index = 0;

	while(pch != NULL) {
	  rank_ind(rank_index++) = atoi(pch);
	  pch = strtok(NULL, ",\n");
	}

	free(line);
	free(pch);
	assert(rank_index == rdata_size);
      }


      for (size_t j = 0; j < k; j++) {


	if ((*indices)(j, i) != -1) {
	  // assert((*indices)(j, i) != -1);
	  size_t rank = rank_ind((*indices)(j, i)); 
	  all_ranks_list( i * k + j ) = rank;
	  if (rank < k + 1)
	    all_correct++;
	} else {
	  all_ranks_list( i * k + j ) = rdata_size;
	}	
      }

      double pdone = i * 100 / indices->n_cols;

      if (pdone >= done_sky * perc_done) {
	if (done_sky > 1) {
	  printf("\b\b\b=%zu%%", (size_t) pdone); fflush(NULL); 
	} else {
	  printf("=%zu%%", (size_t) pdone); fflush(NULL);
	}
	done_sky++;
      }
    } // query-loop


    fclose(rank_fp);

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
      / (double) (k * indices->n_cols);

    size_t median_rank = arma::median(all_ranks_list);


    Log::Info << "Avg. Precision@" << k << ": " 
	      << avg_precision << endl 
	      << "Median Rank@" << k << ": "
	      << median_rank << endl;

  }

};

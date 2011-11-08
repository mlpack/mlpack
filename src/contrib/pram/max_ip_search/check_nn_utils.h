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

    size_t total_considerable_candidates
      = indices->n_cols * indices->n_rows;

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

	assert(i * k + j < qdata->n_cols * indices->n_rows);

	if ((*indices)(j, i) != (size_t) -1) {

	  size_t rank = (*rank_ind)((*indices)(j, i));
	  all_ranks_list( i * k + j ) = rank;
	  if (rank < k + 1)
	    all_correct++;
	} else {
	  // if no result found, just penalize the worst rank
//	  all_ranks_list( i * k + j ) = rdata->n_cols;
	  all_ranks_list( i * k + j ) = -1;
	  total_considerable_candidates--;
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


//     double avg_precision = (double) all_correct 
//       / (double) (k * qdata->n_cols);

//     size_t median_rank = arma::median(all_ranks_list);

    double avg_precision = (double) all_correct 
      / (double) (total_considerable_candidates);

    arma::uvec tcc = find(all_ranks_list != -1);

    assert(tcc.n_elem == total_considerable_candidates);

    arma::Col<size_t> hemmed_all_ranks_list;
    hemmed_all_ranks_list.set_size(tcc.n_elem);

    for (size_t i = 0; i < tcc.n_elem; i++) 
      hemmed_all_ranks_list(i) = all_ranks_list(tcc(i));

    size_t median_rank = arma::median(hemmed_all_ranks_list);


    Log::Warn << "Avg. Precision@" << k << ": " 
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


    size_t total_considerable_candidates
      = indices->n_cols * indices->n_rows;

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


	if ((*indices)(j, i) != (size_t) -1) {
	  size_t rank = rank_ind((*indices)(j, i)); 
	  all_ranks_list( i * k + j ) = rank;
	  if (rank < k + 1)
	    all_correct++;
	} else {
// 	  all_ranks_list( i * k + j ) = rdata_size;
	  all_ranks_list( i * k + j ) = -1;
	  total_considerable_candidates--;
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



//     double avg_precision = (double) all_correct 
//       / (double) (k * indices->n_cols);

//     size_t median_rank = arma::median(all_ranks_list);

    double avg_precision = (double) all_correct 
      / (double) (total_considerable_candidates);

    arma::uvec tcc = find(all_ranks_list != -1);

    assert(tcc.n_elem == total_considerable_candidates);

    arma::Col<size_t> hemmed_all_ranks_list;
    hemmed_all_ranks_list.set_size(tcc.n_elem);

    for (size_t i = 0; i < tcc.n_elem; i++) 
      hemmed_all_ranks_list(i) = all_ranks_list(tcc(i));

    size_t median_rank = arma::median(hemmed_all_ranks_list);


    Log::Warn << "Avg. Precision@" << k << ": " 
	      << avg_precision << endl 
	      << "Median Rank@" << k << ": "
	      << median_rank << endl
	      << "TCC: " << total_considerable_candidates
	      << endl;

    fclose(rank_fp);

  }



  // Computing errors for the setting where you have 
  // multiple solutions and you want to load up/compute the 
  // ranks only once for every query.
  void compute_error(string rank_file, size_t rdata_size,
		     vector< arma::Mat<size_t>* > solutions,
		     vector<double>* precisions,
		     vector<size_t>* median_ranks) {

    vector< arma::Col<size_t>* > all_ranks_lists;
    vector<size_t> all_corrects;
    vector<size_t> total_considerable_candidates;

    // set up the list first
    for (size_t i = 0; i < solutions.size(); i++) {

      arma::Col<size_t>* all_ranks_list = new arma::Col<size_t>();
      //size_t all_correct = 0;

      all_ranks_list->set_size(solutions[i]->n_cols * solutions[i]->n_rows);
      all_ranks_lists.push_back(all_ranks_list);

      //all_corrects.push_back(all_correct);
      all_corrects.push_back(0);

      total_considerable_candidates.push_back(solutions[i]->n_cols 
					      * solutions[i]->n_rows);
    }


    // 1. Pick up the rank list
    // 2. Go through the solution for that query
    // 3. precision and ranks

    double perc_done = 10.0;
    double done_sky = 1.0;
        
    FILE *rank_fp = fopen(rank_file.c_str(), "r");

    size_t num_queries = solutions[1]->n_cols;
    size_t num_solutions = solutions.size();

    // do it with a loop over the queries.
    for (size_t i = 0; i < num_queries; i++) {

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


      // Going through all the solutions
      for (size_t ind = 0; ind < num_solutions; ind++) {

	size_t k = solutions[ind]->n_rows;

	for (size_t j = 0; j < k; j++) {
	  
	  if ((*solutions[ind])(j, i) != (size_t) -1) {

	    size_t rank = rank_ind((*solutions[ind])(j, i));

	    // for the median
	    (*all_ranks_lists[ind])( i * k + j ) = rank;

	    // precision
	    if (rank < k + 1)
	      all_corrects[ind]++;

	  } else {
	    // if no result found, just penalize the worst rank
// 	    (*all_ranks_lists[ind])( i * k + j ) = rdata_size;
	    (*all_ranks_lists[ind])( i * k + j ) = -1;
	    total_considerable_candidates[ind]--;
	  }

	} // top k neighbors
      } // all solutions for this query

      double pdone = i * 100 / num_queries;

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

    for (size_t ind = 0; ind < num_solutions; ind++) {

//       precisions->push_back((double) all_corrects[ind]
// 			    / (double) all_ranks_lists[ind]->n_elem);
//       median_ranks->push_back(arma::median(*all_ranks_lists[ind]));
      precisions->push_back((double) all_corrects[ind]
			    / (double) total_considerable_candidates[ind]);

      arma::uvec tcc = find((*all_ranks_lists[ind]) != -1);

      assert(tcc.n_elem == total_considerable_candidates[ind]);

      arma::Col<size_t> hemmed_all_ranks_list;
      hemmed_all_ranks_list.set_size(tcc.n_elem);

      for (size_t i = 0; i < tcc.n_elem; i++) 
	hemmed_all_ranks_list(i) = (*all_ranks_lists[ind])(tcc(i));

      median_ranks->push_back(arma::median(hemmed_all_ranks_list));
    
      delete(all_ranks_lists[ind]);
    }

    return;
  }


  void compute_error(arma::mat* rdata, arma::mat* qdata,
		     vector< arma::Mat<size_t>* > solutions,
		     vector<double>* precisions,
		     vector<size_t>* median_ranks) {


    vector< arma::Col<size_t>* > all_ranks_lists;
    vector<size_t> all_corrects;
    vector<size_t> total_considerable_candidates;

    // set up the list first
    for (size_t i = 0; i < solutions.size(); i++) {

      arma::Col<size_t>* all_ranks_list = new arma::Col<size_t>();
      //size_t all_correct = 0;

      all_ranks_list->set_size(solutions[i]->n_cols * solutions[i]->n_rows);
      all_ranks_lists.push_back(all_ranks_list);

      all_corrects.push_back(0);
      total_considerable_candidates.push_back(solutions[i]->n_cols
					      * solutions[i]->n_rows);
    }


    // 1. Pick up the rank list
    // 2. Go through the solution for that query
    // 3. precision and ranks

    double perc_done = 10.0;
    double done_sky = 1.0;
        
    size_t num_queries = solutions[1]->n_cols;
    assert(num_queries == qdata->n_cols);
    size_t num_solutions = solutions.size();

    // do it with a loop over the queries.
    for (size_t i = 0; i < num_queries; i++) {

      // obtaining the rank list

      // obtaining the ips
      arma::vec ip_q = arma::trans(arma::trans(qdata->col(i)) 
				   * (*rdata));
      
      assert(ip_q.n_elem == rdata->n_cols);

      // obtaining the ranks
      arma::uvec* srt_ind = new arma::uvec(arma::sort_index(ip_q, 1));
      arma::uvec* rank_ind = new arma::uvec();
      invert_index(*srt_ind, rank_ind);

      // Going through all the solutions
      for (size_t ind = 0; ind < num_solutions; ind++) {

	size_t k = solutions[ind]->n_rows;

	for (size_t j = 0; j < k; j++) {

	  if ((*solutions[ind])(j, i) != (size_t) -1) {
	    size_t rank = (*rank_ind)((*solutions[ind])(j, i));

	    // median rank
	    (*all_ranks_lists[ind])( i * k + j ) = rank;
	    // precision
	    if (rank < k + 1)
	      all_corrects[ind]++;
	  } else {
// 	    (*all_ranks_lists[ind])( i * k + j ) = rdata->n_cols;
	    (*all_ranks_lists[ind])( i * k + j ) = -1;
	    total_considerable_candidates[ind]--;
	  }
	} // top k neighbors
      } // all solutions for this query

      double pdone = i * 100 / num_queries;

      if (pdone >= done_sky * perc_done) {
	if (done_sky > 1) {
	  printf("\b\b\b=%zu%%", (size_t) pdone); fflush(NULL); 
	} else {
	  printf("=%zu%%", (size_t) pdone); fflush(NULL);
	}
	done_sky++;
      }

      delete(rank_ind);
      delete(srt_ind);

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

    for (size_t ind = 0; ind < num_solutions; ind++) {

//       precisions->push_back((double) all_corrects[ind]
// 			    / (double) all_ranks_lists[ind]->n_elem);
//       median_ranks->push_back(arma::median(*all_ranks_lists[ind]));

      precisions->push_back((double) all_corrects[ind]
			    / (double) total_considerable_candidates[ind]);

      arma::uvec tcc = find((*all_ranks_lists[ind]) != -1);

      assert(tcc.n_elem == total_considerable_candidates[ind]);

      arma::Col<size_t> hemmed_all_ranks_list;
      hemmed_all_ranks_list.set_size(tcc.n_elem);

      for (size_t i = 0; i < tcc.n_elem; i++) 
	hemmed_all_ranks_list(i) = (*all_ranks_lists[ind])(tcc(i));

      median_ranks->push_back(arma::median(hemmed_all_ranks_list));

      delete(all_ranks_lists[ind]);
    }

    return;
  }





  void check_rank_bound(string rank_file, size_t rdata_size,
			double epsilon, double alpha,
			arma::Mat<size_t>* indices) {

    size_t all_correct = 0;

    size_t k = indices->n_rows;

    size_t rank_error = (size_t) ( epsilon * (double) rdata_size / 100.0);

    size_t total_considerable_candidates = indices->n_cols;

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


//       for (size_t j = 0; j < k; j++) {


      if ((*indices)(k-1, i) != (size_t) -1) {
	size_t rank = rank_ind((*indices)(k -1, i)); 
	if (rank < rank_error +1)
	  all_correct++;
      } else {
	total_considerable_candidates--;
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


//     double avg_precision = (double) all_correct 
//       / (double) (k * indices->n_cols);

//     size_t median_rank = arma::median(all_ranks_list);

    double avg_precision = (double) all_correct 
      / (double) (indices->n_cols);


    Log::Warn << "Actual Alpha @" << k << ": " 
	      << avg_precision << endl 
	      << "TCC: " << total_considerable_candidates
	      << endl << "Alpha: " << alpha << endl
	      << "Rank Error: " << rank_error << endl;
  }

}; // end check_nn_utils

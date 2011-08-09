#include <string>
#include "approx_nn_dual_with_stats.h"
//#include "fastlib/fx/fx.h"

const fx_entry_doc approx_nn_main_dual_entries[] = {
  {"r", FX_REQUIRED, FX_STR, NULL,
   " A file containing the reference set.\n"},
  {"q", FX_PARAM, FX_STR, NULL,
   " A file containing the query set"
   " (defaults to the reference set).\n"},
  {"lr", FX_PARAM, FX_STR, NULL,
   " A file containing the labels for"
   " the reference dataset.\n"},
  {"lq", FX_PARAM, FX_STR, NULL,
   " A file containing the labels for"
   " the query dataset.\n"},
  {"donaive", FX_PARAM, FX_BOOL, NULL,
   " A variable which decides whether we do"
   " the naive computation(defaults to false).\n"},
  {"doexact", FX_PARAM, FX_BOOL, NULL,
   " A variable which decides whether we do"
   " the exact computation"
   "(defaults to false).\n"},
  {"doapprox", FX_PARAM, FX_BOOL, NULL,
   " A variable which decides whether we do"
   " the approximate computation(defaults to false).\n"},
  {"doapprox_nt", FX_PARAM, FX_BOOL, NULL,
   " A variable which decides whether we do"
   " the approximate computation without stratified "
   "sampling (defaults to false).\n"},
  {"compute_error", FX_PARAM, FX_BOOL, NULL,
   " Whether to compute the rank and distance error"
   " for the approximate results.\n"},
  {"rank_matrix", FX_PARAM, FX_STR, NULL,
   " The file containing the rank of the reference points"
   " with respect to the set of queries. Required only "
   "for computing the true rank errors and failure "
   "probabilities.\n"},
  {"result_file", FX_PARAM, FX_STR, NULL,
   " The file in which the nearest neighbor results are to be output"
   " (defaults to 'results.txt')\n"},
  {"print_results", FX_PARAM, FX_BOOL, NULL,
   " Whether to print out results or not\n"},
  {"reps", FX_PARAM, FX_INT, NULL,
   "The number of times the RANN should be computed"
   " (defaults to 1).\n"},
  {"epsilons", FX_PARAM, FX_STR, NULL,
   "Different values of epsilon to be tried.\n"},
  {"alphas", FX_PARAM, FX_STR, NULL,
   "Different values of alpha to be tried.\n"},
  FX_ENTRY_DOC_DONE
};

const fx_submodule_doc approx_nn_main_dual_submodules[] = {
  {"ann", &approx_nn_dual_doc,
   " Responsible for doing approximate nearest neighbor"
   " search using sampling on kd-trees.\n"},
  FX_SUBMODULE_DOC_DONE
};

const fx_module_doc approx_nn_main_dual_doc = {
  approx_nn_main_dual_entries, approx_nn_main_dual_submodules,
  "This is a program to test run the approx "
  " nearest neighbors using sampling on kd-trees.\n"
  "It performs the exact, approximate"
  " and the naive computation.\n"
};

void compute_classification_error(Matrix& lrdata, Matrix& lqdata,
				  ArrayList<index_t>& neighbors,
				  index_t knns, FILE* fout);

int main (int argc, char *argv[]) {

  srand( time(NULL));

  fx_module *root
    = fx_init(argc, argv, &approx_nn_main_dual_doc);

  Matrix qdata, rdata;
  Matrix lqdata, lrdata;

  std::string rfile = fx_param_str_req(root, "r");
  NOTIFY("Loading files...");
  data::Load(rfile.c_str(), &rdata);
  if (fx_param_exists(root, "lr")) {
    std::string lrfile = fx_param_str_req(root, "lr");
    data::Load(lrfile.c_str(), &lrdata);
    DEBUG_ASSERT(lrdata.n_rows() == 1);
    DEBUG_ASSERT(lrdata.n_cols() == rdata.n_cols());
  } else {
    lqdata.Init(0,0);
    lrdata.Init(0,0);
  }

  if (fx_param_exists(root, "q")) {
    std::string qfile = fx_param_str_req(root, "q");
    data::Load(qfile.c_str(), &qdata);
    if (fx_param_exists(root, "lr")) {
      std::string lqfile = fx_param_str_req(root, "lq");
      data::Load(lqfile.c_str(), &lqdata);
      DEBUG_ASSERT(lqdata.n_rows() == 1);
      DEBUG_ASSERT(lqdata.n_cols() == qdata.n_cols());
    }
  } else {
    qdata.Copy(rdata);
    if (fx_param_exists(root, "lr")) {
      lqdata.Copy(lrdata);
    }
  }

  NOTIFY("File loaded...");

  struct datanode *ann_module
    = fx_submodule(root, "ann");

  ArrayList<index_t> nac, exc, apc, ap_nt_c;
  ArrayList<double> din, die, dia, di_nt_a;
  index_t knns = fx_param_int(ann_module, "knns", 1);
  std::string result_file = fx_param_str(root, "result_file", "result.txt");


  // Naive computation
  if (fx_param_bool(root, "donaive", false)) {
    ApproxNN naive_nn;
    NOTIFY("Naive");
    NOTIFY("Init");
    if (fx_param_exists(root, "q")) {
      fx_timer_start(ann_module, "naive_init");
      naive_nn.InitNaive(qdata, rdata, knns);
      fx_timer_stop(ann_module, "naive_init");
    } else {
      fx_timer_start(ann_module, "naive_init");
      naive_nn.InitNaive(rdata, knns);
      fx_timer_stop(ann_module, "naive_init");
    }

    NOTIFY("Compute");
    fx_timer_start(ann_module, "naive");
    naive_nn.ComputeNaive(&nac, &din);
    fx_timer_stop(ann_module, "naive");

    if (fx_param_bool(root, "print_results", true)) {
      FILE *fp=fopen(result_file.c_str(), "w");
      if (fp==NULL)
	FATAL("Error while opening %s...%s", result_file.c_str(),
	      strerror(errno));

      for(index_t i=0 ; i < nac.size()/knns ; i++) {
	fprintf(fp, "%"LI"d", i);
	for(index_t j=0; j<knns; j++)
	  fprintf(fp, ",%"LI"d,%lg", 
		  nac[i*knns+j], din[i*knns+j]);
	fprintf(fp, "\n");
      }
      fclose(fp);
    }
  }

  // Exact computation
  if (fx_param_bool(root, "doexact", false)) {
    ApproxNN exact_nn;
    NOTIFY("Exact");
    NOTIFY("Init");
    if (fx_param_exists(root, "q")) {
      fx_timer_start(ann_module, "exact_init");
      exact_nn.Init(qdata, rdata, ann_module);
      fx_timer_stop(ann_module, "exact_init");
    } else {
      fx_timer_start(ann_module, "exact_init");
      exact_nn.Init(rdata, ann_module);
      fx_timer_stop(ann_module, "exact_init");
    }

    NOTIFY("Compute");
    fx_timer_start(ann_module, "exact");
    exact_nn.ComputeNeighbors(&exc, &die);
    fx_timer_stop(ann_module, "exact");

    if (fx_param_bool(root, "print_results", true)) {
      FILE *fp=fopen(result_file.c_str(), "w");
      if (fp==NULL)
	FATAL("Error while opening %s...%s", result_file.c_str(),
	      strerror(errno));

      for(index_t i=0 ; i < exc.size()/knns ; i++) {
	fprintf(fp, "%"LI"d", i);
	for(index_t j=0; j<knns; j++)
	  fprintf(fp, ",%"LI"d,%lg",
		  exc[i*knns+j], die[i*knns+j]);
	fprintf(fp, "\n");
      }
      fclose(fp);
    }
		
    /**
     * computing the kNN classification error 
     * if the labels are provided */ 
    if (fx_param_exists(root, "lr")) {
      FILE *fout = fopen(fx_param_str(root, "lab_res_file",
				      "lab_res_file.txt"),"w");
      compute_classification_error(lrdata, lqdata, exc, knns, fout);
      fclose(fout);
    }
  }



  // NOTIFY("HERE");
  std::vector<double> eps;
  if (fx_param_exists(root, "epsilons")) {
    if(fx_param_bool(root, "doapprox", false) 
       || fx_param_bool(root, "doapprox_nt", false)) {

      // take the string and split on commas
      // and store in an array.
      NOTIFY("HERE");
      std::string epsilons = fx_param_str_req(root, "epsilons");
      NOTIFY("%s", epsilons.c_str());
      char *temp = (char *) epsilons.c_str();
      char *epsilon = strtok(temp, ","); 
      while (epsilon != NULL) {
	double e = atof(epsilon);
	eps.push_back(e);
	epsilon = strtok(NULL, ",");
      }
    }
  }

  NOTIFY("%"LI"d epsilons", (index_t) eps.size());


  // Approximate computation
  if (fx_param_bool(root, "doapprox", false)) {
    ApproxNN approx_nn;
    NOTIFY("Approx");
    NOTIFY("Init");
    if (fx_param_exists(root, "q")) {

      fx_timer_start(ann_module, "approx_init");
      approx_nn.InitApprox(qdata, rdata, ann_module);
      fx_timer_stop(ann_module, "approx_init");
    } else {
      fx_timer_start(ann_module, "approx_init");
      approx_nn.InitApprox(rdata, ann_module);
      fx_timer_stop(ann_module, "approx_init");
    }

    // exit(0);

    GenMatrix<index_t> rank_matrix;
    bool ranks_given = false;
    if (fx_param_bool(root, "compute_error", false)) {
      Matrix temp;
      if (fx_param_exists(root, "rank_matrix")) {
	ranks_given = true;
	std::string rank_file = fx_param_str_req(root, "rank_matrix");
	NOTIFY("Loading rank file...huge file will take time");
	data::Load(rank_file.c_str(), &temp);
	
	rank_matrix.Init(temp.n_rows(), temp.n_cols());
      
	for (index_t i = 0; i < temp.n_cols(); i++)
	  for (index_t j = 0; j < temp.n_rows(); j++)
	    rank_matrix.set(j, i, (index_t) temp.get(j, i));

	NOTIFY("Done loading Rank file");
      } else {
	temp.Init(0,0);
	rank_matrix.Init(0,0);
      }
    }

    // exit(0);
    for (index_t eps_it = 0; eps_it < (index_t) eps.size(); eps_it++) { 

      fx_set_param_double(ann_module, "epsilon", eps[eps_it]);

      fx_timer_start(ann_module, "approx_init");
      approx_nn.InitSampleSizes();
      fx_timer_stop(ann_module, "approx_init");

      for (index_t i = 0; i < fx_param_int(root, "reps", 1); i++) {

	NOTIFY("Compute %"LI"d", i+1);
	fx_timer_start(ann_module, "approx");
	approx_nn.ComputeApprox(&apc, &dia);
	fx_timer_stop(ann_module, "approx");

	if (fx_param_bool(root, "print_results", false)) {
	  FILE *fp=fopen(result_file.c_str(), "w");
	  if (fp==NULL)
	    FATAL("Error while opening %s...%s", result_file.c_str(),
		  strerror(errno));
      
	  for(index_t i=0 ; i < apc.size()/knns ; i++) {
	    fprintf(fp, "%"LI"d", i);
	    for(index_t j=0; j<knns; j++)
	      fprintf(fp, ",%"LI"d,%lg", 
		      apc[i*knns+j], dia[i*knns+j]);
	    fprintf(fp, "\n");
	  }
	  fclose(fp);
	}

	/**
	 * computing the kNN classification error 
	 * if the labels are provided 
	 */ 
	if (fx_param_exists(root, "lr")) {
	  FILE *fout = fopen(fx_param_str(root, "lab_res_file",
					  "lab_res_file.txt"),"w");
	  compute_classification_error(lrdata, lqdata, apc, knns, fout);
	  fclose(fout);
	}


	/**
	 * Computing the error of the approximate neighbors
	 * Must provide the file containing the rank matrix 
	 */
	if (fx_param_bool(root, "compute_error", false)) {

	  double epsilon
	    = fx_param_double_req(ann_module, "epsilon");
	  index_t rank_error
	    = (index_t) (epsilon * (double) rdata.n_cols() / 100);
	  double alpha
	    = fx_param_double_req(ann_module, "alpha");
      
	  index_t avg_rank_error = 0,
	    max_rank = -1, min_rank = rdata.n_cols(),
	    max_k_rank = -1, min_k_rank = rdata.n_cols();

	  for (index_t i = 0; i < apc.size() / knns ; i++) {

	    if (i % 1000 == 0) 
	      if (ranks_given && !fx_param_exists(root, "q")) 
		DEBUG_ASSERT(rank_matrix.get(i,i) == 0);
				
	    for (index_t j = 0; j < knns; j++) {
	      if (ranks_given) {
		index_t knn_rank = rank_matrix.get(apc[i*knns+j], i);
		avg_rank_error += knn_rank - (j+1);
						
		if (j == 0) {
		  if (knn_rank > max_rank)
		    max_rank = knn_rank;
		  if (knn_rank < min_rank)
		    min_rank = knn_rank;
		}
						
		if (j == knns-1) {
		  if (knn_rank > max_k_rank)
		    max_k_rank = knn_rank;
		  if (knn_rank < min_k_rank)
		    min_k_rank = knn_rank;
		}
	      }
	    }
	  }
			
	  // NOTIFY("Rank Error: %"LI"d", rank_error);
	  if (ranks_given) {
	    // 	NOTIFY("XR: %"LI"d NR: %"LI"d XKR: %"LI"d NKR: %"LI"d",
	    // 	       max_rank, min_rank, max_k_rank, min_k_rank);
	    // NOTIFY("ARE: %lg", (double) avg_rank_error 
	    // / (double) (knns * qdata.n_cols()));
	  }

	  // computing average rank error
	  // and probability of failure
	  index_t re = 0, failed = 0;
	  index_t max_er = 0, min_er = rdata.n_cols();
	  for (index_t i = 0; i < apc.size() / knns; i++) {
	    if (rank_matrix.get(apc[(i+1)*knns -1], i) > max_er)
	      max_er = rank_matrix.get(apc[(i+1)*knns -1], i) -1;
	
	    if (rank_matrix.get(apc[(i+1)*knns -1], i) < min_er)
	      min_er = rank_matrix.get(apc[(i+1)*knns -1], i) -1;
					
	    if (rank_matrix.get(apc[(i+1)*knns -1], i) -1 > rank_error)
	      failed++;
					
	    re += rank_matrix.get(apc[(i+1)*knns -1], i) -1;
	  }
	  double avg_rank = (double) re / (double) qdata.n_cols();
	  double success_prob = (double) (qdata.n_cols() - failed)
	    / (double) qdata.n_cols();
      
	  //NOTIFY("Required rank error: %"LI"d,"
	  // " Required success Prob = %1.2lf",
	  // rank_error, alpha);

	  //NOTIFY("True Avg k Rank error: %6.2lf,"
	  // " True success prob = %1.2lf,",
	  // avg_rank, success_prob);
	  printf("%0.2lf,%"LI"d,%1.2lf\n", avg_rank, max_er, 
		 success_prob);
      
	  //NOTIFY("Max error: %"LI"d,"
	  // " Min error: %"LI"d",
	  // max_er, min_er);
	} // compute_error

	// NOTIFY("Quick Clean Up");
	approx_nn.QuickCleanUp();
	apc.Renew();
	dia.Renew();
      } // reps

    } // for eps_it

  } // doapprox

  // Approximate computation without stratified sampling
  if (fx_param_bool(root, "doapprox_nt", false)) {
    ApproxNN approx_nn;
    NOTIFY("Approx without Stratified Sampling");
    NOTIFY("Init");
    if (fx_param_exists(root, "q")) {
      fx_timer_start(ann_module, "approx_init");
      approx_nn.InitApprox(qdata, rdata, ann_module);
      fx_timer_stop(ann_module, "approx_init");
    } else {
      fx_timer_start(ann_module, "approx_init");
      approx_nn.InitApprox(rdata, ann_module);
      fx_timer_stop(ann_module, "approx_init");
    }

    GenMatrix<index_t> rank_matrix;
    bool ranks_given = false;
    if (fx_param_bool(root, "compute_error", false)) {
      Matrix temp;
      if (fx_param_exists(root, "rank_matrix")) {
	ranks_given = true;
	std::string rank_file = fx_param_str_req(root, "rank_matrix");
	NOTIFY("Loading rank file...huge file will take time");
	data::Load(rank_file.c_str(), &temp);
	
	rank_matrix.Init(temp.n_rows(), temp.n_cols());
      
	for (index_t i = 0; i < temp.n_cols(); i++)
	  for (index_t j = 0; j < temp.n_rows(); j++)
	    rank_matrix.set(j, i, (index_t) temp.get(j, i));

	NOTIFY("Done loading Rank file");
      } else {
	temp.Init(0,0);
	rank_matrix.Init(0,0);
      }
    }

    // exit(0);
    for (index_t i = 0; i < fx_param_int(root, "reps", 1); i++) {

      NOTIFY("Compute %"LI"d", i+1);
      // exit(0);
      // NOTIFY("Compute");
      fx_timer_start(ann_module, "approx");
      approx_nn.ComputeApproxNoTree(&apc, &dia);
      fx_timer_stop(ann_module, "approx");

      if (fx_param_bool(root, "print_results", true)) {
	FILE *fp=fopen(result_file.c_str(), "w");
	if (fp==NULL)
	  FATAL("Error while opening %s...%s", result_file.c_str(),
		strerror(errno));
      
	for(index_t i=0 ; i < apc.size()/knns ; i++) {
	  fprintf(fp, "%"LI"d", i);
	  for(index_t j=0; j<knns; j++)
	    fprintf(fp, ",%"LI"d,%lg", 
		    apc[i*knns+j], dia[i*knns+j]);
	  fprintf(fp, "\n");
	}
	fclose(fp);
      }

      /**
       * computing the kNN classification error 
       * if the labels are provided 
       */ 
      if (fx_param_exists(root, "lr")) {
	FILE *fout = fopen(fx_param_str(root, "lab_res_file",
					"lab_res_file.txt"),"w");
	compute_classification_error(lrdata, lqdata, apc, knns, fout);
	fclose(fout);
      }


      /**
       * Computing the error of the approximate neighbors
       * Must provide the file containing the rank matrix 
       */
      if (fx_param_bool(root, "compute_error", false)) {

	double epsilon
	  = fx_param_double_req(ann_module, "epsilon");
	index_t rank_error
	  = (index_t) (epsilon * (double) rdata.n_cols() / 100);
	double alpha
	  = fx_param_double_req(ann_module, "alpha");
      
	index_t avg_rank_error = 0,
	  max_rank = -1, min_rank = rdata.n_cols(),
	  max_k_rank = -1, min_k_rank = rdata.n_cols();

	for (index_t i = 0; i < apc.size() / knns ; i++) {

	  if (i % 1000 == 0) 
	    if (ranks_given && !fx_param_exists(root, "q")) 
	      DEBUG_ASSERT(rank_matrix.get(i,i) == 0);
				
	  for (index_t j = 0; j < knns; j++) {
	    if (ranks_given) {
	      index_t knn_rank = rank_matrix.get(apc[i*knns+j], i);
	      avg_rank_error += knn_rank - (j+1);
						
	      if (j == 0) {
		if (knn_rank > max_rank)
		  max_rank = knn_rank;
		if (knn_rank < min_rank)
		  min_rank = knn_rank;
	      }
						
	      if (j == knns-1) {
		if (knn_rank > max_k_rank)
		  max_k_rank = knn_rank;
		if (knn_rank < min_k_rank)
		  min_k_rank = knn_rank;
	      }
	    }
	  }
	}
			
	NOTIFY("Rank Error: %"LI"d", rank_error);
	if (ranks_given) {
	  // 	NOTIFY("XR: %"LI"d NR: %"LI"d XKR: %"LI"d NKR: %"LI"d",
	  // 	       max_rank, min_rank, max_k_rank, min_k_rank);
	  NOTIFY("ARE: %lg", (double) avg_rank_error 
		 / (double) (knns * qdata.n_cols()));
	}

	// computing average rank error
	// and probability of failure
	index_t re = 0, failed = 0;
	index_t max_er = 0, min_er = rdata.n_cols();
	for (index_t i = 0; i < apc.size() / knns; i++) {
	  if (rank_matrix.get(apc[(i+1)*knns -1], i) > max_er)
	    max_er = rank_matrix.get(apc[(i+1)*knns -1], i) -1;
	
	  if (rank_matrix.get(apc[(i+1)*knns -1], i) < min_er)
	    min_er = rank_matrix.get(apc[(i+1)*knns -1], i) -1;
					
	  if (rank_matrix.get(apc[(i+1)*knns -1], i) -1 > rank_error)
	    failed++;
					
	  re += rank_matrix.get(apc[(i+1)*knns -1], i) -1;
	}
	double avg_rank = (double) re / (double) qdata.n_cols();
	double success_prob = (double) (qdata.n_cols() - failed)
	  / (double) qdata.n_cols();
      
	NOTIFY("Required rank error: %"LI"d,"
	       " Required success Prob = %1.2lf",
	       rank_error, alpha);

	NOTIFY("True Avg k Rank error: %6.2lf,"
	       " True success prob = %1.2lf,",
	       avg_rank, success_prob);
      
	NOTIFY("Max error: %"LI"d,"
	       " Min error: %"LI"d",
	       max_er, min_er);
      } // compute_error
      NOTIFY("Quick Clean Up");
      approx_nn.QuickCleanUp();
      apc.Renew();
      dia.Renew();
    } // reps
  } // doapprox_nt

  fx_done(fx_root);
} // end main


/**
 * computing the kNN classification error 
 * if the labels are provided */ 
void compute_classification_error(Matrix& lrdata, Matrix& lqdata,
				  ArrayList<index_t>& neighbors,
				  index_t knns, FILE* fout) {
 
  DEBUG_ASSERT(neighbors.size() / knns == lqdata.n_cols());
  
  NOTIFY("Outputting Labels & Computing Error");

  index_t error = 0;
  index_t no_maj = 0;
  
  for (index_t i = 0; i < neighbors.size() / knns ; i++) {
    index_t true_l = (index_t) lqdata.get(0, i);
	
    ArrayList<index_t> knn_l;
    knn_l.Init(knns);
    fprintf(fout, "%"LI"d L%"LI"d", i, true_l);
	
    for (index_t j = 0; j < knns; j++) {
      knn_l[j] = (index_t) lrdata.get(0, neighbors[i*knns + j]);
      fprintf(fout, ",%"LI"d L%"LI"d", neighbors[i*knns+j], knn_l[j]);
    }
	
    fprintf(fout, "\n");
	
    index_t maj_l = -1, maj_n = 0, temp_n = 0;
    for (index_t j = 0; j < knns; j++) {
      if (maj_l != knn_l[j]) {
	temp_n = 0;
	for (index_t k = 0; k < knns; k++) {
	  if (knn_l[j] == knn_l[k] && j!= k)
	    temp_n++;
	}
	if (temp_n > maj_n) {
	  maj_n = temp_n;
	  maj_l = knn_l[j];
	}
      }
    }
	
    if (maj_l == -1 || maj_l != true_l)
      error++;

    if (maj_l == -1) 
      no_maj++;
  }
			
  NOTIFY("Error: %"LI"d / %"LI"d, NM:%"LI"d",
	 error, lqdata.n_cols(), no_maj);
  printf("Error: %"LI"d/ %"LI"d\n, NM:%"LI"d",
	 error, lqdata.n_cols(), no_maj);
  fclose(fout);
}

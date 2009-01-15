#include <string>
#include "approx_nn_dual.h"

const fx_entry_doc approx_nn_main_dual_entries[] = {
  {"r", FX_REQUIRED, FX_STR, NULL,
   " A file containing the reference set.\n"},
  {"q", FX_PARAM, FX_STR, NULL,
   " A file containing the query set"
   " (defaults to the reference set).\n"},
  {"donaive", FX_PARAM, FX_BOOL, NULL,
   " A variable which decides whether we do"
   " the naive computation(defaults to false).\n"},
  {"doexact", FX_PARAM, FX_BOOL, NULL,
   " A variable which decides whether we do"
   " the exact computation"
   "(defaults to true).\n"},
  {"doapprox", FX_PARAM, FX_BOOL, NULL,
   " A variable which decides whether we do"
   " the approximate computation(defaults to true).\n"},
  {"compute_error", FX_PARAM, FX_BOOL, NULL,
   " Whether to compute the rank and distance error"
   " for the approximate results.\n"},
  {"result_file", FX_PARAM, FX_STR, NULL,
   " The file in which the nearest neighbor results are to be output"
   " (defaults to 'results.txt')\n"},
  {"print_results", FX_PARAM, FX_BOOL, NULL,
   " Whether to print out results or not\n"},
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

/**
 * This function checks if the neighbors computed 
 * by two different methods is the same.
 */
void compare_neighbors(ArrayList<index_t>*, ArrayList<double>*, 
                       ArrayList<index_t>*, ArrayList<double>*);

void count_mismatched_neighbors(ArrayList<index_t>*,
				ArrayList<double>*, 
				ArrayList<index_t>*,
				ArrayList<double>*);

void find_rank_dist(Matrix&, Matrix&, ArrayList<index_t>*,
		    ArrayList<double>*, ArrayList<index_t>*,
		    ArrayList<double>*);

int main (int argc, char *argv[]) {

  srand( time(NULL));

  fx_module *root
    = fx_init(argc, argv, &approx_nn_main_dual_doc);

  Matrix qdata, rdata;
  std::string rfile = fx_param_str_req(root, "r");
  NOTIFY("Loading files...");
  data::Load(rfile.c_str(), &rdata);
  if (fx_param_exists(root, "q")) {
    std::string qfile = fx_param_str_req(root, "q");
    data::Load(qfile.c_str(), &qdata);
  } else {
    qdata.Copy(rdata);
  }

  NOTIFY("File loaded...");

  struct datanode *ann_module
    = fx_submodule(root, "ann");


  ArrayList<index_t> nac, exc, apc;
  ArrayList<double> din, die, dia;

  index_t knns = fx_param_int(ann_module, "knns", 1);
  std::string result_file = fx_param_str(root, "result_file", "result.txt");

  // Part where I check how table is made
//   ApproxNN test;
//   ArrayList<index_t> samples;
//   samples.Init(fx_param_int_req(root, "N"));
//   test.ComputeSampleSizes_(fx_param_int_req(root, "n"), 0.90,
// 			   &samples);

//   for (index_t i = 0; i < samples.size(); i++) {
//     printf("%"LI"d ", samples[i]);
//   }

  // This part ends here

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
      if (fp==NULL) {
	FATAL("Error while opening %s...%s", result_file.c_str(),
	      strerror(errno));
      }
      for(index_t i=0 ; i < nac.size()/knns ; i++) {
	for(index_t j=0; j<knns; j++) {
	  fprintf(fp, "%"LI"d %"LI"d %lg\n", i,
		  nac[i*knns+j], din[i*knns+j]);
	}
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
      if (fp==NULL) {
	FATAL("Error while opening %s...%s", result_file.c_str(),
	      strerror(errno));
      }
      for(index_t i=0 ; i < exc.size()/knns ; i++) {
	fprintf(fp, "%"LI"d", i);
	for(index_t j=0; j<knns; j++) {
	  fprintf(fp, ",%"LI"d",
		  exc[i*knns+j]);
	}
	fprintf(fp, "\n");
      }
      fclose(fp);
    }
  }

  //compare_neighbors(&nac, &din, &exc, &die);

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

    NOTIFY("Compute");
    fx_timer_start(ann_module, "approx");
    approx_nn.ComputeApprox(&apc, &dia);
    fx_timer_stop(ann_module, "approx");

    if (fx_param_bool(root, "print_results", true)) {
      FILE *fp=fopen(result_file.c_str(), "w");
      if (fp==NULL) {
	FATAL("Error while opening %s...%s", result_file.c_str(),
	      strerror(errno));
      }
      for(index_t i=0 ; i < apc.size()/knns ; i++) {
	for(index_t j=0; j<knns; j++) {
	  fprintf(fp, "%"LI"d %"LI"d %lg\n", i,
		  apc[i*knns+j], dia[i*knns+j]);
	}
      }
      fclose(fp);
    }

    if (fx_param_bool(root, "compute_error", false)) {
      double epsilon
	= fx_param_double_req(ann_module, "epsilon");
      index_t rank_error
	= (index_t) (epsilon * (double) rdata.n_cols()
		     / 100);
      double alpha
	= fx_param_double_req(ann_module, "alpha");

      ArrayList<index_t> rank_errors;
      ArrayList<double> true_dist;
      find_rank_dist(qdata, rdata, &apc, &dia,
		     &rank_errors, &true_dist);

      // computing average rank error
      // and probability of failure
      index_t re = 0, failed = 0;
      DEBUG_ASSERT(rank_errors.size() == qdata.n_cols());

      index_t max_er = 0, min_er = rdata.n_cols();

      if (fx_param_exists(root, "q")) {
	for (index_t i = 0; i < rank_errors.size(); i++) {
	  if (rank_errors[i] > max_er) {
	    max_er = rank_errors[i];
	  }
	  if (rank_errors[i] < min_er) {
	    min_er = rank_errors[i];
	  }
	  if (rank_errors[i] > rank_error) {
	    failed++;
	  }
	  re += rank_errors[i];
	}
      } else {
	for (index_t i = 0; i < rank_errors.size(); i++) {
	  if (rank_errors[i]-1 > max_er) {
	    max_er = rank_errors[i] -1;
	  }
	  if (rank_errors[i]-1 < min_er) {
	    min_er = rank_errors[i]-1;
	  }
	  if (rank_errors[i] -1> rank_error) {
	    failed++;
	  }
	  re += rank_errors[i] -1;
	}
      }
      double avg_rank = (double) re / (double) qdata.n_cols();
      double success_prob = (double) (qdata.n_cols() - failed)
	/ (double) qdata.n_cols();

      // computing average distance error
      double de = 0.0;
      for (index_t i = 0; i < true_dist.size(); ++i) {
	de += (sqrt(dia[i]) - sqrt(true_dist[i]))
	  / sqrt(true_dist[i]);
      }
      double avg_de = de / (double) qdata.n_cols();
      NOTIFY("Required rank error: %"LI"d,"
	     " Required success Prob = %1.2lf",
	     rank_error, alpha);

      NOTIFY("True Avg Rank error: %6.2lf,"
	     " True success prob = %1.2lf,"
	     " Avg de = %1.2lf",
	     avg_rank, success_prob, avg_de);

      NOTIFY("Max error: %"LI"d,"
	     " Min error: %"LI"d",
	     max_er, min_er);
    }
  }
  
  //  count_mismatched_neighbors(&exc, &die, &apc, &dia);

  fx_done(fx_root);
}

void compare_neighbors(ArrayList<index_t> *a, 
                       ArrayList<double> *da,
                       ArrayList<index_t> *b, 
                       ArrayList<double> *db) {
  
  NOTIFY("Comparing results for %"LI"d queries", a->size());
  DEBUG_SAME_SIZE(a->size(), b->size());
  index_t *x = a->begin();
  index_t *y = a->end();
  index_t *z = b->begin();

  for(index_t i = 0; x != y; x++, z++, i++) {
    DEBUG_WARN_MSG_IF(*x != *z || (*da)[i] != (*db)[i], 
                      "point %"LI"d brute: %"LI"d:%lf"
		      " fast: %"LI"d:%lf",
                      i, *z, (*db)[i], *x, (*da)[i]);
  }
}

void count_mismatched_neighbors(ArrayList<index_t> *a, 
				ArrayList<double> *da,
				ArrayList<index_t> *b, 
				ArrayList<double> *db) {

  NOTIFY("Comparing results for %"LI"d queries", a->size());
  DEBUG_SAME_SIZE(a->size(), b->size());
  index_t *x = a->begin();
  index_t *y = a->end();
  index_t *z = b->begin();
  index_t count_mismatched = 0;

  for(index_t i = 0; x != y; x++, z++, i++) {
    if (*x != *z || (*da)[i] != (*db)[i]) {
      ++count_mismatched;
    }
  }
  NOTIFY("%"LI"d/%"LI"d errors", count_mismatched, a->size());
}

void find_rank_dist(Matrix &query, Matrix &reference,
		    ArrayList<index_t> *in,
		    ArrayList<double> *dist,
		    ArrayList<index_t> *rank_error,
		    ArrayList<double> *true_nn_dist) {

  DEBUG_SAME_SIZE(in->size(), dist->size());
  DEBUG_SAME_SIZE(in->size(), query.n_cols());
  // initialize the rank and dist error vector
  rank_error->Init(in->size());
  true_nn_dist->Init(in->size());

  // Looping over the queries
  for (index_t i = 0; i < query.n_cols(); i++) {
    Vector q, nn_r;
    query.MakeColumnVector(i, &q);
    reference.MakeColumnVector((*in)[i], &nn_r);

    double test_dist
      = la::DistanceSqEuclidean(q, nn_r);

    double present_dist = (*dist)[i],
      best_dist = test_dist;

    DEBUG_ASSERT(test_dist == present_dist);

    index_t rank = 0;

    // Looping over the references
    for (index_t j = 0; j < reference.n_cols(); j++) {

      Vector r;
      reference.MakeColumnVector(j, &r);

      double this_dist = la::DistanceSqEuclidean(q, r);
      if (this_dist < test_dist) {
	rank++;
      }
      if (this_dist < best_dist) {
	best_dist = this_dist;
      }
    }

    // The actual rank error
    (*rank_error)[i] = rank;

    // The true nearest neighbor dist
    (*true_nn_dist)[i] = best_dist;
  }
}

/*
  Things to try:
  * try removing the error distribution strategy and check times
  and average error and success probability

 */

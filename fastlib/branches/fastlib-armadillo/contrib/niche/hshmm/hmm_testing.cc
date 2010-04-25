#include "hmm_testing.h"



int main(int argc, char* argv[]) {
  fx_module* root = fx_init(argc, argv, &hmm_testing_doc);

  srand(time(0));
  srand48(time(0));
  //srand(10);


  bool load_hmms = fx_param_bool(NULL, "load_hmms", false);
  bool save_hmms = fx_param_bool(NULL, "save_hmms", false);
  const char* hmms_filename = fx_param_str(NULL, "hmms", "hmms.dat");

  if(save_hmms == true) {
    struct stat stFileInfo;
    if(stat(hmms_filename, &stFileInfo) == 0) {
      FATAL("Error: File to which learned HMMs are to be saved already exists! We avoid overwriting previously saved HMMs.\nExiting...\n");
    }
  }


  int n_sequences_per_class;
  int n_sequences;

  
  ArrayList<Vector> initial_probs_vectors;
  ArrayList<Matrix> transition_matrices;
  ArrayList<Matrix> emission_matrices;

  Matrix training_data;
  Matrix test_data;



  if(load_hmms == true) {
    LoadHMMs(hmms_filename, &initial_probs_vectors, &transition_matrices,
	     &emission_matrices, &training_data, &test_data);
  }
  else {
    ObtainDataLearnHMMs(&initial_probs_vectors, &transition_matrices,
			&emission_matrices, &training_data, &test_data);
    if(save_hmms) {
      SaveHMMs(hmms_filename, initial_probs_vectors, transition_matrices,
	       emission_matrices, training_data, test_data);
    }
  }

  n_sequences = initial_probs_vectors.size();
  n_sequences_per_class = n_sequences / 2; // we assume balanced data




  
  
  
  // Compute or Load Kernel Matrix

  Matrix kernel_matrix;

  char kernel_matrix_filename[100];
  double lambda = fx_param_double(NULL, "lambda", 0.3);
  sprintf(kernel_matrix_filename, "../kernel_matrix_%f.csv", lambda);

  struct stat stFileInfo;
  if(stat(kernel_matrix_filename, &stFileInfo) == 0) {
    printf("Loading kernel matrix...\n");
    data::Load(kernel_matrix_filename, &kernel_matrix);
  }
  else {
    printf("Computing kernel matrix...\n");
    fx_timer_start(NULL, "computing_kernel_all_pairs");
    ComputeKernelAllPairs(2 * n_sequences_per_class,
			  initial_probs_vectors, transition_matrices,
			  emission_matrices,
			  &kernel_matrix);
    fx_timer_stop(NULL, "computing_kernel_all_pairs");
    
    Vector sqrt_diag;
    sqrt_diag.Init(n_sequences);
    for(int i = 0; i < n_sequences; i++) {
      sqrt_diag[i] = sqrt(kernel_matrix.get(i, i));
    }
    for(int i = 0; i < n_sequences; i++) {
      for(int j = 0; j < n_sequences; j++) {
	kernel_matrix.set(j, i,
			  kernel_matrix.get(j, i) /
			  (sqrt_diag[i] * sqrt_diag[j]));
      }
    }
    //kernel_matrix.PrintDebug("kernel matrix");
    data::Save(kernel_matrix_filename, kernel_matrix);
  }


  int n_test_points;
  int n_correct_class0 = 0;
  int n_correct_class1 = 0;

  datanode* svm_module = fx_submodule(fx_root, "svm");
  
  if(KFOLD == false) {
    Dataset training_set;
    training_set.CopyMatrix(training_data);
  
    // Begin SVM Training | Training and Testing
    //datanode *svm_module = fx_submodule(fx_root, "svm");
    fx_set_param_double(svm_module, "c", 1e-2);
    SVM<SVMRBFKernel> svm; // this should be changed from SVMRBFKernel to something like SVMMMKKernel
    int learner_typeid = 0; // for svm_c
  
    printf("SVM Training... \n");
    svm.InitTrain(learner_typeid, training_set, svm_module, kernel_matrix);
  
    
    Dataset test_set;
    test_set.CopyMatrix(test_data);
    printf("test data dims = (%d, %d)\n", 
	   test_data.n_cols(), test_data.n_rows());
  
    printf("SVM Predicting... \n");
    svm.BatchPredict(learner_typeid, test_set, "predicted_values"); // TODO:param_req
  
    
    
    // Emit results
  
    Matrix predicted_values;
    data::Load("predicted_values", &predicted_values);

    n_test_points = predicted_values.n_cols();
  
    for(int i = 0; i < n_test_points; i++) {
      bool correct =
	((int)predicted_values.get(0, i)) == ((int)test_data.get(1,i));
      if(correct) {
	if(test_data.get(1,i) == 1) {
	  n_correct_class1++;
	}
	else {
	  n_correct_class0++;
	}
      }
    }
  }
  else { // KFOLD
    Dataset cv_set;
    cv_set.CopyMatrix(training_data);
    printf("cv data dims = (%d, %d)\n",
	   training_data.n_cols(), training_data.n_rows());

    int n_points = training_data.n_cols();
    n_test_points = n_points;

    ArrayList<index_t> permutation;
    math::MakeIdentityPermutation(n_points, &permutation);


    ArrayList<double> all_c_tries;
    all_c_tries.Init(2);
    ArrayList<double> val_all_c_tries;
    val_all_c_tries.Init(2);


    ArrayList<double> c_tries;
    c_tries.Init(3);
    ArrayList<int> val_c_tries;
    val_c_tries.Init(3);
    ArrayList<int> val1_c_tries;
    val1_c_tries.Init(3);
    ArrayList<int> val0_c_tries;
    val0_c_tries.Init(3);


    // we should definitely try these first:
    //1e-4,1e-3,1e-2,1e-1,1,1e1,1e2,1e3  ,1e4

    double c_upper = 1e4;
    double c_lower = 1e-7;

    printf("c_upper = %f\n", c_upper);
    int val_c_upper = eval_kfold_svm(c_upper, n_points, permutation, cv_set, svm_module, kernel_matrix, &n_correct_class1, &n_correct_class0);
    c_tries[0] = c_upper;
    all_c_tries[0] = c_upper;
    val_c_tries[0] = val_c_upper;
    val_all_c_tries[0] = val_c_upper;
    val1_c_tries[0] = n_correct_class1;
    val0_c_tries[0] = n_correct_class0;

    printf("c_lower = %f\n", c_lower);
    int val_c_lower = eval_kfold_svm(c_lower, n_points, permutation, cv_set, svm_module, kernel_matrix, &n_correct_class1, &n_correct_class0);
    c_tries[1] = c_lower;
    all_c_tries[1] = c_lower;
    val_c_tries[1] = val_c_lower;
    val_all_c_tries[1] = val_c_lower;
    val1_c_tries[1] = n_correct_class1;
    val0_c_tries[1] = n_correct_class0;

    const double c_first[] = {1e-4,1e-3,1e-2,1e-1,1,1e1,1e2,1e3};
    for(int i = 0; i < 8; i++) {
      double val_c = eval_kfold_svm(c_first[i], n_points, permutation, cv_set, svm_module, kernel_matrix, &n_correct_class1, &n_correct_class0);
      all_c_tries.PushBackCopy(c_first[i]);
      val_all_c_tries.PushBackCopy(val_c);
    }



    int lower_bound;
    if(val_c_upper < val_c_lower) {
      lower_bound = val_c_upper;
    }
    else {
      lower_bound = val_c_lower;
    }

    int val_c_new = lower_bound;
    double c_new;
    int num_guesses = 0;
    bool terminate = false;
    while(val_c_new <= lower_bound) {
      if(num_guesses >= 50) {
	printf("50 guesses without improvement from starting strawmen!\nTerminating...\n");
	terminate = true;
	break;
      }
      double log_c_upper = log(c_upper);
      double log_c_lower = log(c_lower);
      c_new = exp(drand48() * (log_c_upper - log_c_lower) + log_c_lower);
      printf("c_new = %f\n", c_new);
      val_c_new = eval_kfold_svm(c_new, n_points, permutation, cv_set, svm_module, kernel_matrix, &n_correct_class1, &n_correct_class0);
      all_c_tries.PushBackCopy(c_new);
      val_all_c_tries.PushBackCopy(val_c_new);
      num_guesses++;
    }

    c_tries[2] = c_new;
    val_c_tries[2] = val_c_new;
    val1_c_tries[2] = n_correct_class1;
    val0_c_tries[2] = n_correct_class0;

    double c_tol = 1e-8;
    int iteration_num = 0;

    while(((c_upper - c_lower) > c_tol) && !terminate) {
      iteration_num++;
      double c_new2;
      int val_c_new2 = val_c_new;
      int num_guesses = 0;
      while(val_c_new2 == val_c_new) {
	if(num_guesses >= 25) {
	  printf("25 guesses without improvement.\nTerminating...\n");
	  terminate = true;
	  break;
	}
 	double log_c_upper = log(c_upper);
	double log_c_lower = log(c_lower);
	c_new2 = exp(drand48() * (log_c_upper - log_c_lower) + log_c_lower);
	printf("c_new2 = %f\titeration_num = %d\n", c_new2, iteration_num);
	val_c_new2 = eval_kfold_svm(c_new2, n_points, permutation, cv_set, svm_module, kernel_matrix, &n_correct_class1, &n_correct_class0);
	all_c_tries.PushBackCopy(c_new2);
	val_all_c_tries.PushBackCopy(val_c_new2);
	num_guesses++;
      }
      

      c_tries.PushBackCopy(c_new2);
      val_c_tries.PushBackCopy(val_c_new2);
      val1_c_tries.PushBackCopy(n_correct_class1);
      val0_c_tries.PushBackCopy(n_correct_class0);

      if(val_c_new2 > val_c_new) {
	if(c_new2 < c_new) {
	  c_upper = c_new;
	  val_c_upper = val_c_new;
	  c_new = c_new2;
	  val_c_new = val_c_new2;
	}
	else {
	  c_lower = c_new;
	  val_c_lower = val_c_new;
	  c_new = c_new2;
	  val_c_new = val_c_new2;
	}
      }
      else { // val_c_new2 < val_c_new
	if(c_new2 < c_new) {
	  c_lower = c_new2;
	  val_c_lower = val_c_new2;
	}
	else {
	  c_upper = c_new2;
	  val_c_upper = val_c_new2;
	}
      }
    }

    int n_all_tries = all_c_tries.size();
    Matrix c_accuracy_pairs;
    c_accuracy_pairs.Init(2, n_all_tries);
    for(int i = 0; i < n_all_tries; i++) {
      c_accuracy_pairs.set(0, i, all_c_tries[i]);
      c_accuracy_pairs.set(1, i, val_all_c_tries[i]);
    }
    data::Save("c_accuracy.csv", c_accuracy_pairs);

    int n_tries = val_c_tries.size();

    int opt_ind;
    if(val_c_tries[n_tries - 1] > val_c_tries[n_tries - 2]) {
      opt_ind = n_tries - 1;
    }
    else {
      opt_ind = n_tries - 2;
    }

    double opt_c = c_tries[opt_ind];
    n_correct_class1 = val1_c_tries[opt_ind];
    n_correct_class0 = val0_c_tries[opt_ind];
    
    printf("optimal c = %f\n", opt_c);
    fx_result_double(NULL, "optimal_c", opt_c);

    
    
    /*

    printf("LOOCV SVM Training and Testing... \n");
    for(int i = 0; i < n_points; i++) {
      Dataset training_set;
      Dataset test_set;
      cv_set.SplitTrainTest(n_points, i, permutation, &training_set, &test_set);
        
      // Begin SVM Training | Training and Testing
      SVM<SVMRBFKernel> svm; // this should be changed from SVMRBFKernel to something like SVMMMKKernel
      int learner_typeid = 0; // for svm_c
      

      svm.InitTrain(learner_typeid, training_set, svm_module, kernel_matrix);

      Vector test_point;
      test_point.Alias(test_set.point(0), 2);
      
      double prediction = svm.Predict(learner_typeid, test_point);
      int test_label = (int) ((test_set.point(0))[1]);
      bool correct =
	((int)prediction) == test_label;
      if(correct) {
	if(test_label == 1) {
	  n_correct_class1++;
	}
	else {
	  n_correct_class0++;
	}
      }
    }
    */
  }
  
  int n_correct = n_correct_class1 + n_correct_class0;
  
  // we assume n_test_points is even because we assumed balanced classes
  fx_result_double(NULL, "class1_accuracy",
		   ((double)n_correct_class1) / ((double)n_test_points / 2));
  fx_result_double(NULL, "class0_accuracy",
		   ((double)n_correct_class0) / ((double)n_test_points / 2));
  fx_result_double(NULL, "total_accuracy",
		   ((double) n_correct) / ((double) n_test_points));
  
  /*
    printf("n_correct / n_test_points = %d / %d = %f\n",
    n_correct, n_test_points,
    ((double) n_correct) / ((double) n_test_points));
    
    printf("n_correct_class1 / n_test_points_class_1 = %d / %d = %f\n",
    n_correct_class1, n_test_points / 2,
    ((double) n_correct_class1) / ((double) n_test_points / 2));
    
    printf("n_correct_class0 / n_test_points_class_0 = %d / %d = %f\n",
    n_correct_class0, n_test_points / 2,
    ((double) n_correct_class0) / ((double) n_test_points / 2));
  */

  fx_done(root);
}

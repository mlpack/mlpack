#ifndef INSIDE_TEST_ENGINE_IMPL_H
#error "This is not a public header file!"
#endif


void CreateIDLabelPairs(const GenVector<int> &labels,
			Matrix* p_id_label_pairs) {
  Matrix &id_label_pairs = *p_id_label_pairs;

  int n_points = labels.length();
  id_label_pairs.Init(2, n_points);
  for(int i = 0; i < n_points; i++) {
    id_label_pairs.set(0, i, i);
    id_label_pairs.set(1, i, labels[i]);
  }
}

void TestHMMGenMMKClassification(const ArrayList<HMM<Multinomial> > &hmms,
				 const GenVector<int> &labels) {

  int n_hmms = labels.length();
  printf("n_hmms = %d\n", n_hmms);
  
  double lambda = fx_param_double_req(NULL, "lambda");
  printf("lambda = %f\n", lambda);
  int witness_length = fx_param_int(NULL, "witness_length", 70);
  printf("witness_length = %d\n", witness_length);


  Matrix kernel_matrix;
  GenerativeMMKBatch(lambda, witness_length, hmms, &kernel_matrix);
  //PrintDebug("original kernel matrix", kernel_matrix, "%3e");

//   Vector original_eigenvalues;
//   la::EigenvaluesInit(kernel_matrix,
// 		      &original_eigenvalues);
  //PrintDebug("original eigenvalues", original_eigenvalues, "%3e");

  NormalizeKernelMatrix(&kernel_matrix);
  //PrintDebug("normalized kernel matrix", kernel_matrix, "%3e");

//   Vector normalized_eigenvalues;
//   la::EigenvaluesInit(kernel_matrix,
// 		      &normalized_eigenvalues);
  //PrintDebug("normalized eigenvalues", normalized_eigenvalues, "%3e");
  //data::Save("kernel_matrix.csv", kernel_matrix);
  
  Matrix id_label_pairs;
  CreateIDLabelPairs(labels, &id_label_pairs);
  id_label_pairs.PrintDebug("id_label_pairs");

  Vector c_set;
  int min_c_exp = -30;
  int max_c_exp = 8;
  c_set.Init(max_c_exp - min_c_exp + 1);
  for(int i = min_c_exp; i <= max_c_exp; i++) {
    c_set[i - min_c_exp] = pow(2, ((double)i)/2.0);
  }
  PrintDebug("c_set", c_set, "%3e");

  
  SVMKFoldCV(id_label_pairs, kernel_matrix, c_set);
}

void TestHMMLatMMKClassification(const HMM<Multinomial> &hmm,
				 const ArrayList<GenMatrix<int> > &sequences,
				 const GenVector<int> &labels) {

  int n_sequences = labels.length();
  printf("n_sequences = %d\n", n_sequences);
  
  double lambda = fx_param_double_req(NULL, "lambda");
  printf("lambda = %f\n", lambda);


  Matrix kernel_matrix;
  LatentMMKBatch(lambda, hmm, sequences, &kernel_matrix);
  //PrintDebug("original kernel matrix", kernel_matrix, "%3e");

//   Vector original_eigenvalues;
//   la::EigenvaluesInit(kernel_matrix,
// 		      &original_eigenvalues);
  //PrintDebug("original eigenvalues", original_eigenvalues, "%3e");

  NormalizeKernelMatrix(&kernel_matrix);
  //PrintDebug("normalized kernel matrix", kernel_matrix, "%3e");

//   Vector normalized_eigenvalues;
//   la::EigenvaluesInit(kernel_matrix,
// 		      &normalized_eigenvalues);
  //PrintDebug("normalized eigenvalues", normalized_eigenvalues, "%3e");
  //data::Save("kernel_matrix.csv", kernel_matrix);
  
  Matrix id_label_pairs;
  CreateIDLabelPairs(labels, &id_label_pairs);
  id_label_pairs.PrintDebug("id_label_pairs");

  Vector c_set;
  int min_c_exp = -15;
  int max_c_exp = 4;
  c_set.Init(max_c_exp - min_c_exp + 1);
  for(int i = min_c_exp; i <= max_c_exp; i++) {
    c_set[i - min_c_exp] = pow(2, i);
  }
  PrintDebug("c_set", c_set, "%3e");

  
  SVMKFoldCV(id_label_pairs, kernel_matrix, c_set);
}


int EvalKFoldSVM(double c, int n_points,
		 int n_folds,
		 const ArrayList<index_t> &permutation, const Dataset& cv_set,
		 datanode* svm_module, const Matrix &kernel_matrix,
		 int *n_correct_class1, int *n_correct_class0) {
  printf("10-FOLD SVM Training and Testing... \n");

  fx_set_param_double(svm_module, "c", c);

  *n_correct_class1 = 0;
  *n_correct_class0 = 0;
  for(int fold_num = 0; fold_num < n_folds; fold_num++) {
    Dataset training_set;
    Dataset test_set;
    cv_set.SplitTrainTest(n_folds, fold_num, permutation, &training_set, &test_set);
    
    // Begin SVM Training | Training and Testing
    SVM<SVMRBFKernel> svm; // this should be changed from SVMRBFKernel to something like SVMMMKKernel
    int learner_typeid = 0; // for svm_c
    
    
    svm.InitTrain(learner_typeid, training_set, svm_module, kernel_matrix);

    for(int i = 0; i < test_set.n_points(); i++) {
      Vector test_point;
      test_point.Alias(test_set.point(i), 2);
    
      double prediction = svm.Predict(learner_typeid, test_point);
      int test_label = (int) test_point[1];
      bool correct =
	((int)prediction) == test_label;
      if(correct) {
	if(test_label == 1) {
	  (*n_correct_class1)++;
	}
	else {
	  (*n_correct_class0)++;
	}
      }
    }
  }

  int n_correct = (*n_correct_class1) + (*n_correct_class0);
  printf("n_correct = %d\n", n_correct);
  return n_correct;
}

void SVMKFoldCV(const Matrix &id_label_pairs,
		const Matrix &kernel_matrix,
		const Vector &c_set) {

  datanode* svm_module = fx_submodule(fx_root, "svm");

  Dataset cv_set;
  cv_set.CopyMatrix(id_label_pairs);
  printf("cv data dims = (%d, %d)\n",
	 id_label_pairs.n_cols(), id_label_pairs.n_rows());

  int n_points = id_label_pairs.n_cols();

  ArrayList<index_t> permutation;
  math::MakeIdentityPermutation(n_points, &permutation);


  int c_set_size = c_set.length();

  GenVector<int> n_correct_results;
  GenVector<int> n_correct_class1_results;
  GenVector<int> n_correct_class0_results;

  n_correct_results.Init(c_set_size);
  n_correct_class1_results.Init(c_set_size);
  n_correct_class0_results.Init(c_set_size);
  
  int n_folds = 10;

  for(int i = 0; i < c_set_size; i++) {
    printf("\nc = %f\n", c_set[i]);
    n_correct_results[i] =
      EvalKFoldSVM(c_set[i], n_points,
		   n_folds, permutation, cv_set,
		   svm_module, kernel_matrix,
		   &(n_correct_class1_results[i]),
		   &(n_correct_class0_results[i]));
  }

  int n_correct_max = -1;
  int argmax = -1;

  Matrix c_accuracy_pairs;
  c_accuracy_pairs.Init(2, c_set_size);
  char c_result_name[80];
  for(int i = 0; i < c_set_size; i++) {
    int val = n_correct_results[i];

    c_accuracy_pairs.set(0, i, c_set[i]);
    c_accuracy_pairs.set(1, i, val);

    if(val > n_correct_max) {
      n_correct_max = val;
      argmax = i;
    }

    sprintf(c_result_name, "C%f", c_set[i]);
    fx_result_double(NULL, c_result_name, val);
  }
  data::Save("c_accuracy.csv", c_accuracy_pairs);

  double c_opt = c_set[argmax];
  printf("optimal c = %f\n", c_opt);
  double best_accuracy =
    ((double)n_correct_results[argmax]) / ((double)n_points);
  printf("accuracy = %f\n", best_accuracy);
  fx_result_double(NULL, "optimal_c", c_opt);
  fx_result_double(NULL, "best_accuracy", best_accuracy);
  
}



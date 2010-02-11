#ifndef INSIDE_TEST_ENGINE_IMPL_H
#error "This is not a public header file!"
#endif


void EmitResults(const Vector &c_set,
		 const GenVector<int> &n_correct_results,
		 int n_sequences) {

  int c_set_size = c_set.length();

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
    ((double)n_correct_results[argmax]) / ((double)n_sequences);
  printf("accuracy = %f\n", best_accuracy);
  fx_result_double(NULL, "optimal_c", c_opt);
  fx_result_double(NULL, "best_accuracy", best_accuracy);
}

void LoadCommonCSet(Vector* p_c_set) {
  Vector &c_set = *p_c_set;

  int min_c_exp = -60;
  int max_c_exp = 16;
  c_set.Init(max_c_exp - min_c_exp + 1);
  for(int i = min_c_exp; i <= max_c_exp; i++) {
    c_set[i - min_c_exp] = pow(2, ((double)i)/4.0);
  }

  PrintDebug("c_set", c_set, "%3e");
}

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

  double rho = fx_param_double(NULL, "rho", 1);
  printf("rho = %f\n", rho);

  Matrix kernel_matrix;
  GenerativeMMKBatch(lambda, rho, witness_length, hmms, &kernel_matrix);
  NormalizeKernelMatrix(&kernel_matrix);
  
  Matrix id_label_pairs;
  CreateIDLabelPairs(labels, &id_label_pairs);
  id_label_pairs.PrintDebug("id_label_pairs");

  Vector c_set;
  LoadCommonCSet(&c_set);
  
  SVMKFoldCV(id_label_pairs, kernel_matrix, c_set);
}

void TestHMMGenMMK2Classification(const ArrayList<HMM<Multinomial> > &hmms,
				 const GenVector<int> &labels) {

  int n_hmms = labels.length();
  printf("n_hmms = %d\n", n_hmms);
  
  double lambda = fx_param_double_req(NULL, "lambda");
  printf("lambda = %f\n", lambda);

  int witness_length = fx_param_int(NULL, "witness_length", 70);
  printf("witness_length = %d\n", witness_length);

  double rho = fx_param_double(NULL, "rho", 1);
  printf("rho = %f\n", rho);

  Matrix kernel_matrix;
  GenerativeMMKBatch(lambda, rho, witness_length, hmms, &kernel_matrix);


  /*
  ////// - trace normalization - doesn't seem to work
  double trace = 0;
  for(int i = 0; i < n_hmms; i++) {
    trace += kernel_matrix.get(i, i);
  }

  la::Scale(((double)1) / trace, &kernel_matrix);
  /////
  */



  NormalizeKernelMatrix(&kernel_matrix);


  ////////// - gaussianification of the kernel
  //data::Save("orig_kernel_matrix.csv", kernel_matrix);

  Vector k_diagonal;
  k_diagonal.Init(n_hmms);
  for(int i = 0; i < n_hmms; i++) {
    k_diagonal[i] = kernel_matrix.get(i, i);
  }

  la::Scale(-2.0, &kernel_matrix);

  for(int i = 0; i < n_hmms; i++) {
    for(int j = 0; j < n_hmms; j++) {
      kernel_matrix.set(j, i,
			kernel_matrix.get(j, i)
			+ k_diagonal[i]);
    }
  }

  la::TransposeSquare(&kernel_matrix);

  for(int i = 0; i < n_hmms; i++) {
    for(int j = 0; j < n_hmms; j++) {
      kernel_matrix.set(j, i,
			kernel_matrix.get(j, i)
			+ k_diagonal[i]);
    }
  }
  
  double meta_sigma = fx_param_double_req(NULL, "meta_sigma");
  double scaling_factor = -0.5 / (meta_sigma * meta_sigma);


  //data::Save("mmk_dist_matrix.csv", kernel_matrix);

  for(int i = 0; i < n_hmms; i++) {
    for(int j = 0; j < n_hmms; j++) {
      kernel_matrix.set(j, i,
			exp(scaling_factor * kernel_matrix.get(j, i)));
    }
  }
  
  //data::Save("kernel_matrix.csv", kernel_matrix);
  //////////
  
  Matrix id_label_pairs;
  CreateIDLabelPairs(labels, &id_label_pairs);
  id_label_pairs.PrintDebug("id_label_pairs");

  Vector c_set;
  LoadCommonCSet(&c_set);
  
  SVMKFoldCV(id_label_pairs, kernel_matrix, c_set);
}

void TestHMMGenMMK2Classification(const ArrayList<HMM<DiagGaussian> > &hmms,
				  const GenVector<int> &labels) {

  int n_hmms = labels.length();
  printf("n_hmms = %d\n", n_hmms);
  
  double lambda = fx_param_double_req(NULL, "lambda");
  printf("lambda = %f\n", lambda);

  int witness_length = fx_param_int(NULL, "witness_length", 70);
  printf("witness_length = %d\n", witness_length);

  //double rho = fx_param_double(NULL, "rho", 1);
  //printf("rho = %f\n", rho);

  Matrix kernel_matrix;
  GenerativeMMKBatch(lambda, witness_length, hmms, &kernel_matrix);


  /*
  ////// - trace normalization - doesn't seem to work
  double trace = 0;
  for(int i = 0; i < n_hmms; i++) {
    trace += kernel_matrix.get(i, i);
  }

  la::Scale(((double)1) / trace, &kernel_matrix);
  /////
  */



  NormalizeKernelMatrix(&kernel_matrix);


  ////////// - gaussianification of the kernel
  data::Save("orig_kernel_matrix.csv", kernel_matrix);
  
  Vector k_diagonal;
  k_diagonal.Init(n_hmms);
  for(int i = 0; i < n_hmms; i++) {
    k_diagonal[i] = kernel_matrix.get(i, i);
  }

  la::Scale(-2.0, &kernel_matrix);

  for(int i = 0; i < n_hmms; i++) {
    for(int j = 0; j < n_hmms; j++) {
      kernel_matrix.set(j, i,
			kernel_matrix.get(j, i)
			+ k_diagonal[i]);
    }
  }

  la::TransposeSquare(&kernel_matrix);

  for(int i = 0; i < n_hmms; i++) {
    for(int j = 0; j < n_hmms; j++) {
      kernel_matrix.set(j, i,
			kernel_matrix.get(j, i)
			+ k_diagonal[i]);
    }
  }
  
  double meta_sigma = fx_param_double_req(NULL, "meta_sigma");
  double scaling_factor = -0.5 / (meta_sigma * meta_sigma);


  //data::Save("mmk_dist_matrix.csv", kernel_matrix);

  for(int i = 0; i < n_hmms; i++) {
    for(int j = 0; j < n_hmms; j++) {
      kernel_matrix.set(j, i,
			exp(scaling_factor * kernel_matrix.get(j, i)));
    }
  }
  
  //data::Save("kernel_matrix.csv", kernel_matrix);
  //////////
  
  Matrix id_label_pairs;
  CreateIDLabelPairs(labels, &id_label_pairs);
  id_label_pairs.PrintDebug("id_label_pairs");

  Vector c_set;
  LoadCommonCSet(&c_set);
  
  SVMKFoldCV(id_label_pairs, kernel_matrix, c_set);
}

void TestHMMGenMMK2ClassificationLog(const ArrayList<HMM<DiagGaussian> > &hmms,
				     const GenVector<int> &labels) {

  int n_hmms = labels.length();
  printf("n_hmms = %d\n", n_hmms);
  
  double lambda = fx_param_double_req(NULL, "lambda");
  printf("lambda = %f\n", lambda);

  int witness_length = fx_param_int(NULL, "witness_length", 70);
  printf("witness_length = %d\n", witness_length);

  //double rho = fx_param_double(NULL, "rho", 1);
  //printf("rho = %f\n", rho);

  Matrix kernel_matrix_log;
  GenerativeMMKBatchLog(lambda, witness_length, hmms, &kernel_matrix_log);


  // each entry kernel_matrix(i,j) = log(K(i,j))


  


  /*
  ////// - trace normalization - doesn't seem to work
  double trace = 0;
  for(int i = 0; i < n_hmms; i++) {
    trace += kernel_matrix.get(i, i);
  }

  la::Scale(((double)1) / trace, &kernel_matrix);
  /////
  */


  // we can normalize each entry as:
  // new_K(i,j) = exp(log(K(i,j)) - 1/2(log(K(i,i)) - log(K(j,j))))

  NormalizeKernelMatrixLog(&kernel_matrix_log);

  // rename since now we're back to the non-log form of the kernel matrix
  Matrix& kernel_matrix = kernel_matrix_log;

  ////////// - gaussianification of the kernel
  //data::Save("orig_kernel_matrix.csv", kernel_matrix);

  Vector k_diagonal;
  k_diagonal.Init(n_hmms);
  for(int i = 0; i < n_hmms; i++) {
    k_diagonal[i] = kernel_matrix.get(i, i);
  }

  la::Scale(-2.0, &kernel_matrix);

  for(int i = 0; i < n_hmms; i++) {
    for(int j = 0; j < n_hmms; j++) {
      kernel_matrix.set(j, i,
			kernel_matrix.get(j, i)
			+ k_diagonal[i]);
    }
  }

  la::TransposeSquare(&kernel_matrix);

  for(int i = 0; i < n_hmms; i++) {
    for(int j = 0; j < n_hmms; j++) {
      kernel_matrix.set(j, i,
			kernel_matrix.get(j, i)
			+ k_diagonal[i]);
    }
  }
  
  double meta_sigma = fx_param_double_req(NULL, "meta_sigma");
  double scaling_factor = -0.5 / (meta_sigma * meta_sigma);


  //data::Save("mmk_dist_matrix.csv", kernel_matrix);

  for(int i = 0; i < n_hmms; i++) {
    for(int j = 0; j < n_hmms; j++) {
      kernel_matrix.set(j, i,
			exp(scaling_factor * kernel_matrix.get(j, i)));
    }
  }
  
  //data::Save("kernel_matrix.csv", kernel_matrix);
  //////////
  
  Matrix id_label_pairs;
  CreateIDLabelPairs(labels, &id_label_pairs);
  id_label_pairs.PrintDebug("id_label_pairs");

  Vector c_set;
  LoadCommonCSet(&c_set);
  
  SVMKFoldCV(id_label_pairs, kernel_matrix, c_set);
}



void TestHMMGenMMKClassification(const ArrayList<HMM<DiagGaussian> > &hmms,
				 const GenVector<int> &labels) {

  int n_hmms = labels.length();
  printf("n_hmms = %d\n", n_hmms);
  
  double lambda = fx_param_double_req(NULL, "lambda");
  printf("lambda = %f\n", lambda);

  int witness_length = fx_param_int(NULL, "witness_length", 70);
  printf("witness_length = %d\n", witness_length);

  //double rho = fx_param_double(NULL, "rho", 1);
  //printf("rho = %f\n", rho);

  Matrix kernel_matrix;
  //GenerativeMMKBatch(lambda, rho, witness_length, hmms, &kernel_matrix);
  GenerativeMMKBatch(lambda, witness_length, hmms, &kernel_matrix);
  NormalizeKernelMatrix(&kernel_matrix);
  kernel_matrix.PrintDebug("kernel_matrix");
  
  Matrix id_label_pairs;
  CreateIDLabelPairs(labels, &id_label_pairs);
  id_label_pairs.PrintDebug("id_label_pairs");

  Vector c_set;
  LoadCommonCSet(&c_set);
  
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
  NormalizeKernelMatrix(&kernel_matrix);
  
  Matrix id_label_pairs;
  CreateIDLabelPairs(labels, &id_label_pairs);
  id_label_pairs.PrintDebug("id_label_pairs");

  Vector c_set;
  LoadCommonCSet(&c_set);
  
  SVMKFoldCV(id_label_pairs, kernel_matrix, c_set);
}

void TestHMMLatMMKClassificationKFold(int n_folds,
				      const ArrayList<HMM<Multinomial> > &kfold_hmms,
				      const ArrayList<GenMatrix<int> > &sequences,
				      const GenVector<int> &labels) {
  
  Matrix id_label_pairs;
  CreateIDLabelPairs(labels, &id_label_pairs);
  id_label_pairs.PrintDebug("id_label_pairs");

  Dataset cv_set;
  cv_set.CopyMatrix(id_label_pairs);

  Vector c_set;
  LoadCommonCSet(&c_set);
  int c_set_size = c_set.length();

  GenVector<int> n_correct_results;
  GenVector<int> n_correct_class1_results;
  GenVector<int> n_correct_class0_results;
  n_correct_results.Init(c_set_size);
  n_correct_class1_results.Init(c_set_size);
  n_correct_class0_results.Init(c_set_size);
  n_correct_results.SetZero();
  n_correct_class1_results.SetZero();
  n_correct_class0_results.SetZero();

  int n_sequences = sequences.size();
  ArrayList<index_t> permutation;
  math::MakeIdentityPermutation(n_sequences, &permutation);

  datanode* svm_module = fx_submodule(fx_root, "svm");

  double lambda = fx_param_double_req(NULL, "lambda");
  printf("lambda = %f\n", lambda);

  printf("n_sequences = %d\n", n_sequences);

  for(int fold_num = 0; fold_num < n_folds; fold_num++) {
    printf("fold %d\n", fold_num);
    Dataset training_set;
    Dataset test_set;
    
    cv_set.SplitTrainTest(n_folds, fold_num, permutation, &training_set, &test_set);
    
    Matrix kernel_matrix;
    LatentMMKBatch(lambda, kfold_hmms[fold_num], sequences, &kernel_matrix);
    NormalizeKernelMatrix(&kernel_matrix); // how does SVM perform without this step? I just tried this and the answer is, very badly!

    for(int c_index = 0; c_index < c_set_size; c_index++) {
      fx_set_param_double(svm_module, "c", c_set[c_index]);
  
      // Begin SVM Training | Training and Testing
      SVM<SVMRBFKernel> svm; // this should be changed from SVMRBFKernel to something like SVMMMKKernel
      int learner_typeid = 0; // for svm_c
            
      svm.InitTrain(learner_typeid, training_set, svm_module, kernel_matrix);

      int n_correct_class1 = 0;
      int n_correct_class0 = 0;
      for(int i = 0; i < test_set.n_points(); i++) {
	Vector test_point;
	test_point.Alias(test_set.point(i), 2);
	
	double prediction = svm.Predict(learner_typeid, test_point);
	int test_label = (int) test_point[1];
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

      printf("fold %d\tc = %e\tn_correct = %d\n",
	     fold_num, c_set[c_index],
	     n_correct_class1 + n_correct_class0);

      n_correct_class1_results[c_index] += n_correct_class1;
      n_correct_class0_results[c_index] += n_correct_class0;
      n_correct_results[c_index] += n_correct_class1 + n_correct_class0;
    }
  }
  
  /*
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
    ((double)n_correct_results[argmax]) / ((double)n_sequences);
  printf("accuracy = %f\n", best_accuracy);
  fx_result_double(NULL, "optimal_c", c_opt);
  fx_result_double(NULL, "best_accuracy", best_accuracy);
  */

  EmitResults(c_set, n_correct_results, n_sequences);


}

void TestHMMLatMMK2ClassificationKFold(int n_folds,
				       const ArrayList<HMM<Multinomial> > &kfold_hmms,
				       const ArrayList<GenMatrix<int> > &sequences,
				       const GenVector<int> &labels) {
  
  Matrix id_label_pairs;
  CreateIDLabelPairs(labels, &id_label_pairs);
  id_label_pairs.PrintDebug("id_label_pairs");

  Dataset cv_set;
  cv_set.CopyMatrix(id_label_pairs);

  Vector c_set;
  LoadCommonCSet(&c_set);
  int c_set_size = c_set.length();

  GenVector<int> n_correct_results;
  GenVector<int> n_correct_class1_results;
  GenVector<int> n_correct_class0_results;
  n_correct_results.Init(c_set_size);
  n_correct_class1_results.Init(c_set_size);
  n_correct_class0_results.Init(c_set_size);
  n_correct_results.SetZero();
  n_correct_class1_results.SetZero();
  n_correct_class0_results.SetZero();

  int n_sequences = sequences.size();
  ArrayList<index_t> permutation;
  math::MakeIdentityPermutation(n_sequences, &permutation);

  datanode* svm_module = fx_submodule(fx_root, "svm");

  double lambda = fx_param_double_req(NULL, "lambda");
  printf("lambda = %f\n", lambda);

  printf("n_sequences = %d\n", n_sequences);

  for(int fold_num = 0; fold_num < n_folds; fold_num++) {
    printf("fold %d\n", fold_num);
    Dataset training_set;
    Dataset test_set;
    
    cv_set.SplitTrainTest(n_folds, fold_num, permutation, &training_set, &test_set);
    
    Matrix kernel_matrix;
    LatentMMKBatch(1e9, kfold_hmms[fold_num], sequences, &kernel_matrix);

    Vector k_diagonal;
    k_diagonal.Init(n_sequences);
    for(int i = 0; i < n_sequences; i++) {
      k_diagonal[i] = kernel_matrix.get(i, i);
    }

    la::Scale(-2.0, &kernel_matrix);

    for(int i = 0; i < n_sequences; i++) {
      for(int j = 0; j < n_sequences; j++) {
	kernel_matrix.set(j, i,
			  kernel_matrix.get(j, i)
			  + k_diagonal[i]);
      }
    }

    la::TransposeSquare(&kernel_matrix);

    for(int i = 0; i < n_sequences; i++) {
      for(int j = 0; j < n_sequences; j++) {
	kernel_matrix.set(j, i,
			  kernel_matrix.get(j, i)
			  + k_diagonal[i]);
      }
    }

    for(int i = 0; i < n_sequences; i++) {
      for(int j = 0; j < n_sequences; j++) {
	kernel_matrix.set(j, i,
			  exp(-lambda * kernel_matrix.get(j, i)));
      }
    }
    NormalizeKernelMatrix(&kernel_matrix);

    for(int c_index = 0; c_index < c_set_size; c_index++) {
      fx_set_param_double(svm_module, "c", c_set[c_index]);
  
      // Begin SVM Training | Training and Testing
      SVM<SVMRBFKernel> svm; // this should be changed from SVMRBFKernel to something like SVMMMKKernel
      int learner_typeid = 0; // for svm_c
            
      svm.InitTrain(learner_typeid, training_set, svm_module, kernel_matrix);

      int n_correct_class1 = 0;
      int n_correct_class0 = 0;
      for(int i = 0; i < test_set.n_points(); i++) {
	Vector test_point;
	test_point.Alias(test_set.point(i), 2);
	
	double prediction = svm.Predict(learner_typeid, test_point);
	int test_label = (int) test_point[1];
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

      printf("fold %d\tc = %e\tn_correct = %d\n",
	     fold_num, c_set[c_index],
	     n_correct_class1 + n_correct_class0);

      n_correct_class1_results[c_index] += n_correct_class1;
      n_correct_class0_results[c_index] += n_correct_class0;
      n_correct_results[c_index] += n_correct_class1 + n_correct_class0;
    }
  }
  
  /*
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
    ((double)n_correct_results[argmax]) / ((double)n_sequences);
  printf("accuracy = %f\n", best_accuracy);
  fx_result_double(NULL, "optimal_c", c_opt);
  fx_result_double(NULL, "best_accuracy", best_accuracy);
  */

  EmitResults(c_set, n_correct_results, n_sequences);


}


void TestHMMLatMMKClassificationKFold(int n_folds,
				      const ArrayList<HMM<Multinomial> > &kfold_exon_hmms,
				      const ArrayList<HMM<Multinomial> > &kfold_intron_hmms,
				      const ArrayList<GenMatrix<int> > &sequences,
				      const GenVector<int> &labels) {
  
  Matrix id_label_pairs;
  CreateIDLabelPairs(labels, &id_label_pairs);
  id_label_pairs.PrintDebug("id_label_pairs");

  Dataset cv_set;
  cv_set.CopyMatrix(id_label_pairs);

  Vector c_set;
  LoadCommonCSet(&c_set);
  int c_set_size = c_set.length();

  GenVector<int> n_correct_results;
  GenVector<int> n_correct_class1_results;
  GenVector<int> n_correct_class0_results;
  n_correct_results.Init(c_set_size);
  n_correct_class1_results.Init(c_set_size);
  n_correct_class0_results.Init(c_set_size);
  n_correct_results.SetZero();
  n_correct_class1_results.SetZero();
  n_correct_class0_results.SetZero();

  int n_sequences = sequences.size();
  ArrayList<index_t> permutation;
  math::MakeIdentityPermutation(n_sequences, &permutation);

  datanode* svm_module = fx_submodule(fx_root, "svm");

  double lambda = fx_param_double_req(NULL, "lambda");
  printf("lambda = %f\n", lambda);

  printf("n_sequences = %d\n", n_sequences);

  for(int fold_num = 0; fold_num < n_folds; fold_num++) {
    printf("fold %d\n", fold_num);
    Dataset training_set;
    Dataset test_set;
    
    cv_set.SplitTrainTest(n_folds, fold_num, permutation, &training_set, &test_set);
    
    Matrix exon_kernel_matrix;
    LatentMMKBatch(lambda, kfold_exon_hmms[fold_num], sequences, &exon_kernel_matrix);
    NormalizeKernelMatrix(&exon_kernel_matrix);

    Matrix intron_kernel_matrix;
    LatentMMKBatch(lambda, kfold_intron_hmms[fold_num], sequences, &intron_kernel_matrix);
    NormalizeKernelMatrix(&intron_kernel_matrix);

    Matrix kernel_matrix;
    la::AddInit(exon_kernel_matrix, intron_kernel_matrix, &kernel_matrix);
    la::Scale(0.5, &kernel_matrix);

    for(int c_index = 0; c_index < c_set_size; c_index++) {
      fx_set_param_double(svm_module, "c", c_set[c_index]);
  
      // Begin SVM Training | Training and Testing
      SVM<SVMRBFKernel> svm; // this should be changed from SVMRBFKernel to something like SVMMMKKernel
      int learner_typeid = 0; // for svm_c
            
      svm.InitTrain(learner_typeid, training_set, svm_module, kernel_matrix);

      int n_correct_class1 = 0;
      int n_correct_class0 = 0;
      for(int i = 0; i < test_set.n_points(); i++) {
	Vector test_point;
	test_point.Alias(test_set.point(i), 2);
	
	double prediction = svm.Predict(learner_typeid, test_point);
	int test_label = (int) test_point[1];
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

      printf("fold %d\tc = %e\tn_correct = %d\n",
	     fold_num, c_set[c_index],
	     n_correct_class1 + n_correct_class0);

      n_correct_class1_results[c_index] += n_correct_class1;
      n_correct_class0_results[c_index] += n_correct_class0;
      n_correct_results[c_index] += n_correct_class1 + n_correct_class0;
    }
  }
  
  /*
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
    ((double)n_correct_results[argmax]) / ((double)n_sequences);
  printf("accuracy = %f\n", best_accuracy);
  fx_result_double(NULL, "optimal_c", c_opt);
  fx_result_double(NULL, "best_accuracy", best_accuracy);
  */

  EmitResults(c_set, n_correct_results, n_sequences);


}

void TestHMMLatMMK2ClassificationKFold(int n_folds,
				      const ArrayList<HMM<Multinomial> > &kfold_exon_hmms,
				      const ArrayList<HMM<Multinomial> > &kfold_intron_hmms,
				      const ArrayList<GenMatrix<int> > &sequences,
				      const GenVector<int> &labels) {
  
  Matrix id_label_pairs;
  CreateIDLabelPairs(labels, &id_label_pairs);
  id_label_pairs.PrintDebug("id_label_pairs");

  Dataset cv_set;
  cv_set.CopyMatrix(id_label_pairs);

  Vector c_set;
  LoadCommonCSet(&c_set);
  int c_set_size = c_set.length();

  GenVector<int> n_correct_results;
  GenVector<int> n_correct_class1_results;
  GenVector<int> n_correct_class0_results;
  n_correct_results.Init(c_set_size);
  n_correct_class1_results.Init(c_set_size);
  n_correct_class0_results.Init(c_set_size);
  n_correct_results.SetZero();
  n_correct_class1_results.SetZero();
  n_correct_class0_results.SetZero();

  int n_sequences = sequences.size();
  ArrayList<index_t> permutation;
  math::MakeIdentityPermutation(n_sequences, &permutation);

  datanode* svm_module = fx_submodule(fx_root, "svm");

  double lambda = fx_param_double_req(NULL, "lambda");
  printf("lambda = %f\n", lambda);

  printf("n_sequences = %d\n", n_sequences);

  for(int fold_num = 0; fold_num < n_folds; fold_num++) {
    printf("fold %d\n", fold_num);
    Dataset training_set;
    Dataset test_set;
    
    cv_set.SplitTrainTest(n_folds, fold_num, permutation, &training_set, &test_set);
    
    Matrix exon_kernel_matrix;
    LatentMMKBatch(1e9, kfold_exon_hmms[fold_num], sequences, &exon_kernel_matrix);
    //NormalizeKernelMatrix(&exon_kernel_matrix);

    Vector k_diagonal;
    k_diagonal.Init(n_sequences);
    for(int i = 0; i < n_sequences; i++) {
      k_diagonal[i] = exon_kernel_matrix.get(i, i);
    }

    la::Scale(-2.0, &exon_kernel_matrix);

    for(int i = 0; i < n_sequences; i++) {
      for(int j = 0; j < n_sequences; j++) {
	exon_kernel_matrix.set(j, i,
			       exon_kernel_matrix.get(j, i)
			       + k_diagonal[i]);
      }
    }

    la::TransposeSquare(&exon_kernel_matrix);

    for(int i = 0; i < n_sequences; i++) {
      for(int j = 0; j < n_sequences; j++) {
	exon_kernel_matrix.set(j, i,
			       exon_kernel_matrix.get(j, i)
			       + k_diagonal[i]);
      }
    }


    Matrix intron_kernel_matrix;
    LatentMMKBatch(1e9, kfold_intron_hmms[fold_num], sequences, &intron_kernel_matrix);
    //NormalizeKernelMatrix(&intron_kernel_matrix);

    k_diagonal.Destruct();
    k_diagonal.Init(n_sequences);
    for(int i = 0; i < n_sequences; i++) {
      k_diagonal[i] = intron_kernel_matrix.get(i, i);
    }

    la::Scale(-2.0, &intron_kernel_matrix);

    for(int i = 0; i < n_sequences; i++) {
      for(int j = 0; j < n_sequences; j++) {
	intron_kernel_matrix.set(j, i,
				 intron_kernel_matrix.get(j, i)
				 + k_diagonal[i]);
      }
    }

    la::TransposeSquare(&intron_kernel_matrix);

    for(int i = 0; i < n_sequences; i++) {
      for(int j = 0; j < n_sequences; j++) {
	intron_kernel_matrix.set(j, i,
			       intron_kernel_matrix.get(j, i)
			       + k_diagonal[i]);
      }
    }

    Matrix kernel_matrix;
    la::AddInit(exon_kernel_matrix, intron_kernel_matrix, &kernel_matrix);

    for(int i = 0; i < n_sequences; i++) {
      for(int j = 0; j < n_sequences; j++) {
	kernel_matrix.set(j, i,
			  exp(-lambda * kernel_matrix.get(j, i)));
      }
    }

    NormalizeKernelMatrix(&kernel_matrix);
    

    for(int c_index = 0; c_index < c_set_size; c_index++) {
      fx_set_param_double(svm_module, "c", c_set[c_index]);
  
      // Begin SVM Training | Training and Testing
      SVM<SVMRBFKernel> svm; // this should be changed from SVMRBFKernel to something like SVMMMKKernel
      int learner_typeid = 0; // for svm_c
            
      svm.InitTrain(learner_typeid, training_set, svm_module, kernel_matrix);

      int n_correct_class1 = 0;
      int n_correct_class0 = 0;
      for(int i = 0; i < test_set.n_points(); i++) {
	Vector test_point;
	test_point.Alias(test_set.point(i), 2);
	
	double prediction = svm.Predict(learner_typeid, test_point);
	int test_label = (int) test_point[1];
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

      printf("fold %d\tc = %e\tn_correct = %d\n",
	     fold_num, c_set[c_index],
	     n_correct_class1 + n_correct_class0);

      n_correct_class1_results[c_index] += n_correct_class1;
      n_correct_class0_results[c_index] += n_correct_class0;
      n_correct_results[c_index] += n_correct_class1 + n_correct_class0;
    }
  }
  
  /*
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
    ((double)n_correct_results[argmax]) / ((double)n_sequences);
  printf("accuracy = %f\n", best_accuracy);
  fx_result_double(NULL, "optimal_c", c_opt);
  fx_result_double(NULL, "best_accuracy", best_accuracy);
  */

  EmitResults(c_set, n_correct_results, n_sequences);


}

void TestHMMFisherKernelClassification(const HMM<Multinomial> &hmm,
				       const ArrayList<GenMatrix<int> > &sequences,
				       const GenVector<int> &labels) {
  
  int n_sequences = labels.length();
  printf("n_sequences = %d\n", n_sequences);
  
  Matrix kernel_matrix;
  FisherKernelBatch(hmm, sequences, &kernel_matrix);
  NormalizeKernelMatrix(&kernel_matrix);
  
  Matrix id_label_pairs;
  CreateIDLabelPairs(labels, &id_label_pairs);
  id_label_pairs.PrintDebug("id_label_pairs");

  Vector c_set;
  LoadCommonCSet(&c_set);
  
  SVMKFoldCV(id_label_pairs, kernel_matrix, c_set);
}

void TestHMMFisherKernelClassificationKFold(int n_folds,
					    const ArrayList<HMM<Multinomial> > &kfold_hmms,
					    const ArrayList<GenMatrix<int> > &sequences,
					    const GenVector<int> &labels) {
  
  Matrix id_label_pairs;
  CreateIDLabelPairs(labels, &id_label_pairs);
  id_label_pairs.PrintDebug("id_label_pairs");

  Dataset cv_set;
  cv_set.CopyMatrix(id_label_pairs);

  Vector c_set;
  LoadCommonCSet(&c_set);
  int c_set_size = c_set.length();

  GenVector<int> n_correct_results;
  GenVector<int> n_correct_class1_results;
  GenVector<int> n_correct_class0_results;
  n_correct_results.Init(c_set_size);
  n_correct_class1_results.Init(c_set_size);
  n_correct_class0_results.Init(c_set_size);
  n_correct_results.SetZero();
  n_correct_class1_results.SetZero();
  n_correct_class0_results.SetZero();

  int n_sequences = sequences.size();
  ArrayList<index_t> permutation;
  math::MakeIdentityPermutation(n_sequences, &permutation);

  datanode* svm_module = fx_submodule(fx_root, "svm");

  double lambda = fx_param_double_req(NULL, "lambda");
  printf("lambda = %f\n", lambda);

  printf("n_sequences = %d\n", n_sequences);

  for(int fold_num = 0; fold_num < n_folds; fold_num++) {
    printf("fold %d\n", fold_num);
    Dataset training_set;
    Dataset test_set;
    
    cv_set.SplitTrainTest(n_folds, fold_num, permutation, &training_set, &test_set);
    
    Matrix kernel_matrix;
    FisherKernelBatch(lambda, kfold_hmms[fold_num], sequences, &kernel_matrix);
    NormalizeKernelMatrix(&kernel_matrix);

    for(int c_index = 0; c_index < c_set_size; c_index++) {
      fx_set_param_double(svm_module, "c", c_set[c_index]);
  
      // Begin SVM Training | Training and Testing
      SVM<SVMRBFKernel> svm; // this should be changed from SVMRBFKernel to something like SVMMMKKernel
      int learner_typeid = 0; // for svm_c
            
      svm.InitTrain(learner_typeid, training_set, svm_module, kernel_matrix);

      int n_correct_class1 = 0;
      int n_correct_class0 = 0;
      for(int i = 0; i < test_set.n_points(); i++) {
	Vector test_point;
	test_point.Alias(test_set.point(i), 2);
	
	double prediction = svm.Predict(learner_typeid, test_point);
	int test_label = (int) test_point[1];
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

      printf("fold %d\tc = %e\tn_correct = %d\n",
	     fold_num, c_set[c_index],
	     n_correct_class1 + n_correct_class0);

      n_correct_class1_results[c_index] += n_correct_class1;
      n_correct_class0_results[c_index] += n_correct_class0;
      n_correct_results[c_index] += n_correct_class1 + n_correct_class0;
    }
  }
  
  EmitResults(c_set, n_correct_results, n_sequences);
}

void TestHMMFisherKernelClassificationKFold(int n_folds,
					    const ArrayList<HMM<Multinomial> > &kfold_exon_hmms,
					    const ArrayList<HMM<Multinomial> > &kfold_intron_hmms,
					    const ArrayList<GenMatrix<int> > &sequences,
					    const GenVector<int> &labels) {
  
  Matrix id_label_pairs;
  CreateIDLabelPairs(labels, &id_label_pairs);
  id_label_pairs.PrintDebug("id_label_pairs");

  Dataset cv_set;
  cv_set.CopyMatrix(id_label_pairs);

  Vector c_set;
  LoadCommonCSet(&c_set);
  int c_set_size = c_set.length();

  GenVector<int> n_correct_results;
  GenVector<int> n_correct_class1_results;
  GenVector<int> n_correct_class0_results;
  n_correct_results.Init(c_set_size);
  n_correct_class1_results.Init(c_set_size);
  n_correct_class0_results.Init(c_set_size);
  n_correct_results.SetZero();
  n_correct_class1_results.SetZero();
  n_correct_class0_results.SetZero();

  int n_sequences = sequences.size();
  ArrayList<index_t> permutation;
  math::MakeIdentityPermutation(n_sequences, &permutation);

  datanode* svm_module = fx_submodule(fx_root, "svm");

  printf("n_sequences = %d\n", n_sequences);

  for(int fold_num = 0; fold_num < n_folds; fold_num++) {
    printf("fold %d\n", fold_num);
    Dataset training_set;
    Dataset test_set;
    
    cv_set.SplitTrainTest(n_folds, fold_num, permutation, &training_set, &test_set);
    
    Matrix exon_kernel_matrix;
    FisherKernelBatch(kfold_exon_hmms[fold_num], sequences, &exon_kernel_matrix);
    NormalizeKernelMatrix(&exon_kernel_matrix);

    Matrix intron_kernel_matrix;
    FisherKernelBatch(kfold_intron_hmms[fold_num], sequences, &intron_kernel_matrix);
    NormalizeKernelMatrix(&intron_kernel_matrix);

    Matrix kernel_matrix;
    la::AddInit(exon_kernel_matrix, intron_kernel_matrix, &kernel_matrix);
    la::Scale(0.5, &kernel_matrix);

    for(int c_index = 0; c_index < c_set_size; c_index++) {
      fx_set_param_double(svm_module, "c", c_set[c_index]);
  
      // Begin SVM Training | Training and Testing
      SVM<SVMRBFKernel> svm; // this should be changed from SVMRBFKernel to something like SVMMMKKernel
      int learner_typeid = 0; // for svm_c
            
      svm.InitTrain(learner_typeid, training_set, svm_module, kernel_matrix);

      int n_correct_class1 = 0;
      int n_correct_class0 = 0;
      for(int i = 0; i < test_set.n_points(); i++) {
	Vector test_point;
	test_point.Alias(test_set.point(i), 2);
	
	double prediction = svm.Predict(learner_typeid, test_point);
	int test_label = (int) test_point[1];
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

      printf("fold %d\tc = %e\tn_correct = %d\n",
	     fold_num, c_set[c_index],
	     n_correct_class1 + n_correct_class0);

      n_correct_class1_results[c_index] += n_correct_class1;
      n_correct_class0_results[c_index] += n_correct_class0;
      n_correct_results[c_index] += n_correct_class1 + n_correct_class0;
    }
  }
  
  EmitResults(c_set, n_correct_results, n_sequences);
}

void TestHMMBayesClassificationKFold(int n_folds,
				     const ArrayList<HMM<Multinomial> > &kfold_exon_hmms,
				     const ArrayList<HMM<Multinomial> > &kfold_intron_hmms,
				     const ArrayList<GenMatrix<int> > &sequences,
				     const GenVector<int> &labels) {
  
  Matrix id_label_pairs;
  CreateIDLabelPairs(labels, &id_label_pairs);
  id_label_pairs.PrintDebug("id_label_pairs");

  Dataset cv_set;
  cv_set.CopyMatrix(id_label_pairs);

  int n_correct = 0;
  int n_exons_correct = 0;
  int n_introns_correct = 0;

  int n_sequences = sequences.size();
  ArrayList<index_t> permutation;
  math::MakeIdentityPermutation(n_sequences, &permutation);

  printf("n_sequences = %d\n", n_sequences);

  for(int fold_num = 0; fold_num < n_folds; fold_num++) {
    printf("fold %d\n", fold_num);
    Dataset training_set;
    Dataset test_set;
    
    cv_set.SplitTrainTest(n_folds, fold_num, permutation, &training_set, &test_set);
    
    for(int i = 0; i < test_set.n_points(); i++) {
      int test_sequence_index = (int)(test_set.get(0, i));
      ///
      Matrix p_x_given_q;
      ArrayList<Matrix> p_qq_t;
      Matrix p_qt;
      double exon_neg_likelihood;
      double intron_neg_likelihood;

      const HMM<Multinomial> &fold_exon_hmm = kfold_exon_hmms[fold_num];
      const HMM<Multinomial> &fold_intron_hmm = kfold_intron_hmms[fold_num];

      fold_exon_hmm.ExpectationStepNoLearning(sequences[test_sequence_index],
					      &p_x_given_q,
					      &p_qq_t,
					      &p_qt,
					      &exon_neg_likelihood);
      p_x_given_q.Destruct();
      p_qq_t.Renew();
      p_qt.Destruct();
      fold_intron_hmm.ExpectationStepNoLearning(sequences[test_sequence_index],
						&p_x_given_q,
						&p_qq_t,
						&p_qt,
						&intron_neg_likelihood);
      if(exon_neg_likelihood < intron_neg_likelihood) { // predict exon (1)
	if(test_set.get(1, i) == 1) {
	  n_exons_correct++;
	}
      }
      else { // predict intron (0)
	if(test_set.get(1, i) == 0) {
	  n_introns_correct++;
	}
      }
    }
  }

  n_correct = n_exons_correct + n_introns_correct;
  double accuracy =
    ((double)n_correct) / ((double)n_sequences);
  printf("accuracy = %f\n", accuracy);
  fx_result_double(NULL, "best_accuracy", accuracy);
}


void TestHMMBayesClassificationKFold(int n_folds,
				     const ArrayList<HMM<DiagGaussian> > &kfold_class1_hmms,
				     const ArrayList<HMM<DiagGaussian> > &kfold_class0_hmms,
				     const ArrayList<GenMatrix<double> > &sequences,
				     const GenVector<int> &labels) {
  
  Matrix id_label_pairs;
  CreateIDLabelPairs(labels, &id_label_pairs);
  id_label_pairs.PrintDebug("id_label_pairs");

  Dataset cv_set;
  cv_set.CopyMatrix(id_label_pairs);

  int n_correct = 0;
  int n_class1_correct = 0;
  int n_class0_correct = 0;

  int n_sequences = sequences.size();
  ArrayList<index_t> permutation;
  math::MakeIdentityPermutation(n_sequences, &permutation);

  printf("n_sequences = %d\n", n_sequences);

  for(int fold_num = 0; fold_num < n_folds; fold_num++) {
    printf("fold %d\n", fold_num);
    Dataset training_set;
    Dataset test_set;
    
    cv_set.SplitTrainTest(n_folds, fold_num, permutation, &training_set, &test_set);
    
    for(int i = 0; i < test_set.n_points(); i++) {
      int test_sequence_index = (int)(test_set.get(0, i));
      ///
      Matrix p_x_given_q;
      ArrayList<Matrix> p_qq_t;
      Matrix p_qt;
      double class1_neg_likelihood;
      double class0_neg_likelihood;

      const HMM<DiagGaussian> &fold_class1_hmm = kfold_class1_hmms[fold_num];
      const HMM<DiagGaussian> &fold_class0_hmm = kfold_class0_hmms[fold_num];

      fold_class1_hmm.ExpectationStepNoLearning(sequences[test_sequence_index],
						&p_x_given_q,
						&p_qq_t,
						&p_qt,
						&class1_neg_likelihood);
      p_x_given_q.Destruct();
      p_qq_t.Renew();
      p_qt.Destruct();
      fold_class0_hmm.ExpectationStepNoLearning(sequences[test_sequence_index],
						&p_x_given_q,
						&p_qq_t,
						&p_qt,
						&class0_neg_likelihood);
      if(class1_neg_likelihood < class0_neg_likelihood) { // predict class 1
	if(test_set.get(1, i) == 1) {
	  n_class1_correct++;
	}
      }
      else { // predict class 0
	if(test_set.get(1, i) == 0) {
	  n_class0_correct++;
	}
      }
    }
  }

  n_correct = n_class1_correct + n_class0_correct;
  double accuracy =
    ((double)n_correct) / ((double)n_sequences);
  printf("accuracy = %f\n", accuracy);
  fx_result_double(NULL, "best_accuracy", accuracy);
}



void TestMarkovMMKClassification(int n_symbols,
				 const ArrayList<GenMatrix<int> > &sequences,
				 const GenVector<int> &labels) {
  int n_sequences = labels.length();
  printf("n_sequences = %d\n", n_sequences);
  
  double lambda = fx_param_double_req(NULL, "lambda");
  printf("lambda = %f\n", lambda);

  int order = fx_param_int_req(NULL, "order");
  printf("order = %d\n", order);

  Matrix kernel_matrix;
  MarkovEmpiricalMMKBatch(lambda, order, n_symbols, sequences, &kernel_matrix);
  NormalizeKernelMatrix(&kernel_matrix);
  
  Matrix id_label_pairs;
  CreateIDLabelPairs(labels, &id_label_pairs);
  id_label_pairs.PrintDebug("id_label_pairs");

  Vector c_set;
  LoadCommonCSet(&c_set);
  
  SVMKFoldCV(id_label_pairs, kernel_matrix, c_set);
}

int EvalKFoldSVM(double c, int n_points,
		 int n_folds,
		 const ArrayList<index_t> &permutation, const Dataset& cv_set,
		 datanode* svm_module, const Matrix &kernel_matrix,
		 int* p_n_correct_class1, int* p_n_correct_class0) {
  int n_correct_class1 = *p_n_correct_class1;
  int n_correct_class0 = *p_n_correct_class0;

  printf("10-FOLD SVM Training and Testing... \n");

  fx_set_param_double(svm_module, "c", c);

  n_correct_class1 = 0;
  n_correct_class0 = 0;
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
	  n_correct_class1++;
	}
	else {
	  n_correct_class0++;
	}
      }
    }
  }

  int n_correct = n_correct_class1 + n_correct_class0;
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

  EmitResults(c_set, n_correct_results, n_points);
  /*
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
  */
}

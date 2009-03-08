#include "fastlib/fastlib.h"
#include "hmm_testing.h"
#include "contrib/niche/mmf/discreteHMM.h"
#include "contrib/niche/mmf/mmf3.cc"
#include "test_hmm_multinomial.cc"
#include "contrib/niche/svm/smo.h"
#include "contrib/niche/svm/svm.h"

#define KFOLD true

const fx_entry_doc hmm_testing_entries_doc[] = {
  {"lambda", FX_PARAM, FX_DOUBLE, NULL,
   "  Kernel bandwidth"},
  {"trans1", FX_PARAM, FX_STR, NULL,
   "  Filename for class 1 transition matrix"},
  {"trans0", FX_PARAM, FX_STR, NULL,
   "  Filename for class 0 transition matrix"},
  {"emis1", FX_PARAM, FX_STR, NULL,
   "  Filename for class 1 emission matrix"},
  {"emis0", FX_PARAM, FX_STR, NULL,
   "  Filename for class 0 emission matrix"},
  {"n_sequences_per_class", FX_PARAM, FX_INT, NULL,
   "  Number of sequences per class. There are two classes total"},
  {"n_symbols", FX_PARAM, FX_INT, NULL,
   "  Number of symbols. Only used when loading sequences from file."},
  {"save_hmms", FX_PARAM, FX_BOOL, NULL,
   "  Whether or not to save the sufficient statistics for the HMMs learned from the sequences."},
  {"load_hmms", FX_PARAM, FX_BOOL, NULL,
   "  Whether or not to load the sufficient statistics for the HMMs learned in a previous program execution."},
  {"hmms", FX_PARAM, FX_STR, NULL,
   "  Filename for the HMMS (either already learned and to be loaded, or to be learned and saved."},
  FX_ENTRY_DOC_DONE
};

const fx_submodule_doc hmm_testing_submodules[] = {
  {"svm", &svm_main_doc,
   " does svm stuff\n"},
  FX_SUBMODULE_DOC_DONE
};

const fx_module_doc hmm_testing_doc = {
  hmm_testing_entries_doc, hmm_testing_submodules,
  "MMK testing for HMM with Multinomial state distributions\n"
};



void GenerateAndTrainSequences(bool load_data,
			       const char* sequences_filename,
			       const char* transition_filename,
			       const char* emission_filename,
			       int n_sequences,
			       ArrayList<Vector>* p_initial_probs_vectors,
			       ArrayList<Matrix>* p_transition_matrices,
			       ArrayList<Matrix>* p_emission_matrices,
			       int first_index) {

  ArrayList<Vector> &initial_probs_vectors = *p_initial_probs_vectors;
  ArrayList<Matrix> &transition_matrices = *p_transition_matrices;
  ArrayList<Matrix> &emission_matrices = *p_emission_matrices;

  
  
  int max_iter_mmf = fx_param_int(NULL, "max_iter_mmf", 500);
  int max_rand_iter = fx_param_int(NULL, "max_rand_iter", 60);
  double tol_mmf = fx_param_double(NULL, "tolerance_mmf", 1e-4);

  int max_iter_bw = fx_param_int(NULL, "max_iter_bw", 500);
  double tol_bw = fx_param_double(NULL, "tolerance_bw", 1e-3);

  int last_index = first_index + n_sequences;

  if(load_data == false) {
    Matrix transition;
    data::Load(transition_filename, &transition);
    
    Matrix emission;
    data::Load(emission_filename, &emission);
    
    int n_symbols = emission.n_cols();

    DiscreteHMM hmm_gen;
    hmm_gen.Init(transition, emission);

    const int sequence_length = fx_param_int(NULL, "length", 100);
    int n_states = transition.n_cols() - 1; // subtract one for the start state
    //int n_states = fx_param_int(NULL, "n_states", 4);
    
    for(int k = first_index; k < last_index; k++) {
      Vector observed_sequence;
      Vector state_sequence;
      
      hmm_gen.GenerateSequence(sequence_length, &observed_sequence, &state_sequence);
      
      GetHMMSufficientStats(observed_sequence, n_states, n_symbols,
			    &(initial_probs_vectors[k]),
			    &(transition_matrices[k]),
			    &(emission_matrices[k]),
			    max_iter_mmf, max_rand_iter, tol_mmf,
			    max_iter_bw, tol_bw);
    }
  }
  else {
    Matrix sequences;
    data::Load(sequences_filename, &sequences);

    int sequence_length = sequences.n_rows();
    int n_symbols = fx_param_int_req(NULL, "n_symbols");
    double ratio = fx_param_double(NULL, "ratio", 0.1);
    int n_states = 
      ((int)floor(0.5 * sqrt(n_symbols * n_symbols
		 	     + 4 * (sequence_length * ratio + n_symbols + 1))
		  - 0.5 * n_symbols))
      + 1;
    n_states = 11; // override of formula to see if we results improve
    printf("n_states = %d\n", n_states);
    
    int sequence_num = 0;
    for(int k = first_index; k < last_index; k++) {
      Vector observed_sequence;

      sequences.MakeColumnVector(sequence_num, &observed_sequence);
      
      GetHMMSufficientStats(observed_sequence, n_states, n_symbols,
			    &(initial_probs_vectors[k]),
			    &(transition_matrices[k]),
			    &(emission_matrices[k]),
			    max_iter_mmf, max_rand_iter, tol_mmf,
			    max_iter_bw, tol_bw);
      sequence_num++;
    }
  }
}


void GetHMMSufficientStats(const Vector &observed_sequence,
			   int n_states, int n_symbols,
			   Vector* p_initial_probs_vector,
			   Matrix* p_transition_matrix,
			   Matrix* p_emission_matrix,
			   int max_iter_mmf, int max_rand_iter, double tol_mmf,
			   int max_iter_bw, double tol_bw) {
  
  DiscreteHMM learned_hmm;
  learned_hmm.Init(n_states, n_symbols);
  
  ArrayList<Vector> seqs;
  seqs.Init(1);
  seqs[0].Copy(observed_sequence);
    
    
  learned_hmm.TrainMMF(seqs, max_rand_iter, max_iter_mmf, tol_mmf);
  
  Vector stationary_probabilities;
  ComputeStationaryProbabilities(learned_hmm.transmission(),
				 &stationary_probabilities);
  
  //stationary_probabilities.PrintDebug("stationary probabilities");
  
  /* // a quick check to verify the stationary probabilities
     Matrix result, result2, result4, result16;
     la::MulInit(learned_hmm.transmission(), learned_hmm.transmission(),
     &result);
     la::MulInit(result, result, &result2);
     la::MulInit(result2, result2, &result4);
     la::MulInit(result4, result4, &result16);
     result16.PrintDebug("A^16");
  */
  
  Matrix new_transition_matrix;
  new_transition_matrix.Init(n_states + 1, n_states + 1);
  for(int i = 0; i < n_states + 1; i++) {
    new_transition_matrix.set(i, 0, 0);
  }
  for(int j = 0; j < n_states; j++) {
    new_transition_matrix.set(0, j + 1, stationary_probabilities[j]);
    for(int i = 0; i < n_states; i++) {
      new_transition_matrix.set(i + 1, j + 1,
				learned_hmm.transmission().get(i, j));
    }
  }
  
  //learned_hmm.transmission().PrintDebug("transition matrix");
  //new_transition_matrix.PrintDebug("augmented transition matrix");
  
  
  Matrix new_emission_matrix;
  new_emission_matrix.Init(n_states + 1, n_symbols);
  
  double uniform_p = 1. / ((double) n_symbols);
  
  for(int j = 0; j < n_symbols; j++) {
    new_emission_matrix.set(0, j, uniform_p);
    for(int i = 0; i < n_states; i++) {
      new_emission_matrix.set(i + 1, j,
			      learned_hmm.emission().get(i, j));
    }
  }

  //learned_hmm.emission().PrintDebug("emission matrix");
  //new_emission_matrix.PrintDebug("new emission matrix");

  DiscreteHMM augmented_learned_hmm;
  augmented_learned_hmm.Init(new_transition_matrix,
			     new_emission_matrix);
  
  /*
    learned_hmm.TrainBaumWelch(seqs, max_iter_bw, tol_bw);
    
    
    learned_hmm.transmission().PrintDebug("BW learned transition matrix");
    learned_hmm.emission().PrintDebug("BW learned emission matrix");
    
    transition_matrices[k].Copy(learned_hmm.transmission());
    emission_matrices[k].Copy(learned_hmm.emission());
  */
  printf("training HMM via Baum Welch\n");
  augmented_learned_hmm.TrainBaumWelch(seqs, max_iter_bw, tol_bw);

 
  //augmented_learned_hmm.transmission().PrintDebug("BW learned transition matrix");
  //augmented_learned_hmm.emission().PrintDebug("BW learned emission matrix");
  
  const Matrix &augmented_transition = augmented_learned_hmm.transmission();
  
  Vector &initial_probs = *p_initial_probs_vector;
  initial_probs.Init(n_states);
  for(int i = 0; i < n_states; i++) {
    initial_probs[i] = augmented_transition.get(0, i + 1);
  }
  
  Matrix &reduced_transition = *p_transition_matrix;
  reduced_transition.Init(n_states, n_states);
  for(int j = 0; j < n_states; j++) {
    for(int i = 0; i < n_states; i++) {
      reduced_transition.set(i, j,
			     augmented_transition.get(i + 1, j + 1));
    }
  }
  
  const Matrix &augmented_emission = augmented_learned_hmm.emission();
  
  Matrix &reduced_emission = *p_emission_matrix;
  reduced_emission.Init(n_states, n_symbols);
  for(int j = 0; j < n_symbols; j++) {
    for(int i = 0; i < n_states; i++) {
      reduced_emission.set(i, j,
			   augmented_emission.get(i + 1, j));
    }
  }
  
  /*
    transition_matrices[k].PrintDebug("transition matrix");
    initial_probs_vectors[k].PrintDebug("initial probs");
    emission_matrices[k].PrintDebug("emission matrix");
  */
}



/* transition_matrix is a stochastic matrix (each row sums to 1) */
void ComputeStationaryProbabilities(const Matrix &transition_matrix,
				    Vector* stationary_probabilities) {
  int n_states = transition_matrix.n_cols();

  Matrix transition_matrix_transpose;
  la::TransposeInit(transition_matrix, 
		    &transition_matrix_transpose);

  Matrix eigenvectors;
  Vector eigenvalues;
  la::EigenvectorsInit(transition_matrix_transpose,
		       &eigenvalues, &eigenvectors);

  // we use Perron-Frobenius by which the maximum eigenvalue of a stochastic matrix is 1
  double max_eigenvalue = -1;
  int argmax_eigenvalue = -1;
  for(int i = 0; i < n_states; i++) {
    if(eigenvalues[i] > max_eigenvalue) {
      max_eigenvalue = eigenvalues[i];
      argmax_eigenvalue = i;
    }
  }

  Vector one_eigenvector;
  eigenvectors.MakeColumnVector(argmax_eigenvalue,
				&one_eigenvector);
  
  double sum = 0;
  for(int i = 0; i < n_states; i++) {
    sum += one_eigenvector[i];
  }
  
  la::ScaleInit(1 / sum, one_eigenvector, stationary_probabilities);
}


void SetToRange(int x[], int start, int end) {
  int cur = start;
  for(int i = 0; cur < end; i++) {
    x[i] = cur;
    cur++;
  }
}

void RandPerm(int x[], int length) {
  int length_minus_1 = length - 1;

  int swap_index;
  int temp;
  for(int i = 0; i < length_minus_1; i++) {
    swap_index = rand() % (length - i);
    temp = x[i];
    x[i] = x[swap_index];
    x[swap_index] = temp;
  }
}

template<typename T>
void WriteOut(const T &object, FILE* file) {
  index_t size = ot::FrozenSize(object);
  char* buf = mem::Alloc<char>(size);
  ot::Freeze(buf, object);
  fwrite(&size, sizeof(size), 1, file);
  fwrite(buf, 1, size, file);
  mem::Free(buf);
}

template<typename T>
void ReadIn(T* object, FILE* file) {
  index_t size;
  fread(&size, sizeof(size), 1, file);
  char* buf = mem::Alloc<char>(size);
  fread(buf, 1, size, file);
  ot::InitThaw(object, buf);
  mem::Free(buf);
}

void SaveHMMs(const char* filename,
	      const ArrayList<Vector> &initial_probs_vectors,
	      const ArrayList<Matrix> &transition_matrices,
	      const ArrayList<Matrix> &emission_matrices,
	      const Matrix &training_data,
	      const Matrix &test_data) {
  FILE* file = fopen(filename, "wb");

  WriteOut(initial_probs_vectors, file);
  WriteOut(transition_matrices, file);
  WriteOut(emission_matrices, file);
  WriteOut(training_data, file);
  WriteOut(test_data, file);

  fclose(file);
}

void LoadHMMs(const char* filename,
	      ArrayList<Vector> *initial_probs_vectors,
	      ArrayList<Matrix> *transition_matrices,
	      ArrayList<Matrix> *emission_matrices,
	      Matrix *training_data,
	      Matrix *test_data) {
  FILE* file = fopen(filename, "rb");
  
  ReadIn(initial_probs_vectors, file);
  ReadIn(transition_matrices, file);
  ReadIn(emission_matrices, file);
  ReadIn(training_data, file);
  ReadIn(test_data, file);
  
  fclose(file);
}


void ObtainDataLearnHMMs(ArrayList<Vector> *p_initial_probs_vectors,
			 ArrayList<Matrix> *p_transition_matrices,
			 ArrayList<Matrix> *p_emission_matrices,
			 Matrix *p_training_data,
			 Matrix *p_test_data) {

  ArrayList<Vector> &initial_probs_vectors = *p_initial_probs_vectors;
  ArrayList<Matrix> &transition_matrices = *p_transition_matrices;
  ArrayList<Matrix> &emission_matrices = *p_emission_matrices;
  Matrix &training_data = *p_training_data;
  Matrix &test_data = *p_test_data;
  
  int n_sequences_per_class = fx_param_int(NULL, "n_sequences_per_class", 10);
  int n_sequences  = 2 * n_sequences_per_class;
  
  initial_probs_vectors.Init(2 * n_sequences_per_class);
  
  transition_matrices.Init(2 * n_sequences_per_class);
  
  emission_matrices.Init(2 * n_sequences_per_class);

  bool load_data = fx_param_bool(NULL, "load_data", false);
  
  if(load_data == false) {
    const char* class1_trans = 
      fx_param_str(NULL, "trans1", "../../../../class_1_transition.csv");
    const char* class0_trans =
      fx_param_str(NULL, "trans0", "../../../../class_0_transition.csv");
    const char* class1_emis = 
      fx_param_str(NULL, "emis1", "../../../../class_1_emission.csv");
    const char* class0_emis = 
      fx_param_str(NULL, "emis0", "../../../../class_0_emission.csv");
    
    printf("Generating sequences and training HMMs...\n");
    GenerateAndTrainSequences(false,
			      NULL,
			      class1_trans, class1_emis,
			      n_sequences_per_class,
			      &initial_probs_vectors, 
			      &transition_matrices,
			      &emission_matrices,
			      0);
    
    GenerateAndTrainSequences(false,
			      NULL,
			      class0_trans, class0_emis,
			      n_sequences_per_class,
			      &initial_probs_vectors,
			      &transition_matrices,
			      &emission_matrices,
			      n_sequences_per_class);
  }
  else {
    const char* class1_sequences_filename =
      fx_param_str(NULL, "class1_seq", "GC_50-55_coding.900bps_small.dat");
    
    const char* class0_sequences_filename =
      fx_param_str(NULL, "class0_seq", "GC_45-50_noncoding.900bps_small.dat");

    printf("Using sequences in files:\n\t%s\n\t%s\n",
	   class1_sequences_filename,
	   class0_sequences_filename);
    
    printf("Loading sequences and training HMMs...\n");
    GenerateAndTrainSequences(true,
			      class1_sequences_filename,
			      NULL, NULL,
			      n_sequences_per_class,
			      &initial_probs_vectors, 
			      &transition_matrices,
			      &emission_matrices,
			      0);
    
    GenerateAndTrainSequences(true,
			      class0_sequences_filename,
			      NULL, NULL,
			      n_sequences_per_class,
			      &initial_probs_vectors,
			      &transition_matrices,
			      &emission_matrices,
			      n_sequences_per_class);
  }
  
  Vector labels;
  labels.Init(n_sequences);
  for(int i = 0; i < n_sequences_per_class; i++) {
    labels[i] = 1;
  }
  for(int i = n_sequences_per_class; i < n_sequences; i++) {
    labels[i] = 0;
  }
  
  /*
    printf("\n\n\n\n\n\n\n\n\n\n");
    for(int k = 0; k < n_sequences; k++) {
    printf("sequence %d\n", k);
    initial_probs_vectors[k].PrintDebug("initial probs");
    transition_matrices[k].PrintDebug("transition matrix");
    emission_matrices[k].PrintDebug("emission matrix");
    }
  */
  
  
  /* Load training data */

  if(KFOLD == false) {
    int indices_class1[n_sequences_per_class];
    SetToRange(indices_class1, 0, n_sequences_per_class);
    int indices_class0[n_sequences_per_class];
    SetToRange(indices_class0, n_sequences_per_class, n_sequences);
  
    RandPerm(indices_class1, n_sequences_per_class);
    RandPerm(indices_class0, n_sequences_per_class);
  
    int n_training_points = n_sequences / 2;
    int n_test_points = n_sequences - n_training_points;
  
    int training_indices[n_training_points];
    int test_indices[n_test_points];
  
    int n_training_points_per_class = n_training_points / 2;
    int n_test_points_per_class = n_test_points / 2;
  
    int k_training = 0;
    int k_test = 0;
    int i;
    for(i = 0; k_training < n_training_points_per_class; i++) {
      training_indices[k_training] = indices_class1[i];
      k_training++;
    }
    for(; k_test < n_test_points_per_class; i++) {
      test_indices[k_test] = indices_class1[i];
      k_test++;
    }
    for(i = 0; k_training < n_training_points; i++) {
      training_indices[k_training] = indices_class0[i];
      k_training++;
    }
    for(; k_test < n_test_points; i++) {
      test_indices[k_test] = indices_class0[i];
      k_test++;
    }
  
    RandPerm(training_indices, n_training_points);
    RandPerm(test_indices, n_test_points);
  
    training_data.Init(2, n_training_points);
    for(int i = 0; i < n_training_points; i++) {
      training_data.set(0, i, training_indices[i]);
      training_data.set(1, i, labels[training_indices[i]]);
    }
  
    test_data.Init(2, n_test_points);
    for(int i = 0; i < n_test_points; i++) {
      test_data.set(0, i, test_indices[i]);
      test_data.set(1, i, labels[test_indices[i]]);
    }
  }
  else { // KFOLD
    int indices_class1[n_sequences_per_class];
    SetToRange(indices_class1, 0, n_sequences_per_class);
    int indices_class0[n_sequences_per_class];
    SetToRange(indices_class0, n_sequences_per_class, n_sequences);
  
    RandPerm(indices_class1, n_sequences_per_class);
    RandPerm(indices_class0, n_sequences_per_class);

    int training_indices[n_sequences];
    for(int i = 0; i < n_sequences_per_class; i++) {
      training_indices[i] = indices_class1[i];
    }
    int k = 0;
    for(int i = n_sequences_per_class; i < n_sequences; i++) {
      training_indices[i] = indices_class0[k];
      k++;
    }

    printf("\n\nindices\n");
    for(int i = 0; i < n_sequences; i++) {
      printf("i = %d ", training_indices[i]);
    }
    printf("\n\n\n");
    
    training_data.Init(2, n_sequences);
    for(int i = 0; i < n_sequences; i++) {
      training_data.set(0, i, training_indices[i]);
      training_data.set(1, i, labels[training_indices[i]]);
    }

    test_data.Init(2, 0);
  }
}




int eval_kfold_svm(double c, int n_points, const ArrayList<index_t> &permutation, const Dataset& cv_set, datanode* svm_module, const Matrix &kernel_matrix, int *n_correct_class1, int *n_correct_class0) {
  printf("10-FOLD SVM Training and Testing... \n");

  fx_set_param_double(svm_module, "c", c);

  int n_folds = 10;

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





int main(int argc, char* argv[]) {
  fx_module* root = fx_init(argc, argv, &hmm_testing_doc);

  srand(time(0));
  //srand(10);


  bool load_hmms = fx_param_bool(NULL, "load_hmms", false);
  bool save_hmms = fx_param_bool(NULL, "save_hmms", false);
  const char* hmms_filename = fx_param_str(NULL, "hmms", "hmms.dat");

  if(save_hmms == true) {
    struct stat stFileInfo;
    if(stat(hmms_filename, &stFileInfo) == 0) {
      fprintf(stderr, "Error: File to which learned HMMs are to be save already exists! We avoid overwriting previously saved HMMs.\nExiting...\n");
      return 1;
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

    ArrayList<double> c_tries;
    c_tries.Init(3);
    ArrayList<int> val_c_tries;
    val_c_tries.Init(3);
    ArrayList<int> val1_c_tries;
    val1_c_tries.Init(3);
    ArrayList<int> val0_c_tries;
    val0_c_tries.Init(3);


    double c_upper = 1e4;
    double c_lower = 1e-7;

    printf("c_upper = %f\n", c_upper);
    int val_c_upper = eval_kfold_svm(c_upper, n_points, permutation, cv_set, svm_module, kernel_matrix, &n_correct_class1, &n_correct_class0);
    c_tries[0] = c_upper;
    val_c_tries[0] = val_c_upper;
    val1_c_tries[0] = n_correct_class1;
    val0_c_tries[0] = n_correct_class0;

    printf("c_lower = %f\n", c_lower);
    int val_c_lower = eval_kfold_svm(c_lower, n_points, permutation, cv_set, svm_module, kernel_matrix, &n_correct_class1, &n_correct_class0);
    c_tries[1] = c_lower;
    val_c_tries[1] = val_c_lower;
    val1_c_tries[1] = n_correct_class1;
    val0_c_tries[1] = n_correct_class0;

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

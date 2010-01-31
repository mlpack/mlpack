#ifndef HMM_TESTING_H
#define HMM_TESTING_H

#include "fastlib/fastlib.h"
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


void GenerateAndTrainSequences(const char* transition_filename,
			       const char* emission_filename,
			       int n_sequences,
			       ArrayList<Vector>* p_initial_probs_vectors,
			       ArrayList<Matrix>* p_transition_matrices,
			       ArrayList<Matrix>* p_emission_matrices,
			       int first_index);

void GetHMMSufficientStats(const Vector &observed_sequence,
			   int n_states, int n_symbols,
			   Vector* p_initial_probs_vector,
			   Matrix* p_transition_matrix,
			   Matrix* p_emission_matrix,
			   int max_iter_mmf, int max_rand_iter, double tol_mmf,
			   int max_iter_bw, double tol_bw);

void ObtainDataLearnHMMs(ArrayList<Vector> *p_initial_probs_vectors,
			 ArrayList<Matrix> *p_transition_matrices,
			 ArrayList<Matrix> *p_emission_matrices,
			 Matrix *p_training_data,
			 Matrix *p_test_data);

void SaveHMMs(const char* filename,
	      const ArrayList<Vector> &initial_probs_vectors,
	      const ArrayList<Matrix> &transition_matrices,
	      const ArrayList<Matrix> &emission_matrices,
	      const Matrix &training_data,
	      const Matrix &test_data);

void LoadHMMs(const char* filename,
	      ArrayList<Vector> *initial_probs_vectors,
	      ArrayList<Matrix> *transition_matrices,
	      ArrayList<Matrix> *emission_matrices,
	      Matrix *training_data,
	      Matrix *test_data);

void ComputeStationaryProbabilities(const Matrix &transition_matrix,
				    Vector* stationary_probabilities);

void SetToRange(int x[], int start, int end);

void RandPerm(int x[], int length);

int eval_loocv_svm(double c, int n_points, const ArrayList<index_t> &permutation, const Dataset& cv_set, datanode* svm_module, const Matrix &kernel_matrix, int *n_correct_class1, int *n_correct_class0);

void DoMMFBaumWelch(const ArrayList<Vector> &sequences, DiscreteHMM* p_augmented_learned_hmm,
		    int n_states, int n_symbols,
		    int max_rand_iter, int max_iter_mmf, double tol_mmf,
		    int max_iter_bw, double tol_bw);

int GenerativeHMMClassifier(const ArrayList<Vector> &sequences,
			     const Dataset &training_set,
			     const Dataset &test_set);

void LoadVaryingLengthData(const char* filename, ArrayList<Vector>* p_data);






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

    Matrix sequences;
    sequences.Init(sequence_length, n_sequences);
    
    int sequence_num = 0;
    for(int k = first_index; k < last_index; k++) {
      printf("sequence %d\n", k);
      Vector observed_sequence;
      Vector state_sequence;
      
      hmm_gen.GenerateSequence(sequence_length, &observed_sequence, &state_sequence);
      sequences.CopyVectorToColumn(sequence_num, observed_sequence);
      
      GetHMMSufficientStats(observed_sequence, n_states, n_symbols,
			    &(initial_probs_vectors[k]),
			    &(transition_matrices[k]),
			    &(emission_matrices[k]),
			    max_iter_mmf, max_rand_iter, tol_mmf,
			    max_iter_bw, tol_bw);
      sequence_num++;
    }
    struct stat stFileInfo;
    if(stat(sequences_filename, &stFileInfo) == 0) {
      FATAL("Error: File to which saved sequences are to be saved already exists! We avoid overwriting previously saved sequences. Exiting...");
    }
    else {
      data::Save(sequences_filename, sequences);
    }
  }
  else {
    //Matrix sequences;
    //data::Load(sequences_filename, &sequences);
    ArrayList<Vector> sequences;
    LoadVaryingLengthData(sequences_filename, &sequences);
    
    
    int n_symbols = fx_param_int_req(NULL, "n_symbols");
    //int sequence_length = sequences.n_rows();
    /*
    double ratio = fx_param_double(NULL, "ratio", 0.1);
    int n_states = 
      ((int)floor(0.5 * sqrt(n_symbols * n_symbols
		 	     + 4 * (sequence_length * ratio + n_symbols + 1))
		  - 0.5 * n_symbols))
		  + 1;
    int n_states = 9; // override of formula to see if results improve
    printf("n_states = %d\n", n_states);
    */
    int n_states = -1;
    
    int sequence_num = 0;
    for(int k = first_index; k < last_index; k++) {
      printf("sequence %d\n", k);
      //Vector observed_sequence;
      //sequences.MakeColumnVector(sequence_num, &observed_sequence);
      Vector &observed_sequence = sequences[sequence_num];
      
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
  
  //observed_sequence.PrintDebug("observed_sequence");
  if(n_states == -1) {
    int sequence_length = observed_sequence.length();
    double ratio = fx_param_double(NULL, "ratio", 0.1);
    n_states = 
      ((int)floor(0.5 * sqrt(n_symbols * n_symbols
			     + 4 * (sequence_length * ratio + n_symbols + 1))
		  - 0.5 * n_symbols))
      + 1;
    //int n_states = 9; // override of formula to see if results improve
  }
  printf("n_states = %d\n", n_states);


  
  ArrayList<Vector> seqs;
  seqs.Init(1);
  seqs[0].Copy(observed_sequence);

  DiscreteHMM learned_hmm;
  learned_hmm.Init(n_states, n_symbols);
  printf("training HMM via MMF\n");
  learned_hmm.TrainMMF(seqs, max_rand_iter, max_iter_mmf, tol_mmf);
  //printf("finished MMF\n");
  Vector stationary_probabilities;
  ComputeStationaryProbabilities(learned_hmm.transmission(),
				 &stationary_probabilities);
  
  //stationary_probabilities.PrintDebug("stationary probabilities");
  
  
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



void DoMMFBaumWelch(const ArrayList<Vector> &sequences, DiscreteHMM* p_augmented_learned_hmm,
		    int n_states, int n_symbols,
		    int max_rand_iter, int max_iter_mmf, double tol_mmf,
		    int max_iter_bw, double tol_bw) {


  DiscreteHMM learned_hmm;
  learned_hmm.Init(n_states, n_symbols);

  printf("MMF on %d sequences\n", sequences.size());
  learned_hmm.TrainMMF(sequences, max_rand_iter, max_iter_mmf, tol_mmf);
  
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

  DiscreteHMM &augmented_learned_hmm = *p_augmented_learned_hmm;
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
  
  augmented_learned_hmm.TrainBaumWelch(sequences, max_iter_bw, tol_bw);

 
  augmented_learned_hmm.transmission().PrintDebug("BW learned transition matrix");
  augmented_learned_hmm.emission().PrintDebug("BW learned emission matrix");



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
    swap_index = (rand() % (length - i)) + i;
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

  const char* class1_sequences_filename =
    fx_param_str(NULL, "class1_seq", "GC_50-55_coding.900bps_small.dat");
  
  const char* class0_sequences_filename =
    fx_param_str(NULL, "class0_seq", "GC_45-50_noncoding.900bps_small.dat");

  
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
			      class1_sequences_filename,
			      class1_trans, class1_emis,
			      n_sequences_per_class,
			      &initial_probs_vectors, 
			      &transition_matrices,
			      &emission_matrices,
			      0);
    
    GenerateAndTrainSequences(false,
			      class0_sequences_filename,
			      class0_trans, class0_emis,
			      n_sequences_per_class,
			      &initial_probs_vectors,
			      &transition_matrices,
			      &emission_matrices,
			      n_sequences_per_class);
  }
  else {

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
  
    //RandPerm(indices_class1, n_sequences_per_class);
    //RandPerm(indices_class0, n_sequences_per_class);

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


int GenerativeHMMClassifier(const ArrayList<Vector> &sequences,
			    const Dataset &training_set,
			    const Dataset &test_set,
			    const char* class1_hmm_filename,
			    const char* class0_hmm_filename) {
  int n_symbols = fx_param_int_req(NULL, "n_symbols");  
  int n_states = fx_param_int_req(NULL, "n_states");

  

  int max_iter_mmf = fx_param_int(NULL, "max_iter_mmf", 500);
  int max_rand_iter = fx_param_int(NULL, "max_rand_iter", 60);
  double tol_mmf = fx_param_double(NULL, "tolerance_mmf", 1e-4);

  int max_iter_bw = fx_param_int(NULL, "max_iter_bw", 500);
  double tol_bw = fx_param_double(NULL, "tolerance_bw", 1e-3);

  int n_training_points = training_set.n_points();
  int n_test_points = test_set.n_points();

  int n_class1_training_points = 0;
  int n_class0_training_points = 0;

  for(int i = 0; i < n_training_points; i++) {
    if(((int)(training_set.point(i)[1])) == 1) {
      n_class1_training_points++;
    }
    else {
      n_class0_training_points++;
    }
  }


  ArrayList<Vector> class1_training_sequences;
  ArrayList<Vector> class0_training_sequences;
  ArrayList<Vector> test_sequences;

  class1_training_sequences.Init(n_class1_training_points);
  class0_training_sequences.Init(n_class0_training_points);
  test_sequences.Init(n_test_points);

  int class1_training_num = 0;
  int class0_training_num = 0;
  printf("\ntraining on ");
  for(int i = 0; i < n_training_points; i++) {
    if(((int)(training_set.point(i)[1])) == 1) {
      class1_training_sequences[class1_training_num].Copy(sequences[(int)(training_set.point(i)[0])]);
      printf("%d ", (int)(training_set.point(i)[0]));
      class1_training_num++;
    }
    else {
      class0_training_sequences[class0_training_num].Copy(sequences[(int)(training_set.point(i)[0])]);
      printf("%d ", (int)(training_set.point(i)[0]));
      class0_training_num++;
    }
  }

  printf("\n testing on ");
  for(int i = 0; i < n_test_points; i++) {
    test_sequences[i].Copy(sequences[(int)(test_set.point(i)[0])]);
    printf("%d ", (int)(test_set.point(i)[0]));
  }
  printf("\n");

  DiscreteHMM class1_learned_hmm;
  DoMMFBaumWelch(class1_training_sequences, &class1_learned_hmm,
		 n_states, n_symbols,
		 max_rand_iter, max_iter_mmf, tol_mmf,
		 max_iter_bw, tol_bw);
  class1_learned_hmm.SaveProfile(class1_hmm_filename);

  DiscreteHMM class0_learned_hmm;
  DoMMFBaumWelch(class0_training_sequences, &class0_learned_hmm,
		 n_states, n_symbols,
		 max_rand_iter, max_iter_mmf, tol_mmf,
		 max_iter_bw, tol_bw);
  class0_learned_hmm.SaveProfile(class0_hmm_filename);
  
  int n_correct = 0;
  for(int i = 0; i < n_test_points; i++) {
    double class0_ll = class0_learned_hmm.ComputeLogLikelihood(test_sequences[i]);
    double class1_ll = class1_learned_hmm.ComputeLogLikelihood(test_sequences[i]);
    //test_sequences[i].PrintDebug("test sequence");


    double ll_diff = class1_ll - class0_ll;

    printf("ll_diff = %f\n", ll_diff);

    if(ll_diff > 0) {
      if(((int)(test_set.point(i)[1])) == 1) {
	printf("one ");
	n_correct++;
      }
    }
    else {
      if(((int)(test_set.point(i)[1])) == 0) {
	printf("zero ");
	n_correct++;
      }
    }
  }
  printf("\n");

  return n_correct;

}

void LoadVaryingLengthData(const char* filename, ArrayList<Vector>* p_data) {
  ArrayList<Vector> &data = *p_data;

  data.Init();

  FILE* file = fopen(filename, "r");

  char* buffer = (char*) malloc(sizeof(char) * 70000);
  size_t len = 70000;


  int n_read;
  while((n_read = getline(&buffer, &len, file)) != -1) {
    int sequence_length = (int) ((n_read - 1) / 2);
  
    Vector sequence;
    sequence.Init(sequence_length);
    for(int i = 0; i < sequence_length; i++) {
      sscanf(buffer + (2 * i), "%lf", &(sequence[i]));
    }
    
    //sequence.PrintDebug("sequence");
    
    data.PushBackCopy(sequence);
  }

  free(buffer);

  fclose(file);
  
}



#endif /* HMM_TESTING_H */

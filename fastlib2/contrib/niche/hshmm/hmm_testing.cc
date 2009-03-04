#include "fastlib/fastlib.h"
#include "hmm_testing.h"
#include "contrib/niche/mmf/discreteHMM.h"
#include "contrib/niche/mmf/mmf3.cc"
#include "test_hmm_multinomial.cc"
#include "contrib/niche/svm/smo.h"
#include "contrib/niche/svm/svm.h"


void GenerateAndTrainSequences(const char* transition_filename,
			       const char* emission_filename,
			       int n_sequences,
			       ArrayList<Vector>* p_initial_probs_vectors,
			       ArrayList<Matrix>* p_transition_matrices,
			       ArrayList<Matrix>* p_emission_matrices,
			       int first_index) {

  ArrayList<Vector> &initial_probs_vectors = *p_initial_probs_vectors;
  ArrayList<Matrix> &transition_matrices = *p_transition_matrices;
  ArrayList<Matrix> &emission_matrices = *p_emission_matrices;

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
  
  int max_iter_mmf = fx_param_int(NULL, "max_iter_mmf", 500);
  int max_rand_iter = fx_param_int(NULL, "max_rand_iter", 60);
  double tol_mmf = fx_param_double(NULL, "tolerance_mmf", 1e-4);

  int max_iter_bw = fx_param_int(NULL, "max_iter_bw", 500);
  double tol_bw = fx_param_double(NULL, "tolerance_bw", 1e-3);

  int last_index = first_index + n_sequences;

  for(int k = first_index; k < last_index; k++) {
    Vector observed_sequence;
    Vector state_sequence;
    
    hmm_gen.GenerateSequence(sequence_length, &observed_sequence, &state_sequence);
    
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
    
    augmented_learned_hmm.TrainBaumWelch(seqs, max_iter_bw, tol_bw);

 
    //augmented_learned_hmm.transmission().PrintDebug("BW learned transition matrix");
    //augmented_learned_hmm.emission().PrintDebug("BW learned emission matrix");

    const Matrix &augmented_transition = augmented_learned_hmm.transmission();
    
    Vector &initial_probs = initial_probs_vectors[k];
    initial_probs.Init(n_states);
    for(int i = 0; i < n_states; i++) {
      initial_probs[i] = augmented_transition.get(0, i + 1);
    }

    Matrix &reduced_transition = transition_matrices[k];
    reduced_transition.Init(n_states, n_states);
    for(int j = 0; j < n_states; j++) {
      for(int i = 0; i < n_states; i++) {
	reduced_transition.set(i, j,
			       augmented_transition.get(i + 1, j + 1));
      }
    }

    const Matrix &augmented_emission = augmented_learned_hmm.emission();
    
    Matrix &reduced_emission = emission_matrices[k];
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

void SaveData(const char* name,
	      const ArrayList<Vector> &initial_probs_vectors,
	      const ArrayList<Matrix> &transition_matrices,
	      const ArrayList<Matrix> &emission_matrices,
	      const Matrix &training_data,
	      const Matrix &test_data) {
  FILE* file = fopen(name, "wb");

  WriteOut(initial_probs_vectors, file);
  WriteOut(transition_matrices, file);
  WriteOut(emission_matrices, file);
  WriteOut(training_data, file);
  WriteOut(test_data, file);

  fclose(file);
}

void LoadData(const char* name,
	      ArrayList<Vector> *initial_probs_vectors,
	      ArrayList<Matrix> *transition_matrices,
	      ArrayList<Matrix> *emission_matrices,
	      Matrix *training_data,
	      Matrix *test_data) {
  FILE* file = fopen(name, "rb");
  
  ReadIn(initial_probs_vectors, file);
  ReadIn(transition_matrices, file);
  ReadIn(emission_matrices, file);
  ReadIn(training_data, file);
  ReadIn(test_data, file);
  
  fclose(file);
}




int main(int argc, char* argv[]) {
  fx_init(argc, argv, NULL);

  srand(time(0));
  //srand(10);

  int n_sequences_per_class;
  int n_sequences;

  
  ArrayList<Vector> initial_probs_vectors;
  ArrayList<Matrix> transition_matrices;
  ArrayList<Matrix> emission_matrices;

  Matrix training_data;
  Matrix test_data;

  bool load_data = fx_param_bool(NULL, "load_data", false);
  bool save_data = fx_param_bool(NULL, "save_data", false);
  const char* data_filename = fx_param_str(NULL, "data", "dataset.dat");

  if(load_data == true) {
    LoadData(data_filename, &initial_probs_vectors, &transition_matrices,
	     &emission_matrices, &training_data, &test_data);
    n_sequences = initial_probs_vectors.size();
    n_sequences_per_class = n_sequences / 2; // we assume balanced data
  }
  else {
    n_sequences_per_class = fx_param_int(NULL, "n_sequences", 10);
    n_sequences  = 2 * n_sequences_per_class;

    initial_probs_vectors.Init(2 * n_sequences_per_class);
    
    transition_matrices.Init(2 * n_sequences_per_class);
    
    emission_matrices.Init(2 * n_sequences_per_class);
    
    fprintf(stderr, "Generating sequences and training HMMs...\n");
    GenerateAndTrainSequences("class_0_transition.csv", "class_0_emission.csv",
			      n_sequences_per_class,
			      &initial_probs_vectors, 
			      &transition_matrices,
			      &emission_matrices,
			      0);
    
    GenerateAndTrainSequences("class_1_transition.csv", "class_1_emission.csv",
			      n_sequences_per_class,
			      &initial_probs_vectors,
			      &transition_matrices,
			      &emission_matrices,
			      n_sequences_per_class);
    
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
    
    int indices_class_1[n_sequences_per_class];
    SetToRange(indices_class_1, 0, n_sequences_per_class);
    int indices_class_0[n_sequences_per_class];
    SetToRange(indices_class_0, n_sequences_per_class, n_sequences);
    
    RandPerm(indices_class_1, n_sequences_per_class);
    RandPerm(indices_class_0, n_sequences_per_class);
    
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
      training_indices[k_training] = indices_class_1[i];
      k_training++;
    }
    for(; k_test < n_test_points_per_class; i++) {
      test_indices[k_test] = indices_class_1[i];
      k_test++;
    }
    for(i = 0; k_training < n_training_points; i++) {
      training_indices[k_training] = indices_class_0[i];
      k_training++;
    }
    for(; k_test < n_test_points; i++) {
      test_indices[k_test] = indices_class_0[i];
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
  
    if(save_data) {
      SaveData(data_filename, initial_probs_vectors, transition_matrices,
	       emission_matrices, training_data, test_data);
    }
  }


  
  // Compute Kernel Matrix
  Matrix kernel_matrix;
  fprintf(stderr, "Computing kernel matrix...\n");
  ComputeKernelAllPairs(2 * n_sequences_per_class,
			initial_probs_vectors, transition_matrices,
			emission_matrices,
			&kernel_matrix);
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
  data::Save("kernel_matrix.csv", kernel_matrix);



  

  Dataset training_set;
  training_set.CopyMatrix(training_data);
    
  // Begin SVM Training | Training and Testing
  datanode *svm_module = fx_submodule(fx_root, "svm");

  SVM<SVMRBFKernel> svm; // this should be changed from SVMRBFKernel to something like SVMMMKKernel
  int learner_typeid = 0; // for svm_c
  fprintf(stderr, "SVM Training... \n");
  svm.InitTrain(learner_typeid, training_set, svm_module, kernel_matrix);
    

  Dataset test_set;
  test_set.CopyMatrix(test_data);

  fprintf(stderr, "SVM Predicting... \n");
  svm.BatchPredict(learner_typeid, test_set, "predicted_values"); // TODO:param_req


  Matrix predicted_values;
  data::Load("predicted_values", &predicted_values);
  
  int n_test_points = predicted_values.n_rows();
  int n_correct = 0;
  for(int i = 0; i < n_test_points; i++) {
    n_correct += (((int)predicted_values.get(0, i)) == ((int)test_data.get(1,i)));
  }

  printf("n_correct / n_test_points = %d / %d = %f\n",
	 n_correct, n_test_points,
	 ((float) n_correct) / ((float) n_test_points));
    
  
  fx_done(NULL);

  return SUCCESS_PASS;
}

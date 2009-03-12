#include "hmm_testing.h"

int main(int argc, char* argv[]) {

  fx_module* root = fx_init(argc, argv, &hmm_testing_doc);

  int n_sequences_per_class = fx_param_int(NULL, "n_sequences_per_class", 10);
  int n_sequences  = 2 * n_sequences_per_class;

  const int n_folds = 10;

  ArrayList<Vector> sequences;
  sequences.Init(n_sequences);

  const char* class1_sequences_filename =
    fx_param_str(NULL, "class1_seq", "GC_50-55_coding.900bps_small.dat");
  const char* class0_sequences_filename =
    fx_param_str(NULL, "class0_seq", "GC_45-50_noncoding.900bps_small.dat");

  
  ArrayList<Vector> class1_sequences;
  LoadVaryingLengthData(class1_sequences_filename, &class1_sequences);
  //Matrix class1_sequences_matrix;
  //data::Load(class1_sequences_filename, &class1_sequences_matrix);
  for(int i = 0; i < n_sequences_per_class; i++) {
    //Vector observed_sequence;
    //class1_sequences_matrix.MakeColumnVector(i, &observed_sequence);
    //sequences[i].Copy(observed_sequence);
    sequences[i].Copy(class1_sequences[i]);
  }

  ArrayList<Vector> class0_sequences;
  LoadVaryingLengthData(class0_sequences_filename, &class0_sequences);
  //Matrix class0_sequences_matrix;
  //data::Load(class0_sequences_filename, &class0_sequences_matrix);
  for(int i = 0; i < n_sequences_per_class; i++) {
    //Vector observed_sequence;
    //class0_sequences_matrix.MakeColumnVector(i, &observed_sequence);
    //sequences[i + n_sequences_per_class].Copy(observed_sequence);
    sequences[i + n_sequences_per_class].Copy(class0_sequences[i]);
  }

  /*
  int sequence_length = class1_sequences_matrix.n_rows();
  int n_symbols = fx_param_int_req(NULL, "n_symbols");
  double ratio = fx_param_double(NULL, "ratio", 0.1);
  
  int n_states = 
    ((int)floor(0.5 * sqrt(n_symbols * n_symbols
			   + 4 * (sequence_length * ratio + n_symbols + 1))
		- 0.5 * n_symbols))
    + 1;
  */
  int n_states = 9; // override of formula to see if results improve
  printf("n_states = %d\n", n_states);

  Matrix cv_data;
  cv_data.Init(2, n_sequences_per_class * 2);

  for(int i = 0; i < n_sequences_per_class; i++) {
    cv_data.set(0, i, i);
    cv_data.set(1, i, 1);
  }
  for(int i = n_sequences_per_class; i < n_sequences; i++) {
    cv_data.set(0, i, i);
    cv_data.set(1, i, 0);
  }
 

  Dataset cv_set;
  cv_set.CopyMatrix(cv_data);


  ArrayList<index_t> permutation;
  math::MakeIdentityPermutation(n_sequences, &permutation);

  printf("sequences.size() = %d\n", sequences.size());

  const char* class1_hmm_filename = fx_param_str_req(NULL, "class1_hmm_filename");
  const char* class0_hmm_filename  = fx_param_str_req(NULL, "class0_hmm_filename");

  int n_correct = 0;
  int fold_num = 1;
  /*for(int fold_num = 0; fold_num < n_folds; fold_num++)*/ {
    Dataset training_set;
    Dataset test_set;
    cv_set.SplitTrainTest(n_folds, fold_num, permutation, &training_set, &test_set);
    printf("training_set.n_points() = %d\n", training_set.n_points());

    char class1_hmm_fold_filename[100];
    sprintf(class1_hmm_fold_filename, "%s_%d.dis", class1_hmm_filename, fold_num);
    char class0_hmm_fold_filename[100];
    sprintf(class0_hmm_fold_filename, "%s_%d.dis", class0_hmm_filename, fold_num);
    

    int n_correct_fold = GenerativeHMMClassifier(sequences, training_set, test_set,
						 class1_hmm_fold_filename,
						 class0_hmm_fold_filename);
    printf("fold %d\n\tn_correct = %d\n", fold_num, n_correct_fold);
    n_correct += n_correct_fold;
  }

  printf("total n_correct = %d\n", n_correct);

  fx_done(root);
}

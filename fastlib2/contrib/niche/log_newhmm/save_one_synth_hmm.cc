#include "fastlib/fastlib.h"
#include "test_engine.h"
#include "utils.h"


void GetModelClasses(bool *p_model_class1, bool *p_model_class0) {
  bool &model_class1 = *p_model_class1;
  bool &model_class0 = *p_model_class0;
  
  const char* model_classes = fx_param_str_req(NULL, "model_classes");
  if(strcmp(model_classes, "both") == 0) {
    model_class1 = true;
    model_class0 = true;
  }
  else if(strcmp(model_classes, "class1") == 0) {
    model_class1 = true;
  }
  else if(strcmp(model_classes, "class0") == 0) {
    model_class0 = true;
  }
  else {
    FATAL("Error: Parameter 'model_classes' must be set to \"both\", \"class1\", or \"class0\". Exiting...");
  }
}

void GetOneSynthHMM(bool model_class1, bool model_class0,
		    int n_states,
		    HMM<Multinomial> *p_hmm,
		    GenVector<int> *p_labels) {
  HMM<Multinomial> &hmm = *p_hmm;
  GenVector<int> &labels = *p_labels;

  ArrayList<GenMatrix<int> > sequences;
  sequences.Init(0);

  int n_dims = 4;

  const char* class1_filename = "synth1000_pos.dat";
  const char* class0_filename = "synth1000_neg.dat";

  ArrayList<GenMatrix<int> > class1_sequences;
  LoadVaryingLengthData(class1_filename, &class1_sequences);
  int n_class1 = class1_sequences.size();

  ArrayList<GenMatrix<int> > class0_sequences;
  LoadVaryingLengthData(class0_filename, &class0_sequences);
  int n_class0 = class0_sequences.size();
  
  if(model_class1) {
    sequences.AppendSteal(&class1_sequences);
  }

  if(model_class0) {
    sequences.AppendSteal(&class0_sequences);
  }

  
//   char seq_name[80];
//   for(int i = 0; i < sequences.size(); i++) {
//     sprintf(seq_name, "sequences[%d]", i);    
//     PrintDebug(seq_name, sequences[i], "%d");
//   }
  

  hmm.Init(n_states, n_dims, MULTINOMIAL);
  hmm.InitParameters(sequences);
  //hmm.PrintDebug("hmm after calling InitParameters(sequences)");
  hmm.ViterbiUpdate(sequences);
  //hmm.PrintDebug("hmm after calling ViterbiUpdate(sequences)");
  printf("%d sequences\n", sequences.size());
  hmm.BaumWelch(sequences,
		1e-6 * ((double)1),
		1000);
  //hmm.PrintDebug("hmm after calling BaumWelch");
  
  int n_sequences = n_class1 + n_class0;
  labels.Init(n_sequences);
  for(int i = 0; i < n_class1; i++) {
    labels[i] = 1;
  }
  for(int i = n_class1; i < n_sequences; i++) {
    labels[i] = 0;
  }
}

void SaveOneSynthHMM() {
  int n_states = fx_param_int_req(NULL, "n_states");
  const char* labels_filename = "frozen_synth_labels";
  const char* one_hmm_partial_filename = "frozen_synth_one_hmm_topo";
  char one_hmm_filename[80];

  bool model_class1 = false;
  bool model_class0 = false;
  GetModelClasses(&model_class1, &model_class0);
  if(model_class1 && model_class0) {
    sprintf(one_hmm_filename, "%s%d_model_both", one_hmm_partial_filename, n_states);
  }
  else if(model_class1) {
    sprintf(one_hmm_filename, "%s%d_model_class1", one_hmm_partial_filename, n_states);
  }
  else { // model_class0
    sprintf(one_hmm_filename, "%s%d_model_class0", one_hmm_partial_filename, n_states);
  }
  printf("one_hmm_filename = \"%s\"\n", one_hmm_filename);
  struct stat stFileInfo;
  if(stat(one_hmm_filename, &stFileInfo) == 0) {
    FATAL("Error: File to which HMM is to be saved already exists! Bypassing learning and exiting...");
  }

  HMM<Multinomial> hmm;
  GenVector<int> labels;
  
  GetOneSynthHMM(model_class1, model_class0, n_states, &hmm, &labels);
  
  WriteOutOTObject(one_hmm_filename, hmm);
  WriteOutOTObject(labels_filename, labels);
}

void SaveKFoldSynthHMMs(int n_folds) {

  const char* class1_filename = "synth1000_pos.dat";
  const char* class0_filename = "synth1000_neg.dat";

  const int class1_label = 1;
  const int class0_label = 0;

  int n_dims = 2;
  int n_states = fx_param_int_req(NULL, "n_states");
  
  const char* model_classes = fx_param_str_req(NULL, "model_classes");
  int selected_label;
  if(strcmp(model_classes, "class1") == 0) {
    selected_label = class1_label;
  }
  else if(strcmp(model_classes, "class0") == 0) {
    selected_label = class0_label;
  }
  else {
    FATAL("Error: For k-fold cross-validation, parameter 'model_classes' must be set to \"class1\" or \"class0\". Exiting...");
  }

  ArrayList<GenMatrix<int> > sequences;
  LoadVaryingLengthData(class1_filename, &sequences);
  int n_class1 = sequences.size();

  ArrayList<GenMatrix<int> > class0_sequences;
  LoadVaryingLengthData(class0_filename, &class0_sequences);
  int n_class0 = class0_sequences.size();

  sequences.AppendSteal(&class0_sequences);
  int n_sequences = n_class1 + n_class0;

  Matrix id_label_pairs;
  id_label_pairs.Init(2, n_sequences);
  for(int i = 0; i < n_class1; i++) {
    id_label_pairs.set(0, i, i);
    id_label_pairs.set(1, i, class1_label);
  }
  for(int i = n_class1; i < n_sequences; i++) {
    id_label_pairs.set(0, i, i);
    id_label_pairs.set(1, i, class0_label);
  }
  Dataset cv_set;
  cv_set.CopyMatrix(id_label_pairs);

  ArrayList<index_t> permutation;
  math::MakeIdentityPermutation(n_sequences, &permutation);


  for(int fold_num = 0; fold_num < n_folds; fold_num++) {
    Dataset training_set;
    Dataset test_set;
    
    cv_set.SplitTrainTest(n_folds, fold_num, permutation, &training_set, &test_set);
    const Matrix &training_set_matrix = training_set.matrix();

    ArrayList<GenMatrix<int> > selected_training_sequences;
    selected_training_sequences.Init(0);
    // should set capacity to n_sequences / n_folds
    for(int i = 0; i < training_set.n_points(); i++) {
      if(training_set_matrix.get(1, i) == selected_label) {
	selected_training_sequences.PushBackCopy(sequences[(int)(training_set_matrix.get(0, i))]);
      }
    }

    char hmm_filename[80];
    if(selected_label == class1_label) {
      sprintf(hmm_filename, "frozen/frozen_synth_one_hmm_topo%d_model_class1_fold%dof%d", n_states, fold_num, n_folds);
    }
    else {
      sprintf(hmm_filename, "frozen/frozen_synth_one_hmm_topo%d_model_class0_fold%dof%d", n_states, fold_num, n_folds);
    }

    printf("Fold %d hmm_filename = \"%s\"\n", fold_num, hmm_filename);
    struct stat stFileInfo;
    if(stat(hmm_filename, &stFileInfo) == 0) {
      FATAL("Error: File to which HMM is to be saved already exists! Bypassing learning and exiting...");
    }

    HMM<Multinomial> hmm;
    hmm.Init(n_states, n_dims, MULTINOMIAL);
    hmm.InitParameters(selected_training_sequences);
    //hmm.PrintDebug("hmm after calling InitParameters(sequences)");
    hmm.ViterbiUpdate(selected_training_sequences);
    //hmm.PrintDebug("hmm after calling ViterbiUpdate(sequences)");
    printf("%d sequences\n", selected_training_sequences.size());
    hmm.BaumWelch(selected_training_sequences,
		  1e-6 * ((double)1),
		  1000);

    WriteOutOTObject(hmm_filename, hmm);
  }
}



int main(int argc, char* argv[]) {
  fx_init(argc, argv, NULL);

  srand48(time(0));
  srand(time(0));

  const char* mode = fx_param_str_req(NULL, "mode");
  if(strcmp(mode, "full") == 0) {
    SaveOneSynthHMM();
  }
  else if(strcmp(mode, "kfold") == 0) {
    SaveKFoldSynthHMMs(10);
  }
  else {
    FATAL("Error: Parameter 'mode' must be set to \"full\" or \"kfold\". Exiting...");
  }    
  
  fx_done(fx_root);
}

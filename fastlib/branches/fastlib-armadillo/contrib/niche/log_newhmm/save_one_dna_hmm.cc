#include "fastlib/fastlib.h"
#include "test_engine.h"
#include "utils.h"

void GetModelClasses(bool *p_model_exons, bool *p_model_introns) {
  bool &model_exons = *p_model_exons;
  bool &model_introns = *p_model_introns;
  
  const char* model_classes = fx_param_str_req(NULL, "model_classes");
  if(strcmp(model_classes, "both") == 0) {
    model_exons = true;
    model_introns = true;
  }
  else if(strcmp(model_classes, "exons") == 0) {
    model_exons = true;
  }
  else if(strcmp(model_classes, "introns") == 0) {
    model_introns = true;
  }
  else {
    FATAL("Error: Parameter 'model_classes' must be set to \"both\", \"exons\", or \"introns\". Exiting...");
  }
}

void GetOneDNAHMM(bool model_exons, bool model_introns,
		  int n_states,
		  HMM<Multinomial> *p_hmm,
		  GenVector<int> *p_labels) {
  HMM<Multinomial> &hmm = *p_hmm;
  GenVector<int> &labels = *p_labels;

  ArrayList<GenMatrix<int> > sequences;
  sequences.Init(0);

  int n_dims = 4;

  const char* exons_filename = "exons_small.dat";
  const char* introns_filename = "introns_small.dat";

  ArrayList<GenMatrix<int> > exon_sequences;
  LoadVaryingLengthData(exons_filename, &exon_sequences);
  int n_exons = exon_sequences.size();

  ArrayList<GenMatrix<int> > intron_sequences;
  LoadVaryingLengthData(introns_filename, &intron_sequences);
  int n_introns = intron_sequences.size();
  
  if(model_exons) {
    sequences.AppendSteal(&exon_sequences);
  }

  if(model_introns) {
    sequences.AppendSteal(&intron_sequences);
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
  
  int n_sequences = n_exons + n_introns;
  labels.Init(n_sequences);
  for(int i = 0; i < n_exons; i++) {
    labels[i] = 1;
  }
  for(int i = n_exons; i < n_sequences; i++) {
    labels[i] = 0;
  }
}

void SaveOneDNAHMM() {
  int n_states = fx_param_int_req(NULL, "n_states");
  const char* labels_filename = "frozen_dna_labels";
  const char* one_hmm_partial_filename = "frozen_dna_one_hmm_topo";
  char one_hmm_filename[80];

  bool model_exons = false;
  bool model_introns = false;
  GetModelClasses(&model_exons, &model_introns);
  if(model_exons && model_introns) {
    sprintf(one_hmm_filename, "%s%d_model_both", one_hmm_partial_filename, n_states);
  }
  else if(model_exons) {
    sprintf(one_hmm_filename, "%s%d_model_exons", one_hmm_partial_filename, n_states);
  }
  else { // model_introns
    sprintf(one_hmm_filename, "%s%d_model_introns", one_hmm_partial_filename, n_states);
  }
  printf("one_hmm_filename = \"%s\"\n", one_hmm_filename);
  struct stat stFileInfo;
  if(stat(one_hmm_filename, &stFileInfo) == 0) {
    FATAL("Error: File to which HMM is to be saved already exists! Bypassing learning and exiting...");
  }

  HMM<Multinomial> hmm;
  GenVector<int> labels;
  
  GetOneDNAHMM(model_exons, model_introns, n_states, &hmm, &labels);
  
  WriteOutOTObject(one_hmm_filename, hmm);
  WriteOutOTObject(labels_filename, labels);
}

void SaveKFoldDNAHMMs(int n_folds) {

  const char* exons_filename = "exons_small.dat";
  const char* introns_filename = "introns_small.dat";

  const int exon_label = 1;
  const int intron_label = 0;

  int n_dims = 4;
  int n_states = fx_param_int_req(NULL, "n_states");
  
  const char* model_classes = fx_param_str_req(NULL, "model_classes");
  int selected_label;
  if(strcmp(model_classes, "exons") == 0) {
    selected_label = exon_label;
  }
  else if(strcmp(model_classes, "introns") == 0) {
    selected_label = intron_label;
  }
  else {
    FATAL("Error: For k-fold cross-validation, parameter 'model_classes' must be set to \"exons\" or \"introns\". Exiting...");
  }

  ArrayList<GenMatrix<int> > sequences;
  LoadVaryingLengthData(exons_filename, &sequences);
  int n_exons = sequences.size();

  ArrayList<GenMatrix<int> > intron_sequences;
  LoadVaryingLengthData(introns_filename, &intron_sequences);
  int n_introns = intron_sequences.size();

  sequences.AppendSteal(&intron_sequences);
  int n_sequences = n_exons + n_introns;

  Matrix id_label_pairs;
  id_label_pairs.Init(2, n_sequences);
  for(int i = 0; i < n_exons; i++) {
    id_label_pairs.set(0, i, i);
    id_label_pairs.set(1, i, exon_label);
  }
  for(int i = n_exons; i < n_sequences; i++) {
    id_label_pairs.set(0, i, i);
    id_label_pairs.set(1, i, intron_label);
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
    if(selected_label == exon_label) {
      sprintf(hmm_filename, "frozen/frozen_dna_one_hmm_topo%d_model_exons_fold%dof%d", n_states, fold_num, n_folds);
    }
    else {
      sprintf(hmm_filename, "frozen/frozen_dna_one_hmm_topo%d_model_introns_fold%dof%d", n_states, fold_num, n_folds);
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
    SaveOneDNAHMM();
  }
  else if(strcmp(mode, "kfold") == 0) {
    SaveKFoldDNAHMMs(10);
  }
  else {
    FATAL("Error: Parameter 'mode' must be set to \"full\" or \"kfold\". Exiting...");
  }    
  
  fx_done(fx_root);
}

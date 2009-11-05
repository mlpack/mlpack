#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include "loghmm.h"
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

void LoadSequencesAndLabels(ArrayList<GenMatrix<double> >* p_sequences,
			    GenVector<int>* p_labels) {
  ArrayList<GenMatrix<double> > &sequences = *p_sequences;
  GenVector<int> &labels = *p_labels;

  const char* class1_filename =
    fx_param_str_req(NULL, "data_pos");
  const char* class0_filename =
    fx_param_str_req(NULL, "data_neg");


  LoadVaryingLengthData(class1_filename, &sequences);
  int n_class1 = sequences.size();

  ArrayList<GenMatrix<double> > class0_sequences;
  LoadVaryingLengthData(class0_filename, &class0_sequences);
  int n_class0 = class0_sequences.size();

  sequences.AppendSteal(&class0_sequences);

  int n_sequences = n_class1 + n_class0;
  labels.Init(n_sequences);
  for(int i = 0; i < n_class1; i++) {
    labels[i] = 1;
  }
  for(int i = n_class1; i < n_sequences; i++) {
    labels[i] = 0;
  }
}

void LoadOneHMMAndSequences(HMM<DiagGaussian>* p_hmm,
			    ArrayList<GenMatrix<double> >* p_sequences,
			    GenVector<int>* p_labels) {
  ArrayList<GenMatrix<double> > &sequences = *p_sequences;
  
  int n_states = fx_param_int_req(NULL, "n_states");
  const char* one_hmm_partial_filename = "../../../../frozen_one_hmm_topo";
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
  const char* labels_filename = "../../../../frozen_labels";

  ReadInOTObject(one_hmm_filename, p_hmm);
  ReadInOTObject(labels_filename, p_labels);
  

  const char* class1_filename =
    fx_param_str(NULL, "data_class1", "../../../../data_pos.dat");
  const char* class0_filename =
    fx_param_str(NULL, "data_class0", "../../../../data_neg.dat");


  LoadVaryingLengthData(class1_filename, &sequences);

  ArrayList<GenMatrix<double> > class0_sequences;
  LoadVaryingLengthData(class0_filename, &class0_sequences);

  sequences.AppendSteal(&class0_sequences);
}

void LoadKFoldHMMAndSequences(int n_folds,
				   ArrayList<HMM<DiagGaussian> >* p_kfold_hmms,
				   ArrayList<GenMatrix<double> >* p_sequences,
				   GenVector<int>* p_labels) {
  ArrayList<HMM<DiagGaussian> > &kfold_hmms = *p_kfold_hmms;
  ArrayList<GenMatrix<double> > &sequences = *p_sequences;
  GenVector<int> &labels = *p_labels;

  const char* experiment_name = fx_param_str_req(NULL, "exp");

  int n_states = fx_param_int_req(NULL, "n_states");

  const int class1_label = 1;
  const int class0_label = 0;

  bool model_class1 = false;
  bool model_class0 = false;
  const char* model_classes = fx_param_str_req(NULL, "model_classes");
  if(strcmp(model_classes, "pos") == 0) {
    model_class1 = true;
  }
  else if(strcmp(model_classes, "neg") == 0) {
    model_class0 = true;
  }
  else {
    FATAL("Error: For k-fold cross-validation, parameter 'model_classes' must be set to \"class1\" or \"class0\". Exiting...");
  }

  kfold_hmms.Init(n_folds);
  for(int fold_num = 0; fold_num < n_folds; fold_num++) {
    char hmm_filename[80];
    if(model_class1) {
      sprintf(hmm_filename, "../../../../frozen/frozen_%s_one_hmm_topo%d_model_class1_fold%dof%d", experiment_name, n_states, fold_num, n_folds);
    }
    else { // model_class0
      sprintf(hmm_filename, "../../../../frozen/frozen_%s_one_hmm_topo%d_model_class0_fold%dof%d", experiment_name, n_states, fold_num, n_folds);
    }
    
    printf("Fold %d hmm_filename = \"%s\"\n", fold_num, hmm_filename);
    ReadInOTObject(hmm_filename, &(kfold_hmms[fold_num]));
  }

  const char* class1_filename =
    fx_param_str_req(NULL, "data_pos");
  const char* class0_filename =
    fx_param_str_req(NULL, "data_neg");

  LoadVaryingLengthData(class1_filename, &sequences);
  int n_class1 = sequences.size();

  ArrayList<GenMatrix<double> > class0_sequences;
  LoadVaryingLengthData(class0_filename, &class0_sequences);
  int n_class0 = class0_sequences.size();

  sequences.AppendSteal(&class0_sequences);

  int n_sequences = n_class1 + n_class0;

  labels.Init(n_sequences);
  for(int i = 0; i < n_class1; i++) {
    labels[i] = class1_label;
  }
  for(int i = n_class1; i < n_sequences; i++) {
    labels[i] = class0_label;
  }
}

void LoadKFoldHMMPairAndSequences(int n_folds,
				  ArrayList<HMM<DiagGaussian> >* p_kfold_class1_hmms,
				  ArrayList<HMM<DiagGaussian> >* p_kfold_class0_hmms,
				  ArrayList<GenMatrix<double> >* p_sequences,
				  GenVector<int>* p_labels) {
  ArrayList<HMM<DiagGaussian> > &kfold_class1_hmms = *p_kfold_class1_hmms;
  ArrayList<HMM<DiagGaussian> > &kfold_class0_hmms = *p_kfold_class0_hmms;
  ArrayList<GenMatrix<double> > &sequences = *p_sequences;
  GenVector<int> &labels = *p_labels;

  const char* experiment_name = fx_param_str_req(NULL, "exp");
  
  int n_states_class1 = fx_param_int_req(NULL, "n_states_pos");
  int n_states_class0 = fx_param_int_req(NULL, "n_states_neg");

  const int class1_label = 1;
  const int class0_label = 0;

  kfold_class1_hmms.Init(n_folds);
  kfold_class0_hmms.Init(n_folds);
  for(int fold_num = 0; fold_num < n_folds; fold_num++) {
    char class1_hmm_filename[80];
    char class0_hmm_filename[80];
    sprintf(class1_hmm_filename, "../../../../frozen/frozen_%s_one_hmm_topo%d_model_class1_fold%dof%d", experiment_name, n_states_class1, fold_num, n_folds);
    sprintf(class0_hmm_filename, "../../../../frozen/frozen_%s_one_hmm_topo%d_model_class0_fold%dof%d", experiment_name, n_states_class0, fold_num, n_folds);
    
    printf("Fold %d class1_hmm_filename = \"%s\"\n",
	   fold_num, class1_hmm_filename);
    printf("Fold %d class0_hmm_filename = \"%s\"\n",
	   fold_num, class0_hmm_filename);
    ReadInOTObject(class1_hmm_filename, &(kfold_class1_hmms[fold_num]));
    ReadInOTObject(class0_hmm_filename, &(kfold_class0_hmms[fold_num]));
  }

  const char* class1_filename =
    fx_param_str_req(NULL, "data_pos");
  const char* class0_filename =
    fx_param_str_req(NULL, "data_neg");

  LoadVaryingLengthData(class1_filename, &sequences);
  int n_class1 = sequences.size();

  ArrayList<GenMatrix<double> > class0_sequences;
  LoadVaryingLengthData(class0_filename, &class0_sequences);
  int n_class0 = class0_sequences.size();

  sequences.AppendSteal(&class0_sequences);

  int n_sequences = n_class1 + n_class0;

  labels.Init(n_sequences);
  for(int i = 0; i < n_class1; i++) {
    labels[i] = class1_label;
  }
  for(int i = n_class1; i < n_sequences; i++) {
    labels[i] = class0_label;
  }
}


#endif /* TEST_UTILS_H */

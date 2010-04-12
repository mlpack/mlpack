#ifndef TEST_DNA_UTILS_H
#define TEST_DNA_UTILS_H

#include "hmm.h"
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

void LoadSequencesAndLabels(ArrayList<GenMatrix<int> >* p_sequences,
			    GenVector<int>* p_labels) {
  ArrayList<GenMatrix<int> > &sequences = *p_sequences;
  GenVector<int> &labels = *p_labels;

  const char* exons_filename = "../../../../exons_small.dat";
  const char* introns_filename = "../../../../introns_small.dat";

  //const char* exons_filename = "exons_small.dat";
  //const char* introns_filename = "introns_small.dat";


  LoadVaryingLengthData(exons_filename, &sequences);
  int n_exons = sequences.size();

  ArrayList<GenMatrix<int> > intron_sequences;
  LoadVaryingLengthData(introns_filename, &intron_sequences);
  int n_introns = intron_sequences.size();

  sequences.AppendSteal(&intron_sequences);

  int n_sequences = n_exons + n_introns;
  labels.Init(n_sequences);
  for(int i = 0; i < n_exons; i++) {
    labels[i] = 1;
  }
  for(int i = n_exons; i < n_sequences; i++) {
    labels[i] = 0;
  }
}

void LoadOneDNAHMMAndSequences(HMM<Multinomial>* p_hmm,
			       ArrayList<GenMatrix<int> >* p_sequences,
			       GenVector<int>* p_labels) {
  ArrayList<GenMatrix<int> > &sequences = *p_sequences;

  int n_states = fx_param_int_req(NULL, "n_states");
  const char* one_hmm_partial_filename = "../../../../frozen_dna_one_hmm_topo";
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
  const char* labels_filename = "../../../../frozen_dna_labels";

  ReadInOTObject(one_hmm_filename, p_hmm);
  ReadInOTObject(labels_filename, p_labels);
  

  const char* exons_filename = "../../../../exons_small.dat";
  const char* introns_filename = "../../../../introns_small.dat";

  LoadVaryingLengthData(exons_filename, &sequences);

  ArrayList<GenMatrix<int> > intron_sequences;
  LoadVaryingLengthData(introns_filename, &intron_sequences);

  sequences.AppendSteal(&intron_sequences);
}

void LoadKFoldDNAHMMAndSequences(int n_folds,
				 ArrayList<HMM<Multinomial> >* p_kfold_hmms,
				 ArrayList<GenMatrix<int> >* p_sequences,
				 GenVector<int>* p_labels) {
  ArrayList<HMM<Multinomial> > &kfold_hmms = *p_kfold_hmms;
  ArrayList<GenMatrix<int> > &sequences = *p_sequences;
  GenVector<int> &labels = *p_labels;

  int n_states = fx_param_int_req(NULL, "n_states");

  const int exon_label = 1;
  const int intron_label = 0;

  bool model_exons = false;
  bool model_introns = false;
  const char* model_classes = fx_param_str_req(NULL, "model_classes");
  if(strcmp(model_classes, "exons") == 0) {
    model_exons = true;
  }
  else if(strcmp(model_classes, "introns") == 0) {
    model_introns = true;
  }
  else {
    FATAL("Error: For k-fold cross-validation, parameter 'model_classes' must be set to \"exons\" or \"introns\". Exiting...");
  }

  kfold_hmms.Init(n_folds);
  for(int fold_num = 0; fold_num < n_folds; fold_num++) {
    char hmm_filename[80];
    if(model_exons) {
      sprintf(hmm_filename, "../../../../frozen/frozen_dna_one_hmm_topo%d_model_exons_fold%dof%d", n_states, fold_num, n_folds);
    }
    else { // model_introns
      sprintf(hmm_filename, "../../../../frozen/frozen_dna_one_hmm_topo%d_model_introns_fold%dof%d", n_states, fold_num, n_folds);
    }
    
    printf("Fold %d hmm_filename = \"%s\"\n", fold_num, hmm_filename);
    ReadInOTObject(hmm_filename, &(kfold_hmms[fold_num]));
  }

//   const char* exons_filename = "../../../../exons_small.dat";
//   const char* introns_filename = "../../../../introns_small.dat";
  const char* exons_filename = "../../../../exons_small.dat";
  const char* introns_filename = "../../../../introns_small.dat";

  LoadVaryingLengthData(exons_filename, &sequences);
  int n_exons = sequences.size();

  ArrayList<GenMatrix<int> > intron_sequences;
  LoadVaryingLengthData(introns_filename, &intron_sequences);
  int n_introns = intron_sequences.size();

  sequences.AppendSteal(&intron_sequences);

  int n_sequences = n_exons + n_introns;

  labels.Init(n_sequences);
  for(int i = 0; i < n_exons; i++) {
    labels[i] = exon_label;
  }
  for(int i = n_exons; i < n_sequences; i++) {
    labels[i] = intron_label;
  }
}

void LoadKFoldDNAHMMPairAndSequences(int n_folds,
				     ArrayList<HMM<Multinomial> >* p_kfold_exon_hmms,
				     ArrayList<HMM<Multinomial> >* p_kfold_intron_hmms,
				     ArrayList<GenMatrix<int> >* p_sequences,
				     GenVector<int>* p_labels) {
  ArrayList<HMM<Multinomial> > &kfold_exon_hmms = *p_kfold_exon_hmms;
  ArrayList<HMM<Multinomial> > &kfold_intron_hmms = *p_kfold_intron_hmms;
  ArrayList<GenMatrix<int> > &sequences = *p_sequences;
  GenVector<int> &labels = *p_labels;

  int n_states_exon = fx_param_int_req(NULL, "n_states_exon");
  int n_states_intron = fx_param_int_req(NULL, "n_states_intron");

  const int exon_label = 1;
  const int intron_label = 0;

  kfold_exon_hmms.Init(n_folds);
  kfold_intron_hmms.Init(n_folds);
  for(int fold_num = 0; fold_num < n_folds; fold_num++) {
    char exon_hmm_filename[80];
    char intron_hmm_filename[80];
    sprintf(exon_hmm_filename, "../../../../frozen/frozen_dna_one_hmm_topo%d_model_exons_fold%dof%d", n_states_exon, fold_num, n_folds);
    sprintf(intron_hmm_filename, "../../../../frozen/frozen_dna_one_hmm_topo%d_model_introns_fold%dof%d", n_states_intron, fold_num, n_folds);
    
    printf("Fold %d exon_hmm_filename = \"%s\"\n",
	   fold_num, exon_hmm_filename);
    printf("Fold %d intron_hmm_filename = \"%s\"\n",
	   fold_num, intron_hmm_filename);
    ReadInOTObject(exon_hmm_filename, &(kfold_exon_hmms[fold_num]));
    ReadInOTObject(intron_hmm_filename, &(kfold_intron_hmms[fold_num]));
  }

//   const char* exons_filename = "../../../../exons_small.dat";
//   const char* introns_filename = "../../../../introns_small.dat";
  const char* exons_filename = "../../../../exons_small.dat";
  const char* introns_filename = "../../../../introns_small.dat";

  LoadVaryingLengthData(exons_filename, &sequences);
  int n_exons = sequences.size();

  ArrayList<GenMatrix<int> > intron_sequences;
  LoadVaryingLengthData(introns_filename, &intron_sequences);
  int n_introns = intron_sequences.size();

  sequences.AppendSteal(&intron_sequences);

  int n_sequences = n_exons + n_introns;

  labels.Init(n_sequences);
  for(int i = 0; i < n_exons; i++) {
    labels[i] = exon_label;
  }
  for(int i = n_exons; i < n_sequences; i++) {
    labels[i] = intron_label;
  }
}



#endif /* TEST_DNA_UTILS_H */

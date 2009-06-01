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

/*
void SaveOneDNAHMM() {
  int n_states = fx_param_int_req(NULL, "n_states");
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
  const char* labels_filename = "frozen_dna_labels";
  
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
*/

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


int main(int argc, char* argv[]) {
  fx_init(argc, argv, NULL);

  ArrayList<HMM<Multinomial> > kfold_hmms;
  ArrayList<GenMatrix<int> > sequences;
  GenVector<int> labels;
  int n_folds = 10;
  LoadKFoldDNAHMMAndSequences(n_folds, &kfold_hmms, &sequences, &labels);
  TestHMMLatMMKClassificationKFold(n_folds, kfold_hmms, sequences, labels);
  
  fx_done(fx_root);
}

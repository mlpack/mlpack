#include "fastlib/fastlib.h"
#include "test_dna_utils.h"
#include "test_engine.h"
//#include "utils.h"



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

/*
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
*/


int main(int argc, char* argv[]) {
  fx_init(argc, argv, NULL);

  const char* mode = fx_param_str_req(NULL, "mode");
  if(strcmp(mode, "full") == 0) {
    HMM<Multinomial> hmm;
    ArrayList<GenMatrix<int> > sequences;
    GenVector<int> labels;
    LoadOneDNAHMMAndSequences(&hmm, &sequences, &labels);
    TestHMMLatMMKClassification(hmm, sequences, labels);
  }
  else if(strcmp(mode, "kfold") == 0) {
    ArrayList<HMM<Multinomial> > kfold_hmms;
    ArrayList<GenMatrix<int> > sequences;
    GenVector<int> labels;
    int n_folds = fx_param_int(NULL, "n_folds", 10);
    LoadKFoldDNAHMMAndSequences(n_folds, &kfold_hmms, &sequences, &labels);
    TestHMMLatMMKClassificationKFold(n_folds, kfold_hmms, sequences, labels);
  }
  else {
    FATAL("Error: Parameter 'mode' must be set to \"full\" or \"kfold\". Exiting...");
  }    
  
  fx_done(fx_root);
}

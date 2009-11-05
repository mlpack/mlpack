#include "fastlib/fastlib.h"
#include "test_dna_utils.h"
#include "test_engine.h"
#include "utils.h"


void GetClassHMMsAndLabels(int n_states,
			   HMM<Multinomial>* p_exons_hmm,
			   HMM<Multinomial>* p_introns_hmm,
			   GenVector<int>* p_labels,
			   ArrayList<GenMatrix<int> >* p_test_exon_sequences,
			   ArrayList<GenMatrix<int> >* p_test_intron_sequences) {
  HMM<Multinomial> &exons_hmm = *p_exons_hmm;
  HMM<Multinomial> &introns_hmm = *p_introns_hmm;
  GenVector<int> &labels = *p_labels;
  ArrayList<GenMatrix<int> > &test_exon_sequences = *p_test_exon_sequences;
  ArrayList<GenMatrix<int> > &test_intron_sequences = *p_test_intron_sequences;


  int n_dims = 4;

  const char* exons_filename = "exons_small.dat";
  const char* introns_filename = "introns_small.dat";


  ArrayList<GenMatrix<int> > exon_sequences;
  LoadVaryingLengthData(exons_filename, &exon_sequences);
  int n_exons = exon_sequences.size();
  exon_sequences.SegmentInit((int)(0.2 * n_exons),
			     &test_exon_sequences);
  n_exons = exon_sequences.size();

  ArrayList<GenMatrix<int> > intron_sequences;
  LoadVaryingLengthData(introns_filename, &intron_sequences);
  int n_introns = intron_sequences.size();
  intron_sequences.SegmentInit((int)(0.2 * n_introns),
			       &test_intron_sequences);
  n_introns = intron_sequences.size();
  
  n_states = 5;
  exons_hmm.Init(n_states, n_dims, MULTINOMIAL);
  exons_hmm.InitParameters(exon_sequences);
  //exons_hmm.PrintDebug("exons hmm after calling InitParameters(sequences)");
  exons_hmm.ViterbiUpdate(exon_sequences);
  //exons_hmm.PrintDebug("exons hmm after calling ViterbiUpdate(sequences)");
  printf("%d exon sequences\n", exon_sequences.size());
  exons_hmm.BaumWelch(exon_sequences,
		      1e-6 * ((double)1),
		      1000);
  //exons_hmm.PrintDebug("exons hmm after calling BaumWelch");

  n_states = 6;
  introns_hmm.Init(n_states, n_dims, MULTINOMIAL);
  introns_hmm.InitParameters(intron_sequences);
  //introns_hmm.PrintDebug("introns hmm after calling InitParameters(sequences)");
  introns_hmm.ViterbiUpdate(intron_sequences);
  //introns_hmm.PrintDebug("introns hmm after calling ViterbiUpdate(sequences)");
  printf("%d intron sequences\n", intron_sequences.size());
  introns_hmm.BaumWelch(intron_sequences,
		      1e-6 * ((double)1),
		      1000);
  //introns_hmm.PrintDebug("introns hmm after calling BaumWelch");

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
  
//   PrintDebug("labels", *p_labels, "%d");
//   char hmm_name[80];
//   for(int i = 0; i < 1000; i++) {
//     sprintf(hmm_name, "hmm %d", i);
//     (*p_hmms)[i].PrintDebug(hmm_name);
//   }


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

  srand48(time(0));
  srand(time(0));


  ArrayList<HMM<Multinomial> > kfold_exon_hmms;
  ArrayList<HMM<Multinomial> > kfold_intron_hmms;
  ArrayList<GenMatrix<int> > sequences;
  GenVector<int> labels;
  int n_folds = fx_param_int(NULL, "n_folds", 10);
  LoadKFoldDNAHMMPairAndSequences(n_folds,
				  &kfold_exon_hmms, &kfold_intron_hmms,
				  &sequences, &labels);
  TestHMMBayesClassificationKFold(n_folds, kfold_exon_hmms, kfold_intron_hmms,
				  sequences, labels);

  fx_done(fx_root);
}


/*



  int n_states = fx_param_int_req(NULL, "n_states");

  HMM<Multinomial> exons_hmm;
  HMM<Multinomial> introns_hmm;
  GenVector<int> labels;
  ArrayList<GenMatrix<int> > test_exon_sequences;
  ArrayList<GenMatrix<int> > test_intron_sequences;

  GetClassHMMsAndLabels(n_states,
			&exons_hmm,
			&introns_hmm,
			&labels,
			&test_exon_sequences,
			&test_intron_sequences);
  
  printf("%d test exon sequences\n", test_exon_sequences.size());
  printf("%d test intron sequences\n", test_intron_sequences.size());


  int n_exons_correct = 0;  
  for(int i = 0; i < test_exon_sequences.size(); i++) {
    Matrix p_x_given_q;
    ArrayList<Matrix> p_qq_t;
    Matrix p_qt;
    double exon_neg_likelihood;
    double intron_neg_likelihood;
    exons_hmm.ExpectationStepNoLearning(test_exon_sequences[i],
					&p_x_given_q,
					&p_qq_t,
					&p_qt,
					&exon_neg_likelihood);
    p_x_given_q.Destruct();
    p_qq_t.Renew();
    p_qt.Destruct();
    introns_hmm.ExpectationStepNoLearning(test_exon_sequences[i],
					  &p_x_given_q,
					  &p_qq_t,
					  &p_qt,
					  &intron_neg_likelihood);
    if(exon_neg_likelihood < intron_neg_likelihood) {
      n_exons_correct++;
    }
  }

  int n_introns_correct = 0;
  for(int i = 0; i < test_intron_sequences.size(); i++) {
    Matrix p_x_given_q;
    ArrayList<Matrix> p_qq_t;
    Matrix p_qt;
    double exon_neg_likelihood;
    double intron_neg_likelihood;
    exons_hmm.ExpectationStepNoLearning(test_intron_sequences[i],
					&p_x_given_q,
					&p_qq_t,
					&p_qt,
					&exon_neg_likelihood);
    p_x_given_q.Destruct();
    p_qq_t.Renew();
    p_qt.Destruct();
    introns_hmm.ExpectationStepNoLearning(test_intron_sequences[i],
					  &p_x_given_q,
					  &p_qq_t,
					  &p_qt,
					  &intron_neg_likelihood);
    if(intron_neg_likelihood <= exon_neg_likelihood) {
      n_introns_correct++;
    }
  }

  printf("exons accuracy: %f\n",
	 ((double)n_exons_correct)
	 / ((double)(test_exon_sequences.size())));

  printf("introns accuracy: %f\n",
	 ((double)n_introns_correct)
	 / ((double)(test_intron_sequences.size())));

  printf("overall accuracy: %f\n",
	 ((double)(n_exons_correct + n_introns_correct))
	 / ((double)(test_exon_sequences.size() + test_intron_sequences.size())));

  fx_done(fx_root);
}
*/

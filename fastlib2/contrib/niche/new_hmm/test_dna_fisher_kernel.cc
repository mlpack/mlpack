#include "fastlib/fastlib.h"
#include "test_engine.h"
#include "utils.h"

int main(int argc, char* argv[]) {
  fx_init(argc, argv, NULL);

  HMM<Multinomial> hmm;
  ArrayList<GenMatrix<int> > sequences;
  GenVector<int> labels;

  ReadInOTObject("frozen_dna_one_hmm_topo4_model_exons", &hmm);
  ReadInOTObject("frozen_dna_labels", &labels);

  const char* exons_filename = "exons_small.dat";
  const char* introns_filename = "introns_small.dat";

  LoadVaryingLengthData(exons_filename, &sequences);

  ArrayList<GenMatrix<int> > intron_sequences;
  LoadVaryingLengthData(introns_filename, &intron_sequences);
  sequences.AppendSteal(&intron_sequences);

  double val = FisherKernel(hmm, sequences[0], sequences[1]);
  printf("fisher kernel = %f\n", val);
  
  TestHMMFisherKernelClassification(hmm, sequences, labels);
  
  fx_done(fx_root);
}

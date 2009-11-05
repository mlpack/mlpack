#include "fastlib/fastlib.h"
#include "test_utils.h"
#include "test_engine.h"
#include "utils.h"


int main(int argc, char* argv[]) {
  fx_init(argc, argv, NULL);

  ArrayList<GenMatrix<double> > sequences;
  GenVector<int> labels;
  LoadSequencesAndLabels(&sequences, &labels);
  TestMarkovMMKClassification(sequences, labels);

  fx_done(fx_root);
}

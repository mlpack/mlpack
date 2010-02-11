#include "fastlib/fastlib.h"
#include "dtw.h"

int main(int argc, char* argv[]) {
  fx_init(argc, argv, NULL);

  Vector ts1;
  LoadTimeSeries("ts1.dat", &ts1);
  ts1.PrintDebug("ts1");

  Vector ts2;
  LoadTimeSeries("ts2.dat", &ts2); 
  ts2.PrintDebug("ts2");
  
  ArrayList< GenVector<int> > best_path;

  double score = ComputeDTWAlignmentScore(ts1, ts2, &best_path);
  
  printf("score = %f\n", score);

  fx_done(fx_root);
}

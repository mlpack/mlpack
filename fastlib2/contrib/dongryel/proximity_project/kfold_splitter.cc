#include "kfold_splitter.h"

int main(int argc, char *argv[]) {

  fx_init(argc, argv);
  KFoldSplitter ks;

  ks.Init();
  ks.Split();
  fx_done();
  return 0;
}

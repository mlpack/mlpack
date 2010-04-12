#include "kfold_splitter.h"

int main(int argc, char *argv[]) {

  fx_init(argc, argv, NULL);
  KFoldSplitter ks;

  ks.Init();
  ks.Split();
  fx_done(NULL);
  return 0;
}

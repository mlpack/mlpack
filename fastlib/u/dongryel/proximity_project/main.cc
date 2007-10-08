#include "fastlib/fastlib_int.h"
#include "spill_kdtree.h"

typedef GeneralBinarySpaceTree<DHrectBound<2>, Matrix> Tree;

int main(int argc, char *argv[]) {
  fx_init(argc, argv);
  const char *fname = fx_param_str(NULL, "data", NULL);
  Dataset dataset_;
  dataset_.InitFromFile(fname);
  Matrix data_;
  data_.Own(&(dataset_.matrix()));
  int leaflen = fx_param_int(NULL, "leaflen", 20);
  Tree *root_ = 
    proximity::MakeSpillKdTreeMidpoint<Tree>(data_, leaflen, NULL);
  
  root_->Print();
  fx_done();
  return 0;
}

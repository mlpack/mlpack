#include <fastlib/fastlib.h>
#include "cover_tree.h"
#include "ctree.h"

int main(int argc, char *argv[]) {

  fx_init(argc, argv);

  const char *ref_data = fx_param_str(NULL, "R", "my_ref_data.data");
  Matrix r_set;
  data::Load(ref_data, &r_set);

  CoverTreeNode *root;

  //NOTIFY("Entering tree construction\n");
  root = ctree::MakeCoverTree(r_set);
  //NOTIFY("Tree construction done\n");
  ctree::PrintTree(root);

  fx_silence();
  fx_done();

  return 0;
}

#include "fastlib/fastlib.h"
#include "fastlib/tree/statistic.h"
#include "general_spacetree.h"
#include "contrib/dongryel/fast_multipole_method/fmm_stat.h"
#include "subspace_stat.h"
#include "cfmm_tree.h"
#include "mlpack/kde/dataset_scaler.h"

int main(int argc, char *argv[]) {

  fx_init(argc, argv, NULL);

  const char *fname = fx_param_str_req(NULL, "data");
  Matrix dataset;
  data::Load(fname, &dataset);

  int leaflen = fx_param_int(NULL, "leaflen", 30);
  int min_required_ws_index = fx_param_int(NULL, "min_ws_index", 2);

  printf("Constructing the tree...\n");
  fx_timer_start(NULL, "cfmm_tree_build");

  ArrayList<Matrix *> matrices;
  matrices.Init(1);
  matrices[0] = &dataset;

  ArrayList<Vector *> targets;
  const char *target_fname = fx_param_str_req(NULL, "target");

  Matrix target_incoming;

  data::Load(target_fname, &target_incoming);

  Vector target;
  target.Init(target_incoming.n_cols());

  for (index_t i = 0; i < target.length(); i++) {
    target[i] = target_incoming.get(0, i);
  }
  targets.Init(1);
  targets[0] = &target;

  ArrayList< ArrayList<index_t> > old_from_new;
  ArrayList< ArrayList<proximity::CFmmTree< EmptyStatistic<Matrix> > *> > nodes_in_each_level;
  proximity::CFmmTree<EmptyStatistic<Matrix> > *root;
  root = proximity::MakeCFmmTree
         (matrices, targets, leaflen, min_required_ws_index, 2,
          &nodes_in_each_level, &old_from_new);

  fx_timer_stop(NULL, "cfmm_tree_build");

  for (index_t i = 0; i < nodes_in_each_level.size(); i++) {
    for (index_t j = 0; j < nodes_in_each_level[i].size(); j++) {
      printf("%u ", (nodes_in_each_level[i][j])->node_index());
    }
    printf("\n");
  }

  printf("Finished constructing the tree...\n");

  // Print the tree.
  root->Print();

  // Clean up the memory used by the tree...
  delete root;

  fx_done(fx_root);
  return 0;
}

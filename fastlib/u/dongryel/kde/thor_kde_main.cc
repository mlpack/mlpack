#include "fastlib/fastlib_int.h"
#include "thor/thor.h"
#include "thor_kde.h"
#include "u/dongryel/series_expansion/kernel_aux.h"

typedef ThorKde<GaussianKernel, GaussianKernelAux> GThorKde;

int main(int argc, char *argv[]) {
  fx_init(argc, argv);
  GThorKde thor_kde;

  thor::RpcDualTree<GThorKde, DualTreeRecursiveBreadth<GThorKde> >
    (fx_submodule(fx_root, "gnp", "gnp"), thor_kde.GNP_CHANNEL,
     thor_kde.parameters_, thor_kde.q_tree_, thor_kde.r_tree_, 
     &thor_kde.q_results_, &thor_kde.global_result_);

  fx_done();
  return 0;
}

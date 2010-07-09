#include <iostream>
#include <fastlib/fastlib.h>

#include "FunctionTemplate.h"
#include "NelderMead.h"

using namespace std;
using namespace optim;

const fx_entry_doc optimization_entries[] = {

  {"seed", FX_PARAM, FX_INT, NULL,
   "Number of seeds (default 10).\n"},
  FX_ENTRY_DOC_DONE
};

const fx_submodule_doc optimization_submodules[] = {
  FX_SUBMODULE_DOC_DONE
};

const fx_module_doc optimization_doc = {
  optimization_entries, optimization_submodules,
  "This is a program testing optimization codes.\n"
};

void testNelderMead(fx_module* module) {
  cout << "Nelder-Mead test ..." << endl;

  int n_seed = fx_param_int(module, "seed", 10);
  ArrayList<Vector> vX;
  vX.Init();
  for (int i = 0; i < n_seed; i++) {
    Vector x; x.Init(2);
    x[0] = 10+10*(double)rand()/RAND_MAX;
    x[1] = 20+10*(double)rand()/RAND_MAX;
    vX.PushBackCopy(x);
  }
  LengthEuclidianSquare f;

  NelderMead<Vector, optim::LengthEuclidianSquare> algo(f);

  algo.addSeed(vX);

  Vector sol;
  sol.Init(2);
  double v = algo.optimize(sol);

  cout << "Best value = " << v << endl;
  ot::Print(sol, "Solution", stdout);

  cout << "Nelder-Mead test succeeded." << endl;
}

int main(int argc, char** argv) {
  fx_module* root = fx_init(argc, argv, &optimization_doc);
  cout << "Optimization tests" << endl;

  testNelderMead(root);

  fx_done(root);
  return 0;
}

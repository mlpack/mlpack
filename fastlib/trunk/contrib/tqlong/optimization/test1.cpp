#include <iostream>
#include <fastlib/fastlib.h>

#include "FunctionTemplate.h"
#include "NelderMead.h"
#include "GradientDescent.h"
#include "BFGS.h"
#include "LBFGS.h"

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
  LengthEuclidianSquare f(2);
  NelderMead<optim::LengthEuclidianSquare> algo(f);

  // Seeding
  ArrayList<Vector> vX;
  vX.Init();
  for (int i = 0; i < n_seed; i++) {
    Vector x;
    f.Init(&x);
    x[0] = 10+10*(double)rand()/RAND_MAX;
    x[1] = 20+10*(double)rand()/RAND_MAX;
    vX.PushBackCopy(x);
  }
  algo.addSeed(vX);

  // Optimization
  Vector sol;
  sol.Init(2);
  double v = algo.optimize(sol);

  cout << "Best value = " << v << endl;
  ot::Print(sol, "Solution", stdout);

  cout << "Nelder-Mead test succeeded." << endl;
}

void testGradientDescent(fx_module* module) {
  cout << "GradientDescent test ..." << endl;

  double param[] = {100, 0.00001, 0.001, 1e-4, 0.9, 0.4};
  LengthEuclidianSquare f(2);
  GradientDescent<optim::LengthEuclidianSquare> algo(f, param);

  // Seeding
  Vector x0;
  f.Init(&x0);
  x0[0] = 10+10*(double)rand()/RAND_MAX;
  x0[1] = 20+10*(double)rand()/RAND_MAX;

  // Optimization
  Vector sol;
  f.Init(&sol);
  algo.setX0(x0);
  double v = algo.optimize(sol);

  cout << "Best value = " << v << endl;
  ot::Print(sol, "Solution", stdout);
  //algo.printHistory();

  cout << "GradientDescent test succeeded." << endl;
}

void testBFGS(fx_module* module) {
  cout << "BFGS test ..." << endl;

  double param[] = {100, 0.00001, 0.001, 1e-4, 0.9, 0.4};
  TestFunction f;
  BFGS<optim::TestFunction> algo(f, param);

  // Seeding
  Vector x0;
  f.Init(&x0);
  x0[0] = 1;
  x0[1] = 1;

  // Optimization
  Vector sol;
  f.Init(&sol);
  algo.setX0(x0);
  double v = algo.optimize(sol);

  cout << "Best value = " << v << endl;
  ot::Print(sol, "Solution", stdout);
  //algo.printHistory();

  cout << "BFGS test succeeded." << endl;
}

void testLBFGS(fx_module* module) {
  cout << "LBFGS test ..." << endl;

  double param[7] = {100, 0.00001, 0.001, 1e-4, 0.9, 0.4, 20};
  TestFunction f;
  LBFGS<optim::TestFunction> algo(f, param);

  // Seeding
  Vector x0;
  f.Init(&x0);
  x0[0] = 1;
  x0[1] = 1;

  // Optimization
  Vector sol;
  f.Init(&sol);
  algo.setX0(x0);
  double v = algo.optimize(sol);

  cout << "Best value = " << v << endl;
  ot::Print(sol, "Solution", stdout);
  //algo.printHistory();

  cout << "LBFGS test succeeded." << endl;
}

int main(int argc, char** argv) {
  fx_module* root = fx_init(argc, argv, &optimization_doc);
  cout << "Optimization tests" << endl;

  //testNelderMead(root);
  //testGradientDescent(root);
  //testBFGS(root);
  testLBFGS(root);

  fx_done(root);
  return 0;
}

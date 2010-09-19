#include <fastlib/fastlib.h>
#include "anmf.h"

#include <iostream>

using namespace std;

const fx_entry_doc anmf_test_entries[] = {
//   {"method", FX_PARAM, FX_STR, NULL,
//    "  Inference method: naive (*), sum_product, msg_priority, msg_pending.\n"},
   {"iter", FX_PARAM, FX_INT, NULL,
    "  Maximum number of iterations: default 10.\n"},
   {"ctol", FX_PARAM, FX_DOUBLE, NULL,
    "  Change tolerance: default 1e-12.\n"},
  {"M", FX_PARAM, FX_STR, NULL,
   "The matching measure matrix M_{ia;jb}, default: m.txt"},
   {"size", FX_PARAM, FX_INT, NULL,
    "  size of the graph: default 5.\n"},
  FX_ENTRY_DOC_DONE
};

const fx_submodule_doc anmf_test_submodules[] = {
  FX_SUBMODULE_DOC_DONE
};

const fx_module_doc anmf_test_doc = {
  anmf_test_entries, anmf_test_submodules,
  "This is a program testing affine NMF functionalities.\n"
};

void testMaxWeightMatching(fx_module*);
void testGraphMatching(fx_module*);
void randomData(fx_module* module);

int main(int argc, char** argv)
{
  fx_module* root = fx_init(argc, argv, &anmf_test_doc);
  testMaxWeightMatching(root);
//  if (fx_param_exists(root, "size"))
//    randomData(root);
//  testGraphMatching(root);
  fx_done(root);
}

void testGraphMatching(fx_module* module)
{
  const char* inputFile = fx_param_str(module, "M", "m.txt");
  fx_param_int(module, "iter", 10);
  fx_param_double(module, "ctol", 1e-12);
  Matrix M;
  if (data::Load(inputFile, &M) != SUCCESS_PASS) 
    return;
  Vector sol;
  anmf::ipfpGraphMatching(module, M, sol);
  ot::Print(sol, "solution");
}

void randomData(fx_module* module)
{
  const char* filename = fx_param_str(module, "M", "m.txt");
  int n = fx_param_int(module, "size", 5);
  int n2 = n*n;
  Matrix M;
  M.Init(n2, n2);
  for (int i = 0; i < n2; i++)
  {
    for (int j = i; j < n2; j++)
    {
      M.ref(i,j) = (math::Random() * 10);
      M.ref(j,i) = M.get(i,j);
    }
  }
  data::Save(filename, M);
}

void testMaxWeightMatching(fx_module* module)
{
  const char* inputFile = fx_param_str(module, "M", "m.txt");
  Matrix M;
  if (data::Load(inputFile, &M) != SUCCESS_PASS)
    return;

  anmf::MaxWeightMatching<Matrix> a(M);

  cout << "Left-->Right" << endl;
  for (int i = 0; i < M.n_rows(); i++)
    cout << i << " --> " << a.matchLeft(i) << endl;

  cout << "Right-->Left" << endl;
  for (int i = 0; i < M.n_cols(); i++)
    cout << i << " --> " << a.matchRight(i) << endl;
}

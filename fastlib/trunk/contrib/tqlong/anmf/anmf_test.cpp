#include <fastlib/fastlib.h>
#include "anmf.h"

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
  FX_ENTRY_DOC_DONE
};

const fx_submodule_doc anmf_test_submodules[] = {
  FX_SUBMODULE_DOC_DONE
};

const fx_module_doc anmf_test_doc = {
  anmf_test_entries, anmf_test_submodules,
  "This is a program testing affine NMF functionalities.\n"
};

void testGraphMatching(fx_module*);

int main(int argc, char** argv)
{
  fx_module* root = fx_init(argc, argv, &anmf_test_doc);
  testGraphMatching(root);
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

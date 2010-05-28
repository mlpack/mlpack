
#include <fastlib/fastlib.h>
#include "affineNMF.h"

const fx_entry_doc anmf_entries[] = {
  {"i1", FX_PARAM, FX_STR, NULL,
   "  HMM type : discrete | gaussian | mixture.\n"},
  {"i2", FX_PARAM, FX_STR, NULL,
   "  A file containing HMM transition.\n"},
  /*
  {"fileE", FX_REQUIRED, FX_STR, NULL,
   "  A file containing HMM emission.\n"},
  {"length", FX_PARAM, FX_INT, NULL,
   "  Sequence length, default = 10.\n"},
  {"lenmax", FX_PARAM, FX_INT, NULL,
   "  Maximum sequence length, default = length\n"},
  {"numseq", FX_PARAM, FX_INT, NULL,
   "  Number of sequance, default = 10.\n"},
  {"fileSEQ", FX_PARAM, FX_STR, NULL,
   "  Output file for the generated sequences.\n"},
  */
  //{"statefile", FX_PARAM, FX_STR, NULL,
  // "  Output file for the generated state sequences.\n"},
  FX_ENTRY_DOC_DONE
};

const fx_submodule_doc anmf_submodules[] = {
  FX_SUBMODULE_DOC_DONE
};

const fx_module_doc anmf_doc = {
  anmf_entries, anmf_submodules,
  "This is a program generating sequences from HMM models.\n"
};

int main(int argc, char* argv[]) {
  fx_module* root = fx_init(argc, argv, &anmf_doc);

  Matrix I1, I2;
  data::Load(fx_param_str(root, "i1", "i1"), &I1);
  data::Load(fx_param_str(root, "i2", "i2"), &I2);

  Vector m;
  projective_register(I1, I2, &m);

  ot::Print(m);
  fx_done(root);
  return 0;
}

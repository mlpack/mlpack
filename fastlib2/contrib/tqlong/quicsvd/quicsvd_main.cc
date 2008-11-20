#include <fastlib/fastlib.h>
#include "quicsvd.h"

const fx_entry_doc quicsvd_main_entries[] = {
  {"A_in", FX_REQUIRED, FX_STR, NULL,
   " File consists of matrix A to be decomposed A = U S VT. \n"},
  {"relErr", FX_PARAM, FX_DOUBLE, NULL,
   " Target relative error |A-A'|/|A|, default = 0.1.\n"},
  {"U_out", FX_PARAM, FX_STR, NULL,
   " File to hold matrix U.\n"},
  {"s_out", FX_PARAM, FX_STR, NULL,
   " File to hold the singular values vector s.\n"},
  {"VT_out", FX_PARAM, FX_STR, NULL,
   " File to hold matrix VT (V transposed).\n"},
  FX_ENTRY_DOC_DONE
};

const fx_submodule_doc quicsvd_main_submodules[] = {
  FX_SUBMODULE_DOC_DONE
};

const fx_module_doc quicsvd_main_doc = {
  quicsvd_main_entries, quicsvd_main_submodules,
  "This is a program calculating an approximated Singular "
  "Value Decomposition using QUIC-SVD method.\n"
};

int main(int argc, char* argv[]) {
  fx_init(argc, argv, &quicsvd_main_doc);

  Matrix A, U, VT;
  Vector s;

  // parse input file to get matrix A
  const char* A_in = fx_param_str(NULL, "A_in", NULL);
  data::Load(A_in, &A);

  // parse target relative error, default = 0.1
  const double targetRelErr = fx_param_double(NULL, "relErr", 0.1);

  // call the QUIC-SVD method
  QuicSVD::SVDInit(A, targetRelErr, &s, &U, &VT);
  
  if (fx_param_exists(NULL, "U_out"))
    data::Save(fx_param_str(NULL, "U_out", NULL), U);
  else // use OT to write to standard output
    ot::Print(U, "U", stdout);

  if (fx_param_exists(NULL, "s_out")) {
    Matrix S;
    S.AliasColVector(s);
    data::Save(fx_param_str(NULL, "s_out", NULL), S);
  }
  else // use OT to write to standard output
    ot::Print(s, "s", stdout);

  if (fx_param_exists(NULL, "VT_out"))
    data::Save(fx_param_str(NULL, "VT_out", NULL), VT);
  else // use OT to write to standard output
    ot::Print(VT, "VT", stdout);

  fx_done(NULL);
}

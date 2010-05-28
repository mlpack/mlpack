
#include <fastlib/fastlib.h>
#include "affineNMF.h"

const fx_entry_doc anmf_entries[] = {
  {"i1", FX_PARAM, FX_STR, NULL,
   "  input file 1.\n"}, 
  {"i2", FX_PARAM, FX_STR, NULL,
   "  input file 2.\n"},
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

void InitRandom01(index_t n_rows, index_t n_cols, Matrix* A_) {
  Matrix& A = *A_;
  A.Init(n_rows, n_cols);
  for (index_t i = 0; i < n_rows; i++) 
    for (index_t j = 0; j < n_cols; j++)
      A.ref(i, j) = math::Random(0.1,1.0000);
}

void nmf_run(const Matrix& V, index_t rank,
	     Matrix* W_, Matrix* H_) {
  Matrix Winit, Hinit;
  InitRandom01(V.n_rows(), rank, &Winit);
  InitRandom01(rank, V.n_cols(), &Hinit);
  
  nmf(V, Winit, Hinit, 10, W_, H_);
}

int main(int argc, char* argv[]) {
  fx_module* root = fx_init(argc, argv, &anmf_doc);

  const char* f1 = fx_param_str(root, "i1", "i1");
  const char* f2 = fx_param_str(root, "i2", "i2");

  // Test registration 
  Matrix I1, I2;
  data::Load(f1, &I1);
  data::Load(f2, &I2);
  Vector m;
  projective_register(I1, I2, &m);
  ot::Print(m);

  /* Test nmf */
  Matrix V;

  V.Init(400,5);
  for (int i = 0; i < 5; i++) {
    Matrix X;
    char fn[100];
    sprintf(fn, "im%d", i+1);
    data::Load(fn, &X);
    for (int j = 0; j < 400; j++) {
      V.ref(j,i) = X.get(j%20, j/20)/255;
    }
  }

  //data::Load("V", &V);
  prepare_for_nmf(V);
  printf("size(V) = %d x %d", V.n_rows(), V.n_cols());
  Matrix W, H;
  nmf_run(V, 2, &W, &H);
  data::Save("basis", W);
  printf("size(W) = %d x %d", W.n_rows(), W.n_cols());
  data::Save("weight", H);
  printf("size(H) = %d x %d", H.n_rows(), H.n_cols());
  fx_done(root);
  return 0;
}

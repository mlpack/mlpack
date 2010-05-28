
#include <fastlib/fastlib.h>
#include "nmf.h"

void nmf(const Matrix& V, const Matrix& Winit, const Matrix& Hinit, 
	 index_t maxiter, Matrix* W_, Matrix* H_) {
  Matrix& W = *W_;
  Matrix& H = *H_;

  W.Copy(Winit);
  H.Copy(Hinit);
  
  for (index_t iter = 0; iter < maxiter; iter++) {
    //ot::Print(H);
    // update H
    Matrix WtV, WtW, WtWH;
    la::MulTransAInit(W, V, &WtV);
    la::MulTransAInit(W, W, &WtW);
    la::MulInit(WtW, H, &WtWH);
    //ot::Print(WtWH, "WtWH");
    //ot::Print(WtV, "WtV");
    for (index_t i = 0; i < H.n_rows(); i++)
      for (index_t j = 0; j < H.n_cols(); j++) {
	if (WtWH.get(i,j) < 1e-10)
	  printf("WtWh bad");
	H.ref(i, j) *= WtV.get(i, j) / WtWH.get(i, j);
      }

    // update W
    Matrix VHt, HHt, WHHt;
    la::MulTransBInit(V, H, &VHt);
    la::MulTransBInit(H, H, &HHt);
    la::MulInit(W, HHt, &WHHt);
    for (index_t i = 0; i < W.n_rows(); i++)
      for (index_t j = 0; j < W.n_cols(); j++) {
	if (WHHt.get(i, j) < 1e-10) 
	  printf("WHHt bad.");
	W.ref(i, j) *= VHt.get(i, j) / WHHt.get(i, j);
      }
  }
}


void prepare_for_nmf(Matrix& V) {
  for (index_t i = 0; i < V.n_rows(); i++)
    for (index_t j = 0; j < V.n_cols(); j++)
      if (V.get(i, j) < 1e-8) V.ref(i, j) = 1e-4;
}

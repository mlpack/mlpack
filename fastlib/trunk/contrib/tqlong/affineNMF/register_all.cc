#include <fastlib/fastlib.h>
#include "image_register.h"

void CalculateSum(const Vector& w, const ArrayList<ImageType>& B, ImageType& WiB) {
  DEBUG_ASSERT(w.length() == B.size());
  for (index_t j = 0; j < w.length(); j++)
    WiB.Add(B[j], w[j]);
}

void CalculateTransformBasis(const ArrayList<ImageType>& B, const Transformation& T, 
			     ArrayList<ImageType>& BT) {
  for (index_t j = 0; j < B.size(); j++) 
    B[j].Transform(BT[j], T);
}

void register_all(const ArrayList<ImageType>& X, ArrayList<Transformation>& T, 
		  ArrayList<Vector>& W, ArrayList<ImageType>& B) {
  DEBUG_ASSERT(T.size() == X.size() && W.size() == X.size());
  int maxIter = fx_param_int(NULL, "maxIter", 100);
  ArrayList<ImageType> BT;
  BT.InitCopy(B);
  for (index_t iter = 0; iter < maxIter; iter++) {
    // register transformations
    for (index_t iT = 0; iT < T.size(); iT++) {
      ImageType WiB;
      CalculateSum(W[iT], B, WiB);
      register_transform(X[iT], WiB, T[iT]);
    }

    // register weights
    for (index_t iW = 0; iW < W.size(); iW++) {
      CalculateTransformBasis(B, T[iW], BT);
      register_weights(X[iW], BT, W[iW]);
    }

    // register basis 
    register_basis(X, T, W, B);
    //Save(stdout, "B", B); fflush(stdin); getchar();
  }
}

void CalculateRecovery(const ArrayList<Transformation>& T, 
		       const ArrayList<Vector>& W, 
		       const ArrayList<ImageType>& B, 
		       ArrayList<ImageType>& XRecover) {
  DEBUG_ASSERT(T.size() == W.size());
  XRecover.Init();
  for (index_t i = 0; i < T.size(); i++) {
    ImageType S, Xi;
    for (index_t j = 0; j < B.size(); j++) S.Add(B[j], W[i][j]);
    S.Transform(Xi, T[i]);
    XRecover.PushBackCopy(Xi);
  }
}

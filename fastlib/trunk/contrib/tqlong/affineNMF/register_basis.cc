#include <fastlib/fastlib.h>
#include "image_register.h"

void SetImageZero(ArrayList<ImageType>& X) {
  for (index_t i = 0; i < X.size(); i++)
    for (index_t j = 0; j < X[i].n_points(); j++)
      X[i].pList[j].setValue(0.0, 0.0, 0.0);
}

void ComputeContribution(const PointType& pXi, const Transformation& Ti,
			 double wij, const PointType& pBj, PointType& p) {
  PointType wTpj = pBj.Transform(Ti, wij);
  double dr, dc, df;
  d_exp_kernel(pXi, wTpj, dr, dc, df);
  p.setValue(dr*Ti.m[0] + dc*Ti.m[3], dr*Ti.m[1] + dc*Ti.m[4], df*wij); 
}

void register_basis(const ArrayList<ImageType>& X, 
		    const ArrayList<Transformation>& T, const ArrayList<Vector>& W,
		    const ArrayList<ImageType>& BInit, ArrayList<ImageType>& BOut) {
  DEBUG_ASSERT(T.size() == X.size() && W.size() == X.size());
  BOut.InitCopy(BInit);
  ArrayList<ImageType> dB;
  dB.InitCopy(BOut);

  index_t maxIter = 10;
  double lambda = 0.1;
  for (index_t iter = 0; iter < maxIter; iter++) {
    SetImageZero(dB);
    for (index_t iX = 0; iX < X.size(); iX++)
      for (index_t pX = 0; pX < X[iX].n_points(); pX++)
	for (index_t iB = 0; iB < BOut.size(); iB++)
	  for (index_t pB = 0; pB < BOut[iB].n_points(); pB++) {
	    PointType p;
	    ComputeContribution(X[iX].pList[pX], T[iX], W[iX][iB], 
				BOut[iB].pList[pB], p);
	    dB[iB].pList[pB].AddNumeric(p);
	  }	    
    
    for (index_t iB = 0; iB < BOut.size(); iB++)
      for (index_t pB = 0; pB < BOut[iB].n_points(); pB++) {  
	BOut[iB].pList[pB].AddNumeric(dB[iB].pList[pB], -lambda);
      }
  }
}

void register_basis(const ArrayList<ImageType>& X, 
		    const ArrayList<Transformation>& T, const ArrayList<Vector>& W,
		    ArrayList<ImageType>& BOut) {
  DEBUG_ASSERT(T.size() == X.size() && W.size() == X.size());
  ArrayList<ImageType> dB;
  dB.InitCopy(BOut);

  index_t maxIter = 10;
  double lambda = 0.1;
  for (index_t iter = 0; iter < maxIter; iter++) {
    SetImageZero(dB);
    for (index_t iX = 0; iX < X.size(); iX++)
      for (index_t pX = 0; pX < X[iX].n_points(); pX++)
	for (index_t iB = 0; iB < BOut.size(); iB++)
	  for (index_t pB = 0; pB < BOut[iB].n_points(); pB++) {
	    PointType p;
	    ComputeContribution(X[iX].pList[pX], T[iX], W[iX][iB], 
				BOut[iB].pList[pB], p);
	    dB[iB].pList[pB].AddNumeric(p);
	  }	    
    
    for (index_t iB = 0; iB < BOut.size(); iB++)
      for (index_t pB = 0; pB < BOut[iB].n_points(); pB++) {  
	BOut[iB].pList[pB].AddNumeric(dB[iB].pList[pB], -lambda);
      }
  }  
}

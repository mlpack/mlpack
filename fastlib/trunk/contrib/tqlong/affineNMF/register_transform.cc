
#include <fastlib/fastlib.h>
#include "image_register.h"
#include "image_type.h"

void ComputeContribution(const ImageType& I1, const ImageType& I2, 
			 const ImageType& tmp, index_t i, index_t j, 
			 Vector& dt) {
  double dr, dc, df;
  d_exp_kernel(I1.pList[i], tmp.pList[j], dr, dc, df);
  dt[0] += dr * I2.pList[i].r;
  dt[1] += dr * I2.pList[i].c;
  dt[2] += dr;
  dt[3] += dc * I2.pList[i].r;
  dt[4] += dc * I2.pList[i].c;
  dt[5] += dc;
  dt[6] += 0;
  dt[7] += 0;
}

void register_transform(const ImageType& I1, const ImageType& I2,
			const Transformation& tInit, Transformation& tOut) {
  tOut = tInit;
  index_t maxIter = 1;
  double lambda = 0.1;
  for (index_t iter = 0; iter < maxIter; iter++) {
    ImageType tmp;
    Vector dt;
    dt.Init(8); dt.SetZero();
    I2.Transform(tmp, tOut);
    for (index_t i = 0; i < I1.n_points(); i++)
      for (index_t j = 0; j < I2.n_points(); j++)
	ComputeContribution(I1, I2, tmp, i, j, dt);
    la::AddExpert(-lambda, dt, &tOut.m);
  }
}

void register_transform(const ImageType& I1, const ImageType& I2,
			Transformation& tOut) {
  index_t maxIter = 10;
  double lambda = 0.1;
  for (index_t iter = 0; iter < maxIter; iter++) {
    ImageType tmp;
    Vector dt;
    dt.Init(8); dt.SetZero();
    I2.Transform(tmp, tOut);
    for (index_t i = 0; i < I1.n_points(); i++)
      for (index_t j = 0; j < I2.n_points(); j++)
	ComputeContribution(I1, I2, tmp, i, j, dt);
    la::AddExpert(-lambda, dt, &tOut.m);
  }
}

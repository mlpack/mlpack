#include "fastlib/fastlib.h"
#include<stdlib.h>
#include<time.h>

const int n = 100;
const int d = 20;
const int k = 5;
const int M = 3;

const double pData = .5;
const double pProject = .1;




void generateData(Matrix &data) {
 
 
  data.Init(d,n);
  for(index_t i = 0; i < d; i++) {
    for(index_t j = 0; j < n; j++) {
      data.set(i, j, drand48() < pData);
    }
  }


 
}


void generateProjectionMatrix(Matrix &projectionMatrix) {
  projectionMatrix.Init(k, d);
  
  for(index_t i = 0; i < k; i++) {
    for(index_t j = 0; j < d; j++) {
      projectionMatrix.set(i, j, drand48() < pProject);
    }
  }
}




Vector projectPoint(Matrix projectionMatrix, Vector point) {
  	     
  Vector projectedPoint;
  la::MulInit(projectionMatrix, point, &projectedPoint);

  // this could be sped up by performing the modulus during the multiplication
  for(index_t i = 0; i < k; i++) {
    projectedPoint[i] = ((int)(projectedPoint[i])) % 2;
  }
  
  return projectedPoint;
}



int main(int argc, char *argv[]) {
  fx_init(argc, argv);

  srand48(time(NULL));


  Matrix data;
  generateData(data);

  Vector point;
  data.MakeColumnVector(0, &point);


    
  
  Matrix projectionMatrices[d][M];
  for(index_t i = 0; i < d; i++) {
    for(index_t j = 0; j < M; j++) {
      generateProjectionMatrix(projectionMatrices[i][j]);
    }
  }



  // project each data point to a cube
  


  point.PrintDebug("point", stdout);

  Vector projectedPoint = projectPoint(projectionMatrices[0][0], point);

  projectedPoint.PrintDebug("projectedPoint", stdout);



  Vector cube;
  cube.Init((int)pow(2,k));
  




}

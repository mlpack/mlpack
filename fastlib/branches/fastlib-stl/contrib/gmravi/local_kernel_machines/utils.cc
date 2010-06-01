
#include "fastlib/fastlib.h"
#include "utils.h"
void RemoveLastRowFromMatrixInit( Matrix &mat,Vector &last_row){
  
  Matrix mat_trans;
  Matrix dest;

  la::TransposeInit(mat,&mat_trans);
  mat_trans.MakeColumnSlice(0,mat_trans.n_cols()-1,&dest);
  

  //Collect the elements of the last column
  last_row.Init(mat_trans.n_rows());
  last_row.CopyValues(mat_trans.GetColumnPtr(mat_trans.n_cols()-1));
  // The matrix dest has the last column removed. Now transpose this
  // matrix and copy it back to mat

  // Destruct mat and copy dest transposed back into mat

  mat.Destruct();
  la::TransposeInit(dest,&mat);

}

void AppendVectorToMatrixAsLastRow(Vector &vec,Matrix &mat,Matrix &stiched){
  
  // First tranpose this matrix

  
  Matrix mat_trans;
  la::TransposeInit(mat,&mat_trans);


  // Now append the vector as the last column.
  
  Matrix temp;
  temp.Init(mat_trans.n_rows(),mat_trans.n_cols()+1);
  temp.CopyColumnFromMat(0,0,mat_trans.n_cols(),mat_trans);
  
  
  // Now populate the last column
  
  
  for(index_t i=0;i<mat_trans.n_rows();i++){
    temp.set(i,mat_trans.n_cols(),vec[i]);
  }
  
  //Finally tranpose temp and store it in stiched
  la::TransposeInit(temp,&stiched);

}

void ConvertOneColumnMatrixToVector(Matrix &mat, Vector &vec){
  
  vec.Alias(mat.GetColumnPtr(0),mat.n_cols());
}

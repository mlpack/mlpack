#include "fastlib/fastlib.h"

// This routine simply appends a vector to the last column of a data matrix.
// Since in fastlibs data matrices are transposed, hence we shall append it 
// to the last row

// TODO: Improve this function, where you 
void AppendVectorToMatrix(Vector &vec, Matrix &mat, Matrix &final_mat){
  
 
  // First transpose mat

  Matrix mat_trans;
  la::TransposeInit(mat,&mat_trans);
  
  index_t num_rows=mat_trans.n_rows();
  index_t num_cols=mat_trans.n_cols();

  printf("num_rows=%d...\n",num_rows);
  printf("num_cols=%d...\n",num_cols);
  
  printf("Original mat trans is...\n");
  mat_trans.PrintDebug();

  Matrix final_mat_trans;
  final_mat_trans.Copy(mat_trans);

  printf("Final mat trans is..\n");
  final_mat_trans.PrintDebug();
  
  // Resize final_mat_trans

  final_mat_trans.ResizeNoalias(num_cols+1);
  printf("Resized...\n"); 

  printf("vec is...\n");
  vec.PrintDebug();
 
  for(index_t row=0;row<num_rows;row++){

    final_mat_trans.set(row,num_cols,vec[row]);
    printf("Adding..\n");
  }
  
  printf("Addded the rows..\n");
  // Now finally tranpose it to make it usable for fastlab.

  la::TransposeInit(final_mat_trans,&final_mat);

  final_mat.PrintDebug();

}

void ConvertMatrixToVector(Matrix &mat, Vector &vec){
  
  index_t length=mat.n_cols();
  vec.Alias(mat.GetColumnPtr(0),length);
}

void RemoveLastRowFromMatrixInit_(Matrix &mat, Vector &vec){

  index_t num_cols=mat.n_cols();
  index_t num_rows=mat.n_rows();

  vec.Init(num_cols);

  

  for(index_t i=0;i<num_cols;i++){

    vec[i]=mat.get(num_rows-1,i);
  }

  // Now strip of the last row
  
  Matrix mat_trans;
  la::TransposeInit(mat,&mat_trans);

  
  // Now kick the last column. 
  // Remember the num_cols of mat_trans=num_rows of mat
  mat_trans.ResizeNoalias(num_rows-1);
  
  

  // Copy it back to mat. Note mat will have 1 row less.
  // However before I can do this I will have to delete mat
  mat.Destruct();
  la::TransposeInit(mat_trans,&mat);
}

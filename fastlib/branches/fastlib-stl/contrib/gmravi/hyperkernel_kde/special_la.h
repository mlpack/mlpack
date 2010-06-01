#ifndef SPECIAL_LA_H
#define SPECIAL_LA_H
#define SUCCESS 1
#define FAILURE -1
#define EPSILON pow(10,-6)
#include "fastlib/fastlib.h"
#include "special_la.h"
#include "mlpack/fastica/lin_alg.h"

//This is a class of special linear algebra routines that are not
//provided by lapack

class special_la{

 public:  
  
  //This function appends the matrix m1 to m2
  
  static index_t AppendMatrixInit(Matrix &m1, Matrix &m2, Matrix *m3){
    
    
    index_t num_rows1=m1.n_rows();
    index_t num_rows2=m2.n_rows();
    
    index_t num_cols1=m1.n_cols();
    index_t num_cols2=m2.n_cols();

    index_t total_num_cols=
      num_cols1+num_cols2;

    if(num_rows1!=num_rows2){

      return FAILURE;

    }
    m3->Init(num_rows1,total_num_cols);

    for(index_t i=0;i<num_rows1;i++){

      index_t j;
      for(j=0;j<num_cols1;j++){
	
	m3->set(i,j,m1.get(i,j));

      }

      for(index_t k=0;k<num_cols2;k++){

	m3->set(i,j+k,m2.get(i,k));
      }

    }

    return SUCCESS;
   }
   
   static void AppendMatrixWithZerosInit(Matrix &m1, 
					 index_t num_zeros, 
					 Matrix *m3){
     
     
     index_t num_rows1=m1.n_rows();
          
     index_t num_cols1=m1.n_cols();

     index_t total_num_cols=
       num_cols1+num_zeros;
     
     m3->Init(num_rows1,total_num_cols);
     
     for(index_t i=0;i<num_rows1;i++){
       
       index_t j;
       for(j=0;j<num_cols1;j++){
	 
	 m3->set(i,j,m1.get(i,j));
	 
       }

       for(index_t k=0;k<num_zeros;k++){
	 
	 m3->set(i,j+k,0);
       }
       
     }
   }

   static index_t StackMatrixMatrixInit(Matrix m1,Matrix m2,Matrix *m3){

     printf("Will stack matrices...\n");

    
     index_t num_cols1=m1.n_cols();
     index_t num_cols2=m2.n_cols();

     if(num_cols1!=num_cols2){

       printf("Coluns1=%d and Columns2=%d...\n",num_cols1,num_cols2);

       return FAILURE;
     }

     index_t num_rows1=m1.n_rows();
     index_t num_rows2=m2.n_rows();

     m3->Init(num_rows1+num_rows2,num_cols1);

     index_t i=0;
     for(i=0;i<num_rows1;i++){

       for(index_t j=0;j<num_cols1;j++){

	 m3->set(i,j,m1.get(i,j));
       }
     }
     for(index_t k=0;k<num_rows2;k++){
       for(index_t j=0;j<num_cols2;j++){
	 
	 m3->set(i+k,j,m2.get(k,j));
       }
     }
     return SUCCESS;
   }

 static index_t StackVectorVectorInit(Vector v1,Vector v2,Matrix *m3){

    
     index_t num_cols1=v1.length();
     index_t num_cols2=v2.length();

     if(num_cols1!=num_cols2){
       
       return FAILURE;
     }

     //Initialize m3
     m3->Init(2,num_cols1);
     for(index_t j=0;j<num_cols1;j++){
       
       m3->set(0,j,v1[j]);
     }
     
     for(index_t k=0;k<num_cols1;k++){
       
       m3->set(1,k,v2[k]);
     }
     return SUCCESS;
 }
 
 static index_t StackMatrixVectorInit(Matrix m1,Vector v2,Matrix *m3){
   
   printf("Came to matrix vector stacking...\n");
   index_t num_cols1=m1.n_cols();
   index_t num_cols2=v2.length();
   
   if(num_cols1!=num_cols2){
     
     printf("Cols1=%d and cols2=%d...\n",num_cols1,num_cols2);
     
     return FAILURE;
   }
   
   printf("Passed the test...\n");
   index_t num_rows1=m1.n_rows();
   
   m3->Init(num_rows1+1,num_cols1);
   
   index_t i=0;
   for(i=0;i<num_rows1;i++){
     
     for(index_t j=0;j<num_cols1;j++){
       
       m3->set(i,j,m1.get(i,j));
     }
   }

   for(index_t j=0;j<num_cols2;j++){
     
     m3->set(i,j,v2[j]);
   }
   return SUCCESS;
 }

 static index_t StackVectorMatrixInit(Vector v1,Matrix m1,Matrix *m2){
   
   
   index_t num_cols1=m1.n_cols();
   index_t num_cols2=v1.length();
   
   if(num_cols1!=num_cols2){
     
     printf("Cols1=%d and cols2=%d...\n",num_cols1,num_cols2);
     
     return FAILURE;
   }
   
   index_t num_rows1=m1.n_rows();
   m2->Init(num_rows1+1,num_cols1);
  

   index_t i=0;
   for(index_t j=0;j<num_cols2;j++){
     
     m2->set(i,j,v1[j]);
   }
   for(index_t i=1;i<=num_rows1;i++){
     
     for(index_t j=0;j<num_cols1;j++){
       
       m2->set(i,j,m1.get(i-1,j));
     }
   }

   return SUCCESS;
 }

 // Low complexity Post multiplication of a matrix with a diagonal matrix

 static void PostMultiplyDiagonalMatrix(Matrix &mat1, Vector diag_elem, 
				     Matrix *res){

   //Make a check to make sure that the multiplication is feasible
   
   index_t num_cols1=mat1.n_cols();

   index_t num_rows2=diag_elem.length();


   if(num_cols1!=num_rows2){

     printf("Matrices to be multiplied are of improper size...\n");
     return;
   }

   // printf("Will scale rows now...\n");

   //This multiplication is basically scaling up the columns of the
   //matrix with the diagonal elements of the diagonal matrix. This
   //can be seen as scaling up the rows of the transpose and then
   //transposing the resultant matrix back

   Matrix mat1_trans;
   la::TransposeInit(mat1,&mat1_trans);
   la::ScaleRows(diag_elem,&mat1_trans);

   //printf("Scaled rows...\n");
   //printf("mult1_trans is...\n");
   la::TransposeOverwrite(mat1_trans,res);
   //printf("Result has been set to..\n");
   //res->PrintDebug(); 

   //printf("Returning from the function...\n");
 }
 
 static void PostMultiplyDiagonalMatrixInit(Matrix &mat1, Vector diag_elem, 
				 Matrix *res){
   
   //Make a check to make sure that the multiplication is feasible
   
   index_t num_rows1=mat1.n_rows();
   index_t num_cols1=mat1.n_cols();
   
   index_t num_rows2=diag_elem.length();
  
   
   if(num_cols1!=num_rows2){
     
     printf("Matrices to be multiplied are of improper size...\n");
     return;
   }

   //Initialize the matrix
   res->Init(num_rows1,num_cols1);

   PostMultiplyDiagonalMatrix(mat1,diag_elem,res);
  
   //   res->PrintDebug();

 }

 
 static void PostMultiplyMatrixWithVectorGivenCholesky(Matrix &chol_factor,
						       ArrayList <index_t> 
						       &perm_mat,
						       Vector &beta,
						       Vector *res){
   //First make a compatibility check
   
   index_t length_beta=beta.length();
   index_t size_of_matrix=perm_mat.size();
   
   if(length_beta!=size_of_matrix){
     
     printf("Post Multiply with matrix: Sizes are incompatible..\n");
   }
   
   ArrayList <index_t> perm_mat_trans;

   //The tranpose of the permutation matrix is the same as its inverse
   PermutationMatrixTransposeInit(perm_mat,&perm_mat_trans);


   Vector temp1,temp2,temp3;
   PreMultiplyVectorWithPermutationMatrixInit(perm_mat_trans,beta,&temp1);//P^T\beta

   Matrix chol_factor_trans;

   la::TransposeInit(chol_factor,&chol_factor_trans);

   la::MulInit(chol_factor_trans,temp1,&temp2); //L^T(P^T\beta)
   
   la::MulInit(chol_factor,temp2,&temp3); //LL^TP^T\beta

   PreMultiplyVectorWithPermutationMatrixInit(perm_mat,temp3,res);

 }


 //This function does M\beta where M=PLL^TP^T

 static void PostMultiplyMatrixWithVectorGivenCholeskyInit(Matrix &chol_factor,
						       ArrayList <index_t> 
						       &perm_mat,
						       Vector &beta,
						       Vector *res){
   //First make a compatibility check

   index_t length_beta=beta.length();
   index_t size_of_matrix=perm_mat.size();

   if(length_beta!=size_of_matrix){

     printf("Post Multiply with matrix: Sizes are incompatible..\n");
   }

   PostMultiplyMatrixWithVectorGivenCholesky(chol_factor,
					     perm_mat,beta,
					     res);
 }

 //A low complexity operation which exploits the fact that one of the
 //matrix being multiplied is a lower triangle matrix.

 static void PreMultiplyLowerTriangMatrix(Matrix &lower_triang,
					 Matrix &mat,Matrix *res){
     printf("Came to this function\n");
  //$a_ij=\sum_k a_ik a_kj$

   index_t num_rows1=lower_triang.n_rows();
   index_t num_cols1=lower_triang.n_cols();

   index_t num_rows2=mat.n_rows();
   index_t num_cols2=mat.n_cols();

   printf("num_rows1=%d num_rows2=%d..\n",num_rows1,num_rows2);
   
   printf("num_cols1=%d num_cols2=%d..\n",num_cols1,num_cols2);
   for(index_t i=0;i<num_rows1;i++){
     
     for(index_t j=0;j<num_cols2;j++){
       
       double val=0;
       
       for(index_t k=0;k<=min(i,num_cols1-1);k++){
	 printf("i is %d and j is %d and k=%d.....\n",i,j,k);
	 val+=lower_triang.get(i,k)*mat.get(k,j);
       }
       
       //a_ij=val
     
       res->set(i,j,val);
     }
   }
 }

 static void PreMultiplyVectorWithDiagonalMatrix(Vector &vec, Vector &diag, 
						 Vector *res){

   index_t length_diag=diag.length();
   index_t length_vec=vec.length();
   //make a compatibility check

   if(length_vec!=length_diag){

     printf("PreMultiplyVectorWithDiagonalMatrix:Sizes not comaptible");
     return;
   }

   for(index_t i=0;i<length_vec;i++){

     (*res)[i]=diag[i]*vec[i];
   }
 }

 static void  PreMultiplyVectorWithDiagonalMatrixInit(Vector diag,Vector vec, 
						      Vector *res){
   
   //Make A compatibility check
   index_t length_diag=diag.length();
   index_t length_vec=vec.length();
   
   if(length_vec!=length_diag){

     printf("PreMultiplyVectorWithDiagonalMatrix:Sizes not comaptible");
     return;
   }
   
   //Initalize the vector now
   res->Init(length_vec);
   
   PreMultiplyVectorWithDiagonalMatrix(vec,diag,res);
 }



 // Assumes that the result matrix has not been initialized
 static void PreMultiplyLowerTriangMatrixInit(Matrix &lower_triang,
					     Matrix &mat,Matrix *res){
   
   index_t num_rows1=lower_triang.n_rows();
   index_t num_cols1=lower_triang.n_cols();

   index_t num_rows2=mat.n_rows();
   index_t num_cols2=mat.n_cols();

   if(num_cols1!=num_rows2){

     printf("Pre Multiply with Lower Triangular matrix cannot be done because of size mismatch...\n");
     return;
     
   }
   res->Init(num_rows1,num_cols2);
   PreMultiplyLowerTriangMatrix(lower_triang,mat,res);
 }



 static void SubtractFromDiagonalMatrix(Vector &diag, Matrix &mat,Matrix *res){


   index_t num_rows1=diag.length();

   index_t num_rows2=mat.n_rows();
   index_t num_cols2=mat.n_cols();

   if(num_rows1!=num_rows2 || num_rows2!=num_cols2){

     printf("Subtract form diagonal matrix not successful: Improper matrices");
     return;
   }
   //First invert the sign of all elements
   la::ScaleOverwrite(-1,mat,res); 
   
   printf("Scaled by -1..\n");
   
   //Now add the diagonal elements of the diagonal matrix to the matrix mat   
   
   for(index_t i=0;i<num_rows1;i++){
     
     res->set(i,i,diag[i]+res->get(i,i)); 
   }
 }

 
 static void SubtractFromDiagonalMatrixInit(Vector &diag,Matrix &mat, 
					    Matrix *res){
   
   index_t num_rows1=diag.length();

   index_t num_rows2=mat.n_rows();
   index_t num_cols2=mat.n_cols();

   if(num_rows1!=num_rows2 || num_rows2!=num_cols2){

     printf("Subtract form diagonal matrix not successful: Improper matrices");
     return;
   }

   //Inialize res
   res->Init(num_rows1,num_rows1);
   SubtractFromDiagonalMatrix(diag,mat,res);
 }

 static void PostMultiplyLowerTriangMatrix(Matrix &mat,
					   Matrix &lower_triang, 
					   Matrix *res){
   
   //First make a check to see if the matrices are compatible for
   //multiplication

   index_t num_rows1=mat.n_rows();
   index_t num_cols1=mat.n_cols();

   index_t num_rows2=lower_triang.n_rows();
   index_t num_cols2=lower_triang.n_cols();

   if(num_cols1!=num_rows2){

     printf("Post Multiply With Lower Triangular Matrix:Incompatible matrices");
     return;
   }

   for(index_t i=0;i<num_rows1;i++){
     
     for(index_t j=0;j<num_cols2;j++){
       
       double val=0;

       for(index_t k=j;k<num_cols1;k++){

	 val+=mat.get(i,k)*lower_triang.get(k,j);

       }

       res->set(i,j,val);
     }
   }
 }

static void  PostMultiplyMatrixWithPermutationMatrix(Matrix &mat,
						     ArrayList <index_t> &perm_mat,
						     Matrix *res){

  //This will permute the columns of the matrix

  index_t num_cols=mat.n_cols(); //n number of columns
  index_t num_rows=mat.n_rows();  //r number of rows

  index_t size_perm_mat=perm_mat.size();
  
  if(num_cols!=size_perm_mat){
    
    printf(" POst Multiply Matrix With perm matr FAILED: Sizes Incompatible");
  }

  //Get the transpose of the permutation matrix and then permnute the
  //columns accordingly

  ArrayList<index_t> perm_mat_trans;
  PermutationMatrixTransposeInit(perm_mat,&perm_mat_trans);

  //permute the columns of the matrix according to permutation matrix
  //transpose

  
  for(index_t i=0;i<size_perm_mat;i++){

    index_t j=perm_mat_trans[i];
    
    //Get the jth column

    double *ptr=mat.GetColumnPtr(j); //This gets me the jth column
   
    //Fill the ith column of the res matrix with this ptr values

    for(index_t k=0;k<num_rows;k++){

      res->set(k,i,ptr[k]);
    }
  }
}

static void  PostMultiplyMatrixWithPermutationMatrixInit(Matrix &mat,ArrayList <index_t> &perm_mat,
							 Matrix *res){
  index_t num_cols=mat.n_cols();
  index_t size_perm_mat=perm_mat.size();

  if(num_cols!=size_perm_mat){

    printf(" POst Multiply Matrix With perm matr FAILED: Sizes Incompatible");
  }

  res->Init(mat.n_rows(),mat.n_cols());

  //printf("The matrix is...\n");
  //mat.PrintDebug();

  //printf("The permutation matrix is...\n");

  /*for(index_t i=0;i<perm_mat.size();i++){

    printf("Permutation Matrix is %d..\n",perm_mat[i]);
    }*/

  PostMultiplyMatrixWithPermutationMatrix(mat,perm_mat,res);
}
 
 static void PreMultiplyMatrixWithPermutationMatrix(ArrayList <index_t> &perm_mat,
						    Matrix &mat, Matrix *res){


   //Check for Compatibility 
   index_t num_rows_perm_mat=perm_mat.size();
   index_t num_rows_mat=mat.n_rows();

   if(num_rows_perm_mat!=num_rows_mat){

     printf("Pre Multiply Matrix Permutation:Sizes are Incompatible");
   }
   
   //Pre Multiplication with a matrix is equivaleent to permuting the
   //rows of the matrix. For computational issues we shall 

   ArrayList <index_t> perm_mat_trans;

   PermutationMatrixTransposeInit(perm_mat,&perm_mat_trans);


  
   Matrix mat_trans;
   la::TransposeInit(mat,&mat_trans);

  
   Matrix temp;
   PostMultiplyMatrixWithPermutationMatrixInit(mat_trans,perm_mat_trans,&temp); //L^TP^T

   //Tranposing this will give the original mat

   la::TransposeOverwrite(temp,res); //This will give back PL

 }


 static void PreMultiplyMatrixWithPermutationMatrixInit(ArrayList <index_t> 
							&perm_mat,
							Matrix &mat,
							Matrix *res){

   //Check for Compatibility 
   index_t num_rows_perm_mat=perm_mat.size();
   index_t num_rows_mat=mat.n_rows();

   if(num_rows_perm_mat!=num_rows_mat){

     printf(" Pre Multiply Matrix Permutation:Sizes are Incompatible");
   }

   res->Init(mat.n_rows(),mat.n_cols());
   PreMultiplyMatrixWithPermutationMatrix(perm_mat,mat,res);
 }

 static void PreMultiplyVectorWithPermutationMatrix(ArrayList <index_t> 
						    &perm_mat, Vector beta, 
						    Vector *res){

   for(index_t i=0;i<res->length();i++){

     (*res)[i]=beta[perm_mat[i]];
   }

 }

 static void  PostMultiplyUpperTriangMatrix(Matrix &mat,
						Matrix &upper_triang,
						Matrix *res){

   //Make a compatibility check

   index_t num_rows1=mat.n_rows();
   index_t num_cols1=mat.n_cols();

   index_t num_rows2=upper_triang.n_rows();
   index_t num_cols2=upper_triang.n_cols();


   if(num_cols1!=num_rows2){
     printf("Post Multiply with Upper Triang Failed:Incompatible matrices");

   }

   for(index_t i=0;i<num_rows1;i++){

     for(index_t j=0;j<num_cols2;j++){

       double val=0;
       for(index_t k=0;k<=min(j,num_cols1-1);k++){
	 
	 val+=mat.get(i,k)*upper_triang.get(k,j);
	 printf("k is %d...\n",k);
       }
       res->set(i,j,val);
     }
   }
 }
 
 static void  PostMultiplyUpperTriangMatrixInit(Matrix &mat,
						Matrix &upper_triang,
						Matrix *res){

   index_t num_rows1=mat.n_rows();
   index_t num_cols1=mat.n_cols();
   
   index_t num_rows2=upper_triang.n_rows();
   index_t num_cols2=upper_triang.n_cols();
   
   if(num_cols1!=num_rows2){
     printf("Post Multiply with Upper Triang Failed:Incompatible matrices");  
   }

   //Now initialize the res matrix

   res->Init(num_rows1,num_cols2);

   PostMultiplyUpperTriangMatrix(mat,upper_triang,res);
 }



 static void PostMultiplyLowerTriangMatrixInit(Matrix &mat,
					   Matrix &lower_triang, 
					   Matrix *res){
   index_t num_rows1=mat.n_rows();
   index_t num_cols1=mat.n_cols();
   
   index_t num_rows2=lower_triang.n_rows();
   index_t num_cols2=lower_triang.n_cols();
   
   if(num_cols1!=num_rows2){
     
     printf("Post Multiply With Lower Triangular Matrix:Incompatible matrices");
     return;
   }
   
   //Initialize the res matrix and then call Post Multiply
   
   res->Init(num_rows1,num_cols2);
   PostMultiplyLowerTriangMatrix(mat,lower_triang,res);
 }

 
 static void AddPermutationMatrix(ArrayList <index_t> &perm, Matrix &mat, 
				      Matrix *res){

   //First check if the sizes are compatible

   index_t size=perm.size();
   index_t rows=mat.n_rows();
   index_t cols=mat.n_cols();

   if(size!=rows||rows!=cols){

     printf("Add Permutation Matrix: Matrix sizes are not compatible");

     printf("Size of permutation matrix is %d\n",size);

     printf("Size of matrix is %d, %d\n",rows,cols);
   }
   
   res->CopyValues(mat);

   for(index_t i=0;i<size;i++){

     index_t index=perm[i];
     double original_val=res->get(i,index);
     res->set(i,index,original_val+1);
   }
 }


 static void AddPermutationMatrixInit(ArrayList <index_t> &perm, Matrix &mat, 
				      Matrix *res){

   //First check if the sizes are compatible

   index_t size=perm.size();
   index_t rows=mat.n_rows();
   index_t cols=mat.n_cols();

   if(size!=rows||rows!=cols){

     printf("Add Permutation Matrix: Matrix sizes are not compatible...\n");
   }

   res->Init(rows,cols);
   AddPermutationMatrix(perm,mat,res);
 }

 //Adds identity matrix to the original matrix

 static void AddIdentityMatrix(Matrix &mat){

   index_t num_rows=mat.n_rows();
   index_t num_cols=mat.n_cols();

   if(num_rows!=num_cols){

     printf("Add Identity Matrix: Matrix is not square...\n");
     return;
   }

   for(index_t i=0;i<num_rows;i++){
     
     double val=mat.get(i,i);
     mat.set(i,i,val+1);
   }
 }


 //This finds the pseudo inverse of the matrix mat and returns it in
 //the same variable

 static void InvertSquareMatrixUsingSVD(Matrix &mat){

   Matrix U;
   Vector s;
   Matrix V_trans;

   //A=UsV^T
   la::SVDInit(mat,&s,&U,&V_trans);

   //A^-1= (V^T)^-1s^-1 U^-1 =V s^(-1) U^T

   Matrix V,U_trans;
   la::TransposeInit(V_trans,&V);
   la::TransposeInit(U,&U_trans);

   for(index_t i=0;i<s.length();i++){
     
     if(fabs(s[i])>EPSILON){

       s[i]=1.0/s[i];
     }
     else{
       
       s[i]=0;
     }
   }

   Matrix diag_s;
   diag_s.InitDiagonal(s);

   //Do the calculation to find the inverse

   Matrix temp1;
   la::MulInit(V,diag_s,&temp1);

   la::MulOverwrite(temp1,U_trans,&mat);
 }

 static void PreMultiplyVectorWithPermutationMatrixInit(ArrayList <index_t>& 
							perm_mat,
							Vector vec,
							Vector *res){


   //Check for compatibility

   if(perm_mat.size()!=vec.length()){

     printf("PreMultiplyVectorWithPermutationMatrixInit: Sizes are incompatible");
     
   }

   res->Init(perm_mat.size());
   //This permutes rows of the matrix

   PreMultiplyVectorWithPermutationMatrix(perm_mat,vec,res);
     
 }
 
   static void PermutationMatrixInverse(ArrayList <index_t> &permutation, 
					ArrayList <index_t> *permutation_inverse){
     
   for(index_t i=0;i<permutation.size();i++){

     
     index_t j=permutation[i];
     
     (*permutation_inverse)[j]=i;
   }
 }

  static void PermutationMatrixInverseInit(ArrayList <index_t> &permutation, 
					  ArrayList <index_t> 
					  *permutation_inverse){
   
   //This requires you to initialize the invese first
   
    
    permutation_inverse->Init(permutation.size());
    PermutationMatrixInverse(permutation, permutation_inverse);
   }

  static void PermutationMatrixTranspose(ArrayList <index_t> &permutation, 
					 ArrayList <index_t> 
					 *permutation_trans){

    //Since permutation matrices are orthogonal hence transpose is the
    //same as Init
    
    PermutationMatrixInverse(permutation,permutation_trans);
  }

  static void PermutationMatrixTransposeInit(ArrayList <index_t> &perm_mat, 
					     ArrayList <index_t> 
					     *perm_mat_trans){

    index_t size=perm_mat.size();
    perm_mat_trans->Init(size);
    PermutationMatrixTranspose(perm_mat,perm_mat_trans);
  }


  static void PreMultiplyMatrixWithDiagonalMatrix(Vector &diag,Matrix &mat,Matrix *res){

    index_t len=diag.length();
    index_t num_rows=mat.n_rows();
    

    //Compatibility Check
    if(len!=num_rows){
      
      printf("Matrix multiplication with diagonal cannot be done: INCOMPATIBLE SIZES");
      return;
    }

    //This is scaling of rows. Hence lets do scaling of columns of the
    //transpose and then tranpose the result


    Matrix mat_trans;
    la::TransposeInit(mat,&mat_trans);

    Matrix res_trans;
    PostMultiplyDiagonalMatrixInit(mat_trans,diag,&res_trans);

    //transpose it back
    la::TransposeInit(res_trans,res);
  }


  static void PreMultiplyMatrixWithDiagonalMatrixInit(Vector &diag,Matrix &mat,Matrix *res){

    index_t len=diag.length();
    index_t num_rows=mat.n_rows();
    

    //Compatibility Check
    if(len!=num_rows){
      
      printf("Matrix multiplication with diagonal cannot be done: INCOMPATIBLE SIZES");
      return;
    }

    PreMultiplyMatrixWithDiagonalMatrix(diag,mat,res);
  }


 //This function does matrix inverse multiplied by a vector. The
 //structure of matrix is D+AA^T, i.e a diagonal plus a low rank psd
 //matrix where $A \in R^{n\times r} and D \in R^{n\times n}. There is
 //no constraint on the vector. We shall use the SMW update to solve
 //this problem 

 // $(D'+AA^T)^-1\alpha=D\alpha-DA\left(I+A^TDA)^-1A^TD\alpha where D'
 // is the diagonal matrix and D=D'^-1. Since no where in my
 // calcuylations I require D', but instead require D which is
 // basically the inverse of D. Hence one of the arguents to this function 
  //is D'^-1=D
 
 static void MatrixInverseTimesVector(Matrix &permuted_chol_factor,
				      Vector diag_inverse,Vector alpha,
				      Vector *res){
 
   //First form A^TD where A is the permuted permuted_chol_factor
   Matrix permuted_chol_factor_trans;
   la::TransposeInit(permuted_chol_factor,&permuted_chol_factor_trans);

   Matrix permuted_chol_factor_trans_D;
   PostMultiplyDiagonalMatrixInit(permuted_chol_factor_trans,diag_inverse,
				  &permuted_chol_factor_trans_D);
   
   Vector permuted_chol_factor_trans_D_alpha;
   la::MulInit(permuted_chol_factor_trans_D,alpha,
	       &permuted_chol_factor_trans_D_alpha);
   
   Matrix permuted_chol_factor_trans_D_permuted_chol_factor;
   la::MulInit(permuted_chol_factor_trans_D,permuted_chol_factor,
	       &permuted_chol_factor_trans_D_permuted_chol_factor);

   //We now have to form the matrix I+A^TDA where A is the permuted_chol_factor

   AddIdentityMatrix(permuted_chol_factor_trans_D_permuted_chol_factor);

   //Now find the inverse of the matrix I+A^TDA. Since this matrix is
   //small we can afford a direct inversion

   index_t flag=
     la::Inverse(&permuted_chol_factor_trans_D_permuted_chol_factor);

   if(flag==SUCCESS_PASS){

     //printf("Inversion done successfully...\n");

     //For the other cases we shall print out warning messages
   }
   else{
     if(flag==SUCCESS_WARN){

       printf("Warning issued");
       exit(0);
     }
     else{
       printf("Inversion failed, but will do pseudo inverse");
       exit(0);
       //       InvertSquareMatrixUsingSVD(permuted_chol_factor_trans_D_permuted_chol_factor);
     }
   }

   //evaluate DA. This is simply the transpose of A^TD

   Matrix D_permuted_chol_factor;
   la::TransposeInit(permuted_chol_factor_trans_D,
		     &D_permuted_chol_factor);

   //We now need to do DA(I+A^TDA)^-1A^TD\alpha

   Matrix temp1;
   la::MulInit (D_permuted_chol_factor,
		permuted_chol_factor_trans_D_permuted_chol_factor,
		&temp1);

   Vector temp2;
   la::MulInit(temp1,permuted_chol_factor_trans_D_alpha,&temp2);

   //Find D\alpha
   Vector D_alpha;
   
   PreMultiplyVectorWithDiagonalMatrixInit(diag_inverse,alpha,
					   &D_alpha);

   //Finally subtract
   la::SubOverwrite(temp2,D_alpha,res);
 }
 
 static void MatrixInverseTimesVectorInit(Matrix &permuted_chol_factor,Vector diag_inverse, Vector alpha,Vector *res){ 
   
   res->Init(diag_inverse.length());  
   
   MatrixInverseTimesVector(permuted_chol_factor,diag_inverse,alpha,res); 
   
 } 


/*  static void WhitenUsingEig(Matrix X, Matrix* X_whitened, Matrix* whitening_matrix) { */
/*    Matrix cov_X, D, D_inv, E; */
/*    Vector D_vector; */
   
/*    Scale(1 / (double) (X.n_cols() - 1), */
/* 	 MulTransBInit(&X, &X, &cov_X)); */
   
   
/*    la::EigenvectorsInit(cov_X, &D_vector, &E);  */
/*    //E.set(0, 1, -E.get(0, 1)); */
/*    //E.set(1, 1, -E.get(1, 1)); */
   
/*    index_t d = D_vector.length(); */
/*    D.Init(d, d); */
/*    D.SetZero(); */
/*    D_inv.Init(d, d); */
/*    D_inv.SetZero(); */
/*    for(index_t i = 0; i < d; i++) { */
/*      double sqrt_val = sqrt(D_vector[i]); */
/*      D.set(i, i, sqrt_val); */
/*      D_inv.set(i, i, 1 / sqrt_val); */
/*    } */
   
/*    MulTransBInit(D_inv, E, whitening_matrix); */
/*    MulInit(*whitening_matrix, X, X_whitened); */
/*  } */
 
/*  static inline void MulTransBInit(const Matrix &A, const Matrix &B, Matrix *C) { */
/*    C->Init(A.n_rows(), B.n_rows()); */
/*    MulTransBOverwrite(A, B, C); */
/*  } */

/*  static Matrix* MulTransBInit(const Matrix* const A, const Matrix* const B, */
/* 		       Matrix* C) { */
/*    la::MulTransBInit(*A, *B, C); */
/*    return C; */
/*  } */

/* static  Matrix* MulTransBOverwrite(const Matrix* const A, Matrix* const B, */
/* 			    Matrix* const C) { */
/*    la::MulTransBOverwrite(*A, *B, C); */
/*    return C; */
/*  } */

/* static  inline void Scale(index_t length, double alpha, double *x) { */
/*      F77_FUNC(dscal)(length, alpha, x, 1); */
/*    } */
 
/*  static Matrix* MulInit(const Matrix* const A, const Matrix* const B, */
/* 		 Matrix* const C) { */
/*    la::MulInit(*A, *B, C); */
/*    return C;    */
/*  } */

 


};

#endif

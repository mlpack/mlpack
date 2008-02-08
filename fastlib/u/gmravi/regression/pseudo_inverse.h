/**
 * @file dataset_scaler.h
 *
 * This file contains utility functions to find the pseudo inverse of
 * a matrix. We accomplish this by SVD. num_of_query_points refers to the 
 * number of matrices in the arraylist of matrices
 *
 * @author Dongryeol Lee (dongryel)
 * @bug No known bugs.
 */

#ifndef PSEUDO_INVERSE_H
#define PSEUDO_INVERSE_H

#include <fastlib/fastlib.h>


class PseudoInverse {

 public:

  static void  FindPseudoInverse(index_t num_query_points, 
				 Matrix &matrix_to_be_inverted ){


   

  /** At the moment the arraylist results_ has the matrix estimate B^TWB. 
     *  However we need (B^TWB)^-1. hence we invert this matrix by 
     *  doing an SVD inversion
     */  
   
 
      /*This is inversion by SVD **********/
      Vector s;
      Matrix U;
      Matrix VT; 
      Matrix V;
      Matrix S_diagonal;
      Matrix U_transpose;
     
      la::SVDInit(matrix_to_be_inverted,&s,&U,&VT); //perform SVD
     
      la::TransposeInit(VT,&V); //Transpose VT to get V 
     
      // S_diagonal is a diagonal matrix formed 
      // from the reciprocal of the elements of s.
     
      // dimensions of S_diagonal are fromed appropriately. 
      // it is (columns in V)X(columns in U)
     
      index_t rows_in_S_diagonal=V.n_cols();
      index_t cols_in_S_diagonal=U.n_cols();
     
      //appropriately initialize s_diagonal

      S_diagonal.Init(rows_in_S_diagonal,cols_in_S_diagonal);
      //Fill up the s_diagonal matrix with the reciprocal elements of s
      
      for(index_t i=0;i<rows_in_S_diagonal;i++){
	
	for(index_t j=0;j<cols_in_S_diagonal;j++){
	  
	  if(i==j){
	    //Only if the singular value is a certain fraction more
	    //than the highest wsingular value do we invert the
	    //singular value. else we make it 0
	    if(s[i]>=0.001*s[0]){
	      S_diagonal.set(i,j,1.0/s[i]);
	    }
	    else{
	      S_diagonal.set(i,j,0.0);
	    }
	  }
	  else{
	    //off diagonal element. hence is equal to 0
	    S_diagonal.set(i,j,0);
	  }
	}
      }
      
      Matrix temp1;
      Matrix temp2;
      la::MulInit (V,S_diagonal,&temp1);
    
      
      //Find transpose of U
      
      la::TransposeInit(U,&U_transpose);
      la::MulInit(temp1, U_transpose, &temp2); 

      //At this point the variable temp holds the 
      //pseudo-inverse of matrix_to_be_inverted[q]
     
      
      //Copy the contents of temp2 to matrix_to_be_inverted
      matrix_to_be_inverted.CopyValues(temp2);
    
  }
};
#endif
    

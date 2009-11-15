#ifndef ICHOL_H
#define ICHOL_H
#include "fastlib/fastlib.h"
#include "hyperkernels.h"
#include "special_la.h"
#include <math.h>


/**This is an implementation of the online incomplpete cholesky
   decomposition by shai fine, schienberg. I am writing this module to
   do a quick online cholesky decomposition of the matrix. I am
   writing it to do an online Cholesky decomposition of the matrix
   \frac{K'}{m^2}+\lambda{K}
**/


class ICholDynamic{
  
  FORBID_ACCIDENTAL_COPIES(ICholDynamic);
 private:

  //Parameters of hyperkernel. 

  double sigma_h_;
  double sigma_;
  double lambda_;

  //The number of train points
  index_t num_train_points_;


  //Number of dimensions of the train data
  index_t num_dims_;

  //Hyperkernels are m^2 x m^2
  index_t sqd_num_train_points_;

  //The threshold. This determines when to stop
  double epsilon_;

  //Online Incomplete Cholesky requires us to store the diagonal
  //elements of the matrix M(whose incomplete cholesky we need) In
  //memory, as they are reused during the run of the code

  Vector diagonal_elements_;

  //The diagonal elements of the Cholesky factor. The diagonal
  //elements are always required

  Vector diagonal_elements_chol_factor_;

  //The train dataset on which the matrices are built

  Matrix train_set_;

  //The lower triangular cholesky matrix
  ArrayList<Vector> chol_factor_dynamic_;

  //Permutation vector. 

  ArrayList <index_t> perm_;

  void ComputeDiagonalElements_(){
    
    index_t row=0;
    for(index_t p=0;p<num_train_points_;p++){
      for(index_t q=0;q<num_train_points_;q++){
	
	double val_diag=ComputeElementOfMMatrix_(p,q,p,q); //r=p,s=q
	diagonal_elements_[row]=val_diag;
	row++;
      }
    }
  }

 public:

  //A dummy constructor and destructor which does nothing

  ICholDynamic(){

  }

  ~ICholDynamic(){

  }

  //Getters
  double get_element_of_m_matrix(index_t p,index_t q,index_t r,index_t s){

    return ComputeElementOfMMatrix_(p,q,r,s);
  }

 private:
  
  void  CreateSquaredVectorFromCholFactor_(index_t start_row,
					   index_t end_col,
					   Vector &squared_vector){
    index_t end_row=sqd_num_train_points_;
    index_t start_col=0;
    index_t i;
    i=0;

    for(index_t row=start_row;row<end_row;row++){
      
      double sum_of_rows=0;
      for(index_t col=start_col;col<end_col;col++){
	
	double val=chol_factor_dynamic_[col][row];
	sum_of_rows+=val*val;
      }
      squared_vector[i]=sum_of_rows; 
      i++;
    }
  }
  
  void  ExtractRowOfOriginalMatrix_(index_t row,index_t start_index_of_perm_vector,Vector &temp){
    
    //This function forms a vector by accessing the original elements ie we neeed

    //We should probably make use of a caching scheme to make it more efficient
    index_t i=0;
    row=perm_[row];
    for(index_t j=start_index_of_perm_vector;j<sqd_num_train_points_;j++){
      
      index_t col=perm_[j];
      
      //Having got the row and column number find p,q,r,s
      index_t p=row/num_train_points_;
      index_t q=row%num_train_points_;
      index_t r=col/num_train_points_;
      index_t s=col%num_train_points_;
      temp[i]=ComputeElementOfMMatrix_(p,q,r,s);
      i++;
    }
  }

  double ComputeElementOfMMatrix_(index_t p, index_t q,index_t r,index_t s){
    
    //$M(p,q,r,s)=K'/m^2 +\lambda K$
    
    //This function will do a naive computation of M Matrix. i.e it
    //will compute an element of the matrix given its position
    
    //Read the data
    double *x_p=train_set_.GetColumnPtr(p);
    double *x_q=train_set_.GetColumnPtr(q);
    double *x_r=train_set_.GetColumnPtr(r);
    double *x_s=train_set_.GetColumnPtr(s);

    Vector vec_x_p;
    vec_x_p.Alias(x_p,num_dims_);

    Vector vec_x_q;
    vec_x_q.Alias(x_q,num_dims_);

    Vector vec_x_r;
    vec_x_r.Alias(x_r,num_dims_);
    
    Vector vec_x_s;
    vec_x_s.Alias(x_s,num_dims_);

    //This is a naive code, which computes the elements of the M
    //matrix naively by a dual summation.

    //TODO: See how this can be done usign dual tree framework

    GaussianKernel gk;
    gk.Init(sigma_*sqrt(2));

    //Squared distance between x_p and x_q
    double sqd_x_p_x_q=la::DistanceSqEuclidean(vec_x_p,vec_x_q);

    //Euclidean distance squared between x_r and x_s
    double sqd_x_r_x_s=la::DistanceSqEuclidean(vec_x_r,vec_x_s);

    //gaussian kernels between x_p x_q and x_r x_s
    double gk_x_p_x_q=gk.EvalUnnormOnSq(sqd_x_p_x_q)/gk.CalcNormConstant(num_dims_);
    double gk_x_r_x_s=gk.EvalUnnormOnSq(sqd_x_r_x_s)/gk.CalcNormConstant(num_dims_);

    //We shall need the values of sigma^4, sigma^2, sigma_h^2 and
    //sigma_h^4. Hence lets compute and store them
    
    double sigma_4=sigma_*sigma_*sigma_*sigma_;
    double sigma_h_4=sigma_h_*sigma_h_*sigma_h_*sigma_h_;
    double sigma_2=sigma_*sigma_;
    double sigma_h_2=sigma_h_*sigma_h_;

    //calculate constants that depend on the dimensions p,q,r,s only

    //Rememember we are working in a multi-d setting and hence these
    //values will be calculated in a loop for each dimension

/*     double c1=7*x_p[0]+7*x_q[0]+x_r[0]+x_s[0]; */
/*     double c2=5*(x_p[0]*x_p[0]+x_q[0]*x_q[0]+x_r[0]*x_r[0]+x_s[0]*x_s[0]); */
/*     double c3=10*x_p[0]*x_q[0]-2*x_p[0]*x_r[0]-2*x_q[0]*x_r[0]-2*x_s[0]*(x_p[0]+x_q[0]-5*x_r[0]); */
/*     double c4=x_p[0]+x_q[0]+7*x_r[0]+7*x_s[0]; */
/*     double c5=3*x_p[0]+3*x_q[0]+x_r[0]+x_s[0]; */
/*     double c6=(x_p[0]+x_q[0])*(x_p[0]+x_q[0]); */
/*     double c7=(x_r[0]+x_s[0])*(x_r[0]+x_s[0]); */
/*     double c8=(x_p[0]+x_q[0]+3*x_r[0]+3*x_s[0]); */

    double c9=(4*math::PI*sqrt(math::PI)*sigma_*sqrt(sigma_2+sigma_h_2)*
		sqrt(3*sigma_2+2*sigma_h_2));

    double c=pow((1.0/c9),num_dims_)*gk_x_p_x_q*gk_x_r_x_s;
    
    

    
    //This constant will work even for multi-dimensional setting
   
   
    
    //printf("c9 is %f..\n",c91/c92);
    //printf("Inverse which was used for 1-d calculations is %f..\n",c92/c91);
    
    /* printf("c1=%f\n",c1);  */
/*     printf("c2=%f\n",c2);  */
/*     printf("c3=%f\n",c3);  */
/*     printf("c4=%f\n",c4);  */
/*     printf("c5=%f\n",c5);  */
/*     printf("c6=%f\n",c6);  */
/*     printf("c7=%f\n",c7);  */
/*     printf("c8=%f\n",c8);  */
/*     printf("c9=%f\n",c9);  */
/*     printf("c=%f\n",c);  */
    
  


    double temp=16*sigma_2*(sigma_2+sigma_h_2)*(3*sigma_2+2*sigma_h_2);
    double k_prime_val=0;
    for(index_t i=0;i<num_train_points_;i++){

      double *x_i=train_set_.GetColumnPtr(i);
      Vector vec_x_i;
      vec_x_i.Alias(x_i,num_dims_);
      for(index_t j=0;j<num_train_points_;j++){

	double *x_j=train_set_.GetColumnPtr(j);
	Vector vec_x_j;
	vec_x_j.Alias(x_j,num_dims_);
	
	//Having got $x_i,x_j$ calculate the following parameters

	double theta_x=0;
	for(index_t d=0;d<num_dims_;d++){
	  
	  double c1=7*x_p[d]+ 7*x_q[d]+ x_r[d]+ x_s[d];
	  double c2=5*(x_p[d]*x_p[d]+ x_q[d]*x_q[d]+ x_r[d]*x_r[d]+ x_s[d]*x_s[d]);
	  double c3=10*x_p[d]*x_q[d]- 2*x_p[d]*x_r[d]- 2*x_q[d]*x_r[d]- 2*x_s[d]*(x_p[d]+x_q[d]-5*x_r[d]);
	  double c4=x_p[d] +x_q[d]+ 7*x_r[d]+ 7*x_s[d];
	  double c5=3*x_p[d]+ 3*x_q[d]+ x_r[d]+ x_s[d];
	  double c6=(x_p[d]+x_q[d])*(x_p[d]+x_q[d]);
	  double c7=(x_r[d]+x_s[d])*(x_r[d]+x_s[d]);
	  double c8=(x_p[d]+x_q[d]+3*x_r[d]+3*x_s[d]);
	  
	  //The values of $\alpha,\beta,\gamma$ along each direction
	  double alpha=17*(x_i[d]*x_i[d])+17*(x_j[d]*x_j[d])-
	    2*x_i[d]*(x_j[d]+c1)+c2+c3-(2*x_j[d]*c4);
	  
	  double beta=(5*x_i[d]*x_i[d])-x_i[d]*(2*x_j[d]+c5)+
	    5*x_j[d]*x_j[d]+c6+c7-x_j[d]*c8;
	  
	  double gamma=(x_i[d]-x_j[d])*(x_i[d]-x_j[d]);
	  
	  //Value of $\theta_x$ for this pair of i,j alogn direction d 
	  theta_x+=alpha*sigma_4+4*beta*sigma_h_2*sigma_2+4*gamma*sigma_h_4;
	 
	  //printf("alpha=%f,beta=%f,gamma=%f,theta_x=%f..\n",alpha,beta,gamma,theta_x);
	}
	//With this the value of $\theta-x$ is calculated
	
	theta_x*=-1/temp;
	k_prime_val+=pow(math::E,theta_x);
      }
    }
    
    
    Vector mean_x_p_x_q,mean_x_r_x_s;
    la::AddInit(vec_x_p,vec_x_q,&mean_x_p_x_q);
    la::Scale(0.5,&mean_x_p_x_q);

    la::AddInit(vec_x_r,vec_x_s,&mean_x_r_x_s);
    la::Scale(0.5,&mean_x_r_x_s);

    
    gk.Init(sqrt(sigma_2+sigma_h_2));
    double sqd_dist_between_means=la::DistanceSqEuclidean(mean_x_p_x_q,
							  mean_x_r_x_s);
    
    double k_val=gk_x_p_x_q*gk_x_r_x_s*gk.EvalUnnormOnSq(sqd_dist_between_means);
    k_val/=gk.CalcNormConstant(num_dims_);

    //printf("Gausian between x_p x_q is %f\n..",gk_x_p_x_q);
    //printf("Gausian between x_r x_s is %f\n..",gk_x_r_x_s);
    //printf("Gaussian between means is %f\n",
    //   gk.EvalUnnormOnSq(sqd_dist_between_means)/gk.CalcNormConstant(num_dims_));
    //printf("Squared distance between means is %f...\n",sqd_dist_between_means);
    //printf("Nomrlaization constatnt=%f..\n",gk.CalcNormConstant(num_dims_));

    //printf("k_val =%f..\n",k_val);
    //printf("k_prime val=%f...\n",k_prime_val);
    //printf("temp=%f..\n",temp);

    double m_val=(c*k_prime_val/sqd_num_train_points_)+(lambda_*k_val);
    //printf("m_val=%f..\n",m_val);
    return m_val;
  }

  double SumOfDiagonalElementsOfCholFactor_(index_t start_row){
    
    double sum_diag_elem=0;
    
    for(index_t i=start_row;i<sqd_num_train_points_;i++){
      
      sum_diag_elem+=diagonal_elements_chol_factor_[i];
    }
    return sum_diag_elem;
  }
  

  //First column of the matrix is same as the first row
  //as the matrix is symmetric
  void GetFirstColOfMMatrix_(Vector &first_col){
    
    
    for(index_t p=0;p<num_train_points_;p++){
      
      for(index_t q=p;q<num_train_points_;q++){
	
	index_t row1=p*num_train_points_+q;
	index_t row2=q*num_train_points_+p;
	double val=ComputeElementOfMMatrix_(p,q,0,0);
	
	first_col[row1]=val;
	first_col[row2]=val;
      }
    }     
  } 
  
  /*index_t CheckIfLastColumnIsZeros_(index_t rank){
    
    double  epsilon;
    epsilon=pow(10,-6);
    
    
    double *last_chol;
    last_chol=chol_factor_.GetColumnPtr(rank-1);
    
    Vector v;
    v.Alias(last_chol,sqd_num_train_points_);
    
    printf("The last column is...\n");
    v.PrintDebug();
    
    for(index_t i=0;i<sqd_num_train_points_;i++){
      
      if(fabs(v[i])>epsilon){
	
	printf("i=%d,value=%6f..\n",i,v[i]);
	return -1;
      }
    }

    return 1;
    }*/
  

  void SwapRowsOfCholFactor_(index_t row1,index_t row2,index_t tillcol){
    
    //Get row1 
    double temp;
    for(index_t i=0;i<tillcol;i++){
      
      temp=chol_factor_dynamic_[i][row1]; //These are elements from ith col row=row1

      chol_factor_dynamic_[i][row1]= 
	chol_factor_dynamic_[i][row2];

      chol_factor_dynamic_[i][row2]=temp;
    }
  }
 
  void FindMaxDiagonalElementInCholFactor_(index_t start_row,
					   index_t *max_index,
					   double *max_value){
    
    *max_value=-INFINITY;
    *max_index=start_row;

    //printf("Diagonal elements at this point are..\n");
  
    for(index_t i=start_row;i<sqd_num_train_points_;i++){
      
      double val=diagonal_elements_chol_factor_[i];
      if(val>*max_value){
	
	*max_value=val;
	*max_index=i;
      }
    }
  }
  

  void ExtractSubMatrixOfCholFactor_(index_t start_row,index_t end_row,
				     index_t start_col,index_t end_col, 
				     Matrix &temp_mat){
    
    index_t i=0;
    index_t j=0;
    
    if(start_row==end_row ||start_col==end_col){
      temp_mat.SetAll(0);
      return;
    }
    for(index_t row=start_row;row<end_row;row++){
      j=0;
      for(index_t col=start_col;col<end_col;col++){
	
	temp_mat.set(i,j,chol_factor_dynamic_[col][row]);
	j++;
      }
      i++;
    }
  }

  //This routine prints arraylist

  void PrintArrayList(){
    

    Matrix temp;
    temp.Init(sqd_num_train_points_,chol_factor_dynamic_.size());
    for(index_t i=0;i<chol_factor_dynamic_.size();i++){
      
      for(index_t j=0;j<sqd_num_train_points_;j++){
	
	temp.set(j,i,chol_factor_dynamic_[i][j]);	  
      }
    }

    printf("The arraylist is...\n");
    temp.PrintDebug();
  }
  
  
  void ComputeIncompleteCholeskyDynamically_(Matrix &chol_factor_in,
					     ArrayList<index_t> &perm_in){
    
    
    //The algorithms requires that we store the diagonal elements of
    //the matrix M in memory. Hence compute diagonal elements
    
    ComputeDiagonalElements_();
    
    //Set the first column of chol factor to the first column of the
    //matrix M_mat_
    
    Vector first_col;
    first_col.Init(sqd_num_train_points_);
    
    GetFirstColOfMMatrix_(first_col);

    // printf("First column of the M matrix was brought in...\n");
    //first_col.PrintDebug();
    
    //Set the first column
    
    index_t i;
    
    Vector new_column_to_be_formed;
    new_column_to_be_formed.Copy(first_col);
    
    for(i=0;i<sqd_num_train_points_;i++){
      
      if(i>=1){
	
	Vector squared_vector;
	squared_vector.Init(sqd_num_train_points_-i);
	
	//start from the i-1 th row and go till column i-1(including)
	CreateSquaredVectorFromCholFactor_(i,i,squared_vector);
	
	for(index_t j=i;j<sqd_num_train_points_;j++){
	  
	  diagonal_elements_chol_factor_[j]=diagonal_elements_[perm_[j]]-squared_vector[j-i];
	}
      }

      //printf("Squared vector is ...\n");
      //squared_vector.PrintDebug();
      
      else{ //this is for i=0
	
	diagonal_elements_chol_factor_.CopyValues(diagonal_elements_);
      }
      

      //Copy the diagonal element of the new column to be formed
      new_column_to_be_formed[i]=diagonal_elements_chol_factor_[i];
      
      
      //printf("chol factor after initial subtraction is %d...\n",i);
      //chol_factor_.PrintDebug();
      
      if(SumOfDiagonalElementsOfCholFactor_(i)<=epsilon_){
      
	break;
      }
      else{ //This is for the continuation of the algorithm
	
	//Find the maximum diagonal element from i to m
	
	index_t max_index;
	double max_value;
	FindMaxDiagonalElementInCholFactor_(i,&max_index,&max_value);
	
	//Take care of the permutations
	
	//p[i] <-> p[max_index]
	//printf("Will swap %d and %d..\n",perm_[i],perm_[max_index]);
	//printf("max_index is %d..\n",max_index);
	
	index_t  temp=perm_[i];
	perm_[i]=perm_[max_index];
	perm_[max_index]=temp;
	
	//swap the contents of the rows i and q only upto columns
	//0:i-1(including) Which is basically the lower traingle of the
	//row i
	
	index_t row1=i;
	index_t row2=max_index;
	index_t tillcol=i;

	//printf("For swapping row1=%d,row2=%d,tillcol=%d...\n",
	//row1,row2,tillcol);

	SwapRowsOfCholFactor_(row1,row2,tillcol);
	
	//NOTE: WE have swapped elements of row i and q only till col
	//=i. Hence none of the diagonal elements change
      


	
	max_value=sqrt(max_value);
      
	new_column_to_be_formed[i]=max_value;
	
	//Remember to synchronize it back
	diagonal_elements_chol_factor_[i]=max_value;
	
	Vector temp1;
	temp1.Init(sqd_num_train_points_-i-1);
	
	index_t start_index_of_perm_vector=i+1;
	//printf("Permutation matrix at this point is..\n");
	/*for(index_t q=i;q<sqd_num_train_points_;q++){
	  
	printf("Permutation[%d]=%d",q,perm_[q]);
	}*/
       
	ExtractRowOfOriginalMatrix_(i,start_index_of_perm_vector,temp1);
	
	if(i>=1){
	  index_t start_row=i+1;
	  index_t end_row=sqd_num_train_points_;
	  index_t start_col=0;
	  index_t end_col=i;
	  
	  Matrix temp_mat;
	  temp_mat.Init(end_row-start_row,end_col-start_col);
	  
	  ExtractSubMatrixOfCholFactor_(start_row,end_row,start_col,end_col,
					temp_mat);
	  
	  Vector temp2;
	  temp2.Init(i);
	  
	  
	  for(index_t j=0;j<i;j++){
	    
	    temp2[j]=chol_factor_dynamic_[j][i];
	  }
	  
	  if(temp_mat.n_rows()!=0&&temp_mat.n_cols()!=0&&temp2.length()!=0){
	    
	    Vector temp3_prod;	   
	    la::MulInit(temp_mat,temp2,&temp3_prod);
	    
	    Vector temp4;
	    la::SubInit(temp3_prod,temp1,&temp4);
	  
	    //printf("The subtractant is %d...\n",i);
	    //temp3_prod.PrintDebug();
	  
	    //L(i+1:m,i)*L(i,1:i-1) <- temp4
	    index_t k=0;
	  
	    //printf("Doing final stuff on chol factor ...\n");
	    for(index_t row=i+1;row<sqd_num_train_points_;row++){
	    
	      new_column_to_be_formed[row]=(temp4[k]/max_value);
	      k++;	   
	    }
	  
	    //printf("Chol factor after final  subtraction is i=%d...\n",i);
	    //chol_factor_.PrintDebug();
	  }
	
	  else{
	  
	    index_t k=0;
	    for(index_t row=i+1;row<sqd_num_train_points_;row++){
	    
	      new_column_to_be_formed[row]=(temp1[k]/max_value);
	      k++;	   
	    }
	  }
	}
	else{ //This will execute if i=0
	
	  //printf("Max value is %f\n",max_value);
	
	  //Fill this up
	  index_t k=0;
	
	  for(index_t row=i+1;row<sqd_num_train_points_;row++){
	  
	  
	    new_column_to_be_formed[row]=temp1[k]/max_value;
	    k++;	   
	  }
	}
      }
      //Add the new column

      chol_factor_dynamic_.AddBackItem(new_column_to_be_formed);

      
      new_column_to_be_formed.SetZero();
    }
    
    //END OF THE ALGORITHM...............
    

    //Determine the rank of the matrix
    
    index_t rank;

    printf("i is %d..\n",i);
    if(i<=sqd_num_train_points_-1){
      
      rank=i+1;
      chol_factor_dynamic_.AddBackItem(new_column_to_be_formed);
    }
    else{
      
      rank=sqd_num_train_points_;
    }

    printf("Cholesky factorization found with rank %d...\n",rank);

    chol_factor_in.Init(sqd_num_train_points_,rank);
    //Copy the cholesky factor which is an arraylist into a matrix

    for(index_t i=0;i<rank;i++){
      
      for(index_t j=0;j<sqd_num_train_points_;j++){
	
	double val=chol_factor_dynamic_[i][j];
	chol_factor_in.set(j,i,val);	  
      }
    }
    ///DONE With this
    
    //Similariy copy permutation matrix
    

    for(index_t i=0;i<sqd_num_train_points_;i++){
      perm_in[i]=perm_[i];
    }
  }
  
 public:
  
  void Compute(Matrix &chol_factor_in,ArrayList <index_t> &perm_in){
    
    /*  Matrix M_mat_;
    M_mat_.Init(sqd_num_train_points_,sqd_num_train_points_);
    
    printf("Wil do incomplete cholesky now...\n");
    
    
    for(index_t p=0;p<num_train_points_;p++){
      
      for(index_t q=0;q<num_train_points_;q++){
	
	index_t row_num1=p*num_train_points_+q;
	index_t row_num2=q*num_train_points_+p;
	
	for(index_t r=0;r<num_train_points_;r++){
	  
	  for(index_t s=0;s<num_train_points_;s++){
	    
	    index_t col_num1=r*num_train_points_+s;
	    index_t col_num2=s*num_train_points_+r;
	    
	    double val=ComputeElementOfMMatrix_(p,q,r,s);
	    
	    M_mat_.set(row_num1,col_num1,val); //p,q,r,s 1,2,3,4
	    //m.set(row_num1,col_num2,val); //p,q,s,r 1,2,4,3
	    //m.set(row_num2,col_num1,val); //q,p,r,s 2,1,3,4
	    //m.set(row_num2,col_num2,val); //q,p,s,r 2,1,4,3
	    
	    //m.set(col_num1,row_num1,val); //r,s,p,q 3,4,1,2
	    //m.set(col_num1,row_num2,val); //r,s,q,p 3,4,2,1
	    //m.set(col_num2,row_num1,val); //s,r,p,q 4,3,1,2
	    //m.set(col_num2,row_num2,val); //s,r,q,p 4,3,2,1
	  }
	}
      }
    }
    
    
    FILE *fp;
    fp=fopen("m_matrix.txt","w");
    M_mat_.PrintDebug(NULL,fp);*/
    
    //Matrix temp;
    //temp.Init(train_set_.n_rows(),train_set_.n_cols());
    //temp.CopyValues(train_set_);
    //la::MulTransBOverwrite(temp,temp,&M_mat_);
    //printf("M matrix is...\n");
    //    M_mat_.PrintDebug(NULL,fp);
    fx_timer_start(NULL,"incomplete_cholesky");
    ComputeIncompleteCholeskyDynamically_(chol_factor_in,perm_in);
    fx_timer_stop(NULL,"incomplete_cholesky");
  }

  void Init(Matrix &train_set,double sigma_h,double sigma,double lambda){
    
    //Initialize all the parameters of the class
    sigma_h_=sigma_h;
    sigma_=sigma;
    lambda_=lambda;
    train_set_.Alias(train_set);
    num_train_points_=train_set.n_cols();
    sqd_num_train_points_=num_train_points_*num_train_points_;
    //Initialize the cholesky factor 
    
    
    printf("Will allocate memory for cholesky factorization...\n");
    printf("Number of training points are...%d\n",num_train_points_);
    printf("lambda is %f..\n",lambda_);
    
    //Chol factor will begin as an arraylist of size 0 and then grow in size
    
    chol_factor_dynamic_.Init();
    diagonal_elements_chol_factor_.Init(sqd_num_train_points_);
    diagonal_elements_.Init(sqd_num_train_points_);
    
    
    //Initialize the permutation vector in a serial order
    
    perm_.Init(sqd_num_train_points_);
    
    for(index_t i=0;i<sqd_num_train_points_;i++){
      
      perm_[i]=i;      
    }
    
    epsilon_=pow(10,-6);
    
    num_dims_=train_set_.n_rows();
  }
  
};
#endif

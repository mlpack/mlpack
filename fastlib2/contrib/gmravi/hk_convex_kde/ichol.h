#ifndef ICHOL_H
#define ICHOL_H
#include "fastlib/fastlib.h"
#include "hyperkernels.h"
#include "special_la.h"
#include <math.h>


/**This is an implementation of the online incomplpete cholesky
   decomposition by shai fine, schienberg. I am writing this module to
   do a quick online cholesky decomposition of the matrix. I am
   writing it to doan online Cholesky decomposition of the matrix
   \frac{K'}{m^2}+\lambda{K}
**/


class IChol{
  
  FORBID_ACCIDENTAL_COPIES(IChol);
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

  //The train dataset on which the matrices are built

  Matrix train_set_;

  //The lower triangular cholesky matrix
  Matrix chol_factor_;

  //Permutation vector. 

  ArrayList <index_t> perm_;

  Matrix M_mat_;

  
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
  

   /* void ComputeDiagonalElements_(){ */
   
/*    index_t row=0; */
/*    for(index_t p=0;p<num_train_points_;p++){ */
/*      for(index_t q=0;q<num_train_points_;q++){ */
       
/*        diagonal_elements_[row]=M_mat_.get(row,row); */
/*        row++; */
/*      } */
/*      } */
/*    printf("The diagoanl eleements are...\n"); */
/*    diagonal_elements_.PrintDebug(); */
/*    } */

 public:

  //A dummy constructor and destructor which does nothing

  IChol(){

  }

  ~IChol(){

  }
  
  void  CreateSquaredVectorFromCholFactor_(index_t start_row,
					   index_t end_col,
					   Vector &squared_vector){
    index_t end_row=sqd_num_train_points_;
    index_t start_col=0;
    index_t i;
    i=0;

    printf("Start row is %d\n",start_row);

    for(index_t row=start_row;row<end_row;row++){
      
      double sum_of_rows=0;
      for(index_t col=start_col;col<end_col;col++){
	
	double val=chol_factor_.get(row,col);
	sum_of_rows+=val*val;
      }
      squared_vector[i]=sum_of_rows; 
      i++;
    }
  }
  
  void FindMaxDiagonalElementInCholFactor_(index_t start_row,
					   index_t *max_index,
					   double *max_value){

    *max_value=-INFINITY;
    *max_index=start_row;
    for(index_t i=start_row;i<sqd_num_train_points_;i++){
      
      if(chol_factor_.get(i,i)>*max_value){
	
	*max_value=chol_factor_.get(i,i);
	*max_index=i;
      }
    }
  }

  void SwapRowsOfCholFactor_(index_t row1,index_t row2,index_t tillcol){

    double temp;

    for(index_t j=0;j<tillcol;j++){
      
      temp=chol_factor_.get(row1,j);
      chol_factor_.set(row1,j,chol_factor_.get(row2,j));
      chol_factor_.set(row2,j,temp);
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
	
	temp_mat.set(i,j,chol_factor_.get(row,col));
	j++;
      }
      i++;
    }
  }

  double SumOfDiagonalElementsOfCholFactor_(index_t start_row){

    double sum_diag_elem=0;

    for(index_t i=start_row;i<sqd_num_train_points_;i++){

      sum_diag_elem+=chol_factor_.get(i,i);
    }

    return sum_diag_elem;
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


  /*void  ExtractRowOfOriginalMatrix_(index_t row,index_t start_index_of_perm_vector,Vector &temp){
    
    //This function forms a vector by accessing the original elements ie we neeed
    
    //We should probably make use of a caching scheme to make it more efficient
    index_t i=0;
    row=perm_[row];
    for(index_t j=start_index_of_perm_vector;j<sqd_num_train_points_;j++){
      
      index_t col=perm_[j];
      
      //Having got the row and column number find p,q,r,s
      //index_t p=row/num_train_points_;
      //index_t q=row%num_train_points_;
      //index_t r=col/num_train_points_;
      //index_t s=col%num_train_points_;
      temp[i]=M_mat_.get(row,col);;
      i++;
      }
    }*/


  double ComputeElementOfMMatrix_(index_t p, index_t q,index_t r,index_t s){


    //printf("p=%d,q=%d,r=%d,s=%d...\n",p,q,r,s);

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
    //matrix naively by a dual summation. Once done with this I shall
    //replace this by a dual tree evaluation

    GaussianKernel gk;
    gk.Init(sigma_*sqrt(2));

    //Squared distance between x_p and x_q
    double sqd_x_p_x_q=la::DistanceSqEuclidean(vec_x_p,vec_x_q);

    //Euclidean distance squared between x_r and x_s
    double sqd_x_r_x_s=la::DistanceSqEuclidean(vec_x_r,vec_x_s);

    //gaussian kernels between x_p x_q and x_r x_s
    double gk_x_p_x_q=gk.EvalUnnormOnSq(sqd_x_p_x_q)/gk.CalcNormConstant(num_dims_);
    double gk_x_r_x_s=gk.EvalUnnormOnSq(sqd_x_r_x_s)/gk.CalcNormConstant(num_dims_);

    
    //calculate constants that depend on the dimensions p,q,r,s only

    //Since this is a one dimensional setting we shall make use of the
    //follwing formula. For a multi-d setting a generalization can be
    //easily made

    double c1=7*x_p[0]+7*x_q[0]+x_r[0]+x_s[0];
    double c2=5*(x_p[0]*x_p[0]+x_q[0]*x_q[0]+x_r[0]*x_r[0]+x_s[0]*x_s[0]);
    double c3=10*x_p[0]*x_q[0]-2*x_p[0]*x_r[0]-2*x_q[0]*x_r[0]-2*x_s[0]*(x_p[0]+x_q[0]-5*x_r[0]);
    double c4=x_p[0]+x_q[0]+7*x_r[0]+7*x_s[0];
    double c5=3*x_p[0]+3*x_q[0]+x_r[0]+x_s[0];
    double c6=(x_p[0]+x_q[0])*(x_p[0]+x_q[0]);
    double c7=(x_r[0]+x_s[0])*(x_r[0]+x_s[0]);
    double c8=(x_p[0]+x_q[0]+3*x_r[0]+3*x_s[0]);

    double c9=(4*math::PI*sqrt(math::PI)*sigma_*sqrt(sigma_*sigma_+sigma_h_*sigma_h_)*
	       sqrt(3*sigma_*sigma_+2*sigma_h_*sigma_h_));

    double c=gk_x_p_x_q*gk_x_r_x_s/c9;

    /* printf("c1=%f\n",c1); */
/*     printf("c2=%f\n",c2); */
/*     printf("c3=%f\n",c3); */
/*     printf("c4=%f\n",c4); */
/*     printf("c5=%f\n",c5); */
/*     printf("c6=%f\n",c6); */
/*     printf("c7=%f\n",c7); */
/*     printf("c8=%f\n",c8); */
/*     printf("c9=%f\n",c9); */
/*     printf("c=%f\n",c); */
    
    //We shall need the values of sigma^4, sigma^2, sigma_h^2 and
    //sigma_h^4. Hence lets compute and store them

    double sigma_4=sigma_*sigma_*sigma_*sigma_;
    double sigma_h_4=sigma_h_*sigma_h_*sigma_h_*sigma_h_;
    double sigma_2=sigma_*sigma_;
    double sigma_h_2=sigma_h_*sigma_h_;


    double theta_x=0;
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

	double alpha=17*(x_i[0]*x_i[0])+17*(x_j[0]*x_j[0])-
	  2*x_i[0]*(x_j[0]+c1)+c2+c3-(2*x_j[0]*c4);

	double beta=(5*x_i[0]*x_i[0])-x_i[0]*(2*x_j[0]+c5)+
	  5*x_j[0]*x_j[0]+c6+c7-x_j[0]*c8;

	double gamma=(x_i[0]-x_j[0])*(x_i[0]-x_j[0]);
	
	theta_x=alpha*sigma_4+4*beta*sigma_h_2*sigma_2+4*gamma*sigma_h_4;
	theta_x*=-1/temp;
	k_prime_val+=pow(math::E,theta_x);
	//printf("alpha=%f,beta=%f,gamma=%f,theta_x=%f..\n",alpha,beta,gamma,theta_x);
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

    double m_val=(c*k_prime_val/sqd_num_train_points_)+lambda_*k_val;
    //printf("m_val=%f..\n",m_val);
    return m_val;
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
  
  /*  index_t CheckIfLastColumnIsZeros(index_t rank){
    
    double  EPSILON;
    EPSILON=pow(10,-10);

    index_t FAILURE=-1;
    index_t SUCCESS=1;

    double *last_chol;
    last_chol=chol_factor_.GetColumnPtr(rank-1);

    Vector v;
    v.Alias(last_chol,sqd_num_train_points_);

    printf("The last column is...\n");
    v.PrintDebug();

    for(index_t i=0;i<sqd_num_train_points_;i++){
      
      if(fabs(v[i])>EPSILON){

	return FAILURE;
      }
    }

    return SUCCESS;
    }*/
  
/*   void GetFirstColOfMMatrix_(Vector &first_col){ */

/*     printf("Squared number of train points are  %d\n",sqd_num_train_points_); */


/*     for(index_t p=0;p<sqd_num_train_points_;p++){ */
      
/*       double val=M_mat_.get(p,0); */
   
/*       first_col[p]=val;  */
/*     }   */
/*   } */
  
  
   void Compute(Matrix &chol_factor_in,ArrayList <index_t> &perm_in){
     
     /*   printf("Wil do incomplete cholesky now...\n");
     
     
      for(index_t p=0;p<num_train_points_;p++){
	
	for(index_t q=0;q<num_train_points_;q++){
	    
	  index_t row_num1=p*num_train_points_+q;
	  index_t row_num2=q*num_train_points_+p;
	    
	  for(index_t r=0;r<num_train_points_;r++){
	      
	    for(index_t s=0;s<num_train_points_;s++){
		
		index_t col_num1=r*num_train_points_+s;
		index_t col_num2=s*num_train_points_+r;
		
		double val=ComputeElementOfMMatrix_(p,q,r,s);

		printf("row_num1=%d,col_num1=%d..\n",row_num1,col_num1);
		
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
     M_mat_.PrintDebug(NULL,fp);
     printf("Computed the matrix also....\n");*/

     //Matrix temp;
     //temp.Init(train_set_.n_rows(),train_set_.n_cols());
     //temp.CopyValues(train_set_);
     //la::MulTransBOverwrite(temp,temp,&M_mat_);
     //printf("M matrix is...\n");
     //M_mat_.PrintDebug();
     ComputeIncompleteCholesky(chol_factor_in,perm_in);
   }
   
   void ComputeIncompleteCholesky(Matrix &chol_factor_in,
				  ArrayList<index_t> &perm_in){
     
     
     //The algorithms requires that we store the diagonal elements of
     //the matrix M in memory. Hence compute diagonal elements
     
     ComputeDiagonalElements_();

     // printf("The diagonal elements are...\n");
     //diagonal_elements_.PrintDebug();
     
     //Set the first column of chol factor to the first column of the
     //matrix M_mat_
     
     Vector first_col;
     first_col.Init(sqd_num_train_points_);
     
     GetFirstColOfMMatrix_(first_col);

     // printf("The first column is...\n");
     //first_col.PrintDebug();

     //Set the first column
     
     for(index_t j=0;j<sqd_num_train_points_;j++){
       
       chol_factor_.set(j,0,first_col[j]);
     }
     
     index_t i;
     
     for(i=0;i<sqd_num_train_points_;i++){
       
       if(i>=1){
	 
	 //printf("Will do subtraction....");   
	 Vector squared_vector;
	 squared_vector.Init(sqd_num_train_points_-i);
	 
	 //start from the i-1 th row and go till column i-1(including)
	 CreateSquaredVectorFromCholFactor_(i,i,squared_vector);
	 
	 for(index_t j=i;j<sqd_num_train_points_;j++){
	   
	   chol_factor_.set(j,j,diagonal_elements_[perm_[j]]-squared_vector[j-i]);
	}

	 //printf("Squared vector is ...\n");
	 //squared_vector.PrintDebug();
       }
       
       else{ //this is for i=0
	 
	 for(index_t j=0;j<sqd_num_train_points_;j++){
	   
	   chol_factor_.set(j,j,diagonal_elements_[j]);
	 }
	 
       }
       
       //printf("chol factor after initial subtraction is %d...\n",i);
       //chol_factor_.PrintDebug();
       
       if(SumOfDiagonalElementsOfCholFactor_(i)<=epsilon_){
	 
	break;
       }
       else{ //THis is for the continuation of the algorithm
	 
	 //Find the maximum diagonal element from i to m
	 
	 index_t max_index;
	 double max_value;
	 FindMaxDiagonalElementInCholFactor_(i,&max_index,&max_value);
	 //printf("The max index is %d..\n",max_index);
	 //	 printf("The starting element of the cholesky factor is %f,%f\n",chol_factor_.get(0,0));
	
	 
	 //Take care of the permutations
	 
	 //p[i] <-> p[max_index]
	
	index_t  temp=perm_[i];
	perm_[i]=perm_[max_index];
	perm_[max_index]=temp;
	
	//swap the contents of the rows i and q only upto columns
	//0:i-1(including) Which is basically the lower traingle of the
	//row i
	
	index_t row1=i;
	index_t row2=max_index;
	index_t tillcol=i;
	SwapRowsOfCholFactor_(row1,row2,tillcol);

	//printf("Before taking square root the mnax value %d is %f..\n",i,max_value);
	max_value=sqrt(max_value);

	//printf("Max value %d is %f\n",i,max_value);

	chol_factor_.set(i,i,max_value);
	
	Vector temp1;
	temp1.Init(sqd_num_train_points_-i-1);
	
	index_t start_index_of_perm_vector=i+1;
	//printf("Permutation matrix at this point is..\n");
	/*for(index_t q=i;q<sqd_num_train_points_;q++){

	  printf("Permutation[%d]=%d",q,perm_[q]);
	  }*/
	printf("\n...");
	ExtractRowOfOriginalMatrix_(i,start_index_of_perm_vector,temp1);

	//printf("Extracted matrix is..\n");
	//temp1.PrintDebug();
	
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
	    
	    temp2[j]=chol_factor_.get(i,j);
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
	      
	      chol_factor_.set(row,i,temp4[k]/max_value);
	      k++;	   
	    }

	    //printf("Chol factor after final  subtraction is i=%d...\n",i);
	    //chol_factor_.PrintDebug();
	  }
	  
	  else{
	    
	    index_t k=0;
	    for(index_t row=i+1;row<sqd_num_train_points_;row++){
	      
	      chol_factor_.set(row,i,temp1[k]/max_value);
	      k++;	   
	    }
	    //printf("The subtractant is %d.......\n",i);
	    //printf("0000000000 vector........\n");
	    
	    //printf("Chol factor after final subtraction is i=%d...\n",i);
	    //chol_factor_.PrintDebug();
	  }
	}
	///...................
	else{//This happens when i=0

	  
	  //Fill this up
	  index_t k=0;
	  
	  for(index_t row=i+1;row<sqd_num_train_points_;row++){
	    
	    
	    chol_factor_.set(row,i,temp1[k]/max_value);
	    k++;	   
	  }

	  // printf("The part from the original matrix is..\n");
	  //temp1.PrintDebug();

	  //printf("The subtractant is  %d.......\n",i);
	  //printf("00000000000 vector.....\n");

	  //printf("Chol factor after final subtraction %d..\n",i);
	  //chol_factor_.PrintDebug();
	}
       }
     }
     
     //END OF THE ALGORITHM...............
    
     //Determine the rank of the matrix
     
     index_t rank;
     if(i<=sqd_num_train_points_-1){
       
       rank=i+1;
     }
     else{
       
       rank=sqd_num_train_points_;
     }

     //printf("The rank of the matrix M is %d..\n",rank);
     
     //It can happen that the last column is full of 0's in which case
     //we just chop off the column.

     
     /*index_t flag=CheckIfLastColumnIsZeros(rank);

     if(flag==1){

       
       //Copy the sliced cholesky factor and the permutation arraylist
       chol_factor_.MakeColumnSlice(0,rank-1,&chol_factor_in);
     }
     else{

       //Copy the sliced cholesky factor and the permutation arraylist
       chol_factor_.MakeColumnSlice(0,rank,&chol_factor_in);
       }*/

     chol_factor_.MakeColumnSlice(0,rank,&chol_factor_in);

     for(index_t i=0;i<sqd_num_train_points_;i++){

       perm_in[i]=perm_[i];
     }

     //printf("Cholesky factor is..\n");
     // chol_factor_in.PrintDebug();

     //make sure all calculations are fine

     /*     Matrix PL;
     special_la::PreMultiplyMatrixWithPermutationMatrixInit(perm_,chol_factor_,&PL);
     
     Matrix chol_factor_trans,PL_chol_factor_trans;
     la::TransposeInit(chol_factor_,&chol_factor_trans);
     la::MulInit(PL,chol_factor_trans,&PL_chol_factor_trans); //PLL^T
     
     ArrayList<index_t> perm_in_trans;
     special_la::PermutationMatrixTransposeInit(perm_,&perm_in_trans); //P^T
     
     Matrix PL_chol_factor_trans_P_trans;
     special_la::PostMultiplyMatrixWithPermutationMatrixInit(PL_chol_factor_trans,perm_in_trans,
     &PL_chol_factor_trans_P_trans); //PLL^TP^T
     
     printf("Post multiplication with permutation matrix done..\n");
     
     Matrix temp1;
     la::SubInit(PL_chol_factor_trans_P_trans,M_mat_,&temp1);
     
     double trace=0;
     for(index_t i=0;i<sqd_num_train_points_;i++){
     
     trace+=temp1.get(i,i);
     }
     
     for(index_t i=0;i<sqd_num_train_points_;i++){
     
     printf("Permutation is %d..\n",perm_[i]);
     }
     printf("Trace is %15f....\n",trace);*/
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


    printf("Will allocated memory for cholesky factorization...\n");
    printf("Number of training points are...%d\n",num_train_points_);
    chol_factor_.Init(sqd_num_train_points_,sqd_num_train_points_);
    chol_factor_.SetAll(0);
    printf("Have been able to successfully allocated memory for the cholesky matrix...\n");
    diagonal_elements_.Init(sqd_num_train_points_);
    

    //Initialize the permutation vector in a serial order

    perm_.Init(sqd_num_train_points_);

    for(index_t i=0;i<sqd_num_train_points_;i++){

      perm_[i]=i;      
    }

    epsilon_=0.0000001;

    num_dims_=train_set_.n_rows();
    //M_mat_.Init(sqd_num_train_points_,sqd_num_train_points_);
  }

};
#endif

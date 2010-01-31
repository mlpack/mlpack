//This .h file contains some commonly used hyperkernel functions
#ifndef HYPERKERNELS_H
#define HYPERKERNELS_H
class GaussianHyperKernel{
 private:
  double  sigma_h_; 
  double sigma_;
  index_t num_dims_;
  GaussianKernel gk_inter_;
  GaussianKernel gk_intra_;

 public:
  void Init(double sigma,double sigma_h,int num_dims){
    
    //Initialize the parameters
    sigma_h_=sigma_h;
    sigma_=sigma;

    //printf("In hyperkernel class sigma_h_=%f\n",
    //   sigma_h_);
    //printf("sigma is %f\n",sigma);
    num_dims_=num_dims;

    //Initialize the gaussian kenels

    //THIS HAS TO CHANGE ACCORDINGLY AS PRODUCT KERNEL FOR
    //MULTIDIMENSIONAL CASE
    gk_intra_.Init(sigma*sqrt(2),num_dims_);

    double sqrt_sum_sqd_bw=sqrt(sigma_h_*sigma_h_+sigma_*sigma_);
    gk_inter_.Init(sqrt_sum_sqd_bw,num_dims_);
  }
  
  double CalcNormConstant(){
    //Calculate the normalization constant
    
      double norm_const1=gk_inter_.CalcNormConstant(num_dims_);
      //printf("norm constant1 is %f..\n",norm_const1);

      double norm_const2=gk_intra_.CalcNormConstant(num_dims_);
      // printf("Norm const2 is %f..\n",norm_const2);

      double norm_const=norm_const2*norm_const2*norm_const1;
      //printf("The normalization constant is %f\n",norm_const);
      return norm_const;
  }

  //Calculate the partial normalization constant. This is the
  //normalization constant

  double CalcNormConstantpartial1(){
    //Calculate the normalization constant
    
    double norm_const1=gk_inter_.CalcNormConstant(num_dims_);
    double norm_const2=gk_intra_.CalcNormConstant(num_dims_);
    double norm_const=norm_const1*norm_const2;
    
    //printf("The normalization constant is %f\n",norm_const);
    return norm_const;
  }
  
  double EvalUnnorm(Vector &x_p, Vector &x_q,Vector &x_r,Vector &x_s){
    
    //THIS WILL Handle even multi-dimensional case
    index_t flagpq=0;
    index_t flagrs=0;
    double unnorm_val1;
    double unnorm_val2;
    double unnorm_val3;

    Vector mean_x_p_x_q;
    Vector mean_x_r_x_s;

    if(x_p.ptr()==x_q.ptr()){

      unnorm_val1=1;
      flagpq=1;
      mean_x_p_x_q.Alias(x_p);
    }
    else{
      
      double sqd_distance=la::DistanceSqEuclidean(x_p,x_q);
      unnorm_val1=gk_intra_.EvalUnnormOnSq(sqd_distance);
      //mean_x_p_x_q <-(x_p+x_q)/2
      la::AddInit(x_p,x_q,&mean_x_p_x_q);
      la::Scale(0.5,&mean_x_p_x_q);
    }
    if(x_r.ptr()==x_s.ptr()){
      unnorm_val2=1;
      flagrs=1;
      mean_x_r_x_s.Alias(x_r);
    }
    else{ 
      double sqd_distance=la::DistanceSqEuclidean(x_r,x_s);      
      unnorm_val2=gk_intra_.EvalUnnormOnSq(sqd_distance);
      la::AddInit(x_r,x_s,&mean_x_r_x_s);
      la::Scale(0.5,&mean_x_r_x_s);
    }
    if(flagpq==1&&flagrs==1&&(x_p.ptr()==x_r.ptr())){
      
      //All the points are the same
      unnorm_val3=1;

      return unnorm_val1*unnorm_val2*unnorm_val3;
    }
    else{       
      if((x_p.ptr()==x_r.ptr())&&(x_q.ptr()==x_s.ptr())){

	//Mean of p,q =Mean of r,s	
	
	unnorm_val3=1;
	return unnorm_val1*unnorm_val2;
      }

      double sqd_distance=la::DistanceSqEuclidean(mean_x_p_x_q,mean_x_r_x_s);
      unnorm_val3=gk_inter_.EvalUnnormOnSq(sqd_distance);
      double hyperkernel_val=
	unnorm_val1*unnorm_val2*unnorm_val3;
      return hyperkernel_val;
    }
    
    return -1; //error statement
  }

  double EvalUnnorm(index_t num_dim,double *x_p, double *x_q,double *x_r,double *x_s){
    
    Vector vec_x_p;
    Vector vec_x_q;
    Vector vec_x_r;
    Vector vec_x_s;

    vec_x_p.Alias (x_p,num_dim);
    vec_x_q.Alias(x_q,num_dim);
    vec_x_r.Alias(x_r,num_dim);
    vec_x_s.Alias(x_s,num_dim);
    double val;
    val=EvalUnnorm(vec_x_p,vec_x_q,vec_x_r,vec_x_s);
    return val;
  }


  //This is a special function that has been created only to optimize
  //calculations. Here the hyperkernel will be calculated as the
  //product of kernels on r,s and between the mean of the points
  
  double EvalUnnormPartial1(Vector &x_p, Vector &x_q,Vector &x_r,Vector &x_s){
    
    //THESE HAVE TO CHANGE IN ORER TO ACCOMODATE FOR MULTIPLICATIVE
    //KERNELS
    double unnorm_val2;
    double unnorm_val3;

    Vector mean_x_p_x_q;
    Vector mean_x_r_x_s;

    /* if(x_p.ptr()==x_q.ptr()){

      unnorm_val1=1;
      flagpq=1;
      mean_x_p_x_q.Alias(x_p);
      }*/
      
    //mean_x_p_x_q <-(x_p+x_q)/2
    la::AddInit(x_p,x_q,&mean_x_p_x_q);
    la::Scale(0.5,&mean_x_p_x_q);
    
    if(x_r.ptr()==x_s.ptr()){
      unnorm_val2=1;
      mean_x_r_x_s.Alias(x_r);
    }
    else{ 
      double sqd_distance=la::DistanceSqEuclidean(x_r,x_s);      
      unnorm_val2=gk_intra_.EvalUnnormOnSq(sqd_distance);
      //printf("Sqd distance between other pairs is %f..\n",sqd_distance);

      //Calculate the mean
      la::AddInit(x_r,x_s,&mean_x_r_x_s);
      la::Scale(0.5,&mean_x_r_x_s);
    }
    double sqd_distance=la::DistanceSqEuclidean(mean_x_p_x_q,mean_x_r_x_s);
    
    unnorm_val3=gk_inter_.EvalUnnormOnSq(sqd_distance);
    double hyperkernel_val=unnorm_val2*unnorm_val3;
    // printf("gaussian between other values  is %f..\n",unnorm_val2);
    return hyperkernel_val;
  }


  double EvalUnnormPartial1(index_t num_dim,double *x_p, double *x_q,double *x_r,double *x_s){
    
    Vector vec_x_p;
    Vector vec_x_q;
    Vector vec_x_r;
    Vector vec_x_s;
    
    vec_x_p.Alias (x_p,num_dim);
    vec_x_q.Alias(x_q,num_dim);
    vec_x_r.Alias(x_r,num_dim);
    vec_x_s.Alias(x_s,num_dim);
    double val=EvalUnnormPartial1(vec_x_p,vec_x_q,vec_x_r,vec_x_s);
    return val;
 }
};
#endif

#ifndef NAIVE_KDE_LOCAL_POLYNOMIAL_H
#define NAIVE_KDE_LOCAL_POLYNOMIAL_H
#include "fastlib/fastlib.h"
#include "vector_kernel.h"

#define EPSILON pow(10,-10)
template <typename MyTKernel> class NaiveKdeLP{

  FORBID_ACCIDENTAL_COPIES(NaiveKdeLP);

 private:

  /* The query set */
  Matrix qset_;

  /* The reference set */
  Matrix rset_;

  /* The kernel which i shall be using */
  MyTKernel my_kernel_;

  /*The vector of densities */
  Vector densities_;

  /** A matrix of bandwidths. Essentialy this is a vector, containing
      bandwidth in each direction*/

  Vector bandwidth_;

  /** A plugin bandwidth */
  double plugin_bandwidth_;

  //Get the vector diff between vec1 and vec2 and return the result in
  //the vector diff


  void GetVectorDifference_(double *vec1, double *vec2,index_t length,
			   Vector &diff){
    
    for(index_t l=0;l<length;l++){

      diff[l]=vec1[l]-vec2[l]; 
    }

  }


 public:

  //Constructors and Destructors

  NaiveKdeLP(){

  }

 ~NaiveKdeLP(){

  }

 ///Getters and Setters /////////


 void get_density_estimates(Vector &results){

   results.CopyValues(densities_);
 }

 //The foreign vector bandwidths has already been initialized
 
 void get_bandwidths(Vector &bandwidths){

   bandwidths.CopyValues(bandwidth_);
 }

 ////////User level functions////////////////

 private:

 double CalculateSigmaHat_(){

   //Calculate the sample variance of the reference set

   double sum=0;
   for(index_t i=0;i<rset_.n_cols();i++){

     sum+=rset_.get(0,i);
   }

   double mean=sum/rset_.n_cols();
  
   //Now calculate the variance

   double sqd_diff=0;
   for(index_t i=0;i<rset_.n_cols();i++){

     sqd_diff+=pow(rset_.get(0,i)-mean,2);
   }
   return sqrt(sqd_diff/rset_.n_cols());
 }

 double FindPluginBandwidth_(){

   //h=1.06*sigma_hat*n^-0.2
   double sigma_hat=CalculateSigmaHat_();
   return 1.06*sigma_hat*pow(rset_.n_cols(),-0.2);
 }

 bool IsSamePoint_(Vector &diff){

   for(index_t i=0;i<diff.length();i++){
     if(fabs(diff[i])>EPSILON){

       return false;
     }
   }
  
   return true;
 }


 public:

 void Compare(Matrix &density, Vector &boundary_points){
   
   //For each query point compare the density by local polynomial with
   //the given density
   
   double mean_sqd_error=0.0;
   double boundary_mean_sqd_error=0;
   index_t number_of_boundary_points=0;
   
   for(index_t q=0;q<qset_.n_cols();q++){
     
     mean_sqd_error+=pow(density.get(0,q)-densities_[q],2);
     
     //Calculate boundary error in case of 1-d dataset
     
     if(qset_.n_rows()==1){
       if(qset_.get(0,q)<boundary_points[1]||
	  qset_.get(0,q)>boundary_points[2]){
	 boundary_mean_sqd_error+=pow(density.get(0,q)-densities_[q],2);
	 number_of_boundary_points++;
       }
     }
   }
 
   mean_sqd_error/=qset_.n_cols();
   boundary_mean_sqd_error/=number_of_boundary_points;
   double root_mean_sqd_error=sqrt(mean_sqd_error);
   double boundary_root_mean_sqd_error=sqrt(boundary_mean_sqd_error);

   printf("RESULTS FOR LOCAL POLYNOMIAL EXPANSION.....\n");

   printf("Total root mean squared error is %f\n",root_mean_sqd_error);
   printf("Boundary root mean squared error is %f\n",
	  boundary_root_mean_sqd_error);
   printf("Number of boundary points are %d\n",number_of_boundary_points);
 }

 void Compute(){

   //We use a plugin bandwidth for the univariate case 

   if(qset_.n_rows()==1&&bandwidth_[0]<0){
     
     //This means the plugin bandwidth needs to be used
     plugin_bandwidth_=FindPluginBandwidth_();
     bandwidth_.SetAll(plugin_bandwidth_);
     printf("Plugin bandwidth as calculated is %f\n",plugin_bandwidth_);
   }
   else{
     printf("Bandwidth was already set and we shall use this value..\n");
   }
   
   Vector diff;
   diff.Init(rset_.n_rows());
   double density_val=0;
   
   
   for(index_t q=0;q<qset_.n_cols();q++){
     
     //get the query point
     double *q_col=qset_.GetColumnPtr(q);
     density_val=0;
     index_t number_of_contributions=0;
     
     //printf("Q is %d...............\n",q);
     for(index_t r=0;r<rset_.n_cols();r++){
       
       //Get the reference point
       double *r_col=rset_.GetColumnPtr(r);
       
       
       //Get the vector difference of the reference and the query  point
       
       GetVectorDifference_(r_col,q_col,rset_.n_rows(),diff);
      
       //Contribution of this reference point
       if(!IsSamePoint_(diff)){
	 double contrib=my_kernel_.EvalUnnormOnVectorDiff(diff);
     
	 density_val+=contrib;
	 number_of_contributions++;
       }
       
     }
     
     
     //Since we calculated the unnormalized value we normalize it with
     //the normalization constant. 
     
     //Divide by the normalizing constant
     //printf("Unnormalized density is %f\n",density_val);
     density_val/=my_kernel_.EvalNormConstant(rset_.n_rows(),
					      number_of_contributions);
     //printf("Norm const is %f\n",
     //	    my_kernel_.EvalNormConstant(rset_.n_rows(),
     //number_of_contributions));
     densities_[q]=density_val;
     //printf("Density estimate of q=%d is %f\n",q,density_val);
   }
 }

 void Init(Matrix &query_set,Matrix &ref_set,double *bwidth,index_t len){

   //Copy the query and the reference sets

   qset_.Init(query_set.n_rows(),query_set.n_cols());
   rset_.Init(ref_set.n_rows(),ref_set.n_cols());

   bandwidth_.Init(len); //set the length of the bandwidth vector
   bandwidth_.CopyValues(bwidth);


   qset_.CopyValues(query_set);
   rset_.CopyValues(ref_set);

   //A vector of bandwidths(one in each direction) for densiy
   //estimation calculations

   bandwidth_.CopyValues(bwidth);

   printf("BANDWIDTH BEING USED FOR LOCAL POLYNOMIAL EXPANSION IS\n");
   bandwidth_.PrintDebug();

   //initialize the kernel

   my_kernel_.Init(bandwidth_);

   //Initialize densities
   densities_.Init(qset_.n_cols());
   densities_.SetZero();

 }
 /*Initialize the object's bandwidth parameter only. This is useful
   when doing cross validation on the same datasets
  */

 void InitBandwidth(Vector &bwidth){

   //bandwidth_ has already been initialized
   printf("Came to set up the bandwidth...\n");
   bandwidth_.CopyValues(bwidth);
   printf("Bandwidth set up...\n");
 }

 void InitBandwidth(double *bandwidth){
   bandwidth_.CopyValues(bandwidth);

 } 
};
#endif
 

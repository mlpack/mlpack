#ifndef AVERAGE_OVER_DATASETS_H
#define AVERAGE_OVER_DATASETS
#include "fast_llde_multi.h"
#include "fastlib/fastlib_int.h"
#include "string.h"
#include "naive_kde.h"


//This particular utility is very useful if u want to run a program
//multiple number of times with the same query set but different
//reference sets and fianlly average out the density contribution. I
//wanted it ot be an independent component which will call the
//required root function to get the job done

class AverageOverDatasets{
  
 private:

  //The query set
  Matrix qset_;

  //The reference set
  Matrix rset_;

  //The bandwidth

  double bandwidth_;

  //The vector of total densities

  Vector total_densities_;

  Vector naive_kde_densities_;

  //NUmber of datasets

  index_t num_of_datasets_;
  
 public:
 

  void Compute(){
    
    char buff[10];
    char temp[10];
    ArrayList<index_t> old_from_new_q;
    old_from_new_q.Init(qset_.n_cols());

    Vector temp_naive_kde;
    temp_naive_kde.Init(qset_.n_cols()); //scratch variable to store densities
			       //fof naive kde calculations

    for(index_t i=0;i<num_of_datasets_;i++){

      //////FIRST GET THE NAME OF THE REFERENCE FILE
      strcpy(temp,"ref");
      index_t succ=sprintf(buff,"%d",i+1);
      strcat(temp,buff);

      //Read the reference datasets

      const char *ref_file_name=fx_param_str_req(NULL,temp);
      data::Load(ref_file_name,&rset_);
      /////////////////////////////////////////////////////


      //Lets create an object of type FastLLDEMULTI 

      FastLLDEMulti fast_llde_multi;
      fast_llde_multi.Init(qset_,rset_,bandwidth_);
      fast_llde_multi.Compute();

      Vector densities;
      fast_llde_multi.get_permuted_density_estimates(densities);

      if(i==0){

	//Only for the first time get the permutation order
	fast_llde_multi.get_old_from_new_q(old_from_new_q);
      }

      //Add the density estimates obtained by using this reference set
      //to densities_
      la::AddTo(densities,&total_densities_);

      printf("REFERNCE SET IS....%d\n",i);
     
    
      printf("Will run naive kde..\n");
      printf("qset is..\n");
      qset_.PrintDebug();

      printf("rset is...\n");
      rset_.PrintDebug();
      //Similarily get naive kde estimates also....
      NaiveKde<EpanKernel> naive_kde;
      naive_kde.Init(qset_,rset_,bandwidth_);
      printf("Initialized/....\n");
      naive_kde.Compute();
      printf("Will add..\n");
      naive_kde.get_density_estimates(temp_naive_kde);
      la::AddTo(temp_naive_kde,&naive_kde_densities_);


      //Destroy the reference set, fast_llde_multi and densities, for
      //the new iteration
      
      fast_llde_multi.Destruct();
      rset_.Destruct();

      //There is no need to destroy densites because we already have
      //destroyed fast_llde_multi and have used the getter function to
      //get densities as a result aliasing densities to the class
      //member of fast_llde_multi

    }

    // In order to avoid the complaint of rset not having been
    // initialized
    rset_.Init(1,1); 


    //Note that the vector total_densities is in a permuted order due
    //to the tree building process. Hence depermute it back

    Vector depermuted;
    depermuted.Init(qset_.n_cols());

    for(index_t q=0;q<qset_.n_cols();q++){

      depermuted[old_from_new_q[q]]=total_densities_[q];

    }

    //Write back the contents of temp to total_densities
    total_densities_.CopyValues(depermuted);
    
    //To get the average over reference sets divide the vector
    //total_densities with the number of datasets used

    for(index_t q=0;q<qset_.n_cols();q++){

      total_densities_[q]/=num_of_datasets_;
      naive_kde_densities_[q]/=num_of_datasets_;
    }

    for(index_t q=0;q<qset_.n_cols();q++){
      
      printf("total_density[%d]=%f ",q,total_densities_[q]);
      
    }

    for(index_t q=0;q<qset_.n_cols();q++){
      
      printf("naive_kde_density[%d]=%f ",q,naive_kde_densities_[q]);
      
    }
  }
  

  void PrintDebug(){

    FILE *fp;
    FILE *gp;

    fp=fopen("avod_fast_local_likelihood_multi.txt","w+");
    gp=fopen("avod_naive_kde.txt","w+");
     for(index_t q=0;q<qset_.n_cols();q++){
      
      fprintf(fp,"%f\n",total_densities_[q]);
      }

       for(index_t q=0;q<qset_.n_cols();q++){
      
      fprintf(gp,"%f\n",naive_kde_densities_[q]);

      }
       fclose(fp);
       fclose(gp);
  }
 
  void Init(Matrix &qset,double bandwidth,index_t num_of_datasets){
    
    //Copy the query set    
    qset_.Alias(qset);

    //Copy the bandwidth
    
    bandwidth_=bandwidth;

    //The number of datasets over which the averaging will take place
    num_of_datasets_=num_of_datasets;

    //initialize total_densities

    total_densities_.Init(qset_.n_cols());
    total_densities_.SetZero();

    naive_kde_densities_.Init(qset_.n_cols());
    naive_kde_densities_.SetZero();
  } 
};
#endif

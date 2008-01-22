/** 
 * @file allknn_main.cc
 * This file contains code to do solve the standalone NN problem
 * The code takes arguments 
 * method--> the method which we want to use
 * qfile --> query file
 * rfile --> reference file
 * k --> number of nearest neighbours
 *
 * @see allknn.h
 */


#include "allknn.h"

int main(int argc, char *argv[])
{ 
  fx_init(argc, argv);

  //Get the method from the user

  const char *method=fx_param_str_req(NULL,"method");
  
  if(strcmp(method,"allnnnaive")==0)
    {
      
      //Get the datafiles
      const char *qfile=fx_param_str_req(NULL,"qfile");
      const char *rfile= fx_param_str_req(NULL,"rfile");
      
 
      //q_matrix_ will hold the query file in matrix format 
      //r_matrix_ will hold the reference file in matrix format
      
      Matrix q_matrix;
      Matrix r_matrix;

      //Load the datafiles
      data::Load(qfile,&q_matrix);
      data::Load(rfile,&r_matrix);
      
      //Declare an object and initialize it by calling the init
      //function. Then it calls the ComputeAllNNNaive 
      AllNNNaive naive;
      naive.Init(q_matrix,r_matrix);
      naive.ComputeAllNNNaive();
      naive.PrintResults();
      
    }
  
  if(strcmp(method,"allnnsingletree")==0)
    {
      
      const char *qfile=fx_param_str_req(NULL,"qfile");
      const char *rfile=fx_param_str_req(NULL,"rfile");
      
      Matrix q_matrix,r_matrix;

      //Load datasets
      data::Load(qfile,&q_matrix);
      data::Load(rfile,&r_matrix); 
  
      AllNNSingleTree ast;
      ast.Init(q_matrix,r_matrix);     
      ast.ComputeAllNNSingleTree();
      //ast.get_results();
      ast.PrintResults();
    } 
  
  if(strcmp(method,"allknnsingletree")==0)
    {
      const char *qfile=fx_param_str_req(NULL,"qfile");
      const char *rfile=fx_param_str_req(NULL,"rfile");
      
      Matrix q_matrix,r_matrix;

      //Load datasets
      data::Load(qfile,&q_matrix);
      data::Load(rfile,&r_matrix); 

      index_t k=fx_param_int_req(NULL,"k");
   
      AllKNNSingleTree akst;
      akst.Init(q_matrix,r_matrix,k);
      akst.ComputeAllKNNSingleTree();
      //akst.get_results();
      akst.PrintResults();
    }
     
  if(strcmp(method,"allknndualtree")==0)
    {
      const char *qfile=fx_param_str_req(NULL,"qfile");
      const char *rfile=fx_param_str_req(NULL,"rfile");
      
      Matrix q_matrix,r_matrix;

      //Load datasets
      data::Load(qfile,&q_matrix);
      data::Load(rfile,&r_matrix); 

      //get calue of k
      index_t k=fx_param_int_req(NULL,"k");
   
      //AllKNNDual tree computations
      AllKNNDualTree akdt;
      akdt.Init(q_matrix,r_matrix,k);
      akdt.ComputeAllKNNDualTree();
      akdt.PrintResults();
    }
  
     
  fx_done();
}



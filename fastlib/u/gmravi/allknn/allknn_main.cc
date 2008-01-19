#include "allknn.h"
int main(int argc, char *argv[])
{ 
  fx_init(argc, argv);
  char method[40];
  strcpy(method,fx_param_str_req(NULL,"method"));
  printf("method is %s\n",method);
  printf("string comparing with allnnnaive is %d\n",strcmp(method,"allnnnaive"));
  if(strcmp(method,"allnnnaive")==0)
    {
      printf("In allnnnaive");
      AllNNNaive naive;
      naive.Init();
      naive.ComputeAllNNNaive();
      naive.PrintResults();
     
    }
  
  if(strcmp(method,"allnnsingletree")==0)
    {
      //AllNNSingleTree computations
      printf("In allnnsingletree");
      AllNNSingleTree ast;
      ast.Init();     
      ast.ComputeAllNNSingleTree();
      //ast.get_results();
      ast.PrintResults();
    } 
  
  if(strcmp(method,"allknnsingletree")==0)
    {
      printf("in allknnsingletree\n");
      //AllKNNSingle tree computations
	 
      
      AllKNNSingleTree akst;
      akst.Init();
      akst.ComputeAllKNNSingleTree();
      //akst.get_results();
      akst.PrintResults();
    }
     
  if(strcmp(method,"allknndualtree")==0)
    {
      printf("In allknndualtree");
      //AllKNNDual tree computations
      AllKNNDualTree akdt;
      akdt.Init();
      akdt.ComputeAllKNNDualTree();
      akdt.PrintResults();
    }
  
     
  fx_done();
}



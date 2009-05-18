#ifndef DO_MCF_H
#define DO_MCF_H

#define PRIMAL
// Note: might have to add more include files when #define'ing DUAL

#include "mcf/mcfdefs.h"
#include "mcf/readmin.c"
#include "mcf/mcfutil.c"
#include "mcf/pbeampp2.c"
#include "mcf/pstart.c"
#include "mcf/pbla.c"
#include "mcf/treeup.c"
#include "mcf/pflowup.c"
#include "mcf/psimplex.c"
#include "mcf/output.c"


int DoMCF(const char* problem_filename, const char* solution_filename) {
  MCF_network_t net;
  
  
  long stat = 0;

  memset((void *)(&net), 0, (size_t)(sizeof(MCF_network_t)));
  stat = MCF_read_dimacs_min( problem_filename, &net);
  if(stat) {
    return -1;
  }
  

#ifdef PRIMAL

  //if( net.m < 10000 )       net.find_bea = MCF_primal_bea_mpp_30_5;
  //else if( net.m > 100000 ) net.find_bea = MCF_primal_bea_mpp_200_20;
  /*else*/                      net.find_bea = MCF_primal_bea_mpp_50_10;

  MCF_primal_start_artificial(&net);
  MCF_primal_net_simplex(&net);

#elif defined DUAL

  //if(net.n < 10000) net.find_iminus = MCF_dual_iminus_mpp_30_5;
  /*else*/                net.find_iminus = MCF_dual_iminus_mpp_50_10;

  dual_start_artificial(&net);
  dual_net_simplex(&net);

#endif   
  
    
  printf("\n%s: %ld nodes / %ld arcs\n", problem_filename, (net.n), (net.m));
  if(net.primal_unbounded) {
    printf("\n   >>> problem primal unbounded <<<\n");
    return -1;
  }
  
  if(net.dual_unbounded) {
    printf("\n   >>> problem dual unbounded <<<\n");
    return -1;
  }
    
  if(net.feasible == 0) {
    printf("\n   >>> problem infeasible or unbounded <<<\n");
    return -1;
  }
    
    
  printf("Iterations                        : %ld\n", net.iterations);
  net.optcost = MCF_primal_obj(&net);
  if(MCF_ABS(MCF_dual_obj(&net) - net.optcost) > (double)MCF_ZERO_EPS)
    printf("NETWORK SIMPLEX: primal-dual objective mismatch!?\n");
#ifdef MCF_FLOAT
  printf("Primal optimal objective          : %10.6f\n",
	  MCF_primal_obj(&net));
  printf("Dual optimal objective            : %10.6f\n", 
	  MCF_dual_obj(&net));
#else
  printf("Primal optimal objective          : %10.0f\n", 
	  MCF_primal_obj(&net));
  printf("Dual optimal objective            : %10.0f\n", 
	  MCF_dual_obj(&net));
#endif
  
  printf("Write solution to file %s\n", solution_filename);
  MCF_write_solution(problem_filename, solution_filename, &net, 0);

  MCF_free(&net); 
  
  return 1;
}

#endif /* DO_MCF_H */

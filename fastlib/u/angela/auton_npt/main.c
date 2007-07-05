/*
   File:        main.c
   Author:      Andrew W. Moore
   Created:     May 2000

   Copyright 2000, The Auton Lab
*/

#include "npt.h"
#include "ammarep.h"
#include "standard.h"
#include "command.h"

extern double Verbosity;

extern bool   Use_Npt2;           /* AG */
extern bool   Use_Npt3;           /* AG */
extern bool   Use_MC;             /* AG */
extern int    Projection;         /* AG */
extern double Eps;                /* AG */
extern double Sig;                /* AG */
extern double Force_p;            /* AG */
extern double Union_p;            /* AG */
extern double Nsamples_block;     /* AG */
extern double Datafrac_crit;      /* AG */
extern double Rerrfrac_crit;      /* AG */
extern int    Num_to_expand;      /* AG */
extern int    Start_secs;         /* AG */

/* Initialize debugging info */
double total_num_inclusions = 0;
double total_num_exclusions = 0;
double total_num_recursions = 0;
double total_num_matches = 0;
double total_num_mismatches = 0;
double max_num_matches = 0;
double total_num_base_cases = 0;
double total_num_iterative_base_cases = 0;
double total_num_missing_ntuples = 0;
double sum_total_ntuples = 0;
double theoretical_total_ntuples = 0;

int iterative = 0;
/* End debugging section */

int main(int argc,char *argv[])
{
  char *s = (argc < 2) ? "help" : argv[1];
  int winsize = int_from_args("winsize",argc,argv,400);
  Verbosity = double_from_args("verbose",argc,argv,0.0);
  Use_Npt2 = bool_from_args("npt2",argc,argv,FALSE);
  Use_Npt3 = bool_from_args("npt3",argc,argv,FALSE);
  Use_MC = bool_from_args("mc",argc,argv,FALSE);
  Sig = double_from_args("stdevs",argc,argv,3.0);
  Eps = double_from_args("errfrac",argc,argv,0.01);
  Force_p = double_from_args("forcep",argc,argv,0.2);
  Union_p = double_from_args("unionp",argc,argv,-1);
  Nsamples_block = double_from_args("nsamples",argc,argv,50000);
  Datafrac_crit = double_from_args("datafrac",argc,argv,0.999);
  Rerrfrac_crit = double_from_args("rerrfrac",argc,argv,1000);
  Num_to_expand = int_from_args("expand",argc,argv,-1);
	iterative = bool_from_args("iterative",argc,argv,FALSE) ? 1 : 0;

  ag_window_shape(winsize,winsize);

  memory_leak_check_args(argc,argv);

  if ( eq_string(s,"npt") )
  {
    void npt_main(int argc,char *argv[]);
    npt_main(argc,argv);
  }
  else if ( eq_string(s,"m2p") )
  {
    void m2p_main(int argc,char *argv[]);
    m2p_main(argc,argv);
  }
  else if ( eq_string(s,"trial") )
  {
    void trial_main(int argc,char *argv[]);
    trial_main(argc,argv);
  }
  else
  {
    printf("%s not recognized.  Use %s [program]\n",s,argv[0]);
    printf("program may be: npt, m2p, or trial\n");
  }
#define AMFAST 1
#ifndef AMFAST
  wait_for_key();
  am_malloc_report();
  really_wait_for_key();
#endif
  return 0;
}


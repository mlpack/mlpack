#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <malloc.h>
#include "ambs.h"
#include "ammarep.h"
#include "cp_args.h"
#include "cp_utils.h"
#include "hrect.h"
#include "mrkd.h"
#include "taylor.h"
#include "batree.h"
#include "kde.h"
#include "regression.h"

void print_header();
void print_help();

int main(int argc,char *argv[])
{
  arg args[num_kde_args+1];
  char *trainfile,*testfile,*dtargetfile,*qtargetfile,*dwgtsfile,*dbwsfile,
    *trainbase;
  char *default_basename,*basename,*logfile=0; bool uselog;
  int numbws; double bwmin, bwmax; bool multi; 
  dym *q = NULL, *d = NULL; /* test dataset, training dataset */
  dyv *w = NULL; /* weights or regression value */
  mrpars *tmp = NULL;
  bwinfo *bws = NULL;
  int allocsize,allocsize2;

  /* Initialize the arguments */
  memory_leak_check_args(argc,argv);
  memset(args, 0x0, sizeof(arg) * (num_kde_args + 1));
  init_args(args);

  /* Get the arguments */
  if (index_of_arg("help",argc,argv) != -1) { 
    LOG = stdout; print_header(); print_help(); 
    print_usage("kde", args, num_kde_args); return 0; 
  }
  if (get_args(argc-1,argv,args,num_kde_args) != 0) return -1;

  trainfile        = str_in_args(Arg_Trainfile,args,"sdss10k.fds");
  testfile         = str_in_args(Arg_Testfile,args,trainfile);
  SELFCASE         = eq_string(testfile,trainfile);
  dtargetfile      = str_in_args(Arg_Train_targetfile,args,"");
  qtargetfile      = str_in_args(Arg_Test_targetfile,args,"");
  dwgtsfile        = str_in_args(Arg_Train_wgtsfile,args,"");
  dbwsfile          = str_in_args(Arg_Train_bwsfile,args,"");

  trainbase = AM_MALLOC_ARRAY(char, strlen(trainfile) + 1); 
  get_base_name(trainfile, trainbase);
  default_basename = make_extended_name(trainbase, ".expt1",&allocsize);
  basename         = str_in_args(Arg_Basename,args,default_basename);
  uselog           = exists_in_args(Arg_Log,args);
  if (!uselog) 
    LOG = stdout; 
  else {
    logfile = make_extended_name(basename, ".log",&allocsize2); 
    LOG = fopen(logfile,"w"); 
  }

  Model  =  lookup( str_in_args(Arg_Model,args,"kde"),
                    Model_names, num_model_opts, "model");
  Method =  lookup( str_in_args(Arg_Method,args,"dual_tree"),
                    Method_names, num_method_opts, "method");
  Task   =  lookup( str_in_args(Arg_Task,args,"all_bw"),
                    Task_names, num_task_opts, "task");
  Prune  =  lookup( str_in_args(Arg_Prune,args,"relative"),
		    Prune_names, num_prune_opts, "prune");
  Scaling=  lookup( str_in_args(Arg_Scaling,args,"standardize"),
                    Scaling_names, num_scaling_opts, "scaling");
  Kernel =  lookup( str_in_args(Arg_Kernel,args,"epanechnikov"),
                    Kernel_names, num_kernel_opts, "kernel");
  Wtpass =  lookup( str_in_args(Arg_Wtpass,args,"delay"), 
                    Wtpass_names, num_wtpass_opts, "wtpass option");
  Search =  lookup( str_in_args(Arg_Search,args,"local_best"),
                    Search_names, num_search_opts, "search option");
  Heur   =  lookup( str_in_args(Arg_Heur,args,"avg_uni_dist"),
                    Heur_names, num_heur_opts, "heuristic");
  Approx =  lookup( str_in_args(Arg_Approx,args,"const_wt"),
                    Approx_names, num_approx_opts, "approximation mechanism");
  Bwstart = lookup( str_in_args(Arg_Bwstart,args,"subsample"),
                    Bwstart_names, num_bwstart_opts, "find_bw start method");

  Tau              = dbl_in_args(Arg_Tau,args,0.1);
  Knn              = int_in_args(Arg_Knn,args,3);
  Rmin             = int_in_args(Arg_Rmin,args,30);

  bwmin            = dbl_in_args(Arg_Bwmin,args,-1);
  bwmax            = dbl_in_args(Arg_Bwmax,args,-1);
  numbws           = int_in_args(Arg_Numbws,args,1);
  multi            = (exists_in_args(Arg_Bwmin,args) ||
                      exists_in_args(Arg_Bwmax,args) || 
                      exists_in_args(Arg_Numbws,args));

  if (!multi) bwmin= dbl_in_args(Arg_Bw,args,-1);

  LOGSCALE         = exists_in_args(Arg_Logscale,args);
  LOO              = !exists_in_args(Arg_Noloo,args) && SELFCASE;
  TRUEDIFF         = exists_in_args(Arg_Truediff,args);

  HELP             = exists_in_args(Arg_Help,args);
  DEBUG            = exists_in_args(Arg_Debug,args);
  Num_timing_iters = int_in_args(Arg_Timingiters,args,1);
  print_header();

  /* determine kind of cv */
  if (SELFCASE) {
    if (GAUSSIAN_STAR_KERNEL || SPHERICAL_STAR_KERNEL || 
	EPANECHNIKOV_STAR_KERNEL) { 
      Xval = Ls_Cv; 
    }
    else { 
      Xval = Lk_Cv; 
    }
  }

  /* Show what was chosen */
  fprintf(LOG,"................................\n");
  fprintf(LOG,"Parameter settings for this run:\n");
  fprintf(LOG,"  Trainfile            = %s\n", trainfile);
  fprintf(LOG,"  Testfile             = %s\n", testfile);
  fprintf(LOG,"  Basename             = %s\n", basename);
  fprintf(LOG,"  Logfile              = %s\n", uselog ? logfile : "stdout");
  fprintf(LOG,"  ........................\n");
  fprintf(LOG,"  Model                = %s\n", Model_names[Model]);
  fprintf(LOG,"  Method               = %s\n", Method_names[Method]);
  fprintf(LOG,"  Prune                = %s\n", Prune_names[Prune]);
  fprintf(LOG,"  Task                 = %s\n", Task_names[Task]);
  fprintf(LOG,"  Scaling              = %s\n", Scaling_names[Scaling]);
  fprintf(LOG,"  Kernel               = %s\n", Kernel_names[Kernel]);
  fprintf(LOG,"  Wtpass               = %s\n", Wtpass_names[Wtpass]);
  fprintf(LOG,"  Search               = %s\n", Search_names[Search]);
  fprintf(LOG,"  Heur                 = %s\n", Heur_names[Heur]);
  fprintf(LOG,"  Approx               = %s\n", Approx_names[Approx]);
  fprintf(LOG,"  ........................\n");
  fprintf(LOG,"  Tau                  = %g\n", Tau);
  
  if (!FIND_BW) {
    if (multi) {
      if (bwmin != -1) fprintf(LOG,"  Bwmin                = %g\n", bwmin);
      else             fprintf(LOG,"  Bwmin                = not specified\n");
      if (bwmax != -1) fprintf(LOG,"  Bwmax                = %g\n", bwmax);
      else             fprintf(LOG,"  Bwmax                = not specified\n");
      fprintf(LOG,"  Numbws        = %d\n", numbws);
    } else {
      if (bwmin != -1) fprintf(LOG,"  Bw                   = %g\n", bwmin);
      else             fprintf(LOG,"  Bw                   = not specified\n");
    }

    
  } 
  else {

    /**
     * Make sure we do not compute KDE in a leave-one-out fashion with
     * least squares cross validation.
     */
    if(LS_CV)
      LOO=FALSE;

    if (bwmax != -1) fprintf(LOG,"  Bwmax              = %g\n", bwmax);
    else             fprintf(LOG,"  Bwmax              = use subsample\n");
  }

  fprintf(LOG,"  ........................\n");
  fprintf(LOG,"  Logscale      = %s\n", LOGSCALE ? "true" : "false");
  fprintf(LOG,"  Leave-one-out = %s\n", LOO ? "true" : "false");
  fprintf(LOG,"  Truediff      = %s\n", TRUEDIFF ? "true" : "false");
  fprintf(LOG,"  ........................\n");
  fprintf(LOG,"  Debug         = %s\n", DEBUG ? "true" : "false");
  fprintf(LOG,"  Iters         = %d\n", Num_timing_iters);
  if (SELFCASE) {
    fprintf(LOG,"  ........................\n");
    fprintf(LOG,"  Xval          = %s\n", Xval_names[Xval]);
  }
  fprintf(LOG,"  ........................\n");

  fprintf(LOG,"\n");

  /**
   * Load in the training/test datasets.
   */
  load_into_dyms(argc,argv,trainfile,testfile,&d,&q);

  if(GAUSSIAN_KERNEL) {
    if(dym_cols(d) == 2)
      PLIMIT = 8;
    else if(dym_cols(d) == 3)
      PLIMIT = 6;
    else if(dym_cols(d) <= 5)
      PLIMIT = 4;
    else if(dym_cols(d) <= 6)
      PLIMIT = 2;
    else
      PLIMIT = 1;
  }
  else {
    PLIMIT = dym_cols(d);
  }
  
  /* in order to build the metric from q */
  tmp = mk_default_mrpars_for_data(q); 
  Metric = mk_copy_dyv(tmp->metric);  free_mrpars(tmp);

  /* This is to prescale the data to avoid doing divisions upon every
   * distance computation. Fix this so that we scale the dataset in the same
   * Euclidean space!!!!
   */
  printf("num data: %d\n",dym_rows(q));
  if (STDIZE_SCALING) { 
    if(SELFCASE)
      scale_data_by_meanstdev(q,NULL);
    else 
      scale_data_by_meanstdev(q,d); 
  }
  else if (RANGE_SCALING) {
    if(SELFCASE)
      scale_data_by_minmax(q,NULL);
    else 
      scale_data_by_minmax(q,d);
  }

  /* Run */
  if ( KDE_MODEL )  {
    bws = mk_bwinfo(bwmin, bwmax, numbws);
    bws->numerator_dim = 1; bws->denominator_dim = 0;
  }
  else if( WKDE_MODEL ) {
    bws = mk_bwinfo(bwmin, bwmax, numbws);
    bws->numerator_dim = 1; bws->denominator_dim = 0;
    
    if(strlen(dwgtsfile) == 0) {
      w = mk_constant_dyv(dym_rows(d),1.0);
    }
    else {
      load_into_dyv(argc,argv,dwgtsfile,&w);
    }
  }
  else if( VKDE_MODEL ) {
    bws = mk_bwinfo(bwmin, bwmax, numbws);
    bws->numerator_dim = 1; bws->denominator_dim = 0;

    if(strlen(dwgtsfile) == 0) {
      w = mk_constant_dyv(dym_rows(d),1.0);
    }
    else {
      load_into_dyv(argc,argv,dwgtsfile,&w);
    }
    
    if(strlen(dbwsfile) == 0) {
    }
    else {
      load_bwsqds_into_dym(argc,argv,dbwsfile,bws);
    }
  }
  else /* print help */   {
    print_usage("kde", args, num_kde_args);
  }

  if(KDE_MODEL || WKDE_MODEL || VKDE_MODEL) {
    kde_main(d,q,w,bws,basename);
    if(bws != NULL) free_bwinfo(bws);
    if(w != NULL) free_dyv(w);
  }

  if(LOG != stdout)
    fclose(LOG); 
  AM_FREE_ARRAY(trainbase,char,strlen(trainfile) + 1); 
  AM_FREE_ARRAY(default_basename,char,allocsize); 

  if(logfile != NULL)
    AM_FREE_ARRAY(logfile,char,allocsize2);
  /* clean up */
  free_dyv(Metric); free_dym(q); if (!SELFCASE) free_dym(d);
  free_args(args,num_kde_args);
  am_malloc_report();
  return 0;
}

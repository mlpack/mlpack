/***** KERNEL DENSITY ESTIMATION *****/

#include <values.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#include "amut.h"
#include "cp_args.h"
#include "allnn.h"
#include "cp_utils.h"
#include "stats.h"
#include "distutils.h"
#include "hrect.h"
#include "mrkd.h"
#include "ballutils.h"
#include "amstr.h"
#include "batree.h"
#include "kde.h"
#include "my_time.h"
#include "taylor.h"
#include "amdmex.h"
#include "integrator.h"
#include "regression.h"
#include "meanshift.h"
#include "stats.h"

/* global sizes */
int    Num_queries, Num_data, Num_dims, Depth;
double Check_iters;
dyv    *Record_l, *Record_u;
/* precomputations */
dyv    *Metric;
/* detected global conditions */
bool   SELFCASE;
/* dataset-guarantee */
dyv_array *LL_l, *LL_u;
/* which overall method/model */
method Method; kernel Kernel; task Task; scaling Scaling; 
model Model=Invalid_kde;
/* algorithmic options */
wtpass Wtpass; search Search; heur Heur; approx Approx; 
xval Xval; bwstart Bwstart;
/* pruning mode - absolute or relative */
prune Prune;
/* number of nearest neighbors for bandwidth setup in variable bw cases */
int Knn;
/* algorithmic parameters */
double Tau, CGiters; int  Rmin;
/* other algorithmic options */
bool   LOGSCALE, LOO;
/* testing/debugging */
bool   TRUEDIFF, DEBUG, HELP;
FILE*  LOG;
/* measurement */
int    Num_timing_iters;
int    PLIMIT;
int    Num_approx_prunes=0, Num_exclud_prunes=0, Num_includ_prunes=0;
int    Num_farfield_prunes=0, Num_far_to_local_conv_prunes = 0,
  Num_direct_local_accum_prunes=0;
int    Num_local_to_local_convs = 0;
int    Num_first_moment_better=0, Num_second_moment_better=0;
int    Num_comparisons=0, Num_node_expansions=0; 
int    Num_if_exhaustive=0;
int numOverTau;
int printMaxerr;

/* trees */
TREE Qtree = NULL; TREE Dtree = NULL;
/* weights */
dyv *Wtsum = NULL;
dyv *Wtsum2 = NULL;

/* names */
char* Model_names[]  = {"kde", "wkde", "vkde"};
char* Method_names[] = {"exhaustive","sngl_tree","dual_tree","treefree"};
char* Task_names[] = {"indep_bw", "all_bw", "peak_bw", "find_bw", "vfind_bw" };
char* Prune_names[]  = {"relative", "relative2", "absolute", "deng_moore"};
char* Scaling_names[]   = {"none", "standardize", "range"};
char* Kernel_names[] = {"spherical", "spherical_star", "epanechnikov", 
			"epanechnikov_star", "gaussian", "gaussian_star", 
			"aitchison_aitken"};
char* Wtpass_names[] = {"delay", "immed"};
char* Search_names[] = {"dfs", "local_best", "global_best"};
char* Heur_names[]   = {"centers_dist", "overlap_factor", "num_overlaps",
                        "min_uni_dist", "max_uni_dist", "avg_uni_dist",
                        "neg_centers_dist"};
char* Approx_names[] = {"const_wt", "datum_guar", "dataset_guar"};
char* Xval_names[]   = {"lk_cv", "ls_cv"};
char* Bwstart_names[]   = {"value", "oversmooth", "subsample"};
char* Func_names[]   = {"none", "first", "sum", "sumsq", "unif", "gaus1", 
                        "anigaus1", "mog1", "mog2", "mog3", "mog4", "mog5", 
                        "summog5"};
char* Bwmode_names[]   = {"all_bw", "find_bw"};
char* Stopmode_names[]   = {"time", "iters", "right", "auto"};

void process_series_expansion(NODE qnode,dym *q,NODE dnode,dym *d,
			      ivec *p_alphaM2Ls,ivec *p_alphaDMs,
			      ivec *p_alphaDLs,int b,dym *l,dym *e,dym *u,
			      dym *weights)
{    
  if(p_alphaM2Ls != NULL) {
    if(ivec_ref(p_alphaM2Ls, b) > 0) {
      if(GAUSSIAN_KERNEL) {
	computeMultipoleCoeffs(Dtree, d, dnode, weights, b,
			       ivec_ref(p_alphaM2Ls, b));
	translateMultipoleToLocal(Dtree, dnode, qnode, b, 
				  ivec_ref(p_alphaM2Ls, b));
      }
      else if(EPANECHNIKOV_KERNEL) {
	computeMultipoleCoeffsEpan(Dtree,d,dnode,weights,b,
				   ivec_ref(p_alphaM2Ls,b));
	translateMultipoleToLocalEpan(Dtree,dnode,qnode,b,
				      ivec_ref(p_alphaM2Ls,b));
      }
    }
    else if(ivec_ref(p_alphaDMs, b) > 0) {
      if(GAUSSIAN_KERNEL) {
	computeMultipoleCoeffs(Dtree,d,dnode,weights,b,ivec_ref(p_alphaDMs,b));
	evaluateMultipoleExpansion(Qtree,q,dnode,qnode,b,
				   ivec_ref(p_alphaDMs, b),l,e,u);
      }
      else if(EPANECHNIKOV_KERNEL) {
	computeMultipoleCoeffsEpan(Dtree,d,dnode,weights,b,
				   ivec_ref(p_alphaDMs, b));
	evaluateMultipoleExpansionEpan(Qtree,q,dnode,qnode,b,
				       ivec_ref(p_alphaDMs, b), l, e, u);
      }
    }
    else if(ivec_ref(p_alphaDLs, b) > 0) {
      if(GAUSSIAN_KERNEL) {
	directLocalAccumulation(Qtree, d, weights, dnode, qnode, b,
				ivec_ref(p_alphaDLs, b));
      }
      else if(EPANECHNIKOV_KERNEL) {
	directLocalAccumulationEpan(Qtree, d, weights, dnode, qnode,
				    b,ivec_ref(p_alphaDLs, b));
      }
    }
  }
}

void dualtree_kde_epan_prune(NODE qnode,dym *q,NODE dnode,dym *d,
			     bwinfo *bws,int b,
			     double requiredError,double dmax,dyv *dl,dyv *du,
			     dyv *dt,ivec *approximated,ivec *p_alphaM2Ls,
			     ivec *p_alphaDMs, ivec *p_alphaDLs,double dl_b,
			     double du_b, double new_dl_b)
{
  double actualErrorM2L, actualErrorDM, actualErrorDL, actualError = 0;
  double bwsqd = dyv_ref((bws->bwsqds).bwsqds_dyv,b);
  int p_alphaM2L = computeRequiredNumTermsDMLEpan(qnode, bwsqd,
						  requiredError, dmax,
						  Qtree, &actualErrorM2L);
  int p_alphaDM = computeRequiredNumTermsDMLEpan(dnode, bwsqd,
						 requiredError, dmax,
						 Qtree, &actualErrorDM);
  int p_alphaDL = p_alphaM2L;
  int costM2L = 0;
  int costDM = 0;
  int costDL = 0;
  int minCost = 0;
  double wtsum_abs = (KDE_MODEL) ? (dnode->num_points):
    dyv_ref((dnode->wtsum_abs).wtsum_abs_dyv,b);

  actualErrorDL = actualErrorM2L;
  
  costDL = INT_MAX;
  costM2L = INT_MAX;
  costDM = 0;
  
  minCost = real_min(costM2L, real_min(costDM, costDL));

  if(minCost == costM2L) {
    ivec_set(p_alphaM2Ls, b, p_alphaM2L);
    actualError = actualErrorM2L;
  }

  else if(minCost == costDM) {
    ivec_set(p_alphaDMs, b, p_alphaDM);
    actualError = actualErrorDM;
  }

  else if(minCost == costDL) {
    ivec_set(p_alphaDLs, b, p_alphaDL);
    actualError = actualErrorDL;
  }
      
  dyv_set(dl, b, dl_b); dyv_set(du, b, du_b);

  if(RELATIVE_PRUNING) {
    if(KDE_MODEL)
      dyv_set(dt, b, wtsum_abs - wtsum_abs * actualError * 
	      (Dtree->root->num_points) / (new_dl_b * Tau));
    else
      dyv_set(dt, b, wtsum_abs - wtsum_abs * actualError *
	      dyv_ref((Dtree->root->wtsum_abs).wtsum_abs_dyv,b) /
	      (new_dl_b * Tau));
  }
  
  ivec_set(approximated, b, 1);
}

void dualtree_kde_gauss_prune(NODE qnode,dym *q,NODE dnode,dym *d,
			      bwinfo *bws,int b,double requiredError, 
			      double dmin, dyv *dl, dyv *du,
			      dyv *dt,ivec *approximated,ivec *p_alphaM2Ls,
			      ivec *p_alphaDMs, ivec *p_alphaDLs,double dl_b,
			      double du_b,double new_dl_b)
{
  double actualErrorM2L, actualErrorDM, actualErrorDL, actualError = 0;
  int p_alphaM2L = -1;
  int p_alphaDM = -1;
  int p_alphaDL = -1;
  bool prunable = FALSE;
  double bwsqd = dyv_ref((bws->bwsqds).bwsqds_dyv,b);
  
  p_alphaM2L = computeRequiredNumTermsM2L(qnode,dnode,bwsqd,requiredError,
					  dmin,Qtree,&actualErrorM2L);
  if(p_alphaM2L < 0) {
    
    if(dnode->num_points < qnode->num_points) {
      p_alphaDL = computeRequiredNumTermsDML(qnode,dnode,bwsqd,requiredError,
					     Qtree,FALSE,&actualErrorDL);
      if(p_alphaDL < 0) {
	p_alphaDM = computeRequiredNumTermsDML(qnode,dnode,bwsqd,
					       requiredError,Qtree,TRUE,
					       &actualErrorDM);
	if(p_alphaDM > 0)
	  prunable = TRUE;
      }
      else
	prunable = TRUE;
    }
    else {
      p_alphaDM = computeRequiredNumTermsDML(qnode,dnode,bwsqd,requiredError,
					     Qtree,TRUE,&actualErrorDM);
      
      if(p_alphaDM < 0) {
	p_alphaDL = computeRequiredNumTermsDML(qnode,dnode,bwsqd,
					       requiredError,Qtree,FALSE,
					       &actualErrorDL);
	
	if(p_alphaDL > 0)
	  prunable = TRUE;
      }
      else
	prunable = TRUE;
    }
  }
  else
    prunable = TRUE;

  if(p_alphaM2L > 0) {
    ivec_set(p_alphaM2Ls, b, p_alphaM2L);
    actualError = actualErrorM2L;
  }
  
  else if(p_alphaDM > 0) {
    ivec_set(p_alphaDMs, b, p_alphaDM);
    actualError = actualErrorDM;
  }

  else if(p_alphaDL > 0) {
    ivec_set(p_alphaDLs, b, p_alphaDL);
    actualError = actualErrorDL;
  }

  if(prunable) {

    double wtsum_abs = (KDE_MODEL) ? (dnode->num_points):
      dyv_ref((dnode->wtsum_abs).wtsum_abs_dyv,b);

    dyv_set(dl, b, dl_b); dyv_set(du, b, du_b);

    if(RELATIVE_PRUNING) {
      if(KDE_MODEL)
	dyv_set(dt, b, wtsum_abs - 
		(wtsum_abs * actualError * (Dtree->root->num_points)) / 
		(new_dl_b * Tau));
      else
	dyv_set(dt, b, wtsum_abs -
		(wtsum_abs * actualError * 
		 dyv_ref((Dtree->root->wtsum_abs).wtsum_abs_dyv,b)) /
		(new_dl_b * Tau));
    }

    ivec_set(approximated, b, 1);
  }
  else
    ivec_set(approximated, b, 0);
}

/* Initialize structures that store command-line argument information. */
void init_args(arg args[])
{
  char s[200];
  bool OPTIONAL=TRUE; //REQUIRED=FALSE;

  init_arg(args,Arg_Trainfile,"data",OPTIONAL,1,
	   make_copy_string("Data points (training set)"));
  init_arg(args,Arg_Testfile,"query",OPTIONAL,1,
	   make_copy_string("Query points (test set)"));
  init_arg(args,Arg_Train_targetfile,"dtarget",OPTIONAL,1,
           make_copy_string("Target values (training y's)"));
  init_arg(args,Arg_Test_targetfile,"qtarget",OPTIONAL,1,
           make_copy_string("Target values (test y's)"));
  init_arg(args,Arg_Train_wgtsfile,"dwgts",OPTIONAL,1,
           make_copy_string("Data weights (train w's)"));
  init_arg(args,Arg_Train_bwsfile,"dbws",OPTIONAL,1,
	   make_copy_string("Data bandwidths (train bw's)"));
  init_arg(args,Arg_Basename,"basename",OPTIONAL,1,
           make_copy_string("Basename to use for output files"));
  init_arg(args,Arg_Log,"log",OPTIONAL,0,
	   make_copy_string("Print to log file instead of screen"));
  
  init_arg(args,Arg_Model,"model",OPTIONAL,1, 
	   mk_build_options_string
	   (s,Model_names,num_model_opts,"Overall model")); 
  init_arg(args,Arg_Method,"method",OPTIONAL,1, mk_build_options_string
	   (s,Method_names,num_method_opts,"Overall algorithm choice"));
  init_arg(args,Arg_Task,"task",OPTIONAL,1, mk_build_options_string
	   (s,Task_names,num_task_opts,"Overall task to perform")); 
  init_arg(args,Arg_Prune,"prune",OPTIONAL,1, mk_build_options_string
	   (s,Prune_names,num_prune_opts,"Pruning options"));
  init_arg(args,Arg_Scaling,"scaling",OPTIONAL,1, mk_build_options_string
	   (s,Scaling_names,num_scaling_opts,"How to scale the data")); 
  init_arg(args,Arg_Kernel,"kernel",OPTIONAL,1, mk_build_options_string
	   (s,Kernel_names,num_kernel_opts,"Kernel function"));
  init_arg(args,Arg_Wtpass,"wtpass",OPTIONAL,1, mk_build_options_string
	   (s,Wtpass_names,num_wtpass_opts,
	    "How to propagate weights (densities) to children"));
  init_arg(args,Arg_Search,"search",OPTIONAL,1, mk_build_options_string
	   (s,Search_names,num_search_opts,"Search control algorithm"));
  init_arg(args,Arg_Heur,"heur",OPTIONAL,1, mk_build_options_string
	   (s, Heur_names,num_heur_opts,"Search heuristic"));
  init_arg(args,Arg_Approx,"approx",OPTIONAL,1, mk_build_options_string
	   (s,Approx_names,num_approx_opts,"Approximation mechanism"));
  init_arg(args,Arg_Bwstart,"bwstart",OPTIONAL,1, mk_build_options_string
	   (s,Bwstart_names,num_bwstart_opts,"How to start bandwidth search"));
  init_arg(args,Arg_Knn,"knn",OPTIONAL,1,
	   make_copy_string("Number of K-nearest neighbors for variable bandwidth crossvalidation"));
								 

  init_arg(args,Arg_Tau,"tau",OPTIONAL,1,
	   make_copy_string("Approximation threshold: tau"));
  init_arg(args,Arg_Rmin,"rmin",OPTIONAL,1,
	   make_copy_string("Max. number points in a node"));
  init_arg(args,Arg_Bw,"bw",OPTIONAL,1,
	   make_copy_string("Bandwidth: h"));
  init_arg(args,Arg_Bwmin,"bwmin",OPTIONAL,1,
	   make_copy_string("Minimum of bandwidth range"));
  init_arg(args,Arg_Bwmax,"bwmax",OPTIONAL,1,
	   make_copy_string("Maximum of bandwidth range"));
  init_arg(args,Arg_Numbws,"numbws",OPTIONAL,1,
           make_copy_string("Number of bandwidths in bandwidth range"));
  init_arg(args,Arg_Bw2,"bw2",OPTIONAL,1,
	   make_copy_string("Bandwidth: h (second class)"));
  init_arg(args,Arg_Bwmin2,"bwmin2",OPTIONAL,1,
           make_copy_string("Minimum of bandwidth range (second class)"));
  init_arg(args,Arg_Bwmax2,"bwmax2",OPTIONAL,1,
           make_copy_string("Maximum of bandwidth range (second class)"));
  init_arg(args,Arg_Numbws2,"numbws2",OPTIONAL,1,
           make_copy_string("Number of bandwidths in bandwidth range (second class)"));

  init_arg(args,Arg_Logscale,"logscale",OPTIONAL,0,
           make_copy_string("Whether to use log-scale for the bandwidth divisions"));
  init_arg(args,Arg_Noloo,"noloo",OPTIONAL,0,
           make_copy_string("Whether to turn off leave-one-out when query set = data set"));
  init_arg(args,Arg_Truediff,"truediff",OPTIONAL,0,
           make_copy_string("Whether to compute true estimate to compare the error"));

  init_arg(args,Arg_Help,"help",OPTIONAL,0,
	   make_copy_string("Help information"));
  init_arg(args,Arg_Debug,"debug",OPTIONAL,0,
	   make_copy_string("Debug printouts"));
  init_arg(args,Arg_Timingiters,"timingiters",OPTIONAL,1,
           make_copy_string("Number of times to run the program, for timing"));


}

void print_header()
{
  fprintf(LOG,
"...............................................................\n"
"Welcome to the Kernel Density Estimator (KDE), Beta Version 0.7\n"
"(for close collaborators only - absolutely not for distribution)\n"
"\n"
"(c) 2001,2002,2003  Alexander Gray\n"
"http://www.cs.cmu.edu/~agray\n"
"Modified by Dongryeol Lee\n"
"\n"
"References:\n"
"[1] Gray and Moore, 'N-Body Problems in Statistical Learning', Neural\n"
"    Information Processing Systems 2000.\n"
"[2] Gray and Moore, 'Nonparametric Density Estimation: Toward Computational\n"
"    Tractability', SIAM Data Mining 2003.\n"
"[3] Gray and Moore, 'Rapid Evaluation of Multiple Density Models',\n"
"    AI & Statistics 2003.\n"
"[4] Gray and Moore, 'Very Fast Multivariate Kernel Density Estimation via\n"
"    Comptuational Geometry', Joint Statistical Meeting 2003.\n"
"[5] Gray and Moore, 'Linear-time Kernel Density Estimation and Regression',\n"
"    to be submitted to JASA (Jour. Amer. Stat. Assn.).\n"
"[6] Lee, Gray and Moore, 'Dual-Tree Fast Gauss Transforms', Neural\n"
"    Information Processing Systems 2005.\n"
"\n"
"Current features:\n"
"- core dual-recursive N-body algorithm\n"
"- finite and infinite kernels\n"
"- fast linear-expectation approximation with optimizing pruning rule\n"
"- fast multipole type pruning for Gaussian kernel\n"
"- hard-guarantee approximation bounds\n"
"- best-first search\n"
"- deferred up-down mass propagation for cross-scale maximization\n"
"- least-squares and likelihood cross-validation\n"
"- multi-bandwidth simultaneous computation\n"
"- high-dimensional sphere-rectangle trees\n"
"- automatic search for optimal bandwidth\n"
"\n");
}

void print_help()
{
  fprintf(LOG,
".........\n"
"EXAMPLES.\n"
"kde -help\n"
"  Print this help screen.\n"
"kde -data sdss10k.fds\n"
"  Use sdss10k.fds for the sample data, and use the same points for the\n"
"  queries, in a leave-one-out fashion; a rough bandwidth guess is computed.\n"
"kde -data sdss10k.fds -task find_bw -kernel spherical\n"
"  Automatically find the optimal bandwidth for the dataset and kernel.\n"
"kde -data sdss10k.fds -bw .01\n"
"  Compute density for bandwidth .01.\n"
"kde -data sdss10k.fds -bwmin .001 -bwmax .01 -numbws 10 -logscale\n"
"  Compute density for 10 bandwidths between .001 and .01, spaced on a\n"
"  logarithmic scale.\n"
"kde -data sdss10k.fds -basename my_first_run -log\n"
"  Specify how to name the output files, and dump the screen output to a log\n"
"  file instead.\n"
"............\n"
"DATA, QUERY.\n"
"The 'training set' is the set of data over which kernels are placed to\n"
"  obtain the density estimates.  Also referred to as the 'sample data'.\n"
"The 'test set' is the set of data for which density values are obtained.\n"
"  Also referred to as the 'query set'.\n"
"The data format is the Auton .ds or .fds format.  If .ds is given and \n"
"  the data doesn't exist in .fds format, a .fds file will automatically\n"
"  be generated with the same base name + '.fds'.\n"
"Whenever the training set and test set are the same, leave-one-out \n"
"  estimation is automatically done.  Because the leave-one-out likelihood\n"
"  is trivially related to the likelihood cross-validation score, that score\n"
"  is automatically reported as well.\n"
"........\n"
"OUTPUTS.\n"
"The program will output the density for each query point, in a file \n"
"  consisting of a column of values, for each point in the same order,\n"
"  called the base name + '.dens'.  When approximation is used,\n"
"  '.dens_lo' will contain the lower bound on the density for each \n"
"  query point, and '.dens_hi' the upper bound.  The guessed density\n"
"  is the midpoint of the lo and hi values.\n"
"All bounds are absolute worst-case bounds, i.e. it is guaranteed that the\n"
"  true value is within the bounds reported.\n"
"The program will also output the log-likelihood of the entire test set.\n"
"  A lower bound and upper bound will be given, in addition to the guess.\n"
"..............\n"
"BASENAME, LOG.\n"
"Specifying 'basename' tells the program how to name its output files.\n"
"  Each output file is named 'basename.suffix', where 'suffix' is hard-coded\n"
"  depending on the type of output, for example 'dens' for the output density\n"
"  values.  This is useful for keeping multiple experiments organized.  The\n"
"  '-log' option tells the program to direct its screen output to a text file\n"
"  instead, which will be called 'basename.log'.\n"
".....\n"
"TASK.\n"
"The main tasks that the program can perform are 'all_bw', which computes\n"
"  the density estimates for all of the bandwidths specified (this is the\n"
"  default task) and 'find_bw', which automatically finds the optimal\n"
"  bandwidth for the given dataset and kernel function.  Other options\n"
"  include 'indep_bw', which performs the density estimates separately\n"
"  rather than simultaneously as a debugging or comparison check, and\n"
"  'peak_bw', which is designed to eliminate sub-optimal bandwidths.\n"
"  (NOTE: The last option is still experimental and currently disabled.)\n"
".........................\n"
"BW, BWMIN, BWMAX, NUMBWS.\n"
"The user can either specify a single bandwidth, using '-bw', or can\n"
"  specify that the density be computed for each bandwidth of 'numbws'\n"
"  bandwidths between 'bwmin' and 'bwmax', either evenly spaced or \n"
"  spaced evenly along a log-scale if the '-logscale' option is specified.\n"
"For example, to specify the bandwidths {.1,.2,.3,....,1}:\n"
"  ./kde -train sdss10k.fds -bwmin .1 -bwmax 1 -numbws 10\n"
"  and to specify the bandwidths {.0001,.001,...,.1}:\n"
"  ./kde -train sdss10k.fds -bwmin .0001 -bwmax .1 -numbws 4 -logscale\n"
"In the multiple-bandwidths case, the density output files will contain\n"
"  one column for each bandwidth, in order.\n"
"........\n"
"SCALING.\n"
"The options for pre-scaling the data are 'standardize', which scales each\n"
"  by subtracting its mean and dividing by its standard deviation, 'range',\n"
"  which subtracts the column's minimum value and divides by its maximum\n"
"  value, and 'none', which just uses the raw data values.  The default\n"
"  is 'standardize'.\n"
".......\n"
"KERNEL.\n"
"Spherical, Epanechnikov, or Gaussian.  There are also kernels with a '-star'\n"
"  ending corresponding to each of these - their purpose is not for density\n"
"  estimation per se, but for least-squares cross-validation when doing\n"
"  scoring for bandwidth selection.\n"
".......\n"
"APPROX.\n"
"There are three choices for the approximation mechanism.  One is to have\n"
"  the computation run as fast as possible but still guarantee a maximum\n"
"  deviation from the true density in the final answer, for each query\n"
"  point.  Another is to run as fast as possible while guaranteeing that\n"
"  the overall log-likelihood of the final answer deviates from the true\n"
"  log-likelihood by less than a specified amount.  The first is in some\n"
"  sense a stronger guarantee, holding on each query datum ('datum_guar'),\n"
"  while the second may allow large errors on individual data to achieve\n"
"  an approximation guarantee on the dataset as a whole ('dataset_guar').\n"
"  In practice, the per-datum guarantee option, while in principle\n"
"  providing a very loose bound on the overall log-likelihood for\n"
"  reasonable values of the approximation threshold ('tau'), tends to also\n"
"  yield low overall log-likelihood error. (NOTE: These two choices are\n"
"  still experimental and currently disabled.)\n"
"In all cases, the resulting actual maximum possible error at both the \n"
"  per-datum level and the dataset level is reported.\n"
"The third option is the constant-weight approximation mechanism, which\n"
"  is a heuristic providing no obvious guarantee on the final answer, but\n"
"  in practice is by far the fastest of the three for a given result\n"
"  accuracy.  It is thus the default.  Relating its approximation parameter\n"
"  ('psi') to the resulting accuracy desired (at either the datum or\n"
"  dataset level) is a matter of experimentation.\n"
".........\n"
"TAU.\n"
"Tau is the guarantee threshold.  In the case of a per-datum guarantee,\n"
"  the percentage error you're willing to accept for any individual\n"
"  query point's density.  For example, 0.1 means you're willing to accept\n"
"  up to 10 percent error from the true density for any individual query\n"
"  point.  The amount of actual error will differ in general for different\n"
"  query points.  The biggest actual error over all the query points is\n"
"  printed at the end of the run.  If the dataset-level guarantee is\n"
"  chosen, Tau is the maximum percentage error you will tolerate in the\n"
"  overall log-likelihood of the density estimate.\n"
".................\n"
"CROSS-VALIDATION.\n"
"Likelihood cross-validation computes the score value CV(h) of Eqn. 3.43 of\n"
"  Silverman p.53.\n"
"Least-squares cross-validation is done by specifying the 'gaussian_star'\n"
"  kernel function, in the Gaussian-kernel case.  ('spherical_star' and\n"
"  'epanechnikov_star' are forthcoming.)  The score function M_1(h)\n"
"  of Eqn. 3.39 of Silverman p. 50 is computed.  Note that when least-\n"
"  squares cross-validation is being done, ie. one of the 'star' kernels\n"
"  is specified, the computed densities are based on this kernel, and \n"
"  thus unlikely to be meaningful.\n"
"BANDWIDTH SEARCH, BWSTART.\n"
"The find_bw mode starts the search for the optimal bandwidth from above,\n"
"  i.e. from a value known to be larger than the true value.  This can be\n"
"  specified in three ways.\n"
"A theoretical oversmoothing bandwidth can be specified as an option of\n"
"  the 'bwstart' parameter.\n"
"Because a smaller sample of the data will have a larger bandwidth, the\n"
"  default option of 'bwstart' is to use the optimal bandwidth determined\n"
"  for a subsample of the data of size 10,000 (assuming the full sample is\n"
"  larger than this).\n"
"Another option of 'bwstart' is to specific a specific starting value.\n"
"  This value is set as the parameter value of 'bwmax'.\n"
".............................\n"
"TIMINGITERS, TRUEDIFF, DEBUG.\n"
"The first option runs the computation several times, for obtaining\n" 
"  sub-second timing accuracy.\n"
"The second option computes the density exhaustively to provide a reference\n"
"  density so that the true error of an approximate density may be evaluated.\n"
"The last option is for debugging the code.\n"
".......\n"
"METHOD.\n"
"'exhaustive' means the naive standard method which loops over each query\n"
"  point and for each those, loops over every training point.\n"
"'dual_tree' means the fast N-body tree-based method.\n"
"'sngl_tree' refers to a less powerful tree-based method which is sometimes\n"
"  of interest for comparative or debugging reasons.  (NOTE: This choice\n"
"  is currently disabled.)\n"
".....................\n"
"WTPASS, SEARCH, HEUR.\n"
"Internal algorithmic options.  Just use the default values unless you are\n"
"  familiar with the underlying algorithms.\n"
".....\n"
"HELP.\n"
"Prints this screen of information.\n"
"\n");
}


/**** UTILITIES **************************************************************/

/**** FILES */
void load_bwsqds_into_dym(int argc,char *argv[],char *bwfile,bwinfo *bws)
{
  datset *tdat=0;
  int i,b;
  double bwsqd,norm,aux;

  /* process files into dat's*/
  tdat = ds_load_with_options_simple(bwfile,argc,argv);

  /* process dat's into dyv's*/
  (bws->bwsqds).bwsqds_dym=mk_dym_from_datset(tdat); free_datset(tdat);
  (bws->norm_consts).norm_consts_dym=
    mk_zero_dym(dym_rows((bws->bwsqds).bwsqds_dym),
		dym_cols((bws->bwsqds).bwsqds_dym));
  (bws->aux_consts).aux_consts_dym=
    mk_zero_dym(dym_rows((bws->bwsqds).bwsqds_dym),
		dym_cols((bws->bwsqds).bwsqds_dym));

  for(i=0; i < dym_rows((bws->bwsqds).bwsqds_dym); i++) {
    for(b=0; b< dym_cols((bws->bwsqds).bwsqds_dym); b++) {
      dym_set((bws->bwsqds).bwsqds_dym,i,b,
	      dym_ref((bws->bwsqds).bwsqds_dym,i,b)*
	      dym_ref((bws->bwsqds).bwsqds_dym,i,b));
      bwsqd =dym_ref((bws->bwsqds).bwsqds_dym,i,b);
      compute_norm_const(Num_dims, bwsqd, &norm, &aux);
      dym_set((bws->norm_consts).norm_consts_dym,i,b,norm);
      dym_set((bws->aux_consts).aux_consts_dym,i,b,aux);
    }
  }
}

void load_into_dym(int argc,char *argv[],char *targetfile, dym **t)
{
  datset *tdat=0; char *target_basename, *f;
  int allocsize;

  target_basename = AM_MALLOC_ARRAY(char, strlen(targetfile) + 1);
  get_base_name(targetfile, target_basename);

  /* process files into dat's*/
  tdat = ds_load_with_options_simple(targetfile,argc,argv);
  if (!filename_is_fast(targetfile)) { 
    /* make fast format if we read slow one */
    f = make_extended_name(target_basename, ".fds",&allocsize);
    ds_save(f, tdat); 
    AM_FREE_ARRAY(f,char,allocsize);
  }
  
  /* process dat's into dyv's*/
  *t  = mk_dym_from_datset(tdat); free_datset(tdat);
  
  AM_FREE_ARRAY(target_basename,char,strlen(targetfile) + 1);
}

void load_into_dyv_array(int argc,char *argv[],char *targetfile, dyv_array **t)
{
  datset *tdat=0; char *target_basename, *f;
  dym *ttmp; int allocsize;

  target_basename = AM_MALLOC_ARRAY(char, strlen(targetfile) + 1);
  get_base_name(targetfile, target_basename);

  /* process files into dat's*/
  tdat = ds_load_with_options_simple(targetfile,argc,argv);
  if (!filename_is_fast(targetfile)) { 
    /* make fast format if we read slow one */
    f = make_extended_name(target_basename, ".fds",&allocsize);
    ds_save(f, tdat); 
    AM_FREE_ARRAY(f,char,allocsize);
  }
  
  /* process dat's into dyv's*/
  ttmp  = mk_dym_from_datset(tdat); free_datset(tdat);
  *t = mk_dyv_array_from_dym(ttmp);
  free_dym(ttmp);
  
  AM_FREE_ARRAY(target_basename,char,strlen(targetfile) + 1);
}

int get_base_name(char *file_name, char *base_name)
{
  int i, len = strlen(file_name);
  for (i = 0; i < len; i++)
    if (file_name[i] != '.') base_name[i] = file_name[i];
    else break;
  base_name[i] = '\0';

  return( i );
}


char *make_extended_name(char *base_name, char *ext_name, int *allocsize)
{
  char *file_name;
  int i, L=strlen(base_name), L_ext=strlen(ext_name);

  file_name = (char*) AM_MALLOC_ARRAY(char, (L + L_ext + 1) );

  strncpy(file_name, base_name,L);
  for (i=0; i<L_ext; i++)
    file_name[i+L]=ext_name[i];
  file_name[L + L_ext]='\0';

  *allocsize = L + L_ext + 1;

  return (file_name);
}


void write_dyv_as_col(char *outfile, dyv *v, char *mode)
{
  FILE *fp = safe_fopen(outfile, mode);
  print_dyv_as_col(fp, v);
  fclose(fp);
}

void print_dyv_as_col(FILE *fp, dyv *v)
{
  int i, n=dyv_size(v);
  for (i=0; i<n; i++)
    fprintf(fp, "%g\n", dyv_ref(v,i));
}

void write_ivec_as_col(char *outfile, ivec *v, char *mode)
{
  FILE *fp = safe_fopen(outfile, mode);
  print_ivec_as_col(fp, v);
  fclose(fp);
}

void print_ivec_as_col(FILE *fp, ivec *v)
{
  int i, n=ivec_size(v);
  for (i=0; i<n; i++)
    fprintf(fp, "%d\n", ivec_ref(v,i));
}

void write_dym(char *outfile, dym *m, char *mode)
{
  FILE *fp = safe_fopen(outfile, mode);
  print_dym(fp, m);
  fclose(fp);
}

void print_dym(FILE *fp, dym *m)
{
  int i, j, n1=dym_rows(m), n2=dym_cols(m);
  for (i=0; i<n1; i++) {
    for (j=0; j<n2; j++)
      fprintf(fp, "%g ", dym_ref(m,i,j));
    fprintf(fp, "\n");
  }
}

void write_dyv_array(char *outfile, dyv_array *va, char *mode)
{
  FILE *fp = safe_fopen(outfile, mode);
  print_dyv_array(fp, va);
  fclose(fp);
}

// note: assumes all dyv's have the same length
void print_dyv_array(FILE *fp, dyv_array *va)
{
  int i, j, n1=dyv_array_size(va), n2=dyv_size(dyv_array_ref(va,0));
  for (i=0; i<n1; i++) {
    for (j=0; j<n2; j++) {
      dyv *v = dyv_array_ref(va,i);
      fprintf(fp, "%g ", dyv_ref(v,j));
    }
    fprintf(fp, "\n");
  }
}

void write_transposed_dyv_array(char *outfile, dyv_array *va, char *mode)
{
  FILE *fp = safe_fopen(outfile, mode);
  print_transposed_dyv_array(fp, va);
  fclose(fp);
}

// note: assumes all dyv's have the same length
void print_transposed_dyv_array(FILE *fp, dyv_array *va)
{
  int i, j, n1=dyv_array_size(va), n2=dyv_size(dyv_array_ref(va,0));
  for (i=0; i<n2; i++) {
    for (j=0; j<n1; j++) {
      dyv *v = dyv_array_ref(va,j);
      fprintf(fp, "%g ", dyv_ref(v,i));
    }
    fprintf(fp, "\n");
  }
}

void load_into_dyv(int argc,char *argv[],char *targetfile, dyv **t)
{
  datset *tdat=0; char *target_basename, *f;
  int allocsize;

  target_basename = AM_MALLOC_ARRAY(char,strlen(targetfile) + 1);
  get_base_name(targetfile, target_basename);

  /* process files into dat's*/
  tdat = ds_load_with_options_simple(targetfile,argc,argv);
  if (!filename_is_fast(targetfile)) { 
    /* make fast format if we read slow one */
    f = make_extended_name(target_basename, ".fds",&allocsize);
    ds_save(f, tdat); 
    AM_FREE_ARRAY(f,char,allocsize);
  }

  /* process dat's into dyv's*/
  *t  = mk_dyv_from_rows(tdat,0,NULL); free_datset(tdat);

  AM_FREE_ARRAY(target_basename,char,strlen(targetfile) + 1);
}

void load_into_dyms(int argc,char *argv[],char *trainfile,char *testfile,
                    dym **d, dym **q)
{
  datset *qdat=0, *ddat=0; char *test_basename, *train_basename; 

  train_basename = AM_MALLOC_ARRAY(char, strlen(trainfile) + 1); 
  get_base_name(trainfile, train_basename);
  test_basename = AM_MALLOC_ARRAY(char,strlen(testfile) + 1); 
  get_base_name(testfile, test_basename);

  /* process files into dat's*/
  qdat = ds_load_with_options_simple(testfile,argc,argv);

  if (!SELFCASE) {
    ddat = ds_load_with_options_simple(trainfile,argc,argv);
  }

  /* process dat's into dym's*/
  *q  = mk_dym_from_datset(qdat); free_datset(qdat); 
  if (!SELFCASE) { *d  = mk_dym_from_datset(ddat); free_datset(ddat); }
  else { *d = *q; }

  AM_FREE_ARRAY(train_basename,char,strlen(trainfile) + 1); 
  AM_FREE_ARRAY(test_basename,char,strlen(testfile) + 1); 
}

/**** DATA */

/* scales each attribute to 0-1 using the min/max values */
void scale_data_by_minmax(dym *m,dym *m2)
{
  int i, num_dims = dym_cols(m);

  for (i=0; i<num_dims; i++) {
    double minv = dym_col_min(m,i), maxv = dym_col_max(m,i);

    if(m2 != NULL) {
      minv = real_min(minv,dym_col_min(m2,i));
      maxv = real_max(maxv,dym_col_max(m2,i));
    }

    printf("In dimension %d, subtracting min (%g) and ",i,minv);
    printf("dividing by range (%g - %g = %g).\n",maxv,minv,maxv-minv);
    dym_col_scalar_add_self(m, i, -minv);
    dym_col_scalar_mult_self(m, i, 1.0/(maxv-minv)); 

    if(m2 != NULL) {
      dym_col_scalar_add_self(m2, i, -minv);
      dym_col_scalar_mult_self(m2, i, 1.0/(maxv-minv));
    }
  }
}

/* scales each attribute to be near both sides of 0 using the mean/stdev */
void scale_data_by_meanstdev(dym *m,dym *m2)
{
  int i, num_dims = dym_cols(m);

  for (i=0; i<num_dims; i++) {
    dyv *v = mk_dyv_from_dym_col(m,i);

    if(m2 != NULL) {
      dyv *m2_dyv = mk_dyv_from_dym_col(m2,i);
      append_to_dyv(v,m2_dyv);
      free_dyv(m2_dyv);
    }

    double s = dyv_sdev(v), u = dyv_mean(v);
    printf("In dimension %d, subtracting mean (%g) and ",i,u);
    printf("dividing by stdev (%g).\n",s);
    free_dyv(v);
    dym_col_scalar_add_self(m, i, -u); 
    dym_col_scalar_mult_self(m, i, 1.0/s); 
    
    if(m2 != NULL) {
      dym_col_scalar_add_self(m2, i, -u);
      dym_col_scalar_mult_self(m2, i, 1.0/s);
    }
  }
}


dym *mk_reorder_data_by_tree(dym *data, TREE tree)
{
  dym *new_data = mk_dym(dym_rows(data),dym_cols(data));
  int curr_row = 0;
  int total = copy_in_dfs_order(data,new_data,tree->root,&curr_row);
  if (total != dym_rows(data)) printf("Um, error...\n");
  return new_data;
}

int copy_in_dfs_order(dym *old, dym *new, NODE node, int *curr_row)
{

  if (is_leaf(node)) {
    int i, n = node->num_points;
    for (i=0; i<n; i++) {
      int row_i = ivec_ref(node->rows,i);
      copy_dym_row_to_dym_row(old, row_i, new, *curr_row);
      ivec_set(node->rows,i,*curr_row);
      (*curr_row)++;
    }
  } else {
    copy_in_dfs_order(old, new, node->left, curr_row);
    copy_in_dfs_order(old, new, node->right, curr_row);
  }
  return *curr_row;
}


/**** VECTORS */

void dym_col_scalar_add_self(dym *m, int col, double val)
{
  int i, num_data = dym_rows(m);
  
  for (i=0; i<num_data; i++)
    dym_set(m, i, col, dym_ref(m, i, col) + val);
}

void dym_col_scalar_mult_self(dym *m, int col, double val)
{
  int i, num_data = dym_rows(m);
  
  for (i=0; i<num_data; i++)
    dym_set(m, i, col, dym_ref(m, i, col) * val);
}

// note: assumes both rows have the same length
void copy_dyv_array_row_to_row(dyv_array *va1,int row1,dyv_array *va2,int row2)
{
  dyv *v1 = dyv_array_ref(va1, row1);
  dyv *v2 = dyv_array_ref(va2, row2);
  copy_dyv(v1, v2);
}

// note: assumes both cols have the same length
void copy_dyv_array_col_to_col(dyv_array *va1,int col1,dyv_array *va2,int col2)
{
  int i;
  for (i=0; i<dyv_array_size(va1); i++)
    dyv_array_ref_set(va2, i, col2, dyv_array_ref_ref(va1, i, col1));
}

dyv *dyv_plus_eq(dyv *d_1, dyv *d_2)
{
  dyv_plus(d_1,d_2,d_1);
  return d_1;
}

dyv *dyv_minus_eq(dyv *d_1, dyv *d_2)
{
  dyv_subtract(d_1,d_2,d_1);
  return d_1;
}

dym *dym_min_eq(dym *v1, dym *v2)
{
  int i, j;
  for(i = 0; i < dym_rows(v1); i++) {
    for(j = 0; j < dym_cols(v1); j++) {
      dym_set(v1, i, j, real_min(dym_ref(v1,i,j),dym_ref(v2,i,j)));
    }
  }
  return v1;
}

dym *dym_max_eq(dym *v1, dym *v2)
{
  int i, j;
  for(i = 0; i < dym_rows(v1); i++) {
    for(j = 0; j < dym_cols(v1); j++) {
      dym_set(v1, i, j, real_max(dym_ref(v1,i,j),dym_ref(v2,i,j)));
    }
  }
  return v1;
}

dym *dym_min_mat(dym *m1, dym *m2, dym *result)
{
  int i, j;
  for(i = 0; i < dym_rows(m1); i++) {
    for(j = 0; j < dym_cols(m1); j++) {
      dym_set(result, i, j, real_min(dym_ref(m1,i,j),dym_ref(m2,i,j)));
    }
  }
  return result;
}

dym *dym_max_mat(dym *m1, dym *m2, dym *result)
{
  int i, j;
  for(i = 0; i < dym_rows(m1); i++) {
    for(j = 0; j < dym_cols(m1); j++) {
      dym_set(result, i, j, real_max(dym_ref(m1,i,j),dym_ref(m2,i,j)));
    }
  }
  return result;
}

dym *dym_abs_max_mat(dym *m1,dym *m2,dym *result)
{
  int i, j;
  for(i = 0; i < dym_rows(m1); i++) {
    for(j = 0; j < dym_cols(m1); j++) {
      dym_set(result, i, j, real_max(fabs(dym_ref(m1,i,j)),
				     fabs(dym_ref(m2,i,j))));
    }
  }
  return result;
}

// note: assumes all vectors are the same length
dyv *dyv_min_eq(dyv *v1, dyv *v2)
{
  int i, n = dyv_size(v2);
  for (i=0; i<n; i++) dyv_set(v1, i, real_min(dyv_ref(v1,i),dyv_ref(v2,i)));
  return v1;
}

// note: assumes all vectors are the same length
dyv *dyv_max_eq(dyv *v1, dyv *v2)
{
  int i, n = dyv_size(v2);
  for (i=0; i<n; i++) dyv_set(v1, i, real_max(dyv_ref(v1,i),dyv_ref(v2,i)));
  return v1;
}

// note: assumes all vectors are the same length
dyv *dyv_min_vec(dyv *v1, dyv *v2, dyv *result)
{
  int i, n = dyv_size(result);
  for (i=0; i<n; i++) dyv_set(result,i,real_min(dyv_ref(v1,i),dyv_ref(v2,i)));
  return result;
}

// note: assumes all vectors are the same length
dyv *dyv_max_vec(dyv *v1, dyv *v2, dyv *result)
{
  int i, n = dyv_size(result);
  for (i=0; i<n; i++) dyv_set(result,i,real_max(dyv_ref(v1,i),dyv_ref(v2,i)));
  return result;
}

dyv *dyv_arg_min_vec(dyv *v1,dyv *av1,dyv *v2,dyv *av2,dyv *min_v,dyv *min_arg)
{
  int i, n = dyv_size(min_arg);
  for(i=0; i<n; i++) {
    if(dyv_ref(v1,i)<dyv_ref(v2,i)) {
      dyv_set(min_v,i,dyv_ref(v1,i));
      dyv_set(min_arg,i,dyv_ref(av1,i));
    }
    else {
      dyv_set(min_v,i,dyv_ref(v2,i));
      dyv_set(min_arg,i,dyv_ref(av2,i));
    }
  }
  return min_arg;
}

dyv *dyv_arg_max_vec(dyv *v1,dyv *av1,dyv *v2,dyv *av2,dyv *max_v,dyv *max_arg)
{
  int i, n = dyv_size(max_arg);
  for(i=0; i<n; i++) {
    if(dyv_ref(v1,i)<dyv_ref(v2,i)) {
      dyv_set(max_v,i,dyv_ref(v2,i));
      dyv_set(max_arg,i,dyv_ref(av2,i));
    }
    else {
      dyv_set(max_v,i,dyv_ref(v1,i));
      dyv_set(max_arg,i,dyv_ref(av1,i));
    }
  }
  return max_arg;
}

void zero_dyv_array(dyv_array *va)
{
  int i, n=dyv_array_size(va);
  for (i=0; i<n; i++)
    zero_dyv(dyv_array_ref(va,i));
}

void constant_dyv_array(dyv_array *va, double C)
{
  int i, n=dyv_array_size(va);
  for (i=0; i<n; i++)
    constant_dyv(dyv_array_ref(va,i), C);
}

/* Returns dym of size ivec_size(rows) in which
    result[i,j] = x[rows[i],j] */
dym *mk_dym_subset(dym *x,ivec *rows)
{
  int nrows = ivec_size(rows), ncols = dym_cols(x), i, j;
  dym *y = mk_dym(nrows,ncols);
  for ( i = 0 ; i < nrows ; i++ )
    for ( j = 0 ; j < ncols ; j++ )
      dym_set(y,i,j,dym_ref(x,ivec_ref(rows,i),j));
  return y;
}

dym *mk_random_dym_subset(dym *m, int size)
{
  ivec *all_rows = mk_sequence_ivec(0, dym_rows(m));
  ivec *subset_rows = mk_random_ivec_subset_fast(all_rows, size);
  dym *dym_subset = mk_dym_subset(m, subset_rows);
  free_ivec(all_rows); free_ivec(subset_rows);
  return dym_subset;
}

dym *mk_two_random_dym_subsets(dym *m1,dym *m2,int size,dym **m2_subset)
{
  ivec *all_rows = mk_sequence_ivec(0, dym_rows(m1));
  ivec *subset_rows = mk_random_ivec_subset_fast(all_rows, size);
  dym *m1_subset = mk_dym_subset(m1, subset_rows);
  *m2_subset = mk_dym_subset(m2, subset_rows);
  free_ivec(all_rows); free_ivec(subset_rows);
  return m1_subset;
}

/**** MATH */

double safe_log(double x)
{
  return log(real_max(FLT_MIN,x));
}


/* Volume of a sphere of radius R in arbitrary dimension d:
                          d/2
                        pi      d
         V_{d}[R]  =   ------  R
                       (d/2)!

                        d      
                       2  ((d-1)/2)!    d/2   d
         V_{d}[R]  =  --------------  pi     R
                               1/2
                       (d)!  pi

    d           V_{d}[R]             k_d (the constant)
  -----     ----------------       --------
   -1         pi^-1 R^-1            0.3183
    0            1                  1.0000
    1            2R                 2.0000
    2          pi R^2               3.1415
    3       (4/3) pi R^3            4.1887
    4       (1/2) pi^2 R^4          4.9348
    5       (8/15) pi^2 R^5         5.2637
    6       (1/6) pi^3 R^6          5.1677
    7       (16/105) pi^3 R^7       4.7247
    8       (1/24) pi^4 R^8         4.0587
    9       (32/945) pi^4 R^9       3.2985

From http://www.seanet.com/%7Eksbrown/kmath163.htm. */
double sphere_volume(int d,double r)
{
  double v;
  if ((d - (double)floor(d)) != 0.0) /* d is non-integer */
    return 0;
  switch(d) {
    case 0: v = 1.0; break;
    case 1: v = 2.0*r; break;
    case 2: v = PI*r*r; break;
    case 3: v = (4.0/3.0)*PI*r*r*r; break;
    case 4: v = 0.5*PI*PI*r*r*r*r; break;
    case 5: v = (8.0/15.0)*PI*PI*r*r*r*r*r; break;
    case 6: v = (1.0/6.0)*PI*PI*PI*r*r*r*r*r*r; break;
    case 7: v = (16.0/105.0)*PI*PI*PI*r*r*r*r*r*r*r; break;
    case 8: v = (1.0/24.0)*PI*PI*PI*PI*r*r*r*r*r*r*r*r; break;
    case 9: v = (32.0/945.0)*PI*PI*PI*PI*r*r*r*r*r*r*r*r*r; break;
    default:
      if (d % 2 == 0) /* d is even */
        v = pow(PI,d/2) * pow(r,d) / factorial(d/2);
      else /* d is odd */
        v = pow(2,d) * pow(r,d) * pow(PI,(d-1)/2) * factorial((d-1)/2) /
            factorial(d);
      break;
  }
  return v;
}

/**
 * Evaluates continued fraction for incomplete beta function by
 * modified Lentz's method. Also taken directly from numerical
 * recipes.
 */
double betacf(double a,double b,double x)
{
  int m,m2;
  double aa,c,d,del,h,qab,qam,qap;
  qab=a+b;
  qap=a+1.0;
  qam=a-1.0;
  c=1.0;
  d=1.0-qab*x/qap;
  if(fabs(d) < 1.0e-30) d=1.0e-30;
  d=1.0/d;
  h=d;
  for(m=1;m<=100;m++) {
    m2=2*m;
    aa=m*(b-m)*x/((qam+m2)*(a+m2));
    d=1.0+aa*d;
    if(fabs(d) < 1.0e-30) d=1.0e-30;
    c=1.0+aa/c;
    if(fabs(c) < 1.0e-30) c=1.0e-30;
    d=1.0/d;
    h*=d*c;
    aa=-(a+m)*(qab+m)*x/((a+m2)*(qap+m2));
    d=1.0+aa*d;
    if(fabs(d)<1.0e-30) d=1.0e-30;
    c=1.0+aa/c;
    if(fabs(c)<1.0e-30) c=1.0e-30;
    d=1.0/d;
    del=d*c;
    h*=del;
    if(fabs(del-1.0) < 3.0e-7) break;
  }
  return h;
}

/**
 * Returns the incomplete beta function: assumes that 0 <= x <=
 * 1. This function uses the continued fraction representation of the
 * incomplete beta function since it offers better convergence than
 * its Taylor expansion. This method is directly modified from the numerical
 * recipes.
 */
double betai(double x,double a,double b)
{
  double bt;  // factors in front of the continued fraction.
  
  if(x==0.0 || x==1.0) bt=0.0;
  else
    bt=exp(a*log(x)+b*log(1.0-x));
    
  if(x < (a+1.0)/(a+b+2.0))
    return bt*betacf(a,b,x)/a;
  else
    return exp((am_lgamma(a)+am_lgamma(b))-am_lgamma(a+b))-
      bt*betacf(b,a,1.0-x)/b;
  return -777;
}


/**** KERNELS */

double spherical_star(double bwsqd,double dsqd)
{
  double retval=0.0;

  if(dsqd<4*bwsqd) {
    /**
     * Base unnormalized value if x^2 <= 4h^2.
     */
    retval=(-2.0*sqrt(dsqd)*pow(bwsqd-dsqd/4.0,Num_dims/2.0)+
	    (Num_dims-1.0)*pow(sqrt(bwsqd),Num_dims+1)*sqrt(4.0-dsqd/bwsqd)*
	    betai(1.0-dsqd/(4.0*bwsqd),0.5*(Num_dims-1),1.5))/
      sqrt(4*bwsqd-dsqd);

    /**
     * Additional subtraction if x^2 <= h^2.
     */
    if(dsqd<bwsqd) {
      retval-=2.0*pow(sqrt(bwsqd),Num_dims)*sqrt(PI)*
	exp(am_lgamma(0.5*(Num_dims+1))-am_lgamma(1+Num_dims/2.0));
    }
  }
  return retval;
}

double epanechnikov_star(double bwsqd,double dsqd)
{
  double retval=0.0;
  
  if(dsqd<=4*bwsqd) {
    retval=pow(2,-Num_dims)*pow(4*bwsqd-dsqd,0.5*(Num_dims-1))*
      (20*bwsqd*pow(dsqd,1.5)+(2*Num_dims-3)*pow(dsqd,2.5))/
       (Num_dims*Num_dims-1.0)+pow(sqrt(bwsqd),2+Num_dims)*
       ((4*bwsqd-5*dsqd)*betai(1-dsqd/(4*bwsqd),0.5*(Num_dims-1),1.5)+
	(5*dsqd-8*bwsqd)*betai(1-dsqd/(4*bwsqd),(1+Num_dims)/2.0,1.5)+
	4*bwsqd*betai(1-dsqd/(4*bwsqd),(3+Num_dims)/2.0,1.5));
    if(dsqd<=bwsqd) {
      retval-=15*pow(2,-1-Num_dims)*Num_dims*(Num_dims+2)*
	pow(sqrt(bwsqd),2.0+Num_dims)*PI*(bwsqd-dsqd)*
	exp(am_lgamma(Num_dims-1)-am_lgamma(2+Num_dims/2.0)*2);
    }
  }
  return retval;
}

double kernel_unnorm(double bwsqd, double dsqd, double auxconst)
{
  if (SPHERICAL_KERNEL) return ((dsqd <= bwsqd) ? 1.0:0.0);
  else if (EPANECHNIKOV_KERNEL) return ((dsqd <= bwsqd) ? (1-dsqd/bwsqd):0.0);
  else if (GAUSSIAN_KERNEL) return (exp(-0.5*dsqd/bwsqd));
  else if (GAUSSIAN_STAR_KERNEL) {
    return (auxconst * exp(-0.25*dsqd/bwsqd)
	    - 2.0 * exp(-0.5 *dsqd/bwsqd));
  }
  else if (AITCHISON_AITKEN_KERNEL) return ( pow(bwsqd,(auxconst - dsqd)) 
                                             * pow((1.0 - bwsqd),dsqd) );
  else return -1;
}

/* we divide by this - it is the normalizing constant for the kernel function
   itself; it doesn't include the number of data */
void compute_norm_const(int num_dims, double bwsqd, double *norm, double *aux)
{
  double norm_const=-777.77, aux_const=-777.77;

  /* Spherical: norm_const = V(h,D) */
  if (SPHERICAL_KERNEL) {
    norm_const = sphere_volume((double)num_dims,sqrt(bwsqd));
  }
  else if(SPHERICAL_STAR_KERNEL) {
    norm_const = pow(sqrt(bwsqd),-2*Num_dims)*pow(PI,-0.5-0.5*Num_dims)*
      exp(am_lgamma(1.0+0.5*Num_dims)*2-am_lgamma((1.0+Num_dims)*0.5));
  }
  /* Epanechnikov: norm_const = 2*V(h,D)/(D+2) */
  else if (EPANECHNIKOV_KERNEL) {
    norm_const = 2.0*sphere_volume((double)num_dims,sqrt(bwsqd))/
      ((double)num_dims+2.0);
  }
  else if (EPANECHNIKOV_STAR_KERNEL) {
    norm_const = (4.0/15.0)*pow(sqrt(bwsqd),-2*(Num_dims+2))*
      pow(PI,-0.5-0.5*Num_dims)*exp(2.0*am_lgamma(2.0+0.5*Num_dims)-
				    am_lgamma(0.5*(Num_dims-1.0)));
  }
  /* Gaussian:  norm_const = (2*PI*h^2)^(D/2)   Bishop p.54 */
  else if (GAUSSIAN_KERNEL) {
    norm_const = pow((2.0*PI*bwsqd),(double)num_dims/2.0);
  }
  /* Gaussian-star:  norm_const = (2*PI*h^2)^(D/2)  see my notes */
  else if (GAUSSIAN_STAR_KERNEL) {
    aux_const = 1.0 / pow(2.0,(double)num_dims/2.0);
    norm_const = pow((2.0*PI*bwsqd),(double)num_dims/2.0);
  }
  *norm = norm_const; *aux = aux_const;
}

/* searches backward; returns the index of the smallest bin containing the
   distance in question */
int get_bin_of_dist(double dist,bwinfo *bws,int row,int b_lo,int b_hi)
{
  int b;
  /* search in reverse */
  if(KDE_MODEL || WKDE_MODEL || NWR_MODEL || MEANSHIFT_MODEL) {
    if ( dist > dyv_ref((bws->bwsqds).bwsqds_dyv,b_hi)) 
      return -1; /* outside of all radii considered */ 
    
    for (b = b_hi; b > b_lo; b--) {
      if ( dist > dyv_ref((bws->bwsqds).bwsqds_dyv,b-1) ) return b;
    }
  }
  else if(VKDE_MODEL || VNWR_MODEL) {
    if ( dist > dym_ref((bws->bwsqds).bwsqds_dym,row,b_hi) ) 
      return -1; /* outside of all radii considered */ 
    
    for (b = b_hi; b > b_lo; b--) {
      if ( dist > dym_ref((bws->bwsqds).bwsqds_dym,row,b-1) ) return b;
    }
  }
  return b_lo;
}

/**** BANDWIDTHS */

/* note: doesn't allocate dyv's inside bwinfo */
bwinfo *mk_bwinfo(double bwmin, double bwmax, int numbws)
{
  bwinfo *bws = AM_MALLOC(bwinfo);
  bws->numbws = numbws; bws->bwmin = bwmin; bws->bwmax = bwmax;
  bws->b_lo = 0; bws->b_hi = numbws - 1;
  (bws->bwsqds).bwsqds_dyv = NULL; 
  (bws->bwsqds).bwsqds_dym = NULL;
  bws->bwmaxs = NULL;
  (bws->norm_consts).norm_consts_dyv = NULL;
  (bws->norm_consts).norm_consts_dym = NULL;
  (bws->aux_consts).aux_consts_dyv = NULL;
  (bws->aux_consts).aux_consts_dym = NULL;

  if(KDE_MODEL || WKDE_MODEL || VKDE_MODEL) {
    bws->numerator_dim = 1;
    bws->denominator_dim = 0;
  }
  else if(NWR_MODEL || VNWR_MODEL) {
    bws->numerator_dim = bws->denominator_dim = 1;
  }
  else if(LPR_MODEL || VLPR_MODEL) {
    /* The following should be changed to extend beyond linear regression */
    bws->numerator_dim = bws->denominator_dim = 1+Num_dims;
  }
  return bws;
}

void free_bwinfo(bwinfo *bws)
{
  if(KDE_MODEL || WKDE_MODEL || LPR_MODEL || NWR_MODEL || MEANSHIFT_MODEL) {
    if ((bws->bwsqds).bwsqds_dyv != NULL) 
      free_dyv((bws->bwsqds).bwsqds_dyv);
    if ((bws->norm_consts).norm_consts_dyv != NULL)
      free_dyv((bws->norm_consts).norm_consts_dyv);
    if ((bws->aux_consts).aux_consts_dyv != NULL)
      free_dyv((bws->aux_consts).aux_consts_dyv);
  }
  else if(VNWR_MODEL || VLPR_MODEL) {
    if ((bws->bwsqds).bwsqds_dym != NULL)
      free_dym((bws->bwsqds).bwsqds_dym);
    if (bws->bwmaxs != NULL)
      free_dyv(bws->bwmaxs);
    if ((bws->norm_consts).norm_consts_dym != NULL)
      free_dym((bws->norm_consts).norm_consts_dym);
    if ((bws->aux_consts).aux_consts_dym != NULL)
      free_dym((bws->aux_consts).aux_consts_dym);
  }
  AM_FREE(bws,bwinfo);
}

/* allocates dyv's inside bwinfo; 
   sets bandwidths and normalization constants for kernels */
void mk_set_bwinfo(bwinfo *bws, dym *data)
{
  double bwmin = bws->bwmin, bwmax = bws->bwmax; int numbws = bws->numbws;
  int b; double bwincr, norm, aux; 

  /* checking consistency and setting values if necessary */
  if(!VKDE_MODEL) {
    if ((bwmin == -1) && (bwmax == -1)) { // no-bw case
      double guess_bw = bandwidth_guess(data);
      bwmax = bwmin = 0.5 * guess_bw;
      fprintf(LOG,"Using %g, theoretical guess for bandwidth.\n",bwmin);  
    }
    if ((bwmin == -1) && (bwmax != -1)) { // only bwmax specified
      fprintf(LOG,"Error: can't specify bwmax without bwmin.\n"); exit(-1);
    }
    if ((bwmin != -1) && (bwmax == -1)) { // single-bw case
      bwmax = bwmin;
    }
    
    // both should be specified by here
    if (bwmax == bwmin) numbws = 1;
    if (bwmax < bwmin) { fprintf(LOG,"Error: bwmax < bwmin.\n"); exit(-1); }
    if (numbws < 1) { fprintf(LOG,"Error: numbws < 1.\n"); exit(-1); }
    bws->bwmin = bwmin; bws->bwmax = bwmax; bws->numbws = numbws;
    bws->b_lo = 0; bws->b_hi = numbws - 1;
    (bws->bwsqds).bwsqds_dyv = mk_dyv(numbws);
    (bws->norm_consts).norm_consts_dyv = mk_dyv(numbws); 
    (bws->aux_consts).aux_consts_dyv = mk_dyv(numbws); 
  }
  else {

    /**
     * For variable bandwidths (VKDE, VNWR), there should be some sort
     * of pilot density estimation using the all-nearest neighbor code here
     * and set up the bandwidths accordingly.
     */
    if((bws->bwsqds).bwsqds_dym == NULL) {
      (bws->bwsqds).bwsqds_dym = mk_zero_dym(dym_rows(data),numbws);
      (bws->aux_consts).aux_consts_dym = mk_zero_dym(dym_rows(data),numbws);
      (bws->norm_consts).norm_consts_dym = mk_zero_dym(dym_rows(data),numbws);
    }

    if(!VFIND_BW && (bws->bwsqds).bwsqds_dym == NULL) {
      int reference;
      ivec_array *k_best_rows = mk_array_of_zero_length_ivecs(dym_rows(data));
      dyv_array *k_best_dists = mk_array_of_zero_length_dyvs(dym_rows(data));
      ivec *kth_best_indx = mk_constant_ivec(dym_rows(data),0);
      dyv *kth_best_dist = mk_constant_dyv(dym_rows(data),FLT_MAX);

      printf("Computing 3-nearest neighbor distances for all reference ");
      printf("points for setting up %d cross-validation bandwidths...\n",
	     bws->numbws);

      batree_allknearest_neighbor(Knn,data,Dtree->root,Dtree->root,FLT_MAX,
				  k_best_rows,k_best_dists,
				  kth_best_indx,kth_best_dist);

      (bws->bwsqds).bwsqds_dym = mk_zero_dym(dym_rows(data),numbws);
          
      /**
       * Loop through each model point and set up bandwidth.
       */
      for(reference = 0; reference < dym_rows(data); reference++) {
	double knn_dist = dyv_ref(kth_best_dist,reference);
	double min_bw, max_bw;

	if(knn_dist > 0) {
	  min_bw = 0.01 * ((knn_dist));
	  max_bw = 12.0 * ((knn_dist));
	}
	else {
	  min_bw = max_bw = FLT_MIN;
	}

	if(numbws > 1) {
	  bwincr= (LOGSCALE ? fabs(log10(max_bw) - log10(min_bw))/(numbws-1) :
		   fabs(max_bw - min_bw)/(numbws-1));
	}
	else
	  bwincr = 0;

	/**
	 * Remember to square the bandwidths.
	 */
	for(b = 0; b < numbws; b++) {
	  double h = (LOGSCALE ? pow(10.0,(log10(min_bw) + 
					   ((double)b * bwincr))) :
		      min_bw + ((double)b * bwincr));
	  dym_set((bws->bwsqds).bwsqds_dym,reference,b,h * h);
	  compute_norm_const(Num_dims, h * h, &norm, &aux);
	  dym_set((bws->norm_consts).norm_consts_dym,reference,b, norm);
	  dym_set((bws->aux_consts).aux_consts_dym,reference,b, aux);
	}
      }
      free_ivec(kth_best_indx);
      free_dyv_array(k_best_dists);
      free_dyv(kth_best_dist);
      free_ivec_array(k_best_rows);
    }
    else if(VFIND_BW) {
      int reference;
      bws->bwmaxs=mk_dyv(numbws);

      if(numbws > 1) {
	bwincr = (LOGSCALE ? fabs(log10(bwmax) - log10(bwmin))/(numbws-1):
		  fabs(bwmax - bwmin)/(numbws-1));
      }
      else
	bwincr = 0;

      /**
       * In case of a variable bandwidth setup,
       */
      for(b = 0; b < numbws; b++) {
	double hmax = (LOGSCALE ? pow(10.0,(log10(bwmin) +
					    ((double)b * bwincr))) :
		       bwmin + ((double)b * bwincr));

	dyv_set(bws->bwmaxs,b,hmax);

	for(reference = 0; reference < dym_rows(data); reference++) {
	  double hr = hmax/(bws->mlbf)*dyv_ref(bws->lbf,reference);
	  dym_set((bws->bwsqds).bwsqds_dym,reference,b,hr * hr);
	  compute_norm_const(Num_dims, hr * hr, &norm, &aux);
	  dym_set((bws->norm_consts).norm_consts_dym,reference,b,norm);
	  dym_set((bws->aux_consts).aux_consts_dym,reference,b,aux);
	}
      }
    }
  }

  if(!VKDE_MODEL && !VNWR_MODEL && !VLPR_MODEL) {
    fprintf(LOG,"  ........................\n");
    if (numbws == 1) fprintf(LOG,"  Bandwidth: "); 
    else fprintf(LOG,"  Bandwidths: ");
    bwincr = (LOGSCALE ? fabs(log10(bwmax) - log10(bwmin))/(numbws-1) :
	      fabs(bwmax - bwmin)/(numbws-1));
    
    for (b = 0; b < numbws; b++) {
      double norm, aux, bw;
      if (numbws == 1) bw = bwmin;
      else {
	bw = (LOGSCALE ? pow(10.0,(log10(bwmin) + ((double)b * bwincr))) :
	      bwmin + ((double)b * bwincr));
      }
      fprintf(LOG,"%g ",bw);
      dyv_set((bws->bwsqds).bwsqds_dyv, b, bw*bw); 
      compute_norm_const(Num_dims, bw*bw, &norm, &aux);
      dyv_set((bws->norm_consts).norm_consts_dyv,b,norm);
      dyv_set((bws->aux_consts).aux_consts_dyv,b,aux);
    }
    fprintf(LOG,"\n");
  }
}

/* Scott p. 152, eq. 6.41; also Silverman p. 47-48, eq. 3.31 
   (univariate only for the Gaussian kernel only).
   Because, as standardly done, we use the same bandwidth across all 
   dimensions,
   we take the average standard deviation across all dimensions. 
   This could be improved by also computing the interquartile range, as 
   suggested by Silverman.
*/
double bandwidth_guess(dym *d)
{
  int num_data = dym_rows(d), num_dims = dym_cols(d), i;
  double avg_sdev = 0, bw;

  for (i=0; i<num_dims; i++) {
    dyv *v = mk_dyv_from_dym_col(d,i);
    double s = dyv_sdev(v);
    avg_sdev += s;
    free_dyv(v);
  }
  avg_sdev /= num_dims;

  bw = pow((4.0/(num_dims+2.0)),1.0/(num_dims+4.0)) * avg_sdev * 
       pow(num_data, -1.0/(num_dims+4.0));

  if(EPANECHNIKOV_KERNEL || EPANECHNIKOV_STAR_KERNEL) {
    bw = 10 * 2.214 * bw;
  }

  return bw;
}


/**** HEURISTIC SEARCH */

nodepair *mk_nodepair(NODE qnode, NODE dnode, double dl, double du, double h)
{
  nodepair *np = AM_MALLOC(nodepair);
  np->qnode = qnode; np->dnode = dnode;
  np->dl = dl; np->du = du; np->h = h;
  return np;
}

void free_nodepair(nodepair *np)
{
  AM_FREE(np,nodepair);
}

int compare_nodepairs(void * x, void * y)
{
  double a = ((nodepair*)x)->h, b = ((nodepair*)y)->h;
  if (a < b) return -1;
  if (a == b) return 0;
  return 1;
}

/* Given a node and two choices of node partner, selects the best
   (partner1) and second best (partner2) for the purposes of 
   all_nearest_search(). */
void best_node_partners(NODE nd, NODE nd1, NODE nd2, 
                        NODE *partner1, NODE *partner2)
{
  double d1 = hrect_dist_heuristic(nd->hr,nd1->hr);
  double d2 = hrect_dist_heuristic(nd->hr,nd2->hr);

  if ( d1 <= d2 ) { *partner1 = nd1; *partner2 = nd2;}
             else { *partner1 = nd2; *partner2 = nd1;}
}

/* single-tree version */
void best_node(dyv *pt, NODE nd1, NODE nd2, NODE *partner1, NODE *partner2)
{
  double d1 = hrect_dyv_min_metric_dsqd(Metric,nd1->hr,pt);
  double d2 = hrect_dyv_min_metric_dsqd(Metric,nd2->hr,pt);

  if ( d1 <= d2 ) { *partner1 = nd1; *partner2 = nd2;}
             else { *partner1 = nd2; *partner2 = nd1;}
}

/* Computes the 'distance' between two rectangles, using one of a number of
   possibilities for the distance function. */
double hrect_dist_heuristic(hrect *hr1, hrect *hr2)
{
  double dist_d, dist;
  int d;

  /* Computes the distance between the centers of two rectangles */
  if (CENTERS_DIST) {
    dyv *mid1 = mk_hrect_middle(hr1), *mid2 = mk_hrect_middle(hr2); 
    dist = dsqd_to_dyv(Metric,mid1,mid2);
    free_dyv(mid1); free_dyv(mid2);
  }
  if (NEG_CENTERS_DIST) {
    dyv *mid1 = mk_hrect_middle(hr1), *mid2 = mk_hrect_middle(hr2); 
    dist = -( dsqd_to_dyv(Metric,mid1,mid2) );
    free_dyv(mid1); free_dyv(mid2);
  }
  else if (OVERLAP_FACTOR) {
    dist = 1;
    for ( d = 0 ; d < hrect_size(hr1) ; d++ )
    {    
      double lo1 = hrect_lo_ref(hr1,d), lo2 = hrect_lo_ref(hr2,d);
      double hi1 = hrect_hi_ref(hr1,d), hi2 = hrect_hi_ref(hr2,d);
      
      if (hi1 < lo2)
        dist_d = -(dyv_ref(Metric,d)/(lo2 - hi1));
      else if (hi1 > lo2) {
        if (lo1 > hi2)
          dist_d = -(dyv_ref(Metric,d)/(lo1 - hi2));
        else if (lo1 < hi2) {
          if (lo1 < lo2) {
            if (hi1 < hi2) dist_d = (hi1 - lo2)/dyv_ref(Metric,d);
            else dist_d = (hi2 - lo2)/dyv_ref(Metric,d);
          }
          else {
            if (hi1 < hi2) dist_d = (hi1 - lo1)/dyv_ref(Metric,d);
            else dist_d = (hi2 - lo1)/dyv_ref(Metric,d);
          }
        }
        else { dist_d = 0; }
      } 
      else { dist_d = 0; }
      dist = dist * (1+dist_d);
    }
    dist = - dist; /* since better overlap is good */
  }
  else if (NUM_OVERLAPS) {
    dist = 0;
    for ( d = 0 ; d < hrect_size(hr1) ; d++ )
    {    
      double lo1 = hrect_lo_ref(hr1,d), lo2 = hrect_lo_ref(hr2,d);
      double hi1 = hrect_hi_ref(hr1,d), hi2 = hrect_hi_ref(hr2,d);
      
      if (hi1 < lo2)
        dist_d = (lo2 - hi1)/dyv_ref(Metric,d);
      else if (hi1 > lo2) {
        if (lo1 > hi2)
          dist_d = (lo1 - hi2)/dyv_ref(Metric,d);
        else if (lo1 < hi2) {
          if (lo1 < lo2) {
            if (hi1 < hi2) dist_d = -(hi1 - lo2)/dyv_ref(Metric,d);
            else dist_d = -(hi2 - lo2)/dyv_ref(Metric,d);
          }
          else {
            if (hi1 > hi2) dist_d = -(hi2 - lo1)/dyv_ref(Metric,d);
            else dist_d = -(hi1 - lo1)/dyv_ref(Metric,d);
          }
        }
        else { dist_d = 0; }
      } 
      else { dist_d = 0; }
      dist = dist + dist_d;
    }
  }
  else {
    /* Computes the dimension-wise distance between two rectangles.
       This is defined here as the maximum dimension-wise distance over all 
       dimensions; if there is overlap in a given dimension, the distance in
       that dimension is zero. */
    dist = 0;
    for ( d = 0 ; d < hrect_size(hr1) ; d++ )
    {    
      if ( hrect_hi_ref(hr1,d) < hrect_lo_ref(hr2,d) )
        dist_d=(hrect_lo_ref(hr2,d) - hrect_hi_ref(hr1,d)) / dyv_ref(Metric,d);
      else if ( hrect_hi_ref(hr2,d) < hrect_lo_ref(hr1,d) )
        dist_d=(hrect_lo_ref(hr1,d) - hrect_hi_ref(hr2,d)) / dyv_ref(Metric,d);
      else
        dist_d = 0;
  
      if (MIN_UNI_DIST)      { if (dist_d < dist) dist = dist_d; }
      else if (MAX_UNI_DIST) { if (dist_d > dist) dist = dist_d; }
      else if (AVG_UNI_DIST) { dist += dist_d; }
    }
    if (AVG_UNI_DIST) { dist /= hrect_size(hr1); }
  }

  return dist;
}

double recursive_hrect_dist(dyv *metric,NODE nd1,NODE nd2,int type,int look)
{
  if ( (look == 0) || (is_leaf(nd1) && is_leaf(nd2)) ) { 
    return hrect_metric_dsqd(metric,nd1->hr,nd2->hr,type);
  }
  else if ( is_leaf(nd1) && !is_leaf(nd2) )
  {
    double d1 = recursive_hrect_dist(metric,nd1,nd2->left,type,look-1);
    double d2 = recursive_hrect_dist(metric,nd1,nd2->right,type,look-1);
    return real_min(d1,d2);
  }
  else if ( !is_leaf(nd1) && is_leaf(nd2) )
  {
    double d1 = recursive_hrect_dist(metric,nd1->left,nd2,type,look-1);
    double d2 = recursive_hrect_dist(metric,nd1->right,nd2,type,look-1);
    return real_min(d1,d2);
  }
  else if ( !is_leaf(nd1) && !is_leaf(nd2) )
  {
    double d1 = recursive_hrect_dist(metric,nd1->left,nd2->left,type,look-1);
    double d2 = recursive_hrect_dist(metric,nd1->left,nd2->right,type,look-1);
    double d3 = recursive_hrect_dist(metric,nd1->right,nd2->left,type,look-1);
    double d4 = recursive_hrect_dist(metric,nd1->right,nd2->right,type,look-1);
    return real_min(real_min(d1,d2),real_min(d3,d4));
  }
  return -777.7;
}


void compute_maxerr(dym *l, dym *u, dyv *maxerr, bwinfo *bws)
{
  int i, b; 

  for (b = 0; b < bws->numbws; b++) {
    double max_error = 0.0;
    for ( i = 0 ; i < Num_queries ; i++ ) {
      double dens_l = dym_ref(l,i,b), 
	dens_u=dym_ref(u,i,b);

      if(RELATIVE_PRUNING) {
	double err = fabs(dens_u - dens_l)/fabs(dens_l);

	if(fabs(dens_l)==fabs(dens_u)) err = 0.0;
	
	if (err > max_error && !isinf(err)) max_error = err;
	
	if(err > Tau) {
	  numOverTau++;
	  
	  /*
	  if(printMaxerr)
	    printf("%g against %g...\n", dens_u, dens_l);
	  */
	}
      }
      else {
	double err = fabs(dens_u - dens_l);

	if(err > max_error) max_error = err;
	
	if(err > Tau) {
	  numOverTau++;
	}
      }
    }
    dyv_set(maxerr, b, max_error);
  }
}

void compute_avgerr(dym *l, dym *u, dyv *avgerr, bwinfo *bws)
{
  int i, b; 
  for (b = 0; b < bws->numbws; b++) {
    double avg_error = 0.0;
    for ( i = 0 ; i < Num_queries ; i++ ) {
      double dens_l = dym_ref(l,i,b), dens_u=dym_ref(u,i,b);
      double err;

      if(RELATIVE_PRUNING) {
	err = fabs(dens_u - dens_l)/fabs(dens_l);
	if (fabs(dens_l)==fabs(dens_u)) err = 0.0;
      }
      else {
	err = fabs(dens_u - dens_l);
      }
      avg_error += err;
    }
    dyv_set(avgerr, b, avg_error / Num_queries);
  }
}


/**** WEIGHT PASSING */

void initialize_wgts_in_nodes(NODE node, dyv *neg_min, dyv *pos_max)
{
  int b;
  
  zero_dyv((node->mass_t).mass_t_dyv);
  zero_dyv((node->mass_e).mass_e_dyv);
  zero_dyv((node->owed_l).owed_l_dyv); 
  zero_dyv((node->owed_u).owed_u_dyv);
  zero_dym((node->coeffs).coeffs_dym);
  zero_ivec((node->coeffOrder).coeffOrder_ivec);
  zero_dym((node->lcoeffs).lcoeffs_dym);
  zero_ivec((node->lcoeffOrder).lcoeffOrder_ivec);

  zero_dyv((node->mass_e2).mass_e2_dyv);
  zero_dyv((node->owed_l2).owed_l2_dyv);
  zero_dyv((node->owed_u2).owed_u2_dyv);
  zero_dym((node->coeffs2).coeffs2_dym);
  zero_ivec((node->coeffOrder2).coeffOrder2_ivec);
  zero_dym((node->lcoeffs2).lcoeffs2_dym);
  zero_ivec((node->lcoeffOrder2).lcoeffOrder2_ivec);

  for(b = 0; b < Qtree->bws->numbws; b++) {
    dyv_set((node->mass_l).mass_l_dyv,b,dyv_ref(neg_min,b));
    dyv_set((node->mass_u).mass_u_dyv,b,0);
    dyv_set((node->mass_l2).mass_l2_dyv,b,0);
    dyv_set((node->mass_u2).mass_u2_dyv,b,dyv_ref(pos_max,b));
  }
  
  if (is_leaf(node)) {
    zero_dyv((node->more_l).more_l_dyv);
    zero_dyv((node->more_u).more_u_dyv);
    zero_dyv((node->more_l2).more_l2_dyv);
    zero_dyv((node->more_u2).more_u2_dyv);
  }
  else {
    initialize_wgts_in_nodes(node->left,neg_min,pos_max);
    initialize_wgts_in_nodes(node->right,neg_min,pos_max);
  }
}

void finalize_wgts_in_nodes(NODE node, dym *neg_e, dym *pos_e, bwinfo *bws,
			    dym *q)
{
  int i, b, n = ivec_size(node->rows);

  /* if a chunk of owed weight was never passed, it must now be 
     incorporated. there is only one chunk of unincorporated weight 
     from root to node in this case, so it is correct to do this. */
  if (is_leaf(node)) {
    for ( i=0; i<n; i++ ) {
      int row_i = ivec_ref(node->rows,i);

      for ( b = bws->b_lo; b <= bws->b_hi; b++ ) {
	int p_alpha = ivec_ref((node->lcoeffOrder).lcoeffOrder_ivec,b);

	dym_increment(neg_e, row_i, b, dyv_ref((node->mass_e).mass_e_dyv, b));
	dym_increment(pos_e, row_i, b, dyv_ref((node->mass_e2).mass_e2_dyv,b));

	if(p_alpha > 0) {
          if(GAUSSIAN_KERNEL)
            dym_increment(pos_e, row_i, b,
                          evaluateTaylorExpansion
                          (Qtree, q, row_i, node, b, p_alpha));
          else if(EPANECHNIKOV_KERNEL)
            dym_increment(pos_e, row_i, b,
                          evaluateTaylorExpansionEpan
                          (Qtree, q, row_i, node, b, p_alpha));
        }
      }
    }
  }
  else {

    /**
     * Propagate down the finite difference approximations!
     */
    dyv_plus((node->left->mass_e).mass_e_dyv, (node->mass_e).mass_e_dyv,
	     (node->left->mass_e).mass_e_dyv);
    dyv_plus((node->right->mass_e).mass_e_dyv, (node->mass_e).mass_e_dyv,
	     (node->right->mass_e).mass_e_dyv);
    dyv_plus((node->left->mass_e2).mass_e2_dyv, (node->mass_e2).mass_e2_dyv,
	     (node->left->mass_e2).mass_e2_dyv);
    dyv_plus((node->right->mass_e2).mass_e2_dyv, (node->mass_e2).mass_e2_dyv,
	     (node->right->mass_e2).mass_e2_dyv);
    
    /**
     * If Taylor coefficients have been accumulated on this node, we need
     * to pass it down to the children.
     */
    if(GAUSSIAN_KERNEL && !VKDE_MODEL)
      translateLocalToLocal(Qtree, node);
    else if(EPANECHNIKOV_KERNEL && !VKDE_MODEL)
      translateLocalToLocalEpan(Qtree, node);

    finalize_wgts_in_nodes(node->left,neg_e,pos_e,bws,q);
    finalize_wgts_in_nodes(node->right,neg_e,pos_e,bws,q);
  }
}

void update_bounds_delay(NODE qnode,dym *q,NODE dnode,dym *d,dym *w,
			 dym *neg_l,dym *neg_e,dym *neg_u,
			 dym *pos_l,dym *pos_e,dym *pos_u,
			 dyv *neg_dl,dyv *neg_de,dyv *neg_du,
			 dyv *pos_dl,dyv *pos_de,
			 dyv *pos_du,ivec *pos_p_alphaM2Ls,
			 ivec *pos_p_alphaDMs,ivec *pos_p_alphaDLs,dyv *dt,
			 int b_lo,int b_hi)
{
  int b;

  // incorporate into self
  for ( b = b_lo; b <= b_hi; b++ ) {
    if(neg_dl != NULL)
      dyv_increment((qnode->mass_l).mass_l_dyv, b, dyv_ref(neg_dl, b));
    
    if(neg_de != NULL)
      dyv_increment((qnode->mass_e).mass_e_dyv, b, dyv_ref(neg_de, b));

    if(neg_du != NULL)
      dyv_increment((qnode->mass_u).mass_u_dyv, b, dyv_ref(neg_du, b));
    
    if(dt != NULL)
      dyv_increment((qnode->mass_t).mass_t_dyv, b, dyv_ref(dt, b));

    if(pos_dl != NULL)
      dyv_increment((qnode->mass_l2).mass_l2_dyv, b, dyv_ref(pos_dl, b));

    if(pos_de != NULL)
      dyv_increment((qnode->mass_e2).mass_e2_dyv, b, dyv_ref(pos_de, b));

    if(pos_du != NULL)
      dyv_increment((qnode->mass_u2).mass_u2_dyv, b, dyv_ref(pos_du, b));

    /**
     * Compute the required multipole pruning stuff here (delayed
     * previously).
     */
    process_series_expansion(qnode,q,dnode,d,pos_p_alphaM2Ls,pos_p_alphaDMs,
			     pos_p_alphaDLs,b,pos_l,pos_e,pos_u,w);
  } /* looping over all bandwidths */
    
  if (is_leaf(qnode)) {
    for ( b = b_lo; b <= b_hi; b++ ) {
      if(neg_dl != NULL)
	dyv_increment((qnode->more_l).more_l_dyv, b, dyv_ref(neg_dl, b));
      if(neg_du != NULL)
	dyv_increment((qnode->more_u).more_u_dyv, b, dyv_ref(neg_du, b));
      if(pos_dl != NULL)
	dyv_increment((qnode->more_l2).more_l2_dyv, b, dyv_ref(pos_dl, b));
      if(pos_du != NULL)
	dyv_increment((qnode->more_u2).more_u2_dyv, b, dyv_ref(pos_du, b));
    }
  } else {
    // pass to direct children only
    for ( b = b_lo; b <= b_hi; b++ ) {
      if(neg_dl != NULL) {
	dyv_increment((qnode->left->owed_l).owed_l_dyv,b,dyv_ref(neg_dl, b));
	dyv_increment((qnode->right->owed_l).owed_l_dyv,b,dyv_ref(neg_dl, b));
      }
      if(neg_du != NULL) {
	dyv_increment((qnode->left->owed_u).owed_u_dyv,b,dyv_ref(neg_du, b));
	dyv_increment((qnode->right->owed_u).owed_u_dyv,b,dyv_ref(neg_du, b));
      }
      if(pos_dl != NULL) {
	dyv_increment((qnode->left->owed_l2).owed_l2_dyv,b,dyv_ref(pos_dl, b));
	dyv_increment((qnode->right->owed_l2).owed_l2_dyv,b,dyv_ref(pos_dl,b));
      }
      if(pos_du != NULL) {
	dyv_increment((qnode->left->owed_u2).owed_u2_dyv,b,dyv_ref(pos_du,b));
	dyv_increment((qnode->right->owed_u2).owed_u2_dyv,b,dyv_ref(pos_du,b));
      }
    }
  }
}


/**** LIKELIHOOD */

void compute_loglike_est(dym *e, dyv *ll_e, bwinfo *bws)
{
  int i, b; 
  zero_dyv(ll_e);
  for (b = 0; b < bws->numbws; b++) {
    double C;
    
    if(VKDE_MODEL) {
      C = (-safe_log(Num_data)) * Num_queries;
    }
    else {
      double norm = dyv_ref((bws->norm_consts).norm_consts_dyv,b);
      C = (-safe_log(norm)-safe_log(Num_data))*Num_queries;
    }

    // add up logs
    for ( i = 0 ; i < Num_queries ; i++ ) {
      double dens_e = dym_ref(e,i,b);
      dyv_increment(ll_e, b, safe_log(dens_e));
    }
    // add constant
    dyv_increment(ll_e, b, C);
  }
}

void normalize_loglike_est(dym *e, bwinfo *bws)
{
  int i, b; 
  for (b = 0; b < bws->numbws; b++) {
    double C;

    if(VKDE_MODEL) {
      C = Num_data;
    }
    else {
      double norm = dyv_ref((bws->norm_consts).norm_consts_dyv,b);
      C = norm * Num_data;
    }

    // divide by constant
    for ( i = 0 ; i < Num_queries ; i++ ) {
      double dens_e = dym_ref(e,i,b);
      dym_set(e,i,b,dens_e/C);
    }
  }
}

void compute_loglike_est_and_normalize(dym *l,dym *e,dym *u,dyv *ll_l,
				       dyv *ll_e,dyv *ll_u,bwinfo *bws)
{
  int i, b;
  zero_dyv(ll_l); zero_dyv(ll_e); zero_dyv(ll_u);
  for (b = 0; b < bws->numbws; b++) {
    double C1;
    double C2;

    if(VKDE_MODEL) {
      C1 = Num_data;
      C2 = (-safe_log(Num_data))*Num_queries;
    }
    else {
      double norm=dyv_ref((bws->norm_consts).norm_consts_dyv,b);
      C1 = norm * Num_data;
      C2 = (-safe_log(norm)-safe_log(Num_data))*Num_queries;
    }

    for ( i = 0 ; i < Num_queries ; i++ ) {
      double dens_l = dym_ref(l,i,b), dens_u=dym_ref(u,i,b);
      double dens_e = dym_ref(e,i,b);

      // add up logs
      dyv_increment(ll_l, b, safe_log(dens_l));
      dyv_increment(ll_e, b, safe_log(dens_e));
      dyv_increment(ll_u, b, safe_log(dens_u));
      // divide by constant
      dym_set(l, i, b, dens_l/C1);
      dym_set(e, i, b, dens_e/C1);
      dym_set(u, i, b, dens_u/C1);
    }
    // add constant
    dyv_increment(ll_l, b, C2);
    dyv_increment(ll_e, b, C2);
    dyv_increment(ll_u, b, C2);
  }
}

void compute_loglike_bounds_only(dym *l, dym *u,dyv *ll_l,dyv *ll_u,
                                 bwinfo *bws, int b_lo, int b_hi)
{
  int i, b;
  zero_dyv(ll_l); zero_dyv(ll_u);
  for (b = b_lo; b <= b_hi; b++) {
    double C2;

    if(VKDE_MODEL) {
      C2 = (-safe_log(Num_data)) * Num_queries;
    }
    else {
      double norm = dyv_ref((bws->norm_consts).norm_consts_dyv,b);
      C2 = (-safe_log(norm)-safe_log(Num_data))*Num_queries;
    }

    for ( i = 0 ; i < Num_queries ; i++ ) {
      double dens_l = dym_ref(l,i,b), dens_u=dym_ref(u,i,b);
      if (dens_u < dens_l) {
        dens_u = dens_l;  //correct rare float errors
      }
      // add up logs
      dyv_increment(ll_l, b, safe_log(dens_l));
      dyv_increment(ll_u, b, safe_log(dens_u));
      // dividing by constant C1 - NOT DONE HERE
    }
    // add constant
    dyv_increment(ll_l, b, C2);
    dyv_increment(ll_u, b, C2);
  }
}

/* computes the importance sampling error */
dyv *mk_iscv(dym *e,dyv *f,bwinfo *bws)
{
  dyv *iscv_e = mk_dyv(bws->numbws); int b;
  for(b = 0; b < bws->numbws; b++) {
    double iscv=0.0; int i;
    double I_est = 0.0;
    
    /**
     * First go through each query density estimate to compute the integral
     * estimate.
     */
    for(i = 0; i < Num_queries; i++) {
      I_est += dym_ref(e,i,b) / dyv_ref(f,i); 
    }
    I_est /= Num_queries;
    for(i = 0; i < Num_queries; i++) {
      double diff = dyv_ref(f,i) - I_est * dym_ref(e,i,b);
      iscv += diff*diff/dym_ref(e,i,b);
    }
    iscv /= (Num_queries * Num_queries);
    dyv_set(iscv_e,b,iscv);
  }
  return iscv_e;
}

/* computes the likelihood score */
dyv *mk_lkcv(dyv *ll_e, bwinfo *bws)
{
  dyv *lkcv_e = mk_dyv(bws->numbws); int b;
  for (b = 0; b < bws->numbws; b++) {
    dyv_set(lkcv_e, b, dyv_ref(ll_e, b)/Num_queries);
  }
  return lkcv_e;
}

/* computes the M_1(h) score of Silverman '86 */
dyv *mk_lscv(dym *e, bwinfo *bws)
{
  dyv *lscv_e = mk_dyv(bws->numbws); int b;
  for (b = 0; b < bws->numbws; b++) {
    double lscv=0.0; int i;
    double bwsqd=dyv_ref((bws->bwsqds).bwsqds_dyv,b);

    for ( i = 0 ; i < Num_queries ; i++ ) {
      lscv += dym_ref(e,i,b);
    }
    // assuming these e values have already been processed by 
    // compute_loglike_est_and_normalize(), they've been divided by norm * N
    lscv /= Num_queries;

    /**
     * Potential confusion here!
     * Because we are using the lscv form of the kernel when this function
     * is called, simply invoking K(0) will use the lscv kernel, not
     * the original kernel, resulting in a wrong sum!
     */
    if(SPHERICAL_STAR_KERNEL) {
      lscv+=2.0/(sphere_volume((double)Num_dims,sqrt(bwsqd))*Num_queries);
    }
    else if(EPANECHNIKOV_STAR_KERNEL) {
      lscv+=2.0/((2.0*sphere_volume((double)Num_dims,sqrt(bwsqd))/
		  ((double)Num_dims+2.0))*Num_queries);
    }
    else if(GAUSSIAN_STAR_KERNEL)
      lscv+=2.0/(pow((2.0*PI*bwsqd),(double)Num_dims/2.0)*Num_queries);
    
    dyv_set(lscv_e, b, lscv);
  }
  return lscv_e;
}


/**** OUTPUT */

void make_plotfile(char *basename, char *suffix, dyv *ll_e, bwinfo *bws)
{
  FILE *PLOG; int b; int allocsize;
  char *f = make_extended_name(basename,suffix,&allocsize); 

  PLOG = fopen(f,"w"); AM_FREE_ARRAY(f,char,allocsize);
  for (b = 0; b < bws->numbws; b++) {
    double bw = sqrt(dyv_ref((bws->bwsqds).bwsqds_dyv,b));
    fprintf(PLOG,"%g %g\n",bw,dyv_ref(ll_e,b));
  }
  fclose(PLOG);
}

void print_output(dym *l, dym *e, dym *u, 
                  dyv *ll_l, dyv *ll_e, dyv *ll_u, 
                  char *basename, dym *d, dym *q, dyv *w, bwinfo *bws)
{
  int b, numbws = bws->numbws; char *f;
  int allocsize;
  dyv *lkcv_l=0, *lkcv_e=0, *lkcv_u=0, *lscv_l=0, *lscv_e=0, *lscv_u=0;
  dyv *maxerr=mk_dyv(numbws), *avgerr=mk_dyv(numbws);
  dyv *true_maxerr=mk_dyv(numbws), *true_avgerr=mk_dyv(numbws);
  dym *true = mk_zero_dym(Num_queries,numbws);
  dyv *ll_true = mk_dyv(numbws); 

  if (LK_CV) { lkcv_l=mk_lkcv(ll_l,bws);lkcv_e=mk_lkcv(ll_e,bws);
               lkcv_u=mk_lkcv(ll_u,bws);}
  if (LS_CV) { lscv_l=mk_lscv(l,bws); lscv_e=mk_lscv(e,bws);
               lscv_u=mk_lscv(u,bws);}

  numOverTau = 0;
  printMaxerr = 0;
  compute_maxerr(l,u,maxerr,bws); compute_avgerr(l,u,avgerr,bws);

  if (TRUEDIFF) {
    fprintf(LOG,"Computing true density exhaustively for comparison.\n");

    run_exhaustive_kde(q,d,w,true,ll_true,bws);
        
    printMaxerr = 1;
    numOverTau = 0;
    compute_maxerr(true,e,true_maxerr,bws);
    compute_avgerr(true,e,true_avgerr,bws);
  }

  if (DEBUG) { write_dyv_as_col("record_l",Record_l,"w"); free_dyv(Record_l);
               write_dyv_as_col("record_u",Record_u,"w"); free_dyv(Record_u); }

  for (b = 0; b < numbws; b++) {
    double bw = 0.0;

    if(!VKDE_MODEL && !VNWR_MODEL)
      bw = sqrt(dyv_ref((bws->bwsqds).bwsqds_dyv,b));
    else {
      int reference;
      for(reference = 0;reference<
	    dym_rows((bws->bwsqds).bwsqds_dym); reference++) {
	bw += sqrt(dym_ref((bws->bwsqds).bwsqds_dym,reference,b));
      }
      bw /= ((double)dym_rows((bws->bwsqds).bwsqds_dym));
    }

    fprintf(LOG,"bw = %g: ll %g [%g %g]  ",bw,dyv_ref(ll_e,b),dyv_ref(ll_l,b),
           dyv_ref(ll_u,b));
    if (LK_CV) fprintf(LOG,"lkcv %g [%g %g]  ",dyv_ref(lkcv_e,b),
                      dyv_ref(lkcv_l,b),dyv_ref(lkcv_u,b));
    if (LS_CV) fprintf(LOG,"lscv %g [%g %g]",dyv_ref(lscv_e, b),
                      dyv_ref(lscv_l,b),dyv_ref(lscv_u,b));
    fprintf(LOG,"\n");
    fprintf(LOG,"     Maximum log-likelihood error: %g\n",
           fabs(dyv_ref(ll_u,b)-dyv_ref(ll_l,b))/fabs(dyv_ref(ll_l,b)) );
    fprintf(LOG,"     Maximum per-datum error: %g\n", dyv_ref(maxerr,b));
    fprintf(LOG,"     Average per-datum error: %g\n", dyv_ref(avgerr,b));
    if (TRUEDIFF) {
      fprintf(LOG,"\n");
      fprintf(LOG,"     True log-likelihood error: %g\n", 
	      fabs(dyv_ref(ll_e,b)-dyv_ref(ll_true,b))/
	      fabs(dyv_ref(ll_true,b)));
      fprintf(LOG,"     True maximum per-datum error: %g\n", 
             dyv_ref(true_maxerr,b));
      fprintf(LOG,"     True average per-datum error: %g\n", 
             dyv_ref(true_avgerr,b));
      fprintf(LOG,"     %d queries went over the estimate...\n", numOverTau);
    }
  }

  if(!VKDE_MODEL && !VNWR_MODEL) {
    if (LS_CV) make_plotfile(basename,".plot_lscv",lscv_e,bws);
    if (LK_CV) make_plotfile(basename,".plot_lkcv",lkcv_e,bws);
    else       make_plotfile(basename,".plot_lkcv",ll_e,bws);
  }
  f = make_extended_name(basename,".dens",&allocsize); write_dym(f,e,"w"); 
  AM_FREE_ARRAY(f,char,allocsize);
  f = make_extended_name(basename,".dens_lo",&allocsize); write_dym(f,l,"w");
  AM_FREE_ARRAY(f,char,allocsize);
  f = make_extended_name(basename,".dens_hi",&allocsize); write_dym(f,u,"w"); 
  AM_FREE_ARRAY(f,char,allocsize);

  if (LK_CV) { free_dyv(lkcv_l); free_dyv(lkcv_e); free_dyv(lkcv_u); }
  if (LS_CV) { free_dyv(lscv_l); free_dyv(lscv_e); free_dyv(lscv_u); }
  free_dyv(maxerr); free_dyv(avgerr); free_dym(true); free_dyv(ll_true);
  free_dyv(true_maxerr); free_dyv(true_avgerr); 
}

/* NO CENTROID **************************************************************
 */

void run_indepbw(dym *q,dym *d,dyv *w,dym *l,dym *e,dym *u,
		 dyv *ll_l,dyv *ll_e,dyv *ll_u,bwinfo *bws)
{
  int b; double time1, time2, tot_time = 0; time1 = get_time();    

  /* loop over all bandwidths */
  for (b = 0; b < bws->numbws; b++) {
    dym *l0 = mk_zero_dym(Num_queries,1); dyv *ll_l0 = mk_dyv(1);
    dym *e0 = mk_zero_dym(Num_queries,1); dyv *ll_e0 = mk_dyv(1);
    dym *u0 = mk_zero_dym(Num_queries,1); dyv *ll_u0 = mk_dyv(1);
    double bw = sqrt(dyv_ref((bws->bwsqds).bwsqds_dyv,b));
    bwinfo *bws0 = mk_bwinfo(bw,bw,1); mk_set_bwinfo(bws0,d);
    time1 = get_time(); 
  
    if ( 0 ) { 
      fprintf(LOG,"Single-tree option is not yet implemented!\n"); 
    } 
    else { 
      run_dualtree_kde(q,d,w,l0,e0,u0,ll_l0,ll_e0,ll_u0,bws0); 
    }
  
    copy_dym_col_to_dym_col(l0,0,l,b); dyv_set(ll_l,b,dyv_ref(ll_l0,0));
    copy_dym_col_to_dym_col(e0,0,e,b); dyv_set(ll_e,b,dyv_ref(ll_e0,0));
    copy_dym_col_to_dym_col(u0,0,u,b); dyv_set(ll_u,b,dyv_ref(ll_u0,0));
    free_dym(l0); free_dym(e0); free_dym(u0); 
    free_dyv(ll_l0); free_dyv(ll_e0); free_dyv(ll_u0); free_bwinfo(bws0);
  }
  time2 = get_time(); tot_time += (time2 - time1);
  fprintf(LOG,"%f sec. elapsed for indep-bw\n",tot_time);
  
}

double run_findbw(dym *q,dym *d,bwinfo *bws)
{
  int i; bool slow=FALSE, turn=FALSE;
  double score_e=0, score_u=0, score_l=0, score_err=0;
  double best_score=0, best_bw = -1, crit_bw = -1, bw, factor = 10;
  double max_bw = bws->bwmax, min_bw = bws->bwmin;
  dyv *the_bws = mk_dyv(0), *the_scores = mk_dyv(0);
  double guess_bw = bandwidth_guess(d); 

  fprintf(LOG,"\nFinding the optimum bandwidth.\n\n");

  if (BWSTART_VALUE) {
    if (max_bw > 0.0) 
      fprintf(LOG,"Using %g, user-specified value, as upper bound.\n",max_bw);
    else { fprintf(LOG,"No start value was specified\n"); 
    Bwstart = Oversmooth; }
  } 
  if (BWSTART_SUBSAMPLE) {
    int subsample_size = 10000;
    if (subsample_size < dym_rows(d)) {
      dym *subsample = mk_random_dym_subset(d, subsample_size);
      TREE Qtree_save = Qtree;
      Bwstart = Oversmooth;
      fprintf(LOG,"Using subsample of size %d.\n",subsample_size);
      make_trees(subsample, subsample, bws);
      max_bw = run_findbw(subsample,subsample,bws);
      Bwstart = Subsample;
      if (max_bw < guess_bw) 
        fprintf(LOG,"Using %g, from subsample, as upper bound.\n",max_bw);
      else Bwstart = Oversmooth;
      free_dym(subsample);
#ifdef USE_KD_TREE
      free_mrkd(Qtree);
#else
      free_batree(Qtree);
#endif
      Qtree = Qtree_save; Dtree = Qtree_save;
    }
    else { fprintf(LOG,"Not subsampling: tiny data\n"); Bwstart = Oversmooth; }
  }
  if (BWSTART_OVERSMOOTH) {
    max_bw = guess_bw;
    fprintf(LOG,"Using %g, theoretical guess, as upper bound.\n",max_bw);
  }

  if (min_bw == -1) { min_bw = .00001 * max_bw; }
  fprintf(LOG,"Using %g, 5 orders below max_bw, as lower bound.\n", min_bw); 
  
  if (LK_CV) best_score = -FLT_MAX; else if (LS_CV) best_score = FLT_MAX; 
  bw = max_bw; i = 0;
  while (bw >= min_bw) {
  
    { /* do this bandwidth */
      dym *l0 = mk_zero_dym(dym_rows(q),1); dyv *ll_l0 = mk_dyv(1);
      dym *e0 = mk_zero_dym(dym_rows(q),1); dyv *ll_e0 = mk_dyv(1);
      dym *u0 = mk_zero_dym(dym_rows(q),1); dyv *ll_u0 = mk_dyv(1);
      bwinfo *bws0 = mk_bwinfo(bw,bw,1); mk_set_bwinfo(bws0,d); 

      Qtree->bws = Dtree->bws = bws0;
      run_dualtree_kde(q,d,NULL,l0,e0,u0,ll_l0,ll_e0,ll_u0,bws0);

      if (LS_CV) { dyv *lscv_l0, *lscv_e0, *lscv_u0;
                   lscv_l0 = mk_lscv(l0,bws0); score_l = dyv_ref(lscv_l0,0);  
                   lscv_e0 = mk_lscv(e0,bws0); score_e = dyv_ref(lscv_e0,0); 
                   lscv_u0 = mk_lscv(u0,bws0); score_u = dyv_ref(lscv_u0,0); 
                   free_dyv(lscv_l0); free_dyv(lscv_e0); free_dyv(lscv_u0); }
      if (LK_CV) { score_l = dyv_ref(ll_l0,0); score_e = dyv_ref(ll_e0,0); 
                   score_u = dyv_ref(ll_u0,0); }
      free_dym(l0); free_dym(e0); free_dym(u0); 
      free_dyv(ll_l0); free_dyv(ll_e0); free_dyv(ll_u0); free_bwinfo(bws0);
    }
    add_to_dyv(the_bws, bw); add_to_dyv(the_scores, score_e);
    score_err = fabs((score_u-score_l)/score_l);
    fprintf(LOG,"  Score (datum-wise midpt): %g  Range: [%g, %g]\n", 
            score_e, score_l, score_u);
    fprintf(LOG,"  Maximum score error: %g\n", score_err);
  
    /* did we stop getting better? */
    if (LK_CV) turn = (score_e < best_score);
    if (LS_CV) turn = (score_e > best_score);
    if (turn) {
      if (slow==FALSE) { crit_bw = bw; slow = TRUE; 
                         if (LK_CV) best_score = -FLT_MAX; 
                         if (LS_CV) best_score = FLT_MAX; }
      else break;
    }
    else { best_score = score_e; best_bw = bw; }
  
    if (slow==TRUE) {
      dym *l0 = mk_zero_dym(dym_rows(q),10); dyv *ll_l0 = mk_dyv(10);
      dym *e0 = mk_zero_dym(dym_rows(q),10); dyv *ll_e0 = mk_dyv(10);
      dym *u0 = mk_zero_dym(dym_rows(q),10); dyv *ll_u0 = mk_dyv(10);
      dyv *lscv_l0=0, *lscv_e0=0, *lscv_u0=0;
      bwinfo *bws0 = mk_bwinfo(crit_bw * 1.1, real_min(crit_bw * 90, 
                               real_max(guess_bw, crit_bw * 1.1)), 10); 
      LOGSCALE = TRUE; mk_set_bwinfo(bws0,d);

      Qtree->bws = bws0; Dtree->bws = bws0;
      run_dualtree_kde(q,d,NULL,l0,e0,u0,ll_l0,ll_e0,ll_u0,bws0);

      if (LS_CV) { lscv_l0 = mk_lscv(l0,bws0); lscv_e0 = mk_lscv(e0,bws0); 
                   lscv_u0 = mk_lscv(u0,bws0); }
      for (i=0; i<10; i++) {
        bw = sqrt(dyv_ref((bws0->bwsqds).bwsqds_dyv,i));
        if (LS_CV) { score_l=dyv_ref(lscv_l0,i);score_e = dyv_ref(lscv_e0,i); 
                     score_u = dyv_ref(lscv_u0,i); }
        if (LK_CV) { score_l = dyv_ref(ll_l0,i); score_e = dyv_ref(ll_e0,i); 
                     score_u = dyv_ref(ll_u0,i); }
        add_to_dyv(the_bws, bw); add_to_dyv(the_scores, score_e);
        score_err = fabs((score_u-score_l)/score_l);
        fprintf(LOG,"  Score (datum-wise midpt): %g  Range: [%g, %g]\n", 
                score_e, score_l, score_u);
        fprintf(LOG,"  Maximum score error: %g\n", score_err);
      }
      if (LS_CV) { best_bw = dyv_ref(the_bws,dyv_argmin(the_scores)); }
      if (LK_CV) { best_bw = dyv_ref(the_bws,dyv_argmax(the_scores)); }

      if (LS_CV) { free_dyv(lscv_l0); free_dyv(lscv_e0); free_dyv(lscv_u0); }
      free_dym(l0); free_dym(e0); free_dym(u0); 
      free_dyv(ll_l0); free_dyv(ll_e0); free_dyv(ll_u0); free_bwinfo(bws0);
      break;
    }
    else bw = bw/factor;
  }
  if (crit_bw == -1) { // no critical bw was found
    fprintf(LOG,"\nPossible breakdown of leave-one-out cross-validation,\n");
    fprintf(LOG,"since no concavity was detected in the scores.\n");
    fprintf(LOG,"Using theoretical bandwidth guess.\n");
    best_bw = bandwidth_guess(d)/2.0; 
  }
  fprintf(LOG,"\n==> Optimum bandwidth is: %g\n\n",best_bw);
  fprintf(LOG,"Summary:\n");
  for (i=0; i<dyv_size(the_bws); i++)
    fprintf(LOG,"%g %g\n",dyv_ref(the_bws,i),dyv_ref(the_scores,i));
  fprintf(LOG,"\n");
  
  free_dyv(the_bws); free_dyv(the_scores);
  return best_bw;
}

void make_trees(dym *d, dym *q, bwinfo *bws)
{
  double time1, time2, tot_time = 0; int numbws;

  if (INDEP_BW) numbws = 1; 
  else if (FIND_BW || VFIND_BW) {
    bws->numbws = numbws = 10;
  }
  else numbws = bws->numbws;

#ifdef USE_KD_TREE
  mrpars *qmrpars = mk_default_mrpars_for_data(q); 
  qmrpars->rmin = Rmin;
  /*
  qmrpars->has_sums = TRUE; qmrpars->has_xxts = TRUE; qmrpars->rmin = Rmin;
  qmrpars->has_sum_sqd_mags = TRUE; qmrpars->has_sq_sum_lengths = TRUE; 
  qmrpars->has_sum_quad_mags = TRUE; qmrpars->has_scaled_sums = TRUE; 
  */
  qmrpars->numbw = numbws;
  
  time1 = get_time(); Qtree = mk_mrkd(qmrpars,q,bws);
  if (SELFCASE) Dtree = Qtree; else Dtree = mk_mrkd(qmrpars,d,bws);
  free_mrpars(qmrpars);
#else
  am_srand(777);

  time1 = get_time(); Qtree = mk_batree_from_dym(q,Rmin,MBW,numbws,bws);
  if (SELFCASE) 
    Dtree=Qtree; 
  else 
    Dtree=mk_batree_from_dym(d,Rmin,MBW,numbws,bws);
  
#endif
  time2=get_time(); tot_time += (time2 - time1);
  fprintf(LOG,"%f sec. elapsed for making tree(s)\n",tot_time); tot_time=0;
  Depth = ((TREE)Qtree)->depth;
}

void reset_meas()
{
  Num_approx_prunes=0; Num_exclud_prunes=0; Num_includ_prunes=0;
  Num_node_expansions=0;

  Num_pt_dists=0; Num_hr_dists=0; Num_farfield_prunes=0;
  Num_far_to_local_conv_prunes = 0; Num_direct_local_accum_prunes=0;
  Num_local_to_local_convs = 0;
}

void print_runstats()
{
  fprintf(LOG,"\n"); 
  fprintf(LOG,"Number of point distance computations: %d\n",Num_pt_dists);
  fprintf(LOG,"Number of hrect distance computations: %d\n",Num_hr_dists);
  fprintf(LOG,"Number of approximation prunes: %d\n",Num_approx_prunes);
  fprintf(LOG,"Number of exclusion prunes: %d\n",Num_exclud_prunes);
  fprintf(LOG,"Number of inclusion prunes: %d\n",Num_includ_prunes);
  fprintf(LOG,"Number of far field prunes: %d\n",Num_farfield_prunes);
  fprintf(LOG,"Number of far field to local conversion prunes: %d\n",
	  Num_far_to_local_conv_prunes);
  fprintf(LOG,"Number of direct local accumulation prunes: %d\n",
	  Num_direct_local_accum_prunes);
  fprintf(LOG,"Number of local to local conversions: %d\n",
	  Num_local_to_local_convs);
  fprintf(LOG,"Number of node expansions: %d\n",Num_node_expansions);
  fprintf(LOG,"Percentage of exhaustive computations: %g%%\n",
         100*(double)(Num_pt_dists + Num_hr_dists)/Num_if_exhaustive);
  fprintf(LOG,"\n"); 
}

/**
 * This function will only be called if there are different weights on
 * each reference point (will not be called in fixed bandwidth KDE case!)
 */
void put_wtsums_bws_in_tree(NODE node, dym *w, bwinfo *bws)
{
  int b;
  if (is_leaf(node)) {
    int num_data = node->num_points, i;
    
    /**
     * Loop over each bandwidth and find out the maximum and minimum
     * weight and bandwidths for this node.
     */
    for(b = bws->b_lo; b <= bws->b_hi; b++) {
      double wtsum_pos = 0.0, maxwt_pos = 0;
      double wtsum_neg = 0.0, minwt_neg = 0;
      double wtsum = 0.0, wtsum_abs = 0.0;

      for ( i = 0 ; i < num_data ; i++ ) {
	
	int row_i = ivec_ref(node->rows,i);
	double refbw = 0, refaux = 0;
	double wt = dym_ref(w,row_i,b);
	
	/* for variable bandwidth, get min/max bandwidth information */
	if(VKDE_MODEL) {
	  refbw = dym_ref((bws->bwsqds).bwsqds_dym,row_i,b);
	  refaux = dym_ref((bws->aux_consts).aux_consts_dym,row_i,b);
	  
	  if(refbw < dyv_ref(node->minbw,b)) {
	    dyv_set(node->minbw,b,refbw);   dyv_set(node->minaux,b,refaux);
	  }
	  if(refbw > dyv_ref(node->maxbw,b)) {
	    dyv_set(node->maxbw,b,refbw);   dyv_set(node->maxaux,b,refaux);
	  }
	}
	  
	wtsum += wt;
	if(wt > 0) {
	  wtsum_abs += wt;    wtsum_pos += wt; 
	  if (wt > maxwt_pos) maxwt_pos = wt;
	}
	else if(wt < 0) {
	  wtsum_abs -= wt;    wtsum_neg += wt;
	  if (wt < minwt_neg) minwt_neg = wt;
	}
      } /* End of looping over reference points */
      
      dyv_set((node->wtsum).wtsum_dyv,b,wtsum);
      dyv_set((node->wtsum_abs).wtsum_abs_dyv,b,wtsum_abs);
      dyv_set((node->wtsum_pos).wtsum_pos_dyv,b,wtsum_pos); 
      dyv_set((node->maxwt_pos).maxwt_pos_dyv,b,maxwt_pos);
      dyv_set((node->wtsum_neg).wtsum_neg_dyv,b,wtsum_neg); 
      dyv_set((node->minwt_neg).minwt_neg_dyv,b,minwt_neg);

    } /* End of looping over bandwidth */
  }
  else {
    put_wtsums_bws_in_tree(node->left,w,bws);
    put_wtsums_bws_in_tree(node->right,w,bws);

    dyv_plus((node->left->wtsum).wtsum_dyv,(node->right->wtsum).wtsum_dyv,
	     (node->wtsum).wtsum_dyv);
    dyv_plus((node->left->wtsum_abs).wtsum_abs_dyv,
	     (node->right->wtsum_abs).wtsum_abs_dyv,
	     (node->wtsum_abs).wtsum_abs_dyv);
    dyv_plus((node->left->wtsum_pos).wtsum_pos_dyv,
	     (node->right->wtsum_pos).wtsum_pos_dyv,
	     (node->wtsum_pos).wtsum_pos_dyv);
    dyv_plus((node->left->wtsum_neg).wtsum_neg_dyv,
	     (node->right->wtsum_neg).wtsum_neg_dyv,
	     (node->wtsum_neg).wtsum_neg_dyv);
 
    for(b = bws->b_lo; b <= bws->b_hi; b++) {
      /** BANDWIDTHS **/
      if(VKDE_MODEL) {
	if(dyv_ref(node->left->minbw,b) < dyv_ref(node->right->minbw,b)) {
	  dyv_set(node->minbw,b,dyv_ref(node->left->minbw,b));
	  dyv_set(node->minaux,b,dyv_ref(node->left->minaux,b));
	}
	else {
	  dyv_set(node->minbw,b,dyv_ref(node->right->minbw,b));
	  dyv_set(node->minaux,b,dyv_ref(node->right->minaux,b));
	}
	if(dyv_ref(node->left->maxbw,b) > dyv_ref(node->right->maxbw,b)) {
	  dyv_set(node->maxbw,b,dyv_ref(node->left->maxbw,b));
	  dyv_set(node->maxaux,b,dyv_ref(node->left->maxaux,b));
	}
	else {
	  dyv_set(node->maxbw,b,dyv_ref(node->right->maxbw,b));
	  dyv_set(node->maxaux,b,dyv_ref(node->right->maxaux,b));
	}
      }

      /** WEIGHTS **/
      /** Now for the weights (numerator) **/
      dyv_set((node->maxwt_pos).maxwt_pos_dyv,b,
	      real_max(dyv_ref((node->left->maxwt_pos).maxwt_pos_dyv,b),
		       dyv_ref((node->right->maxwt_pos).maxwt_pos_dyv,b)));
      dyv_set((node->minwt_neg).minwt_neg_dyv,b,
	      real_min(dyv_ref((node->left->minwt_neg).minwt_neg_dyv,b),
		       dyv_ref((node->right->minwt_neg).minwt_neg_dyv,b)));
    }
  }
}


void dualtree_kde_base(NODE qnode,dym *q,NODE dnode,dym *d,dym *w,
		       dym *neg_l,dym *neg_e,dym *neg_u,
		       dym *pos_l,dym *pos_e,dym *pos_u,
		       bwinfo *bws,int b_lo,int b_hi)
{ 
  int i, j, b; bool SELFNODE = (qnode == dnode);
  int num_data = dnode->num_points, num_queries = qnode->num_points;
  double contrib = 0, contrib2 = 0;

  for ( i = 0 ; i < num_queries ; i++ ) {
    int row_i = ivec_ref(qnode->rows,i);
    
    for ( j = 0 ; j < num_data ; j++ ) {
      int row_j = ivec_ref(dnode->rows,j); double dsqd;
  
      if (LOO && SELFNODE && (row_i == row_j)) continue;
  
      dsqd = row_metric_dsqd(q,d,Metric,row_i,row_j);
      if (SPHERICAL_KERNEL || EPANECHNIKOV_KERNEL) {
        int bin = get_bin_of_dist(dsqd,bws,row_j,b_lo,b_hi);
	
        if (bin == -1) continue; // outside all bw's
	
        for (b = bin; b <= b_hi; b++) {
	  contrib = (VKDE_MODEL) ?
	    KERNEL_UNNORM(dym_ref((bws->bwsqds).bwsqds_dym,row_j,b),dsqd,
			  dym_ref((bws->aux_consts).aux_consts_dym,row_j,b)):
	    KERNEL_UNNORM(dyv_ref((bws->bwsqds).bwsqds_dyv,b),dsqd,
			  dyv_ref((bws->aux_consts).aux_consts_dyv,b));

	  if(WKDE_MODEL || VKDE_MODEL)
	    contrib *= dym_ref(w,row_j,b);

	  if(contrib < 0) {
	    dym_increment(neg_l,row_i,b,contrib);
	    dym_increment(neg_e,row_i,b,contrib);
	    dym_increment(neg_u,row_i,b,contrib);
	  }
	  else {
	    dym_increment(pos_l,row_i,b,contrib);
	    dym_increment(pos_e,row_i,b,contrib);
	    dym_increment(pos_u,row_i,b,contrib);
	  }
	}
      }
      else {
        for (b = b_hi; b >= b_lo; b--) {
	  if(SPHERICAL_STAR_KERNEL) {
	  }
	  else if(EPANECHNIKOV_STAR_KERNEL) {
	  }
	  else if(GAUSSIAN_STAR_KERNEL) {
	    contrib = (VKDE_MODEL) ?
	      dym_ref((bws->aux_consts).aux_consts_dym,row_j,b) *
	      exp(-0.25 * dsqd / dym_ref((bws->bwsqds).bwsqds_dym,row_j,b)):
	      dyv_ref((bws->aux_consts).aux_consts_dyv,b) *
	      exp(-0.25 * dsqd / dyv_ref((bws->bwsqds).bwsqds_dyv,b));
	    contrib2 = (VKDE_MODEL) ?
	      -2 *exp(-0.5 * dsqd / dym_ref((bws->bwsqds).bwsqds_dym,row_j,b)):
	      -2* exp(-0.5 * dsqd / dyv_ref((bws->bwsqds).bwsqds_dyv,b));
	    
	    if(contrib < 0) {
	      dym_increment(neg_l,row_i,b,contrib);
	      dym_increment(neg_e,row_i,b,contrib);
	      dym_increment(neg_u,row_i,b,contrib);
	    }
	    else {
	      dym_increment(pos_l,row_i,b,contrib);
	      dym_increment(pos_e,row_i,b,contrib);
	      dym_increment(pos_u,row_i,b,contrib);
	    }
	    if(contrib2 < 0) {
	      dym_increment(neg_l,row_i,b,contrib2);
	      dym_increment(neg_e,row_i,b,contrib2);
	      dym_increment(neg_u,row_i,b,contrib2);
	    }
	    else {
	      dym_increment(pos_l,row_i,b,contrib2);
	      dym_increment(pos_e,row_i,b,contrib2);
	      dym_increment(pos_u,row_i,b,contrib2);
	    }
	  }
	  else {
	    contrib = (VKDE_MODEL) ?
	      KERNEL_UNNORM(dym_ref((bws->bwsqds).bwsqds_dym,row_j,b),dsqd,
			    dym_ref((bws->aux_consts).aux_consts_dym,row_j,b)):
	      KERNEL_UNNORM(dyv_ref((bws->bwsqds).bwsqds_dyv,b),dsqd,
			    dyv_ref((bws->aux_consts).aux_consts_dyv,b));
	    
	    if(WKDE_MODEL || VKDE_MODEL)
	      contrib *= dym_ref(w,row_j,b);
	    
	    if (contrib == 0.0) break;
	    
	    if(contrib < 0) {
	      dym_increment(neg_l,row_i,b,contrib);
	      dym_increment(neg_e,row_i,b,contrib);
	      dym_increment(neg_u,row_i,b,contrib);
	    }
	    else {
	      dym_increment(pos_l,row_i,b,contrib);
	      dym_increment(pos_e,row_i,b,contrib);
	      dym_increment(pos_u,row_i,b,contrib);
	    }	  
	  }
	}
      }
    }
  }

  /* get min value in l, max value in u over data in this node */
  for (b = b_lo; b <= b_hi; b++) {
    double neg_min_l = 0, neg_max_u = 0;
    double pos_min_l = 0, pos_max_u = 0;
    double wtsum_pos = (KDE_MODEL) ? (dnode->num_points):
      dyv_ref((dnode->wtsum_pos).wtsum_pos_dyv,b);
    double wtsum_neg = (KDE_MODEL) ? 0:
      dyv_ref((dnode->wtsum_neg).wtsum_neg_dyv,b);
    double wtsum_abs = (KDE_MODEL) ? (dnode->num_points):
      dyv_ref((dnode->wtsum_abs).wtsum_abs_dyv,b);

    dyv_increment((qnode->mass_t).mass_t_dyv,b,wtsum_abs);

    
    for ( i=0; i<num_queries; i++ ) {
      int row_i = ivec_ref(qnode->rows,i); double v;
      v = dym_ref(neg_l, row_i, b); if (v < neg_min_l || i==0) neg_min_l = v;
      v = dym_ref(neg_u, row_i, b); if (v > neg_max_u || i==0) neg_max_u = v;
      v = dym_ref(pos_l, row_i, b); if (v < pos_min_l || i==0) pos_min_l = v;
      v = dym_ref(pos_u, row_i, b); if (v > pos_max_u || i==0) pos_max_u = v;
    }

    // note that 'more' vectors are only in leaves
    if(SPHERICAL_STAR_KERNEL) {
    }
    else if(EPANECHNIKOV_STAR_KERNEL) {
    }
    else if(GAUSSIAN_STAR_KERNEL) {
      if(VKDE_MODEL) {
	dyv_increment((qnode->more_l).more_l_dyv,b,
		      -(wtsum_neg*
			dym_ref((bws->aux_consts).aux_consts_dym,0,b)+
			wtsum_pos * (-2)));
	dyv_increment((qnode->more_u2).more_u2_dyv,b,
		      -(wtsum_pos*
			dym_ref((bws->aux_consts).aux_consts_dym,0,b)+
			wtsum_neg * (-2)));
      }
      else {
	dyv_increment((qnode->more_l).more_l_dyv,b,
		      -(wtsum_neg*
			dyv_ref((bws->aux_consts).aux_consts_dyv,b)+
			wtsum_pos*(-2)));
	dyv_increment((qnode->more_u2).more_u2_dyv,b,
		      -(wtsum_pos*
			dyv_ref((bws->aux_consts).aux_consts_dyv,b)+
			wtsum_neg*(-2)));
      }
    }
    else {
      dyv_increment((qnode->more_l).more_l_dyv,b,-wtsum_neg);
      dyv_increment((qnode->more_u2).more_u2_dyv,b,-wtsum_pos);
    }
        
    dyv_set((qnode->mass_l).mass_l_dyv, b,neg_min_l + 
	    dyv_ref((qnode->more_l).more_l_dyv,b));
    dyv_set((qnode->mass_u).mass_u_dyv, b,neg_max_u +
	    dyv_ref((qnode->more_u).more_u_dyv,b));
    dyv_set((qnode->mass_l2).mass_l2_dyv, b,pos_min_l + 
	    dyv_ref((qnode->more_l2).more_l2_dyv,b));
    dyv_set((qnode->mass_u2).mass_u2_dyv, b,pos_max_u +
	    dyv_ref((qnode->more_u2).more_u2_dyv,b));
  }
}

inline double get_bwsqd(bwinfo *bws,int ref_row_num,int b)
{
  if(KDE_MODEL || WKDE_MODEL) {
    return dyv_ref((bws->bwsqds).bwsqds_dyv,b);
  }
  else if(VKDE_MODEL) {
    return dym_ref((bws->bwsqds).bwsqds_dym,ref_row_num,b);
  }
  return -777;
}

inline double get_max_unnorm_kerval(bwinfo *bws,int NUMorDENOM,int b_hi,
				    double dmin,NODE dnode)
{
  if(KDE_MODEL || WKDE_MODEL || NWR_MODEL) {
    return KERNEL_UNNORM(dyv_ref((bws->bwsqds).bwsqds_dyv,b_hi),dmin,
			 dyv_ref((bws->aux_consts).aux_consts_dyv,b_hi));
  }
  else if(VKDE_MODEL) {
    return KERNEL_UNNORM(dyv_ref(dnode->maxbw,b_hi),dmin,
			 dyv_ref(dnode->maxaux,b_hi));
  }
  else if(VNWR_MODEL) {
    return KERNEL_UNNORM(dyv_ref(dnode->maxbw,b_hi),dmin,
			 dyv_ref(dnode->maxaux,b_hi));
  }
  return -777;
}

inline double get_min_unnorm_kerval(bwinfo *bws,int NUMorDENOM,int b_lo,
				    double dmax,NODE dnode)
{
  if(KDE_MODEL || WKDE_MODEL || NWR_MODEL) {
    return KERNEL_UNNORM(dyv_ref((bws->bwsqds).bwsqds_dyv,b_lo),dmax,
			 dyv_ref((bws->aux_consts).aux_consts_dyv,b_lo));
  }
  else if(VKDE_MODEL) {
    return KERNEL_UNNORM(dyv_ref(dnode->minbw,b_lo),dmax,
			 dyv_ref(dnode->minaux,b_lo));
  }
  else if(VNWR_MODEL) {
    return KERNEL_UNNORM(dyv_ref(dnode->minbw,b_lo),dmax,
			 dyv_ref(dnode->minaux,b_lo));
  }
  return -777;
}

void dualtree_kde(NODE qnode,dym *q,NODE dnode,dym *d,dym *w,
		  dym *neg_l,dym *neg_e,dym *neg_u,dym *pos_l,dym *pos_e,
		  dym *pos_u,bwinfo *bws,int b_lo,int b_hi)
{
  NODE try1st; NODE try2nd; bool SELFNODE = (qnode == dnode);
  
  dyv *neg_dl=mk_zero_dyv(bws->numbws), *neg_de=mk_zero_dyv(bws->numbws),
    *neg_du=mk_zero_dyv(bws->numbws), *dt=mk_zero_dyv(bws->numbws);
  dyv *pos_dl=mk_zero_dyv(bws->numbws), *pos_de=mk_zero_dyv(bws->numbws),
    *pos_du=mk_zero_dyv(bws->numbws);
  ivec *p_alphaM2Ls = mk_zero_ivec(bws->numbws),
    *p_alphaDMs = mk_zero_ivec(bws->numbws),
    *p_alphaDLs = mk_zero_ivec(bws->numbws);

  int num_queries = qnode->num_points;
  double dmin, dmax, dens_l = 0, dens_u = 0, dens_l2 = 0, dens_u2 = 0,
    requiredError = 0;
  double neg_dens_l,neg_dens_u,neg_dens_ul,pos_dens_ll,pos_dens_l,pos_dens_u;
  int i, b, numbws = b_hi-b_lo+1;
  double neg_dl_b,neg_dl_b_change,neg_du_b,new_neg_du_b,pos_dl_b,pos_du_b,
    pos_du_b_change,new_pos_dl_b,m=0,new_dl_b,neg_de_b=0,pos_de_b=0;

  Num_node_expansions++;

  /* before anything happens, we can process things sent from above...
     incorporate owed weight and pass it down. */
  {
    update_bounds_delay(qnode,q,dnode,d,w,neg_l,neg_e,neg_u,pos_l,pos_e,pos_u,
			(qnode->owed_l).owed_l_dyv,NULL,
			(qnode->owed_u).owed_u_dyv,
			(qnode->owed_l2).owed_l2_dyv,NULL,
			(qnode->owed_u2).owed_u2_dyv,NULL,NULL,NULL,NULL,
			b_lo,b_hi);
    zero_dyv((qnode->owed_l).owed_l_dyv); 
    zero_dyv((qnode->owed_u).owed_u_dyv);
    zero_dyv((qnode->owed_l2).owed_l2_dyv);
    zero_dyv((qnode->owed_u2).owed_u2_dyv);
  }

  /* tighten based on children */
  if (!is_leaf(qnode)) {
    dyv *res_vec = mk_dyv(dyv_size((qnode->mass_t).mass_t_dyv));

    dyv_max_eq((qnode->mass_l).mass_l_dyv,
	       dyv_min_vec((qnode->left->mass_l).mass_l_dyv,
			   (qnode->right->mass_l).mass_l_dyv, res_vec));
    dyv_min_eq((qnode->mass_u).mass_u_dyv,
	       dyv_max_vec((qnode->left->mass_u).mass_u_dyv,
			   (qnode->right->mass_u).mass_u_dyv, res_vec));

    dyv_max_eq((qnode->mass_t).mass_t_dyv,
	       dyv_min_vec((qnode->left->mass_t).mass_t_dyv,
			   (qnode->right->mass_t).mass_t_dyv, res_vec));
    dyv_subtract((qnode->left->mass_t).mass_t_dyv, res_vec,
		 (qnode->left->mass_t).mass_t_dyv);
    dyv_subtract((qnode->right->mass_t).mass_t_dyv, res_vec,
		 (qnode->right->mass_t).mass_t_dyv);

    dyv_max_eq((qnode->mass_l2).mass_l2_dyv,
	       dyv_min_vec((qnode->left->mass_l2).mass_l2_dyv,
			   (qnode->right->mass_l2).mass_l2_dyv, res_vec));
    dyv_min_eq((qnode->mass_u2).mass_u2_dyv,
	       dyv_max_vec((qnode->left->mass_u2).mass_u2_dyv,
			   (qnode->right->mass_u2).mass_u2_dyv, res_vec));

#ifndef AMFAST
    zero_dyv(res_vec);
#endif
    free_dyv(res_vec);
  }

  /* look at max density contribution */
  dmin = hrect_min_metric_dsqd(Metric,qnode->hr,dnode->hr);

  if(SPHERICAL_STAR_KERNEL) {
  }
  else if(EPANECHNIKOV_STAR_KERNEL) {
  }
  else if(GAUSSIAN_STAR_KERNEL) {
    dens_u = (VKDE_MODEL) ? 
      exp(-0.25 * dmin / dyv_ref(dnode->maxbw,b_hi)):
      exp(-0.25 * dmin / dyv_ref((bws->bwsqds).bwsqds_dyv,b_hi));
  }
  else {
    dens_u = (VKDE_MODEL) ? 
      KERNEL_UNNORM(dyv_ref(dnode->maxbw,b_hi),dmin,
		    dyv_ref(dnode->maxaux,b_hi)):
      KERNEL_UNNORM(dyv_ref((bws->bwsqds).bwsqds_dyv,b_hi),dmin,
		    dyv_ref((bws->aux_consts).aux_consts_dyv,b_hi));
  }

  /* exclusion for bounded kernels, avoiding further computations */
  if (dens_u == 0.0) { 
    for(b=b_lo; b<=b_hi; b++) {
      double wtsum_pos = (KDE_MODEL) ? (dnode->num_points):
	dyv_ref((dnode->wtsum_pos).wtsum_pos_dyv,b);
      double wtsum_neg = (KDE_MODEL) ? 0:
	dyv_ref((dnode->wtsum_neg).wtsum_neg_dyv,b);
      double wtsum_abs = (KDE_MODEL) ? (dnode->num_points):
	dyv_ref((dnode->wtsum_abs).wtsum_abs_dyv,b);
      
      /* for Gaussian convolution, */
      if(GAUSSIAN_STAR_KERNEL) {
	if(VKDE_MODEL) {
	  neg_dens_l=wtsum_neg*dym_ref((bws->aux_consts).aux_consts_dym,b,0) +
	    wtsum_pos * (-2);
	  pos_dens_u=wtsum_pos*dym_ref((bws->aux_consts).aux_consts_dym,b,0) +
	    wtsum_neg * (-2);
	}
	else {
	  neg_dens_l = wtsum_neg*dyv_ref((bws->aux_consts).aux_consts_dyv,b) +
	    wtsum_pos * (-2);
	  pos_dens_u = wtsum_pos*dyv_ref((bws->aux_consts).aux_consts_dyv,b) +
	    wtsum_neg * (-2);
	}
	dyv_set(neg_dl,b,-neg_dens_l);
	dyv_set(pos_du,b,-pos_dens_u);
      }
      /* for non-star convolution kernels, */
      else {
	dyv_set(neg_dl,b,-wtsum_neg);
	dyv_set(pos_du,b,-wtsum_pos);
      }
      dyv_set(dt,b,wtsum_abs);
    }
    
    Num_exclud_prunes += numbws;
    update_bounds_delay(qnode,q,dnode,d,w,neg_l,neg_e,neg_u,pos_l,pos_e,pos_u,
			neg_dl,NULL,NULL,
			NULL,NULL,pos_du,NULL,NULL,NULL,dt,b_lo,b_hi);
    
    free_dyv(neg_dl); free_dyv(neg_de); free_dyv(neg_du); free_dyv(dt);
    free_dyv(pos_dl); free_dyv(pos_de); free_dyv(pos_du);
    free_ivec(p_alphaM2Ls); free_ivec(p_alphaDMs); free_ivec(p_alphaDLs); 
    return;
  }

  /* look at min mass contribution */
  dmax = hrect_max_metric_dsqd(Metric,qnode->hr,dnode->hr);

  if(SPHERICAL_STAR_KERNEL) {
  }
  else if(EPANECHNIKOV_STAR_KERNEL) {
  }
  else if(GAUSSIAN_STAR_KERNEL) {
    dens_l = (VKDE_MODEL) ?
      exp(-0.5 * dmax / dyv_ref(dnode->minbw,b_lo)):
      exp(-0.5 * dmax / dyv_ref((bws->bwsqds).bwsqds_dyv,b_lo));
  }
  else {
    dens_l = (VKDE_MODEL) ?
      KERNEL_UNNORM(dyv_ref(dnode->minbw,b_lo),dmax,
		    dyv_ref(dnode->minaux,b_lo)):
      KERNEL_UNNORM(dyv_ref((bws->bwsqds).bwsqds_dyv,b_lo),dmax,
		    dyv_ref((bws->aux_consts).aux_consts_dyv,b_lo));
  }
  
  /* subsumption for bounded kernels, avoiding further computations */
  if (dens_l == 1.0) {
    for(b=b_lo; b<=b_hi; b++) {
      double wtsum_pos = (KDE_MODEL) ? (dnode->num_points):
	dyv_ref((dnode->wtsum_pos).wtsum_pos_dyv,b);
      double wtsum_pos_l = (LOO && SELFNODE) ? 
	((KDE_MODEL) ? (dnode->num_points - 1):
	 wtsum_pos - dyv_ref((dnode->maxwt_pos).maxwt_pos_dyv,b)):wtsum_pos;
      double wtsum_neg = (KDE_MODEL) ? 0:
	dyv_ref((dnode->wtsum_neg).wtsum_neg_dyv,b);
      double wtsum_neg_l = (LOO && SELFNODE) ?
	((KDE_MODEL) ? 0:wtsum_neg - 
	 dyv_ref((dnode->minwt_neg).minwt_neg_dyv,b)):wtsum_neg;
      double wtsum_abs = (KDE_MODEL) ? (dnode->num_points):
	dyv_ref((dnode->wtsum_abs).wtsum_abs_dyv,b);
      
      if(GAUSSIAN_STAR_KERNEL) {
        if(VKDE_MODEL) {
	  neg_dens_ul=wtsum_neg_l*
	    dym_ref((bws->aux_consts).aux_consts_dym,b,0) + wtsum_pos_l * (-2);
          neg_dens_u=wtsum_neg*dym_ref((bws->aux_consts).aux_consts_dym,b,0) +
            wtsum_pos * (-2);
	  pos_dens_ll=wtsum_pos_l*
	    dym_ref((bws->aux_consts).aux_consts_dym,b,0) + wtsum_neg_l * (-2);
          pos_dens_l=wtsum_pos*dym_ref((bws->aux_consts).aux_consts_dym,b,0) +
            wtsum_neg * (-2);
        }
        else {
	  neg_dens_ul = wtsum_neg_l*
	    dyv_ref((bws->aux_consts).aux_consts_dyv,b) +
            wtsum_pos_l * (-2);
          neg_dens_u = wtsum_neg*dyv_ref((bws->aux_consts).aux_consts_dyv,b) +
            wtsum_pos * (-2);
	  pos_dens_ll = wtsum_pos_l*
	    dyv_ref((bws->aux_consts).aux_consts_dyv,b) +
            wtsum_neg_l * (-2);
          pos_dens_l = wtsum_pos*dyv_ref((bws->aux_consts).aux_consts_dyv,b) +
            wtsum_neg * (-2);
        }
	dyv_set(neg_de,b,neg_dens_u);
        dyv_set(neg_du,b,neg_dens_ul);
        dyv_set(pos_dl,b,pos_dens_ll);
	dyv_set(pos_de,b,pos_dens_l);
      }
      else {
	dyv_set(neg_de,b,wtsum_neg);
	dyv_set(neg_du,b,wtsum_neg_l);
	dyv_set(pos_dl,b,wtsum_pos_l);
	dyv_set(pos_de,b,wtsum_pos);
      }
      dyv_set(dt,b,wtsum_abs);
    }
    
    Num_includ_prunes += numbws;
    update_bounds_delay(qnode,q,dnode,d,w,neg_l,neg_e,neg_u,pos_l,pos_e,pos_u,
			neg_dl,neg_de,neg_du,
			pos_dl,pos_de,pos_du,NULL,NULL,NULL,dt,b_lo,b_hi);
    
    if (LOO && SELFNODE) { // account for self-weights
      for ( i = 0 ; i < num_queries ; i++ ) {
	int row_i = ivec_ref(qnode->rows,i);
	for (b = b_lo; b <= b_hi; b++) {
	  if(dym_ref(w,row_i,b) > 0)
	    dym_increment(pos_e,row_i,b,-dym_ref(w,row_i,b));
	  else
	    dym_increment(neg_e,row_i,b,-dym_ref(w,row_i,b));
	}
      }
    }
    
    free_dyv(neg_dl); free_dyv(neg_de); free_dyv(neg_du); free_dyv(dt);
    free_dyv(pos_dl); free_dyv(pos_de); free_dyv(pos_du);
    free_ivec(p_alphaM2Ls); free_ivec(p_alphaDMs); free_ivec(p_alphaDLs);
    return;
  }

  if (b_lo > b_hi) { fprintf(LOG,"error occurred\n"); exit(0); }

  /* Try approximation */
  {
    int count = 0;

    /**
     * Flags to tell whether for a particular bandwidth, the numerator is
     * approximated or not.
     */
    ivec *approximated = mk_zero_ivec(bws->numbws);

    for (b = b_lo; b <= b_hi; b++) {
      double wtsum_pos =  (KDE_MODEL) ? (dnode->num_points):
	dyv_ref((dnode->wtsum_pos).wtsum_pos_dyv,b);
      double wtsum_pos_l =  (LOO && SELFNODE) ? 
	((KDE_MODEL) ? (dnode->num_points - 1):
	 wtsum_pos - dyv_ref((dnode->maxwt_pos).maxwt_pos_dyv,b)):wtsum_pos;
      double wtsum_neg = (KDE_MODEL) ? 0:
	dyv_ref((dnode->wtsum_neg).wtsum_neg_dyv,b);
      double wtsum_neg_l = (LOO && SELFNODE) ?
	((KDE_MODEL) ? 0:wtsum_neg - 
	 dyv_ref((dnode->minwt_neg).minwt_neg_dyv,b)):wtsum_neg;
      double wtsum_abs = (KDE_MODEL) ? (dnode->num_points):
	dyv_ref((dnode->wtsum_abs).wtsum_abs_dyv,b);

      /**
       * This function call should be changed to handle variable bandwidth
       * multipole pruning!
       */
      if(SPHERICAL_STAR_KERNEL) {
      }
      else if(EPANECHNIKOV_STAR_KERNEL) {
      }
      else if(GAUSSIAN_STAR_KERNEL) {
	dens_l = (VKDE_MODEL) ?
	  exp(-0.25 * dmax / dyv_ref(dnode->minbw,b)):
	  exp(-0.25 * dmax / dyv_ref((bws->bwsqds).bwsqds_dyv,b));
	dens_u = (VKDE_MODEL) ?
	  exp(-0.25 * dmin / dyv_ref(dnode->maxbw,b)):
	  exp(-0.25 * dmin / dyv_ref((bws->bwsqds).bwsqds_dyv,b));
	dens_l2 = (VKDE_MODEL) ?
	  exp(-0.5 * dmax / dyv_ref(dnode->minbw,b)):
	  exp(-0.5 * dmax / dyv_ref((bws->bwsqds).bwsqds_dyv,b));
	dens_u2 = (VKDE_MODEL) ?
	  exp(-0.5 * dmin / dyv_ref(dnode->maxbw,b)):
	  exp(-0.5 * dmin / dyv_ref((bws->bwsqds).bwsqds_dyv,b));
	m = real_max(dens_u - dens_l,dens_u2 - dens_l2);
      }
      else {
	dens_l = (VKDE_MODEL) ?
	  KERNEL_UNNORM(dyv_ref(dnode->minbw,b),dmax,dyv_ref(dnode->minaux,b)):
	  KERNEL_UNNORM(dyv_ref((bws->bwsqds).bwsqds_dyv,b),dmax,
			dyv_ref((bws->aux_consts).aux_consts_dyv,b));
	dens_u = (VKDE_MODEL) ? 
	  KERNEL_UNNORM(dyv_ref(dnode->maxbw,b),dmin,dyv_ref(dnode->maxaux,b)):
	  KERNEL_UNNORM(dyv_ref((bws->bwsqds).bwsqds_dyv,b),dmin,
			dyv_ref((bws->aux_consts).aux_consts_dyv,b));
	m = dens_u - dens_l;
      }

      /**
       * Lowest possible reference node contribution is when the reference
       * points with positive weights are as far as possible from the
       * query node, and those with negative weights are as close as possible 
       * from the query node.
       */
      if(GAUSSIAN_STAR_KERNEL) {
        if(VKDE_MODEL) {
	  neg_dl_b=dens_u * wtsum_neg *
	    dym_ref((bws->aux_consts).aux_consts_dym,0,b) + 
	    dens_u2 * wtsum_pos * (-2);
	  neg_dl_b_change = neg_dl_b -
	    (wtsum_neg*
	     dym_ref((bws->aux_consts).aux_consts_dym,0,b) + wtsum_pos * (-2));
	  neg_de_b = 0.5 * 
	    (wtsum_neg * 
	     dym_ref((bws->aux_consts).aux_consts_dym,0,b)* (dens_l + dens_u) +
	     wtsum_pos * (dens_l2 + dens_u2) * (-2));
          neg_du_b=dens_l * wtsum_neg_l*
	     dym_ref((bws->aux_consts).aux_consts_dym,0,b) + 
	    dens_l2 * wtsum_pos_l * (-2);
	  pos_dl_b=dens_l * wtsum_pos_l*
	    dym_ref((bws->aux_consts).aux_consts_dym,0,b) + 
	    dens_l2 * wtsum_neg_l*(-2);
	  pos_de_b = 0.5 * 
	    (wtsum_pos * 
	     dym_ref((bws->aux_consts).aux_consts_dym,0,b)* (dens_l + dens_u) +
	     wtsum_neg * (dens_l2 + dens_u2) * (-2));
          pos_du_b=dens_u *
	    wtsum_pos * dym_ref((bws->aux_consts).aux_consts_dym,b,0) +
	    dens_u2 * wtsum_neg * (-2);
	  pos_du_b_change=pos_du_b-
	    (wtsum_pos*dym_ref((bws->aux_consts).aux_consts_dym,b,0) +
             wtsum_neg * (-2));
	}
        else {
	  neg_dl_b=dens_u * wtsum_neg *
	    dyv_ref((bws->aux_consts).aux_consts_dyv,b) + 
	    dens_u2 * wtsum_pos * (-2);
	  neg_dl_b_change = neg_dl_b -
	    (wtsum_neg*
	     dyv_ref((bws->aux_consts).aux_consts_dyv,b) + wtsum_pos * (-2));
	  neg_de_b = 0.5 * 
	    (wtsum_neg * 
	     dyv_ref((bws->aux_consts).aux_consts_dyv,b)* (dens_l + dens_u) +
	     wtsum_pos * (dens_l2 + dens_u2) * (-2));
          neg_du_b = dens_l * wtsum_neg_l *
	    dyv_ref((bws->aux_consts).aux_consts_dyv,b) + 
	    dens_l2 * wtsum_pos_l * (-2);
          pos_dl_b=dens_l * wtsum_pos_l*
	    dyv_ref((bws->aux_consts).aux_consts_dyv,b) + 
	    dens_l2 * wtsum_neg_l*(-2);
	  pos_de_b = 0.5 * 
	    (wtsum_pos * 
	     dyv_ref((bws->aux_consts).aux_consts_dyv,b)* (dens_l + dens_u) +
	     wtsum_neg * (dens_l2 + dens_u2) * (-2));
          pos_du_b=dens_u * wtsum_pos * 
	    dyv_ref((bws->aux_consts).aux_consts_dyv,b) + dens_u2 *
	    wtsum_neg * (-2);
	  pos_du_b_change=pos_du_b -
	    (wtsum_pos*dyv_ref((bws->aux_consts).aux_consts_dyv,b) +
             wtsum_neg * (-2));
	}
	new_neg_du_b = neg_du_b + dyv_ref((qnode->mass_u).mass_u_dyv,b);
	new_pos_dl_b = pos_dl_b + dyv_ref((qnode->mass_l2).mass_l2_dyv,b);
	new_dl_b = real_min(-new_neg_du_b,new_pos_dl_b);
      } /* end of Gaussian convolution case */

      /* for non-convolution kernels */
      else {
	neg_dl_b = dens_u * wtsum_neg;
	neg_dl_b_change = neg_dl_b - wtsum_neg;
	neg_de_b = 0.5 * wtsum_neg * (dens_l + dens_u);
	neg_du_b = dens_l * wtsum_neg_l;
	new_neg_du_b = neg_du_b + dyv_ref((qnode->mass_u).mass_u_dyv,b);
	pos_dl_b = dens_l * wtsum_pos_l;
	pos_de_b = 0.5 * wtsum_pos * (dens_l + dens_u);
	pos_du_b = dens_u * wtsum_pos;
	pos_du_b_change = pos_du_b - wtsum_pos;
	new_pos_dl_b = pos_dl_b + dyv_ref((qnode->mass_l2).mass_l2_dyv,b);
	new_dl_b = real_min(-new_neg_du_b,new_pos_dl_b);
      }

      /**
       * Maximum absolute error per weight
       */
      if(RELATIVE_PRUNING) {
	if(!KDE_MODEL) {
	  if(dyv_ref((Dtree->root->wtsum_neg).wtsum_neg_dyv,b) < 0) {
	    requiredError = Tau*-new_neg_du_b*
	      (wtsum_abs + dyv_ref((qnode->mass_t).mass_t_dyv,b))/
	      (dyv_ref((Dtree->root->wtsum_abs).wtsum_abs_dyv,b)*wtsum_abs);
	  }
	  if(dyv_ref((Dtree->root->wtsum_pos).wtsum_pos_dyv,b) > 0) {
	    requiredError =
	      real_min(requiredError, Tau*new_pos_dl_b*
		       (wtsum_abs + dyv_ref((qnode->mass_t).mass_t_dyv,b))/
		       (dyv_ref((Dtree->root->wtsum_abs).wtsum_abs_dyv,b)*
			wtsum_abs));
	  }
	}
	/* for KDE, */
	else {
	  if(GAUSSIAN_STAR_KERNEL) {
	    requiredError = Tau * new_dl_b *
	      (wtsum_abs + dyv_ref((qnode->mass_t).mass_t_dyv,b)) / 
	      ((Dtree->root->num_points)*wtsum_abs);
	  }
	  else {
	    requiredError = Tau*new_pos_dl_b*
	      (wtsum_abs + dyv_ref((qnode->mass_t).mass_t_dyv,b))/
	      ((Dtree->root->num_points)*wtsum_abs);
	  }
	}
      }
      else {
      }
      
      /**
       * If can be approximated by finite-difference, then approximate.
       */
      if(RELATIVE_PRUNING) {
	if(m / 2.0 < requiredError) {
	  dyv_set(neg_dl, b, neg_dl_b_change);
	  dyv_set(neg_de, b, neg_de_b);
	  dyv_set(neg_du, b, neg_du_b);
	  dyv_set(pos_dl, b, pos_dl_b);
	  dyv_set(pos_de, b, pos_de_b);
	  dyv_set(pos_du, b, pos_du_b_change);
	  
	  if(KDE_MODEL)
	    dyv_set(dt, b, wtsum_abs *(1.0-(Dtree->root->num_points) * m /
				       (2.0 * Tau * new_pos_dl_b)));
	  else
	    dyv_set(dt, b, wtsum_abs *
		    (1.0-dyv_ref((Dtree->root->wtsum_abs).wtsum_abs_dyv,b) *m /
		     (2.0 * Tau * new_dl_b)));

	  ivec_set(approximated, b, 1);
	}
	else if(!VKDE_MODEL && !(LOO && SELFNODE) && requiredError > 0) {
	  if(GAUSSIAN_KERNEL) {

            /* prune numerator using fast Gauss transform */
            dualtree_kde_gauss_prune(qnode,q,dnode,d,bws,b,requiredError,
                                     dmin,pos_dl,pos_du,dt,approximated,
				     p_alphaM2Ls,p_alphaDMs,p_alphaDLs,
				     pos_dl_b,pos_du_b_change,
				     new_pos_dl_b);
          } /* end of Gaussian transform pruning */
	  else if(EPANECHNIKOV_KERNEL && 
		  dmax <= dyv_ref((bws->bwsqds).bwsqds_dyv,b)) {
	    
            /* prune numerator using Epanechnikov transform */
	    dualtree_kde_epan_prune(qnode,q,dnode,d,bws,b,
				    requiredError,dmin,pos_dl,pos_du,dt,
				    approximated,
				    p_alphaM2Ls,p_alphaDMs,p_alphaDLs,
				    pos_dl_b,pos_du_b_change,
				    new_pos_dl_b);
          } /* end of Epanechnkov transform pruning */
	}
      }
      else {
      }

      if(ivec_ref(approximated, b))
	count++;

    } /* End of looping through each bandwidth from b_lo to b_hi */

    if (count == numbws) {  // if all bins can be approximated
      Num_approx_prunes += numbws;
      update_bounds_delay(qnode,q,dnode,d,w,neg_l,neg_e,neg_u,
			  pos_l,pos_e,pos_u,neg_dl,neg_de,neg_du,
			  pos_dl,pos_de,pos_du,p_alphaM2Ls,
			  p_alphaDMs,p_alphaDLs,dt,b_lo,b_hi);

      if (LOO && SELFNODE) { // account for self-weights
	for ( i = 0 ; i < num_queries ; i++ ) {
	  int row_i = ivec_ref(qnode->rows,i);
	  for (b = b_lo; b <= b_hi; b++) {
	    if(dym_ref(w,row_i,b) > 0)
	      dym_increment(pos_e,row_i,b,-dym_ref(w,row_i,b));
	    else
	      dym_increment(neg_e,row_i,b,-dym_ref(w,row_i,b));
	  }
	}
      }

      free_dyv(neg_dl); free_dyv(neg_de); free_dyv(neg_du); free_dyv(dt);
      free_dyv(pos_dl); free_dyv(pos_de); free_dyv(pos_du);
      free_ivec(p_alphaM2Ls); free_ivec(p_alphaDMs); free_ivec(p_alphaDLs);
      free_ivec(approximated); 
      return;
    }
  
    /* possibly move the ends of the range to chop off approximated bins */
    if (count > 0) {
      int new_b_lo = b_lo, new_b_hi = b_hi;
      for (b = b_lo; b <= b_hi; b++) {
        if (ivec_ref(approximated,b)==1) new_b_lo++; else break;}
      for (b = b_hi; b >= b_lo; b--) {
        if (ivec_ref(approximated,b)==1) new_b_hi--; else break;}
    
      /* even if we could approximate a bin within the new range, don't yet */
      for (b = new_b_lo; b <= new_b_hi; b++) { 
	dyv_set(neg_dl,b,0); dyv_set(neg_de,b,0); dyv_set(neg_du,b,0); 
	dyv_set(dt,b,0);
	dyv_set(pos_dl,b,0); dyv_set(pos_de,b,0); dyv_set(neg_du,b,0); 
	ivec_set(p_alphaM2Ls,b,0); ivec_set(p_alphaDMs,b,0);
	ivec_set(p_alphaDLs,b,0);
	ivec_set(approximated, b, 0);
      }
      
      if (LOO && SELFNODE) { // account for self-weights
	for ( i = 0 ; i < num_queries ; i++ ) {
	  int row_i = ivec_ref(qnode->rows,i);
	  for (b = b_lo; b <= b_hi; b++) {
	    if(ivec_ref(approximated, b)) {
	      if(dym_ref(w,row_i,b) > 0)
		dym_increment(pos_e,row_i,b,-dym_ref(w,row_i,b));
	      else
		dym_increment(neg_e,row_i,b,-dym_ref(w,row_i,b));
	    }
	  }
	}
      }

      update_bounds_delay(qnode,q,dnode,d,w,neg_l,neg_e,neg_u,
			  pos_l,pos_e,pos_u,neg_dl,neg_de,neg_du,
			  pos_dl,pos_de,pos_du,
			  p_alphaM2Ls,p_alphaDMs,p_alphaDLs,dt,
			  b_lo,b_hi);
      numbws = new_b_hi-new_b_lo+1; Num_approx_prunes += (b_hi-b_lo+1)-numbws;
      b_lo = new_b_lo; b_hi = new_b_hi;
    }
    free_ivec(approximated);
  }
  free_dyv(neg_dl); free_dyv(neg_de); free_dyv(neg_du); free_dyv(dt);
  free_dyv(pos_dl); free_dyv(pos_de); free_dyv(pos_du);
  free_ivec(p_alphaM2Ls); free_ivec(p_alphaDMs); free_ivec(p_alphaDLs);

  /* if both data and query nodes are leaves, do slow density computation */
  if ( is_leaf(qnode) ) {
    if( is_leaf(dnode) ) { 
      dualtree_kde_base(qnode,q,dnode,d,w,neg_l,neg_e,neg_u,pos_l,pos_e,pos_u,
			bws,b_lo,b_hi);
      return;
    }
    
    /* if the query node is a leaf, can still try pruning data node children*/
    else   {
      /* recurse on two halves of data node */
      best_node_partners(qnode,dnode->left,dnode->right,&try1st,&try2nd);
      dualtree_kde(qnode,q,try1st,d,w,neg_l,neg_e,neg_u,pos_l,pos_e,pos_u,
		   bws,b_lo,b_hi);
      dualtree_kde(qnode,q,try2nd,d,w,neg_l,neg_e,neg_u,pos_l,pos_e,pos_u,
		   bws,b_lo,b_hi);
      return;
    }
  }

  /* if data node is leaf but query node still has children, just recurse */
  else {
    if ( is_leaf(dnode) )  {
      /* recurse on two halves of query node */
      best_node_partners(dnode,qnode->left,qnode->right,&try1st,&try2nd);
      dyv_plus((qnode->left->mass_t).mass_t_dyv,(qnode->mass_t).mass_t_dyv,
	       (qnode->left->mass_t).mass_t_dyv);
      dyv_plus((qnode->right->mass_t).mass_t_dyv,(qnode->mass_t).mass_t_dyv,
	       (qnode->right->mass_t).mass_t_dyv);
      zero_dyv((qnode->mass_t).mass_t_dyv);
      dualtree_kde(try1st,q,dnode,d,w,neg_l,neg_e,neg_u,pos_l,pos_e,pos_u,
		   bws,b_lo,b_hi);
      dualtree_kde(try2nd,q,dnode,d,w,neg_l,neg_e,neg_u,pos_l,pos_e,pos_u,
		   bws,b_lo,b_hi);
      return;
    }
    
    /* canonical scenario - both kd-nodes are non-leaves */
    else  {
      /* recurse on each half of data node for each half of query node */
      best_node_partners(qnode->left,dnode->left,dnode->right,&try1st,&try2nd);

      dyv_plus((qnode->left->mass_t).mass_t_dyv,(qnode->mass_t).mass_t_dyv,
	       (qnode->left->mass_t).mass_t_dyv);
      dyv_plus((qnode->right->mass_t).mass_t_dyv,(qnode->mass_t).mass_t_dyv,
	       (qnode->right->mass_t).mass_t_dyv);
      zero_dyv((qnode->mass_t).mass_t_dyv);
      
      dualtree_kde(qnode->left,q,try1st,d,w,neg_l,neg_e,neg_u,
		   pos_l,pos_e,pos_u,bws,b_lo,b_hi);
      dualtree_kde(qnode->left,q,try2nd,d,w,neg_l,neg_e,neg_u,
		   pos_l,pos_e,pos_u,bws,b_lo,b_hi);
      
      best_node_partners(qnode->right,dnode->right,dnode->left,&try1st,
			 &try2nd);
      dualtree_kde(qnode->right,q,try1st,d,w,neg_l,neg_e,neg_u,
		   pos_l,pos_e,pos_u,bws,b_lo,b_hi);
      dualtree_kde(qnode->right,q,try2nd,d,w,neg_l,neg_e,neg_u,
		   pos_l,pos_e,pos_u,bws,b_lo,b_hi);
      return;
    }
  }
}


void run_exhaustive_kde(dym *q,dym *d,dyv *w,dym *e,dyv *ll_e, bwinfo *bws)
{
  int i, j, b;
  double time1, time2, tot_time = 0;              

  printf("Computing the variable standard kernel density estimate...\n");
  Num_queries = dym_rows(q); Num_data = dym_rows(d);
  if(LOO && SELFCASE) Num_data--;
  Num_if_exhaustive = Num_queries * Num_data;

  zero_dym(e);
  time1 = get_time();
  for ( i = 0 ; i < Num_queries ; i++ ) {
  
    for ( j = 0 ; j < Num_data + (LOO && SELFCASE); j++ ) {
      double dsqd;
      
      if (LOO && SELFCASE && (i == j)) continue;
        dsqd = row_metric_dsqd(q,d,Metric,i,j);
	
	for(b = bws->b_lo; b <= bws->b_hi; b++) {
	  double contrib=(KDE_MODEL) ?
	    (KERNEL_UNNORM(dyv_ref((bws->bwsqds).bwsqds_dyv,b),dsqd,
			   dyv_ref((bws->aux_consts).aux_consts_dyv,b))):
	    ((WKDE_MODEL) ?
	     (dyv_ref(w,j) *
	      KERNEL_UNNORM(dyv_ref((bws->bwsqds).bwsqds_dyv,b),dsqd,
			    dyv_ref((bws->aux_consts).aux_consts_dyv,b))):
	     (dyv_ref(w,j) *
	      KERNEL_UNNORM(dym_ref((bws->bwsqds).bwsqds_dym,j,b),dsqd,
			    dym_ref((bws->aux_consts).aux_consts_dym,j,b))/
	      dym_ref((bws->norm_consts).norm_consts_dym,j,b)));
	  
	  dym_increment(e, i, b, contrib);
	}
    }
  }
  time2 = get_time(); tot_time += (time2 - time1);
  
  printf("...Computation done.\n");
  fprintf(LOG,"%f sec. elapsed for %d iters of %s()\n",tot_time,
	  Num_timing_iters,"standard_vkde");

  /* final normalization, for overall log-likelihood */
  compute_loglike_est(e, ll_e, bws);
  normalize_loglike_est(e, bws);
}

void run_dualtree_kde(dym *q,dym *d,dyv *w,dym *l,dym *e,dym *u,
		      dyv *ll_l,dyv *ll_e,dyv *ll_u,bwinfo *bws)
{
  int i, b;
  double time1, time2, tot_time = 0;
  double neg_dens_l=0,pos_dens_u=0;

  /* accumulate negative and positive contribution */
  dym *neg_l=mk_zero_dym(dym_rows(q),bws->numbws);
  dym *neg_e=mk_zero_dym(dym_rows(q),bws->numbws);
  dym *neg_u=mk_zero_dym(dym_rows(q),bws->numbws);
  dym *pos_l=mk_zero_dym(dym_rows(q),bws->numbws);
  dym *pos_e=mk_zero_dym(dym_rows(q),bws->numbws);
  dym *pos_u=mk_zero_dym(dym_rows(q),bws->numbws);

  dyv *neg_lower=mk_zero_dyv(bws->numbws);
  dyv *pos_upper=mk_zero_dyv(bws->numbws);

  /**
   * For VKDE, needs to have different weights for each bandwidth
   * per reference point, since normalizing constant is different for
   * each kernel for each different bandwidth!
   */
  dym *variableweights = mk_constant_dym(dym_rows(d), bws->numbws, 1.0);

  Num_queries = dym_rows(q); Num_data = dym_rows(d);
  if(LOO && SELFCASE) Num_data--;
  Num_if_exhaustive = Num_queries * Num_data;
  
  /**
   * Set up the variable bandwidth weight matrix for passing.
   */
  if(VKDE_MODEL) {
    for(i = 0; i < dym_rows(d); i++) {
      for(b = bws->b_lo; b <= bws->b_hi; b++) {
	dym_set(variableweights, i, b, dyv_ref(w,i) /
		dym_ref((bws->norm_consts).norm_consts_dym,i,b));
      }
    }
  }
  else if(WKDE_MODEL) {
    for(i = 0; i < dym_rows(d); i++) {
      for(b = bws->b_lo; b <= bws->b_hi; b++) {
	dym_set(variableweights, i, b, dyv_ref(w,i));
      }
    }
  }

  printf("Computing the variable kernel density estimate...\n");

  /* initialize, run, finalize */
  time1 = get_time();
  zero_dym(e);
  
  if(!KDE_MODEL) {
    put_wtsums_bws_in_tree(Dtree->root, variableweights, bws);
  }
  
  for(b = bws->b_lo; b <= bws->b_hi; b++) {
    double wtsum_neg = (KDE_MODEL) ? 0:
      dyv_ref((Dtree->root->wtsum_neg).wtsum_neg_dyv,b);
    double wtsum_pos = (KDE_MODEL) ? (Dtree->root->num_points):
      dyv_ref((Dtree->root->wtsum_pos).wtsum_pos_dyv,b);
    
    /* handle the star kernels */
    if(EPANECHNIKOV_STAR_KERNEL) {
    }
    else if(SPHERICAL_STAR_KERNEL) {
	
    }
    else if(GAUSSIAN_STAR_KERNEL) {
      if(VKDE_MODEL) {
	neg_dens_l=wtsum_neg * dym_ref((bws->aux_consts).aux_consts_dym,0,b) +
	  wtsum_pos * (-2);
	pos_dens_u=wtsum_pos * dym_ref((bws->aux_consts).aux_consts_dym,0,b) + 
	  wtsum_neg * (-2);
      }
      else {
	neg_dens_l = wtsum_neg * dyv_ref((bws->aux_consts).aux_consts_dyv,b) + 
	  wtsum_pos * (-2);
	pos_dens_u = wtsum_pos * dyv_ref((bws->aux_consts).aux_consts_dyv,b) +
	  wtsum_neg * (-2);
      }
    }
    /* for non-star kernels */
    else {
      neg_dens_l = wtsum_neg;      pos_dens_u = wtsum_pos;
    }
    
    dyv_set(neg_lower, b, neg_dens_l);
    dyv_set(pos_upper, b, pos_dens_u);
    
    for(i = 0; i < Num_queries; i++) {
      dym_set(neg_l,i,b,dyv_ref(neg_lower,b));
      dym_set(pos_u,i,b,dyv_ref(pos_upper,b));
    }
  }
  
  zero_dym(e);
  initialize_wgts_in_nodes(Qtree->root,neg_lower,pos_upper);
  
  /* COMPUTE */
  {
    dualtree_kde(Qtree->root,q,Dtree->root,d,variableweights,neg_l,neg_e,neg_u,
		 pos_l,pos_e,pos_u,bws,bws->b_lo,bws->b_hi);
    finalize_wgts_in_nodes(Qtree->root,neg_e,pos_e,bws,q);
    
    for(i = 0; i < Num_queries; i++) {
      for(b = 0; b < bws->numbws; b++) {
	dym_set(l,i,b,dym_ref(neg_e,i,b)*(1+Tau)+dym_ref(pos_e,i,b)*(1-Tau));
	dym_set(e,i,b,dym_ref(neg_e,i,b)+dym_ref(pos_e,i,b));
	dym_set(u,i,b,dym_ref(neg_e,i,b)*(1-Tau)+dym_ref(pos_e,i,b)*(1+Tau));
      }
    }
  }
  
  time2 = get_time(); tot_time += (time2 - time1);

  free_dym(variableweights); free_dym(neg_l); free_dym(neg_e); free_dym(neg_u);
  free_dym(pos_l); free_dym(pos_e); free_dym(pos_u);
  free_dyv(neg_lower); free_dyv(pos_upper);

  printf("...Computation done.\n");
  fprintf(LOG,"%f sec. elapsed for %d iters of %s()\n",tot_time,
          Num_timing_iters,"dualtree_kde");

  /* final normalization, for overall log-likelihood */
  compute_loglike_est_and_normalize(l,e,u,ll_l,ll_e,ll_u,bws);
}

/**** MAIN *******************************************************************/

void kde_main(dym *d, dym *q, dyv *w, bwinfo *bws, char *basename)
{
  char *f; int b, numbws;
  int allocsize;

  /* set bandwidth/kernel info */
  Num_dims = dym_cols(d);
  
  if(!EXHAUSTIVE) {
    if (Qtree == NULL) { make_trees(d,q,bws); }  // FIX for SNGLTREE
  }

  if (!FIND_BW) mk_set_bwinfo(bws,d);
  numbws = bws->numbws;

  /* for certain global checks */
  Check_iters = 10 * Num_data;
  if (DEBUG) { Record_l = mk_dyv(0); Record_u = mk_dyv(0); }

  if ( FIND_BW ) {
    run_findbw(q,d,bws);
    if (Qtree != NULL) {
      #ifdef USE_BALL_TREE
      free_batree(Qtree);
      #endif
      #ifdef USE_KD_TREE
      free_mrkd(Qtree);
      #endif
      if (Qtree != Dtree) {
	#ifdef USE_BALL_TREE
	free_batree(Dtree);
	#endif
	#ifdef USE_KD_TREE
	free_mrkd(Dtree);
	#endif
      }
    }
  } 
  else {

    /**** compute density the standard way */
    if ( EXHAUSTIVE ) {
      dym *e = mk_zero_dym(dym_rows(q),numbws); 
      dyv *ll_e = mk_dyv(numbws);
      dyv *lkcv_e=0, *lscv_e=0;
      
      run_exhaustive_kde(q,d,w,e,ll_e,bws);

      if (LK_CV) lkcv_e = mk_lkcv(ll_e,bws);
      if (LS_CV) lscv_e = mk_lscv(e,bws);
      for (b = 0; b < numbws; b++) {
	double bw = sqrt(dyv_ref((bws->bwsqds).bwsqds_dyv,b));
	fprintf(LOG,"bw = %g: ll = %g  ",bw,dyv_ref(ll_e, b));
	if (LK_CV) fprintf(LOG,"lkcv = %g  ",dyv_ref(lkcv_e, b));
	if (LS_CV) fprintf(LOG,"lscv = %g  ",dyv_ref(lscv_e, b));
	fprintf(LOG,"\n");
      }
      if (LK_CV) free_dyv(lkcv_e); if (LS_CV) free_dyv(lscv_e);
      
      print_runstats();
      
      f = make_extended_name(basename,".dens",&allocsize); write_dym(f,e,"w");
      AM_FREE_ARRAY(f,char,allocsize);
      free_dym(e); free_dyv(ll_e);
    } /* end EXHAUSTIVE */
    
    /**** compute density using tree-based methods */
    else {
      dym *l = mk_zero_dym(dym_rows(q),numbws); 
      dyv *ll_l = mk_dyv(numbws);
      dym *e = mk_zero_dym(dym_rows(q),numbws); 
      dyv *ll_e = mk_dyv(numbws);
      dym *u = mk_zero_dym(dym_rows(q),numbws); 
      dyv *ll_u = mk_dyv(numbws);
      
      if ( !INDEP_BW ) {
	
	if ( 0 ) { 
	  fprintf(LOG,"Single tree option not yet implemented!\n"); 
	} 
	else { 
	  run_dualtree_kde(q,d,w,l,e,u,ll_l,ll_e,ll_u,bws);
	}
      } 
      else { 
	run_indepbw(q,d,w,l,e,u,ll_l,ll_e,ll_u,bws); 
      }
      
      print_runstats();
      
      print_output(l,e,u,ll_l,ll_e,ll_u,basename,d,q,w,bws);
      
      free_dyv(ll_e);free_dyv(ll_l);free_dyv(ll_u);
      free_dym(e); free_dym(l); free_dym(u);

      #ifdef USE_BALL_TREE
      free_batree(Qtree); if (!SELFCASE) free_batree(Dtree);
      #endif
      #ifdef USE_KD_TREE
      free_mrkd(Qtree); if (!SELFCASE) free_mrkd(Dtree);
      #endif
    } /* end non-EXHAUSTIVE */
    
    /**** compute density classifier using tree-based methods */
  }
}

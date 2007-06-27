/*
   File:        napi.c
   Author:      Andrew W. Moore
   Created:     July 12, 2000
   Description: Friendly wrappers for Fast N-point computation users

   Copyright 2000, the Auton Lab
*/

#include "napi.h"

/* See next function for comments */
void set_search_indexes_helper(knode *kn,int start)
{
  kn -> lo_index = start;
  kn -> hi_index = start + kn -> num_points;
  if ( !knode_is_leaf(kn) )
  {
    set_search_indexes_helper(kn->left,start);
    set_search_indexes_helper(kn->right,kn->left->hi_index);
    if ( kn->right->hi_index != kn->hi_index )
      my_error("skdskjdbvcskv");
  }
}

/* We associate each datapoint with a "label". Datapoints are labeled
   according to the order that would be visited by a depth-first recursive
   traversal of the tree (the traversal always goes down left branches
   before right branches).

   This function fills up the "lo_index" and "hi_index" fields of all
   knodes in the tree so that "lo_index" = the lowest label of any point
   descended from this node and "hi_index" is one larger than the
   highest label of any point descended from this node. 

   Note this invariant: kn->num_points == kn->hi_index - kn->lo_index */ 
void mrkd_set_search_indexes(mrkd *mr)
{
  set_search_indexes_helper(mr->root,0);
}

datapack *mk_datapack_incfree_xw(char *filename,dym *x,dym *w,int rmin,
				 double min_rel_width,int start_index,
				 mrkd *reuse_mrkd)
{
  datapack *dp = AM_MALLOC(datapack);

  dp -> filename = mk_copy_string(filename);
  dp -> x = x;
  dp->w = w;  /* this pointer will NOT be freed.  the mrkd will free the dym */

  dp -> mps = mk_default_mrpars_for_data(x);
  mrpars_set_all_equal_metric(dp -> mps,x,TRUE);
  dp -> mps -> has_sums = FALSE;
  dp -> mps -> has_xxts = FALSE;
  dp -> mps -> has_weights = (w != NULL);
  //dp -> mps -> has_points = TRUE;
  dp -> mps -> has_points = hpAllNodes; /* AG */
  dp -> mps -> rmin = rmin;
  dp -> mps -> min_rel_width = min_rel_width;

  if (reuse_mrkd) dp->mr = mk_reused_mrkd(reuse_mrkd,x,w);
  else            dp->mr = mk_weighted_mrkd(dp->mps,x,w);

  set_search_indexes_helper(dp->mr->root,start_index);

  dp -> ms = 
    (dym_cols(x)==2) ? mk_mapshape_from_mrkd_for_drawing(dp->mr) : NULL;

  return dp;
}

void read_xw_from_file(char *filename,int argc,char *argv[],dym **x,dym **w)
{
  dym *xx;
  int nweights = int_from_args("nweights",argc,argv,0);

  if (!file_exists(filename)) my_errorf("Can't open datafile %s\n",filename);

  xx = mk_dym_from_ds_file(filename,argc,argv);
  if (nweights)
  {
    int i;
    *x = mk_dym(dym_rows(xx),dym_cols(xx)-nweights);
    *w = mk_dym(dym_rows(xx),nweights);
    for (i=0;i<dym_cols(xx)-nweights;i++) copy_dym_col_to_dym_col(xx,i,*x,i);
    for (i=0;i<nweights;i++) copy_dym_col_to_dym_col(xx,i+dym_cols(*x),*w,i);
    free_dym(xx);
  }
  else *x = xx;
  return;
}

/* start_index added for the indexing of the mrkd-tree.  when multiple data
   sets are used together, their indices should be sequential rather than
   overlapping to allow the searching of only ordered, matching tuples.
   */
datapack *mk_datapack_from_args(char *filekey,int argc,char *argv[],
				char *default_filename,int start_index)
{
  char *filename = string_from_args(filekey,argc,argv,default_filename);
  datapack *dp = NULL;
  dym *x=NULL,*w=NULL;
  //int rmin = int_from_args("rmin",argc,argv,50); /* AG */
  int rmin = int_from_args("rmin",argc,argv,20);
  double min_rel_width = double_from_args("min_rel_width",argc,argv,1e-4);

  read_xw_from_file(filename,argc,argv,&x,&w);

  dp=mk_datapack_incfree_xw(filename,x,w,rmin,min_rel_width,start_index,NULL);

  return dp;
}

/* An awkward way to make the data and randoms in the same kdtree based on
   the union of the data.
*/
void mk_two_datapacks_from_args(char *filekey1, char *filekey2,
				int argc,char *argv[],
				int start_index,
				datapack **dp1, datapack **dp2)
{
  int rmin = int_from_args("rmin",argc,argv,5);
  double min_rel_width = double_from_args("min_rel_width",argc,argv,1e-4);
  char *file1 = string_from_args(filekey1,argc,argv,NULL);
  char *file2 = string_from_args(filekey2,argc,argv,NULL);
  dym *x1 = NULL;
  dym *w1 = NULL;
  dym *x2 = NULL;
  dym *w2 = NULL;
  dym *x = NULL;
  dym *w = NULL;

  *dp1 = NULL;
  *dp2 = NULL;
  if (file1) read_xw_from_file(file1,argc,argv,&x1,&w1);
  if (file2) read_xw_from_file(file2,argc,argv,&x2,&w2);

  if (x1 && !x2)
    *dp1=mk_datapack_incfree_xw(file1,x1,w1,rmin,min_rel_width,start_index,
				NULL);
  else if (!x1 && x2)
    *dp2=mk_datapack_incfree_xw(file2,x2,w2,rmin,min_rel_width,start_index,
				NULL);
  else if (x1 && x2)
  {
    mrpars *mps = NULL; 
    mrkd *reuse_mrkd = NULL;
    x = mk_copy_dym(x1);
    append_dym_to_dym(x,x2);
    if (w1 && w2)
    {
      w = mk_copy_dym(w1);
      append_dym_to_dym(w,w2);
    }

    mps = mk_default_mrpars_for_data(x);
    mrpars_set_all_equal_metric(mps,x,TRUE);
    mps->has_sums = FALSE;
    mps->has_xxts = FALSE;
    mps->has_weights = (w != NULL);
    mps->has_points = TRUE;
    mps->rmin = rmin;
    mps->min_rel_width = min_rel_width;
    reuse_mrkd = mk_weighted_mrkd(mps,x,w);
    *dp1=mk_datapack_incfree_xw(file1,x1,w1,rmin,min_rel_width,start_index,
				reuse_mrkd);
		/* ANG TODO ~> Reset the start index for the second tree to something else */
    *dp2=mk_datapack_incfree_xw(file2,x2,w2,rmin,min_rel_width,
				start_index+dym_rows(x1),reuse_mrkd);
    free_mrkd(reuse_mrkd);
    free_mrpars(mps);
    if (x) free_dym(x);
    /* no need to free w, it went into the reuse_mrkd */
  }
}

void free_datapack(datapack *dp)
{
  if ( dp->ms != NULL ) free_mapshape(dp->ms);
  free_string(dp->filename);
  free_dym(dp->x);
  /* free_dym(dp->w); NO - do not free this!  mr has a copy and will free it */
  free_mrpars(dp->mps);
  free_mrkd(dp->mr);
  AM_FREE(dp,datapack);
}

void explain_datapack(datapack *dp)
{
  printf("\n\n\n*** Summary of the dataset:\n\n"
	 "The dataset from file %s has\n"
	 "%d columns and %d rows. The\n"
	 "resulting mrkd-tree has %d nodes and is bounded by the\n"
	 "following hyper-rectangle:\n",dp->filename,dym_cols(dp->x),
	 dym_rows(dp->x),mrkd_num_nodes(dp->mr));
  fprintf_hrect(stdout,"bounds",dp->mr->root->hr,"\n");
}

dym *datapack_x(datapack *dp)
{
  return dp->x;
}

mrpars *datapack_mrpars(datapack *dp)
{
  return dp->mps;
}

dyv *datapack_metric(datapack *dp)
{
  return datapack_mrpars(dp)->metric;
}

char *mk_string_of_n_ds(int n)
{
  int array_size = n+1;
  char *s = AM_MALLOC_ARRAY(char,array_size);
  int i;
  for ( i = 0 ; i < n ; i++ )
    s[i] = 'd';
  s[n] = '\0';
  return s;
}

twinpack *mk_twinpack_incfree_datapacks(datapack *dp_data,
					datapack *dp_random,
					int n,
					char *format)
{
  twinpack *tp = AM_MALLOC(twinpack);
  int i;

  tp -> n = n;
  tp -> dp_data = dp_data;
  tp -> dp_random = dp_random;
  tp -> format = (format == NULL)?mk_string_of_n_ds(n):mk_copy_string(format);
  tp -> d_used = FALSE;
  tp -> r_used = FALSE;

  if ( strlen(tp->format) != n )
    my_errorf("You gave format %s but also gave n = %d. format should\n"
	      "have exactly n characters in it\n",tp->format,n);

  for ( i = 0 ; i < n ; i++ )
  {
    if ( tp -> format[i] != 'r' && tp->format[i] != 'd' )
      my_errorf("You gave format %s. But formats should only have r's\n"
		"and d's in them\n",tp->format);
    if ( tp -> format[i] == 'r' && tp -> dp_random == NULL )
      my_errorf("You gave a format (%s) that refers to a random dataset\n"
		"but did not supply a random dataset.\n");
    if ( tp->format[i] == 'd' )
      tp->d_used = TRUE;
    if ( tp->format[i] == 'r' )
      tp->r_used = TRUE;
  }

  if ( tp->dp_random != NULL )
  {
    if ( tp->dp_data->ms != NULL )
    {
      free_mapshape(tp->dp_random->ms);
      tp->dp_random->ms = mk_copy_mapshape(tp->dp_data->ms);
    }
    if ( !dyv_equal(tp->dp_data->mps->metric,tp->dp_random->mps->metric) )
      my_error("twinpack components have different metrics");
  }

  return tp;
}

twinpack *mk_twinpack_from_args(params *ps,int argc,char *argv[])
{
  datapack *dp_data = NULL;
  datapack *dp_random = NULL;
  twinpack *tp = NULL;
  char *format = string_from_args("format",argc,argv,NULL);

  if (ps->sametree)
  {
		printf("Trying to make a single kd-tree from 2 datasets.\n");
    mk_two_datapacks_from_args("in","rdata",argc,argv,0,&dp_data,&dp_random);
  }
  else
  {
		int start_index1 = 0;
		int start_index2 = start_index1 + 1;
		
		printf("Building the first tree\n");
    dp_data = mk_datapack_from_args("in",argc,argv,"data.csv",start_index1);
		start_index2 += dym_rows(dp_data->x);
		printf("Building the second tree starting at %d.\n",start_index2);
		/* ANG ~> This is where the trees are made and the start index is specified */		
    dp_random = (index_of_arg("rdata",argc,argv)>=0) ?
                mk_datapack_from_args("rdata",argc,argv,"random.csv",start_index2):
                NULL;
  }
  tp = mk_twinpack_incfree_datapacks(dp_data,dp_random,ps->n,format);
  tp->use_permute = bool_from_args("use_permute",argc,argv,TRUE);

	/* ANG ~> Added for debugging */
	explain_twinpack(tp);
  return tp;
}

bool twinpack_has_random(twinpack *tp)
{
  return tp->dp_random != NULL;
}

void free_twinpack(twinpack *tp)
{
  free_datapack(tp->dp_data);
  if ( twinpack_has_random(tp) )
    free_datapack(tp->dp_random);
  free_string(tp->format);
  AM_FREE(tp,twinpack);
}

void explain_twinpack(twinpack *tp)
{
  explain_datapack(tp->dp_data);
  if ( twinpack_has_random(tp) )
  {
    explain_datapack(tp->dp_random);
    printf("\n*** The search format is %s.\n",tp->format);
    if ( !tp->d_used )
      printf("  That means we only use data from %s in the search\n",
	     tp->dp_random->filename);
    else if ( !tp->r_used )
      printf("  That means we only use data from %s in the search\n",
	     tp->dp_data->filename);
    else
    {
      int i;
      printf("  This means that when we test an n-tuple of points\n"
	     "  {x1,x2..xn} against the matcher (where n = %d), we\n"
	     "  take...\n",tp->n);
      for ( i = 0 ; i < tp->n ; i++ )
	printf("      x%d from the %s dataset (file %s)\n",
	       i+1,(tp->format[i]=='d')?"DATA":"RANDOM",
	       (tp->format[i]=='d')?tp->dp_data->filename :
	                            tp->dp_random->filename);
    }
  }
  else
    printf("There is no random data loaded, and so all n-point operations\n"
	   "solely involve the data from the file %s\n",tp->dp_data->filename);
}  

datapack *twinpack_datapack(twinpack *tp)
{
  return tp->dp_data;
}

dym *twinpack_x(twinpack *tp)
{
  return datapack_x(twinpack_datapack(tp));
}

mrpars *twinpack_mrpars(twinpack *tp)
{
  return datapack_mrpars(twinpack_datapack(tp));
}

dyv *twinpack_metric(twinpack *tp)
{
  return datapack_metric(twinpack_datapack(tp));
}

int twinpack_n(twinpack *tp)
{
  return tp->n;
}

params *mk_default_params()
{
  params *ps = AM_MALLOC(params);
  ps -> n = 2;
  ps -> thresh_ntuples   = 0.0;
  ps -> connolly_thresh  = 0.0;
  ps -> autofind         = FALSE;
  ps -> errfrac          = 0.0;
  ps -> verbosity        = 0.0;
  ps -> rdraw            = FALSE;
  ps->sametree           = FALSE;
  ps->do_wsums            = FALSE;
  ps->do_wsumsqs          = FALSE;
  return ps;
}

params *mk_params_from_args(int argc,char *argv[])
{
  params *ps = mk_default_params();
  ps -> n = int_from_args("n",argc,argv,ps -> n);
  ps -> thresh_ntuples = double_from_args("thresh_ntuples",argc,argv,
					  ps -> thresh_ntuples);
  ps -> connolly_thresh = double_from_args("connolly_thresh",argc,argv,
					  ps -> connolly_thresh);
  ps -> autofind = bool_from_args("autofind",argc,argv,
				  ps -> autofind);
  ps -> errfrac = double_from_args("errfrac",argc,argv,
				   ps -> errfrac);
  ps -> verbosity = double_from_args("verbosity",argc,argv,
				     ps -> verbosity);
  ps -> verbosity = double_from_args("verbose",argc,argv,
				     ps -> verbosity);
  ps -> rdraw = bool_from_args("rdraw",argc,argv,
			       ps -> rdraw);
  ps->sametree = bool_from_args("sametree",argc,argv,ps->sametree);
  ps->do_wsums = bool_from_args("do_wsums",argc,argv,ps->do_wsums);
  ps->do_wsumsqs = bool_from_args("do_wsumsqs",argc,argv,ps->do_wsumsqs);
  return ps;
}

void explain_params(params *ps)
{
  printf("\n*** Summary of the parameters:\n\n");
  fprintf_int(stdout,"n",ps->n,"\n");
  fprintf_double(stdout,"thresh_ntuples",ps->thresh_ntuples,"\n");
  fprintf_double(stdout,"connolly_thresh",ps->connolly_thresh,"\n");
  fprintf_bool(stdout,"autofind",ps->autofind,"\n");
  fprintf_double(stdout,"errfrac",ps->errfrac,"\n");
  fprintf_double(stdout,"verbosity",ps->verbosity,"\n");
  fprintf_bool(stdout,"rdraw",ps->rdraw,"\n");
  fprintf_bool(stdout,"sametree",ps->sametree,"\n");
  fprintf_bool(stdout,"do_wsums",ps->do_wsums,"\n");
  fprintf_bool(stdout,"do_wsumsqs",ps->do_wsumsqs,"\n");
  printf("\n");
}

void free_params(params *ps)
{
  AM_FREE(ps,params);
}

nout *mk_copy_nout(nout *no)
{
  nout *newno = AM_MALLOC(nout);
  newno -> count = no -> count;
  newno -> lo = no -> lo;
  newno -> hi = no -> hi;
  newno->wlobound = (no->wlobound) ? mk_copy_dyv(no->wlobound) : NULL;
  newno->whibound = (no->whibound) ? mk_copy_dyv(no->whibound) : NULL;
  newno->wresult = (no->wresult) ? mk_copy_dyv(no->wresult) : NULL;
  newno->wresult = (no->wresult) ? mk_copy_dyv(no->wresult) : NULL;
  newno->wsum = (no->wsum) ? mk_copy_dyv(no->wsum) : NULL;
  newno->wsumsq = (no->wsumsq) ? mk_copy_dyv(no->wsumsq) : NULL;
  newno -> ferr = no -> ferr;
  newno -> secs = no -> secs;
  return newno;
}

void explain_nout_header()
{
  printf("%11s %10s %10s %10s %10s %7s %10s %10s %10s %10s %10s\n",
	 "","count","lobound","hibound","errfrac","seconds","wtd (prod)","w(p)lobnd","w(p)hibnd","wtd (sum)","wtd (sum sq)");
}

/* On request from Bob, the commas were taken out of the suffix string.
   07-16-02  JGS
*/
void explain_nout_with_suffix(char *suffix,nout *no)
{
  int i;
  char *tsuffix = mk_copy_string(suffix);

  for (i=0;i<strlen(tsuffix);i++) if (tsuffix[i] == ',') tsuffix[i] = ' ';

  printf("%11s %10g %10g %10g %10g %7d",
	 tsuffix,no->count,no->lo,no->hi,no->ferr,no->secs);
  if (no->wresult) fprintf_oneline_dyv(stdout,"",no->wresult,"");
  if (no->wlobound) fprintf_oneline_dyv(stdout,"",no->wlobound,"");
  if (no->whibound) fprintf_oneline_dyv(stdout,"",no->whibound,"");
  if (no->wsum) fprintf_oneline_dyv(stdout,"",no->wsum,"");
  if (no->wsumsq) fprintf_oneline_dyv(stdout,"",no->wsumsq,"");
  printf("\n");
  free_string(tsuffix);
}

void explain_nout(nout *no)
{
  if (Verbosity >= 0.5)
  {
    printf("---------------------------------------------------------------\n");
    printf("Search Result:\n");
  }

  explain_nout_header();
  explain_nout_with_suffix("",no);

  if (Verbosity >= 0.5)
  {
    printf("---------------------------------------------------------------\n");
  }
}

void free_nout(nout *no)
{
  if (no->wlobound) free_dyv(no->wlobound);
  if (no->whibound) free_dyv(no->whibound);
  if (no->wresult) free_dyv(no->wresult);
  if (no->wsum) free_dyv(no->wsum);
  if (no->wsumsq) free_dyv(no->wsumsq);
  AM_FREE(no,nout);
}

nouts *mk_empty_nouts()
{
  nouts *ns = AM_MALLOC(nouts);
  ns -> size = 0;
  return ns;
}

int nouts_size(nouts *ns)
{
  return ns -> size;
}

void add_to_nouts(nouts *ns,nout *no)
{
  if ( nouts_size(ns) >= MAX_NOUTS ) my_error("MAX_NOUTS too small");
  ns->ns[ns->size] = mk_copy_nout(no);
  ns -> size += 1;
}

void free_nouts(nouts *ns)
{
  int i;
  for ( i = 0 ; i < ns -> size ; i++ )
    free_nout(ns->ns[i]);
  AM_FREE(ns,nouts);
}

nout *nouts_ref(nouts *ns,int i)
{
  if ( i < 0 || i >= nouts_size(ns) ) my_error("nouts_ref");
  return ns->ns[i];
}

void explain_nouts(string_array *matcher_strings,nouts *ns)
{
  int i;
  printf("---------------------------------------------------------------\n");
  printf("Multi Search Results:\n");
  explain_nout_header();
  for ( i = 0 ; i < nouts_size(ns) ; i++ )
    explain_nout_with_suffix(string_array_ref(matcher_strings,i),
			     nouts_ref(ns,i));
  printf("---------------------------------------------------------------\n");
}



/*
   File:        m2p.c
   Author:      Andrew W. Moore
   Created:     July 12, 2000
   Description: Multi-radius 2-pt

   Copyright 2000, the Auton Lab
*/

#include "m2p.h"

bucket *mk_bucket(double thresh_ntuples,int lo_tindex,int hi_tindex)
{
  bucket *b = AM_MALLOC(bucket);
  b -> thresh_ntuples = thresh_ntuples;
  b -> lo_tindex = lo_tindex;
  b -> hi_tindex = hi_tindex;
  return b;
}

void free_bucket(bucket *b)
{
  AM_FREE(b,bucket);
}

bucket *mk_copy_bucket(bucket *b)
{
  return mk_bucket(b->thresh_ntuples,b->lo_tindex,b->hi_tindex);
}

int bucket_num_tindexes(bucket *b)
{
  return b -> hi_tindex - b -> lo_tindex;
}

int bucket_lo_tindex(bucket *b)
{
  return b->lo_tindex;
}

int bucket_hi_tindex(bucket *b)
{
  return b->hi_tindex;
}


void make_split_bucket(bucket *b,bucket **r_b1,bucket **r_b2)
{
  int n = bucket_num_tindexes(b);
  int m = bucket_lo_tindex(b)+n/2;
  *r_b1 = mk_bucket(b->thresh_ntuples,bucket_lo_tindex(b),m);
  *r_b2 = mk_bucket(b->thresh_ntuples,m,bucket_hi_tindex(b));
}

double safe_log10(double x)
{
  return log10(real_max(1.0,x));
}

bres *mk_bres(twinpack *tp,dyv *ticks,bucket *b)
{
  nout *no = mk_2pt_nout(tp,b->thresh_ntuples,0.0,
			 pow(10.0,dyv_ref(ticks,bucket_lo_tindex(b))),
			 pow(10.0,dyv_ref(ticks,bucket_hi_tindex(b))));
  bres *br = AM_MALLOC(bres);
  br -> b = mk_copy_bucket(b);
  br -> log_lo_count = safe_log10(no->lo);
  br -> log_hi_count = safe_log10(no->hi);
  free_nout(no);
  return br;
}

void free_bres(bres *br)
{
  free_bucket(br->b);
  AM_FREE(br,bres);
}

bres *mk_copy_bres(bres *br)
{
  bres *n = AM_MALLOC(bres);
  n -> b = mk_copy_bucket(br->b);
  n -> log_lo_count = br -> log_lo_count;
  n -> log_hi_count = br -> log_hi_count;
  return n;
}

void make_split_bres(twinpack *tp,dyv *ticks,
		     bres *br,bres **r_b1,bres **r_b2)
{
  bucket *b1,*b2;
  double lo_all = pow(10.0,br->log_lo_count);
  double hi_all = pow(10.0,br->log_hi_count);
  double lo_1,hi_1,lo_2,hi_2;

  make_split_bucket(br->b,&b1,&b2);

  *r_b1 = mk_bres(tp,ticks,b1);

  lo_1 = pow(10.0,(*r_b1)->log_lo_count);
  hi_1 = pow(10.0,(*r_b1)->log_hi_count);
  
  lo_2 = lo_all - hi_1;
  hi_2 = hi_all - lo_1;
  lo_2 = real_max(0.0,lo_2);

  *r_b2 = AM_MALLOC(bres);
  (*r_b2) -> b = mk_copy_bucket(b2);
  (*r_b2) -> log_lo_count = safe_log10(lo_2);
  (*r_b2) -> log_hi_count = safe_log10(hi_2);

  free_bucket(b1);
  free_bucket(b2);
}

bresses *mk_initial_bresses(twinpack *tp,double thresh_ntuples,dyv *ticks)
{
  bresses *bs = AM_MALLOC(bresses);
  bucket *b = mk_bucket(thresh_ntuples,0,dyv_size(ticks)-1);
  bs -> size = 1;
  bs -> bs[0] = mk_bres(tp,ticks,b);
  bs -> max_log_count = -1.0;
  free_bucket(b);
  return bs;
}

int bresses_size(bresses *bs)
{
  return bs->size;
}

bres *bresses_ref(bresses *bs,int i)
{
  if ( i < 0 || i >= bs->size ) 
    my_error("bresses_ref");
  return bs->bs[i];
}

lingraph *mk_lingraph_from_bresses(dyv *ticks,bresses *bs,bool show_conf)
{
  int i;
  lingraph *lg = mk_empty_lingraph();
  bool dots = bresses_size(bs) < 100;
  set_x_axis_label(lg,"log10(separation)");
  set_y_axis_label(lg,"log10(density of two-point statistic)");

  for ( i = 0 ; i < bresses_size(bs) ; i++ )
  {
    bres *br = bresses_ref(bs,i);
    bucket *b = br->b;
    double x1 = dyv_ref(ticks,bucket_lo_tindex(b));
    double x2 = dyv_ref(ticks,bucket_hi_tindex(b));
    double y1 = br->log_lo_count;
    double y2 = br->log_hi_count;
    double logw = log10(x2 - x1);
    double xmid = (x1 + x2)/2.0;
    double ymid;
    y1 -= logw;
    y2 -= logw;

    y1 = real_max(1.0,y1);
    y2 = real_max(1.0,y2);

    ymid = (y1 + y2)/2.0;

    if ( show_conf )
    {
      add_to_lingraph(lg,1,x1,y1);
      add_to_lingraph(lg,1,x2,y1);
      add_to_lingraph(lg,2,x1,y2);
      add_to_lingraph(lg,2,x2,y2);
    }
    add_to_lingraph(lg,0,xmid,ymid);
    if ( dots ) 
      add_to_lingraph(lg,3,xmid,ymid);
  }

  set_lines_same_color(lg,1,2);
  if ( dots ) 
  {
    set_lines_same_color(lg,0,3);
    set_line_style_dotted(lg,3);
  }

  return lg;
}

void free_bresses(bresses *bs)
{
  int i;
  for ( i = 0 ; i < bresses_size(bs) ; i++ )
    free_bres(bs->bs[i]);
  AM_FREE(bs,bresses);
}

double bresses_max_log_count(bresses *bs)
{
  if ( bs->max_log_count < 0.0 )
  {
    int i;
    bs->max_log_count = bresses_ref(bs,0)->log_hi_count;

    for ( i = 1 ; i < bresses_size(bs) ; i++ )
      bs -> max_log_count = 
	real_max(bs->max_log_count,bresses_ref(bs,i)->log_hi_count);
  }

  return bs->max_log_count;
}

double bres_errfrac(bresses *bs,int index)
{
  bres *br = bresses_ref(bs,index);
  return (br->log_hi_count - br->log_lo_count) / 
         bresses_max_log_count(bs);
}

int index_of_largest_errfrac_bres(bresses *bs)
{
  int result = -1;
  double worst = -77e7;
  int i;
  for ( i = 0 ; i < bresses_size(bs) ; i++ )
  {
    double errfrac = bres_errfrac(bs,i);
    if ( result < 0 || errfrac > worst )
    {
      result = i;
      worst = errfrac;
    }
  }
  return result;
}

int index_of_most_ticks_bres(bresses *bs)
{
  int result = -1;
  double most = -77;
  int i;
  for ( i = 0 ; i < bresses_size(bs) ; i++ )
  {
    int num_ticks = bucket_num_tindexes(bresses_ref(bs,i)->b);
    if ( result < 0 || num_ticks > most )
    {
      result = i;
      most = num_ticks;
    }
  }
  return result;
}

void explain_bresses(dyv *ticks,bresses *bs)
{
  int i;
  printf("%10s %10s %10s %10s %10s %10s\n",
	 "index","thresh","log_lo_r","log_hi_r","log_lo_count","log_hi_count");
  for ( i = 0 ; i < bresses_size(bs) ; i++ )
  {
    bres *br = bresses_ref(bs,i);
    bucket *b = br->b;
    printf("%10d %10g %10.4f %10.4f %10.4f %10.4f\n",
	   i,b->thresh_ntuples,
	   dyv_ref(ticks,bucket_lo_tindex(b)),
	   dyv_ref(ticks,bucket_hi_tindex(b)),
	   br->log_lo_count,br->log_hi_count);
  }
}
	   
void render_bresses(dyv *ticks,bresses *bs,char *agname,bool show_conf)
{
  lingraph *lg = mk_lingraph_from_bresses(ticks,bs,show_conf);
  ag_on(agname);
  render_lingraph(lg);
  ag_off();
  free_lingraph(lg);
  if ( bresses_size(bs) < 10 ) explain_bresses(ticks,bs);
}
    
void split_bresses(twinpack *tp,dyv *ticks,bresses *bs,int index)
{
  bres *br = bresses_ref(bs,index);
  bres *br1,*br2;
  int i;
  make_split_bres(tp,ticks,br,&br1,&br2);
  
  if ( bs->size >= MAX_BRESSES ) my_error("MAX_BRESSES too small");

  free_bres(bs->bs[index]);
  bs->bs[index] = NULL;
  
  for ( i = bs->size ; i > index+1 ; i-- )
    bs->bs[i] = bs->bs[i-1];
    
  bs->bs[index] = br1;
  bs->bs[index+1] = br2;
  bs->size += 1;
  bs->max_log_count = -1.0;
}
  
void halve_bres_thresh(twinpack *tp,dyv *ticks,bresses *bs,int index)
{
  bres *br = bresses_ref(bs,index);
  bucket *b = br->b;
  bres *temp;
  b -> thresh_ntuples /= 5.0;
  temp = mk_bres(tp,ticks,b);
  br -> log_lo_count = temp -> log_lo_count;
  br -> log_hi_count = temp -> log_hi_count;
  free_bres(temp);
  bs->max_log_count = -1.0;
}

bresses *mk_bresses(twinpack *tp,
		    double thresh_ntuples,
		    dyv *ticks,
		    double errfrac)
{
  bool finished = FALSE;
  bresses *bs = mk_initial_bresses(tp,thresh_ntuples,ticks);

  render_bresses(ticks,bs,"",TRUE);
  wait_for_key();
  
  while ( !finished )
  {
    int index = index_of_largest_errfrac_bres(bs);
    double e = bres_errfrac(bs,index);
    if ( e >= errfrac * 
	       pow(2.0,(double)bucket_num_tindexes(bresses_ref(bs,index)->b)))
      halve_bres_thresh(tp,ticks,bs,index);
    else
    {
      index = index_of_most_ticks_bres(bs);
      if ( bucket_num_tindexes(bresses_ref(bs,index)->b) > 1 )
	split_bresses(tp,ticks,bs,index);
      else
	finished = TRUE;
    }

    if ( !finished )
    {
      render_bresses(ticks,bs,"",TRUE);
      wait_for_key();
    }
  }

  return bs;
}

double bres_logcount(bres *br)
{
  return (br->log_lo_count + br->log_hi_count)/2.0;
}

double bresses_logcount_ref(bresses *bs,int i)
{
  bres *br = bresses_ref(bs,i);
  return bres_logcount(br);
}

dyv *mk_logcounts_from_bresses(bresses *bs)
{
  int size = bresses_size(bs);
  int i;
  dyv *logcounts = mk_dyv(size);
  for ( i = 0 ; i < size ; i++ )
    dyv_set(logcounts,i,bresses_logcount_ref(bs,i));
  return logcounts;
}

dyv *mk_logcounts_from_ticks(twinpack *tp,dyv *ticks,double errfrac,bool demo,
			     bool show_conf)
{
  double thresh_ntuples = total_2pt_tuples(tp);
  int secs = global_time();
  bresses *bs;
  char *agname = mk_printf("%s.ps",twinpack_datapack(tp)->filename);
  dyv *logcounts;

  if ( twinpack_datapack(tp)->ms != NULL )
  {
    ag_on("");
    draw_x(twinpack_datapack(tp)->ms,twinpack_x(tp),999999);
    ag_off();
    if ( demo )
      really_wait_for_key();
    else
      wait_for_key();
  }

  bs = mk_bresses(tp,thresh_ntuples,ticks,errfrac);
  secs = global_time() - secs;

  render_bresses(ticks,bs,agname,show_conf);
  explain_bresses(ticks,bs);
  printf("That whole thing took %d seconds\n",secs);
  printf("Final plot in %s\n",agname);

  logcounts = mk_logcounts_from_bresses(bs);

  free_string(agname);
  free_bresses(bs);

  return logcounts;
}

#define MIN_ERRFRAC 0.001

void explain_logcounts(FILE *s,dyv *ticks,dyv *logcounts,double errfrac)
{
  int i;
  char *comment = "# ";
  char *nocomment = "  ";

  if ( errfrac > MIN_ERRFRAC )
    printf("%s%5s %11s %5s | %10s\n",comment,"","","","Approx");
  printf("%s%5s %11s %5s | %10s\n",comment,"","Bucket","","log10(count)");
  printf("%s%10s %1s %10s | %10s\n",comment,"Low","","High","");
  printf("%s%10s %1s %10s | %10s\n",comment,"log10(sep)","","log10(sep)","");
  printf("%s%10s %1s %10s | %10s\n",comment,"---------","-","---------","---------");
  for ( i = 0 ; i < dyv_size(logcounts) ; i++ )
  {
    printf("%s%10g   %10g | %10g\n",nocomment,
	   dyv_ref(ticks,i),dyv_ref(ticks,i+1),dyv_ref(logcounts,i));
  }
  printf("\n");
}

dyv *mk_ticks(int num_buckets,double log_lo_r,double log_hi_r)
{
  int size = num_buckets+1;
  int i;
  dyv *ticks = mk_dyv(size);
  for ( i = 0 ; i < size ; i++ )
  {
    double z = i / (double) num_buckets;
    double log_r = log_lo_r + (log_hi_r - log_lo_r) * z;
    dyv_set(ticks,i,log_r);
  }
  return ticks;
}

dyv *mk_ticks_from_args(int argc,char *argv[])
{
  dyv *ticks = NULL;

  char *tickfile = string_from_args("tickfile",argc,argv,NULL);
  if ( tickfile == NULL )
  {
    double low_log_sep = double_from_args("low_log_sep",argc,argv,-4.0);
    double high_log_sep = double_from_args("high_log_sep",argc,argv,4.0);
    int num_buckets = int_from_args("num_buckets",argc,argv,40);
    ticks = mk_ticks(num_buckets,low_log_sep,high_log_sep);
  }
  else
  {
    ticks = mk_dyv_from_filename_simple(tickfile);
    printf("Loaded ticks from file %s\n",tickfile);
  }

  printf("I will be using %d buckets (thus %d ticks)\n",
	 dyv_size(ticks)-1,dyv_size(ticks));

  return ticks;
}

void m2p_main(int argc,char *argv[])
{
  bool demo = bool_from_args("demo",argc,argv,FALSE);
  bool show_conf = bool_from_args("show_conf",argc,argv,TRUE);
  params *ps = mk_default_params();
  twinpack *tp = mk_twinpack_from_args(ps,argc,argv);
  dyv *ticks = mk_ticks_from_args(argc,argv);
  double errfrac = double_from_args("errfrac",argc,argv,0.02);
  dyv *logcounts = mk_logcounts_from_ticks(tp,ticks,errfrac,demo,show_conf);
  char *outfile = string_from_args("outfile",argc,argv,NULL);
  
  explain_logcounts(stdout,ticks,logcounts,errfrac);

  if ( demo) really_wait_for_key();


  if ( outfile != NULL )
  {
    FILE *s = safe_fopen(outfile,"w");
    explain_logcounts(s,ticks,logcounts,errfrac);
    fclose(s);
    printf("I also saved results to file %s\n",outfile);
  }

  free_dyv(logcounts);
  free_dyv(ticks);
  free_twinpack(tp);
  free_params(ps);
}

void sfs(FILE *s,int code,char *key,int argc,char *argv[],char *value,
	 bool *r_started)
{
  char *k = mk_printf("show_%s",key);
  bool show = index_of_arg(k,argc,argv) >= 0;
  if ( show )
  {
    if ( code == 0 )
      fprintf(s,"c");
    else 
    {
      fprintf(s,"%s%s",(*r_started)?" & ":"",(code==1)?key:value);
      *r_started = TRUE;
    }
    free_string(k);
  }
}

void sfd(FILE *s,int code,char *key,int argc,char *argv[],double value,
	 bool *r_started)
{
  char *vs = mk_printf("%g",value);
  sfs(s,code,key,argc,argv,vs,r_started);
  free_string(vs);
}

void sfi(FILE *s,int code,char *key,int argc,char *argv[],int value,
	 bool *r_started)
{
  sfd(s,code,key,argc,argv,(double) value,r_started);
}

void out_stuff(FILE *s,int code,int argc,char *argv[],
	       params *ps,twinpack *tp,
	       matcher *ma,bool single_tree,nout *no)
{
  bool started = FALSE;
  sfs(s,code,"filename",argc,argv,tp->dp_data->filename,&started);
  sfi(s,code,"points",argc,argv,dym_rows(twinpack_x(tp)),&started);
  sfi(s,code,"dims",argc,argv,dym_cols(twinpack_x(tp)),&started);
  sfi(s,code,"rmin",argc,argv,twinpack_mrpars(tp)->rmin,&started);
  sfs(s,code,"separation",argc,argv,ma->describe_string,&started);
  sfi(s,code,"n",argc,argv,ps->n,&started);
  sfd(s,code,"thresh_ntuples",argc,argv,ps->thresh_ntuples,&started);
  sfd(s,code,"errfrac",argc,argv,ps->errfrac,&started);
  sfs(s,code,"single",argc,argv,(single_tree)?"yes":"no",&started);
  sfd(s,code,"count",argc,argv,no->count,&started);
  sfi(s,code,"ferr",argc,argv,no->ferr,&started);
  sfi(s,code,"secs",argc,argv,no->secs,&started);
  if ( code != 0 )
    fprintf(s," \\\\\n");
}

void trial_main(int argc,char *argv[])
{
  params *ps = mk_params_from_args(argc,argv);
  twinpack *tp = mk_twinpack_from_args(ps,argc,argv);
  bool single_tree = bool_from_args("single_tree",argc,argv,FALSE);
  matcher *ma = mk_matcher_from_args(ps->n,tp->dp_data->mps->metric,
                                     argc,argv);
  matcher *ma2 = mk_matcher2_from_args(ps->n,tp->dp_data->mps->metric,
                                      argc,argv);
  nout *no = mk_run_npt_from_twinpack(tp,ps,ma,ma2);
  //nout *no = mk_run_npt_from_twinpack(tp,ps,ma);
  char *trialname = "trial.txt";
  FILE *s = fopen(trialname,"r");
  bool first_line = s == NULL;
  if ( s != NULL ) fclose(s);
  s = safe_fopen(trialname,"a");
  if ( first_line )
  {
    fprintf(s,"\\begin{tabular}{");
    out_stuff(s,0,argc,argv,ps,tp,ma,single_tree,no);
    fprintf(s,"}\n");
    out_stuff(s,1,argc,argv,ps,tp,ma,single_tree,no);
  }
  out_stuff(s,2,argc,argv,ps,tp,ma,single_tree,no);
  fclose(s);
  printf("Appended results to %s\n",trialname);
  free_params(ps);
  free_twinpack(tp);
  free_matcher(ma);
  free_nout(no);
}





  


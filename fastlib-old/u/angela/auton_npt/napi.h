/*
   File:        napi.h
   Author:      Andrew W. Moore
   Created:     July 12, 2000
   Description: Friendly wrappers for Fast N-point computation users

   Copyright 2000, the Auton Lab
*/

#ifndef NAPI_H
#define NAPI_H


#include "mrkd.h"
#include "dsut.h"
#include "distutils.h"
#include "matcher.h"

/* This file contains useful data structures that wrap up a bunch
   of commonly-used-together substructures. */

/* datapack: 

     This is used to represent one datafile's worth of data. Usually in
     any run of the program there will be only one datapack in existence.
     But if you are doing "data vs random" counts, there'll be two (as we'll
     see later) */
typedef struct datapack
{
  mapshape *ms;      /* Obscure data structure from the mrkd library,
                        solely used for drawing 2-d pictures if rdarw is
                        on or verbosity is high */

  char *filename;    /* A copy of the string specifying where the data
                        came from */
  dym *x;            /* The data. The i'th row of this matrix is the i'th
                        datapoint */
  dym *w;            /* The weight vector assoicated with each point, it
                        may be NULL.  This is only a pointer, DO NOT FREE
                        IT!  DO NOT CHANGE ITS CONTENTS! */
  mrpars *mps;       /* The parameters used to build the mrkd-tree */
  mrkd *mr;          /* The mrkd-tree associated with this datafile */
} datapack;

/* Basic builder of a datapack. Notice that contrary to standard AUTON
   convention, x is NOT copied into the new structure, but a pointer to it
   is placed in the new structure. Hence the use of "incfree" in the
   function name...this function INCludes x, but the programmer should
   treat the variable x as essentially FREEd afterwards. Never directly
   access x or free x again directly. Only access it through datapack->x.
   When datapack is freed THEN x will be freed with the datapack. */
/* See readme.txt or mrkd.h for meaning of rmin and min_rel_width */
datapack *mk_datapack_incfree_x(char *filename,dym *x,int rmin,
				double min_rel_width);

/* If filekey is, say, "in", then searches the command line for two
   strings in a row where the first is "in" and the second is the name
   of the file to load datapack from */
datapack *mk_datapack_from_args(char *filekey,int argc,char *argv[],
				char *default_filename,int start_index);

/* Free datapack and everything in it (including x) */
void free_datapack(datapack *dp);

/* Give about 15 lines to stdout with information about the dataset */
void explain_datapack(datapack *dp);

dym *datapack_x(datapack *dp);

mrpars *datapack_mrpars(datapack *dp);

dyv *datapack_metric(datapack *dp);

/* Params structue holds many miscellaneous parameters the user
   might want to call with. See readme.txt for what they mean. */
typedef struct params
{
  int n;
  double thresh_ntuples;
  double connolly_thresh;
  bool autofind;
  double errfrac;
  double verbosity;
  bool rdraw;
  bool sametree; /* use the same kd-tree for data and random data sets.  it
                    will be built using the union of the two */
  bool do_wsums;  /* calculate sums of sums of weights on matched tuples */
  bool do_wsumsqs;/* calculate sums of sums of squares of weights on ... */
} params;

/* Creates a sensible params (note that it sets verbosity to 0 and rdraw
   to FALSE so that no there will be almost nothing printed to graphics
   or stdout during the search) */
params *mk_default_params();

/* Looks on the commandline for overrides to the defaults */
params *mk_params_from_args(int argc,char *argv[]);

/* Prints the params to stdout */
void explain_params(params *ps);

/* Frees params and all substructres */
void free_params(params *ps);

/* Twinpack either contains one datapack (the DATASET) or two
   datapacks (the DATASET and the RANDOMSET). Also, by means of the
   format field, specifies whether the search is DD DR RR DDR etc... */
typedef struct twinpack
{
  int n; /* The n in npoint */
  datapack *dp_data; /* The primary data */
  datapack *dp_random; /* May be NULL (denoting "do autocorrelation). If
                           non-null is an independent dataset */
  char *format;  /* Must be a null-terminated string with n characters.
                    Each character must be d or r. 

                    If the i'th character is d, means "when testing an
                    n-tuple against the matcher, take the i'th point of
                    the n-tuple from the DATASET.

                    If the i'th character is r, means "when testing an
                    n-tuple against the matcher, take the i'th point of
                    the n-tuple from the RANDOMSET. */
     
  bool d_used;  /* True iff there's a d somewhere in format */
  bool r_used;  /* True iff there's a r somewhere in format */
  bool use_permute; /* default TRUE. should be no need to change. used only
                       to do comparison tests between new and old ways of
                       doing asymmetric searches */
} twinpack;

/* After calling, never touch dp_data or dp_random directly again */
twinpack *mk_twinpack_incfree_datapacks(datapack *dp_data,
					datapack *dp_random,
					int n,
					char *format);

/* Looks for "in <filename>" for the primary DATASET. Also looks
   for "rdata <filename> for the RANDOMSET.
   And also searches for "format <string>"

  Example use:   .... in data.csv rfile random.csv n 3 format ddr */
twinpack *mk_twinpack_from_args(params *ps,int argc,char *argv[]);

/* True if the RANDOMSET exists */
bool twinpack_has_random(twinpack *tp);

void free_twinpack(twinpack *tp);

void explain_twinpack(twinpack *tp);

datapack *twinpack_datapack(twinpack *tp);

dym *twinpack_x(twinpack *dp);

mrpars *twinpack_mrpars(twinpack *dp);

dyv *twinpack_metric(twinpack *dp);

int twinpack_n(twinpack *tp);

/* nout simply contains the results of a search */

typedef struct nout
{
  double count;
  double lo;
  double hi;
  dyv *wlobound;  /* these three refer to accumulating the product of the */
  dyv *whibound;  /* weights of each member in a tuple that matches the   */
  dyv *wresult;   /* template                                             */
  dyv *wsum;      /* accumulates the sum of weights of each matched tuple */
  dyv *wsumsq;    /* same except sum of squares of weights                */
  double ferr;
  int secs;
} nout;

void explain_nout_header();

void explain_nout_with_suffix(char *suffix,nout *no);

void explain_nout(nout *no);

void free_nout(nout *no);

#define MAX_NOUTS 1000

typedef struct nouts
{
  int size;
  nout *ns[MAX_NOUTS];
} nouts;

nouts *mk_empty_nouts();

int nouts_size(nouts *ns);

void add_to_nouts(nouts *ns,nout *no);

void free_nouts(nouts *ns);

nout *nouts_ref(nouts *ns,int i);

void explain_nouts(string_array *matcher_strings,nouts *nouts);

  
#endif

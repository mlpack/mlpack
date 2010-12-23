#ifndef EXAMPLE_H
#define EXAMPLE_H

#include "mmgr.h"

#ifndef O_LARGEFILE //for OSX
#define O_LARGEFILE 0
#endif

typedef unsigned long T_IDX; // type for feature indices
typedef float T_VAL; // type for feature values
typedef float T_LBL; // type for lables

// A word of a sparse vector; e.g. 1:9 10:2.3
typedef struct feature {
  T_IDX widx; // starts from 1
  T_VAL wval;
} FEATURE;

// A sparse vector; e.g. 2:0.2 10: 2.3e-01 4:3.2e-02
typedef struct svec {
  // for a general sparse vector
  FEATURE *feats;
  char *userdefined;
  size_t num_nz_feats; // Number of nonzero features.
  // for parallel computing
  //size_t threads_to_finish;
} SVEC;

// An example with sparse vector and labels
typedef struct example {
  // for a general sparse vector
  FEATURE *feats;
  T_LBL label;
  char *userdefined;
  size_t num_nz_feats; // Number of nonzero features.
  // for parallel computing
  bool in_use;
  //size_t threads_to_finish;
} EXAMPLE;


// create an empty dense vector
SVEC *CreateEmptyDvector(size_t fnum, char *userdefined) {
  SVEC *vec;
  vec = (SVEC *)my_malloc(sizeof(SVEC));
  vec->feats = (FEATURE *)my_malloc(sizeof(FEATURE)*(fnum));
  for(size_t i=0; i<fnum; i++) { 
    vec->feats[i].widx = i + 1; // feature index starts from 1
    vec->feats[i].wval = 0.0;
  }
  vec->num_nz_feats = fnum;
  return vec;
}

// create an empty sparse vector
SVEC *CreateEmptySvector(char *userdefined) {
  SVEC *vec;
  vec = (SVEC *)my_malloc(sizeof(SVEC));
  vec->feats = NULL;
  vec->num_nz_feats = 0;
  return vec;
}

SVEC *CreateSvector(FEATURE *feats, char *userdefined) {
  SVEC *vec;
  long fnum,i;

  fnum = 0;
  while(feats[fnum].widx) {
    fnum++;
  }
  vec = (SVEC *)my_malloc(sizeof(SVEC));
  vec->feats = (FEATURE *)my_malloc(sizeof(FEATURE)*(fnum));
  for(i=0;i<fnum;i++) { 
      vec->feats[i] = feats[i];
  }
  vec->num_nz_feats = fnum;

  fnum=0;
  while(userdefined[fnum]) {
    fnum++;
  }
  vec->userdefined = (char *)my_malloc(sizeof(char)*(fnum));
  for(i=0; i<fnum; i++) { 
      vec->userdefined[i]=userdefined[i];
  }
  return vec;
}

// copy x to v
void CopyFromExample(SVEC *v, EXAMPLE *x) {
  size_t nz_x = x->num_nz_feats;
  v->feats = (FEATURE *)realloc(v->feats, nz_x*sizeof(FEATURE));
  for (size_t i=0; i<nz_x; i++) {
    v->feats[i].widx = x->feats[i].widx;
    v->feats[i].wval = x->feats[i].wval;
  }
  v->num_nz_feats = nz_x;
}

void EmptyFeatures(SVEC *v) {
  if (v->feats) {
    free(v->feats);
  }
  v->feats = NULL;
  v->num_nz_feats = 0;
}

// add a copy of feature (f) at the back of sparse vecotr (v)
void PushBackOne(SVEC *v, FEATURE *f) {
  size_t v_nz = v->num_nz_feats;
  v->feats = (FEATURE *)realloc(v->feats, (v_nz+1)*sizeof(FEATURE));
  v->num_nz_feats = v->num_nz_feats + 1;
  v->feats[v_nz].widx = f->widx;
  v->feats[v_nz].wval = f->wval;
}

// add a copy of feature (f) at a given position (pos) in a sparse vector (v)
void InsertOne(SVEC *v, FEATURE *f, size_t pos) {
  size_t v_nz = v->num_nz_feats;
  // increase capacity
  v->feats = (FEATURE *)realloc(v->feats, (v_nz+1)*sizeof(FEATURE));
  v->num_nz_feats = v->num_nz_feats + 1;
  // move 2nd part
  memmove(v->feats+pos+1, v->feats+pos, (v_nz-pos)*sizeof(FEATURE));
  // insert feats after 1st part
  v->feats[pos].widx = f->widx;
  v->feats[pos].wval = f->wval;
}

void FreeSvec(SVEC *v) {
  if (v) {
    if (v->feats) {
      //free(v->feats);
    }
    if (v->userdefined) {
      free(v->userdefined);
    }
  }
}

EXAMPLE *CreateEmptyExample() {
  EXAMPLE *ex;
  ex = (EXAMPLE *)my_malloc(sizeof(EXAMPLE));
  ex->feats = NULL;
  ex->num_nz_feats = 0;
  return ex;
}

void CreateExample(EXAMPLE *example, FEATURE *feats, T_LBL label, char *userdefined, size_t max_features_example) {
  size_t fnum,i;
  fnum = 0;
  // feature index starts from 1
  for (i=0; i< max_features_example; i++) {
    if(feats[fnum].widx > 0)
      fnum++;
  }
  example->label = label;
  example->in_use = false;
  example->feats = (FEATURE *)my_malloc(sizeof(FEATURE)*(fnum));
  for(i=0;i<fnum;i++) { 
      example->feats[i] = feats[i];
  }
  example->num_nz_feats = fnum;

  fnum=0;
  while(userdefined[fnum]) {
    fnum++;
  }
  example->userdefined = (char *)my_malloc(sizeof(char)*(fnum));
  for(i=0; i<fnum; i++) { 
    example->userdefined[i]=userdefined[i];
  }
}

void FreeExample(EXAMPLE *x) {
  if (x) {
    if (x->feats) {
      free(x->feats);
    }
    if (x->userdefined) {
      free(x->userdefined);
    }
  }
}

void print_svec(SVEC *v) {
  cout <<"v== ";
  for (size_t i=0; i<v->num_nz_feats; i++) {
    cout << v->feats[i].widx << ":" << v->feats[i].wval << ", ";
  }
  cout << endl;
}

void print_ex(EXAMPLE *x) {
  cout << x->label << ", x== ";
  for (size_t i=0; i<x->num_nz_feats; i++) {
    cout << x->feats[i].widx << ":" << x->feats[i].wval << ", ";
  }
  cout << endl;
}



#endif

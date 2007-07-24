#ifndef CP_UTILS
#define CP_UTILS

#include "ambs.h"
#include "cp_args.h"

int int_in_args(int key, arg args[], int default_value);
double dbl_in_args(int key, arg args[], double default_value);
char* str_in_args(int key, arg args[], char *default_value);
int exists_in_args(int key, arg args[]);
int lookup(char *string, char **option_names, int num_options,
           char *option_descrip);
char* mk_build_options_string(char *string, char **option_names, 
                              int num_options, char *option_descrip);
int get_args(int argc, char *argv[], arg args[], int nargs);
void init_arg(arg args[],int i,char *name,bool optflag,int na, char *doc);
void free_args(arg args[],int nargs);

#endif

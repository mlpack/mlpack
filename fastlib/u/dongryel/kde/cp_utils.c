#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <malloc.h>
#include "amma.h"
#include "ammarep.h"
#include "amdmex.h"
#include "ds.h"
#include "cp_args.h"
#include "cp_string.h"
#include "cp_utils.h"



int int_in_args(int key, arg args[], int default_value)
{
  if ( args[key].exists )
    return atoi(args[key].arg_str);
  else 
    return default_value;
}

double dbl_in_args(int key, arg args[], double default_value)
{
  if ( args[key].exists )
    return atof(args[key].arg_str);
  else 
    return default_value;
}

char* str_in_args(int key, arg args[], char *default_value)
{
  if ( args[key].exists )
    return args[key].arg_str;
  else 
    return default_value;
}

int exists_in_args(int key, arg args[])
{
  return ( args[key].exists );
}

/* Given a string and the array of option names, return the index of name the
   string matches, if any; otherwise returns -1. */
int lookup(char *string, char **option_names, int num_options,
           char *option_descrip)
{
  int i, index;

  index = -1;
  for (i = 0; i < num_options; i++)
    if (streq(string, option_names[i]))
    {
      index = i;
      break; /* from for loop */
    }

  /* if an incorrect option was specified */
  if (index == -1)
  {
    printf("Illegal %s option specified: %s\n", option_descrip,
           option_names[index]);

    /* show all legal options */
    printf ("Available %s options: ", option_descrip);
    for (i = 0; i < num_options; i++)
      printf ("%s%s", ((i>0)?", ":""), option_names[i]);
    printf ("\n");
      
    return(-1);
  }
  return( index );
}

/* Build a string containing all the option names for a particular parameter,
   given the array of option names. */
char* mk_build_options_string(char *s, char **option_names, 
                              int num_options, char *option_descrip)
{
  int i;
  char *copy;

  strcpy(s, option_descrip);

  strcat(s, " {");

  for (i = 0; i < num_options; i++)
  {
    if (i>0)
      strcat(s, " | ");
    strcat(s, option_names[i]);
  }
  strcat(s, "}");

  copy = AM_MALLOC_ARRAY(char, strlen(s) + 1);
  strcpy(copy, s);
  return copy;
}

int get_args(int argc, char *argv[], arg args[], int nargs)
{
  return Get_Args(argc,argv,args,nargs);
}

void free_args(arg args[], int nargs)
{
  int i;
  for(i = 0; i < nargs; i++) {
    if(args[i].com_str != NULL)
      AM_FREE_ARRAY(args[i].com_str,char,strlen(args[i].com_str) + 1);
    if(args[i].doc_str != NULL)
      AM_FREE_ARRAY(args[i].doc_str,char,strlen(args[i].doc_str) + 1);
  }
}

void init_arg(arg args[],int i,char *name,bool optflag,int na, char *doc)
{
  args[i].com_str=AM_MALLOC_ARRAY(char,strlen(name) + 1);
  strcpy(args[i].com_str,name);
  args[i].optional=optflag;
  args[i].nargs=na; 
  args[i].doc_str=AM_MALLOC_ARRAY(char,strlen(doc) + 1);
  strcpy(args[i].doc_str,doc);
  AM_FREE_ARRAY(doc,char,strlen(doc) + 1);
}

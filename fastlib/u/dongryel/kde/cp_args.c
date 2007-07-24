/* cp_args.c */

#include <stdio.h>
#include <string.h>
#include "cp_platform.h"
#include "cp_string.h"
#include "cp_args.h"

/* machine dependant stuff */
/* issue 1 */
#if (CP_STRDUP_DEFINED == CP_STRDUP_IN_CP_STRING_H)
#endif
/* issue 2 */
/*#if (CP_NUM_OF_CMD_LINE_ARGS == CP_CMD_LINE_ARGS_ONE)
	#define cpCmdLineArgsExist(A)					(A > CP_CMD_LINE_ARGS_ONE)
	#define cpCmdLineArgsNotExist(A)				(A <= CP_CMD_LINE_ARGS_ONE)
#elif (CP_NUM_OF_CMD_LINE_ARGS == CP_CMD_LINE_ARGS_ONE)
	#define cpCmdLineArgsExist(A)					(A > CP_CMD_LINE_ARGS_ZERO)
	#define cpCmdLineArgsNotExist(A)				(A <= CP_CMD_LINE_ARGS_ZERO)
#endif
*/
/***** MACUNIXDOS...MACUNIXDOS...MACUNIXDOS *****/

/****************************************************************************
 *   function:      get_arg_match
 *   description:   Get the match for the given string within the arg list 
 *   calls:         none
 *   returns:       matched string index
 ****************************************************************************/
int get_arg_match(comstr, args, nargs)
     char *comstr;
     arg args[];
     int nargs;
{
  int i, match = -1;
  
  /* don't skip '-' sign  */
  for (i = 0; i < nargs; i++) {
    if (args[i].com_str != NULL && strcmp(comstr,args[i].com_str) == 0)
    {
      match = i;
      args[i].exists = TRUE;
      break;
    }
  }

  comstr++;	/* skip '-' sign  */
  for (i = 0; i < nargs; i++) {
    if (args[i].com_str != NULL && strcmp(comstr,args[i].com_str) == 0)
    {
      match = i;
      args[i].exists = TRUE;
      break;
    }
  }

  return(match);
}


/****************************************************************************
 *   function:      Print_Args
 *   description:   Prints the list of args as defined by the Init_Args.
 *   calls:         none
 *   returns:       nothing
 ****************************************************************************/
int Print_Args(args, nargs)
     arg args[];
     int nargs;
{
  int i;
  
  printf("\nList of args:\n");
  for(i = 0; i < nargs; i++)
    if (args[i].exists)
      printf("%d:  %s '%s'\n", i, args[i].com_str, args[i].arg_str);

  return(Get_Arg_OK);
}


/****************************************************************************
 *   function:      Get_Args
 *   description:   Gets the args given by the command line with respect to  
 *                  Init_Args' definition.
 *   calls:         print_usage
 *					get_arg_match
 *   returns:       nothing
 ****************************************************************************/
int Get_Args(argc, argv, args, nargs)
     int argc;
     char *argv[];
     arg args[];
     int nargs;
{
  char strbuf[1024];

  int i, j, aindex;
  
  for (i = 0; i < nargs; i++)  {
    args[i].exists = FALSE;
    args[i].arg_str = (char*) NULL;
  }
  
  for (i = 1; i <= argc; i++)  {
    aindex = get_arg_match(argv[i], args, nargs);
    if (aindex == -1)
	{
	  printf("%s: command '%s' is undefined.\n", argv[0], argv[i]);
	  print_usage(argv[0], args, nargs);
	  //return(Get_Arg_ERROR);
	}
    else {
      if (args[aindex].nargs > 0)  {
	i++;
	if (i > argc)  {
	  printf("%s: insufficient arguments for command '-%s'.\n", 
		 argv[0], args[aindex].com_str);
	  print_usage(argv[0], args, nargs);
	  return(Get_Arg_ERROR);
	}        
	
	strcpy(strbuf,argv[i]);
	for(j = 2; j <= args[aindex].nargs; j++)  {
	  strcat(strbuf, " ");
	  i++;
	  if (i > argc)   {
	    printf("%s: insufficient arguments for command '-%s'.\n",
		   argv[0], args[aindex].com_str);
	    print_usage(argv[0], args, nargs);
	    return(Get_Arg_ERROR);
	  }        
	  strcat(strbuf, argv[i]);
	}
	
	args[aindex].arg_str = strdup(strbuf);
      }
      else  /* this command has 0 arguments */
        args[aindex].arg_str = "";
    }
  }
  
  for(i = 1; i < nargs; i++) {
    if (args[i].com_str != NULL && !args[i].optional && !args[i].exists)  {
      printf("%d-th argument...\n", i);
      printf("%s: '-%s' flag missing.\n", argv[0], args[i].com_str);
      printf("%s: insufficient arguments, aborting.\n", argv[0]);
      print_usage(argv[0], args, nargs);
      return(Get_Arg_ERROR);
    }
  }
  
  return(Get_Arg_OK);
}


/****************************************************************************
 *   function:      print_usage
 *   description:   Prints the usage message corresponding to the arg types 
 *                  obtained from the Init_Args definition.
 *   calls:         none
 *   returns:       nothing
 ****************************************************************************/
int print_usage(com_name, args, nargs)
     char *com_name;
     arg args[];
     int nargs;
{
  int i;
  printf("\nUSAGE: %s  ", com_name);
  for(i = 0; i < nargs; i++)
  {
    if(args[i].com_str != NULL) {
      if (args[i].optional) printf("[");
      printf("-%s '%s'", args[i].com_str, args[i].doc_str);
      if (args[i].optional) printf("]");
      printf("\n     ");
    }
  }
  printf("\n");

  return(Get_Arg_OK);
}

/* Eof. */

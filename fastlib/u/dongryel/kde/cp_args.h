/* cp_args.h */

#ifndef CP_ARGS_HDR
#define CP_ARGS_HDR

/* constants */
#define   BOOLEAN        int
#ifndef TRUE
#define   TRUE           1
#define   FALSE          0
#endif
#define   Get_Arg_ERROR  -1
#define   Get_Arg_OK     0

#define   MAX_BUFF       1024

/* typedefs */
typedef struct
     {
     char      *com_str;
     BOOLEAN   optional;
     int       nargs;
     BOOLEAN   exists;
     char      *arg_str;
     char      *doc_str;
     } arg, *arg_ptr;

/* macro accessors */
#define  Init_Arg(args,i,name,optflag,na,doc)    \
		args[i].com_str=name; \
		args[i].optional=optflag; \
		args[i].nargs=na; \
                args[i].doc_str=doc;

/* macros */
#define Set_Arg(args,i,var)						if (args[i].exists) var = args[i].arg_str
#define cpCmdLineArgsExist(A)					(A > CP_CMD_LINE_ARGS_ONE)
#define cpCmdLineArgsNotExist(A)				(A <= CP_CMD_LINE_ARGS_ONE)

#if (CP_NUM_OF_CMD_LINE_ARGS == CP_CMD_LINE_ARGS_ZERO)
	#undef cpCmdLineArgsExist
	#define cpCmdLineArgsExist(A)					(A > CP_CMD_LINE_ARGS_ZERO)
	#undef cpCmdLineArgsNotExist
	#define cpCmdLineArgsNotExist(A)				(A <= CP_CMD_LINE_ARGS_ZERO)
#endif

/* function prototypes */
#ifdef __STDC__
int get_arg_match(char *comstr, arg args[], int nargs);
int Print_Args(arg args[], int nargs);
int Get_Args(int argc, char *argv[], arg args[], int nargs);
int print_usage(char *com_name, arg args[], int nargs);

#endif
void init_args(arg args[]);

#endif

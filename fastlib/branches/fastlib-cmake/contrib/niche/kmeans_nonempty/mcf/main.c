/**************************************************************************
Contains modul MAIN.C of ZIB optimizer MCF

AUTHOR: Andreas Loebel

This software was developed at ZIB Berlin. Maintenance and revisions 
on responsibility of Dr. Andreas Loebel

Dr. Andreas Loebel
Ortlerweg 29b, 12207 Berlin

Konrad-Zuse-Zentrum fuer Informationstechnik Berlin (ZIB)
Scientific Computing - Optimization
Takustr. 7, 14195 Berlin-Dahlem

Copyright (c) 1997-2000 ZIB.            All rights reserved.
Copyright (c) 2000-2003 ZIB & Loebel.   All rights reserved.
**************************************************************************/
/*  LAST EDIT: Tue Jun  3 13:25:31 2003 by Andreas Loebel (opt0.zib.de)  */
#ident "$Id: main.c,v 1.1.1.1 2003/06/03 12:06:54 bzfloebe Exp $"


#include "mcf.h"
#include "mcfutil.h"
#include "parmanag.h"


#define GET_QUIT          10
#define GET_LOAD          20
#define GET_OPTIMIZE      30
#define GET_HELP          40
#define GET_WRITE         50
#define GET_DISPLAY       60
#define GET_PIVOT         70
#define GET_SYSTEM        80
#define GET_STATUS        90
#define GET_DUAL_METHOD   100
#define GET_PRIMAL_METHOD 200

long get_command( char *com );


#define EXEC_READ            1000
#define EXEC_RESET_AND_INIT  2000
#define EXEC_SIMPLEX         3000
#define EXEC_COMPUTE_COSTS   4000
#define EXEC_WRITE_SOLUTION  5000
#define EXEC_FREE_NET        6000
#define EXEC_DISPLAY_STATS   7000
#define EXEC_QUIT_MCF        8000
#define EXEC_PRINT_HELP      9000


long exec_command(
                  long mode, 
                  long (*read)( char *, MCF_network_p ),
                  long (*simplex)( MCF_network_p ),
                  long (*start)( MCF_network_p ),
                  MCF_node_p (*find_iminus)( MCF_DUAL_BLA_PARAMETERS ),
                  MCF_arc_p (*find_bea)( MCF_PRIMAL_BEA_PARAMETERS )
                  );




MCF_network_t net;


char    infile[81];
char    outfile[81];
char    tmpstr[81];
char    cmdstr[200];
char    *command;


double start_cpu_time, simplex_cpu_time;      
#ifndef _WIN32        
double real_time;
#endif



long exit_immediate = 0;
long problem_in = 0;
long optimize  = 0;
long display = 0;
long problem_out = 0;
long pivot_selected = 0;
long messages = 1;
long help = 0;







int main( int argc, char *argv[] )
{
    long (*read)( char *, MCF_network_p )         = &MCF_read_dimacs_min;
    long (*simplex)( MCF_network_p )              = &MCF_primal_net_simplex;
    long (*start)( MCF_network_p )                = &MCF_primal_start_artificial;
    MCF_arc_p (*find_bea)( MCF_PRIMAL_BEA_PARAMETERS )   = 
        &MCF_primal_bea_mpp_30_5;
    MCF_node_p (*find_iminus)( MCF_DUAL_BLA_PARAMETERS ) = 
        &MCF_dual_iminus_mpp_30_5;
            
    long error = 0; 
    long primal;
    





    infile[0] = outfile[0] = '\0';

    if( argc > 1 )
        MCF_par_manager( argc, argv, infile, outfile, &exit_immediate,
            &optimize, &display, &problem_out, &pivot_selected, &messages, 
            &help, &simplex, &start, &find_bea, &find_iminus );

    if( messages )
    {
        printf( "\nNetwork simplex for the minimum-cost " );
        printf( "flow problem Vers. 1.3.%s\n", MCF_ARITHMETIC_TYPE );
        printf( "                Written  by  Andreas  Loebel,\n\n" );
        printf( "Copyright (c) 1997-2000  ZIB.           " );
        printf( "All Rights Reserved.\n" );
        printf( "Copyright (c) 2000-2003  ZIB & Loebel.  " );
        printf( "All Rights Reserved.\n" );
        printf( "\n" );
    }



    primal = (simplex == MCF_primal_net_simplex);
    
    if( help )
        exec_command( EXEC_PRINT_HELP, NULL, NULL, NULL, NULL, NULL );

    if( argc > 1 && *infile )
    {
        /* Call read. */
        if( exec_command( EXEC_READ, read, NULL, NULL, NULL, NULL ) )
        {
            if( messages )
                goto MENU;
            else
                return -1;
        }
        else
            problem_in = 1;
    }

    if( optimize && problem_in )
    {
        /* Call start and simplex. */
#ifndef _WIN32
        real_time = MCF_get_wall_time();
#endif
        if( !exec_command( EXEC_RESET_AND_INIT, NULL, NULL, 
                           start, find_iminus, find_bea ) )
        {
            if( primal )
                error = exec_command( EXEC_SIMPLEX, NULL, simplex, NULL, 
                                 find_iminus, find_bea );
            else
                error = exec_command( EXEC_SIMPLEX, NULL, simplex, NULL,
                                 find_iminus, find_bea );
        }
        if( !error )
            exec_command( EXEC_COMPUTE_COSTS, NULL, NULL, NULL, NULL, NULL );
        else
        {
            if( messages )
                goto MENU;
        }
#ifndef _WIN32
        real_time = MCF_get_wall_time() - real_time;
#endif
    }
        
    if( display && problem_in )
        /* Call display solution to standard out. */
        exec_command( EXEC_DISPLAY_STATS, NULL, simplex, NULL, NULL, NULL );

    if( problem_out && problem_in )
        /* call write_solution */
        if( exec_command( EXEC_WRITE_SOLUTION, NULL, NULL, NULL, NULL, NULL ) )
            if( messages )
                goto MENU;

    if( exit_immediate )
        /* Call exit program. */
        exec_command( EXEC_QUIT_MCF, NULL, NULL, NULL, NULL, NULL );






MENU:
    MCF_SET_ZERO( cmdstr, 200*sizeof(char) ); 
    command = strtok( cmdstr, " \t\n" );
    
    if( !help )
        printf( "\nenter <help> or <?> for help" );

    while( 1 )
    {
        error = (long)NULL;
        if( command )
            command = (char *)strtok( NULL, " \t\n" );
        while( !command )
        {
            if( primal )
                printf( "\nprimal network simplex > " );
            else
                printf( "\ndual network simplex > " );
            if( gets( cmdstr ) )
                command = strtok( cmdstr, " \t\n" );
            else
                command = strcpy( cmdstr, "quit" );
        }

        switch( get_command( command ) )
        {
        case GET_QUIT:                         
            exec_command( EXEC_QUIT_MCF, NULL, NULL, NULL, NULL, NULL );
            break;
            
        case GET_LOAD: 
            if( problem_in )
                exec_command( EXEC_FREE_NET, NULL, NULL, NULL, NULL, NULL );
                    
            if( !(command = strtok( NULL, " \t\n" )) )
            {
                printf( "Name of file to read: " );
                gets( infile );
            }
            else
                strcpy( infile, command );
                        
            if( exec_command( EXEC_READ, read, NULL, NULL, NULL, NULL ) )
                exec_command( EXEC_FREE_NET, NULL, NULL, NULL, NULL, NULL );
            else 
            {
                problem_in = 1;
                problem_out = 0;
            }
            break;

        case GET_OPTIMIZE:
            if( !problem_in )
            {
                printf( "no network in memory\n" );
                break;
            }
#ifndef _WIN32
            real_time = MCF_get_wall_time();
#endif
            if( !exec_command( EXEC_RESET_AND_INIT, NULL, NULL, start, 
                          find_iminus, find_bea ) )
            {
                if( primal )
                    error = exec_command( EXEC_SIMPLEX, NULL, simplex, NULL, 
                                     find_iminus, find_bea );
                else
                    error = exec_command( EXEC_SIMPLEX, NULL, simplex, NULL,
                                     find_iminus, find_bea );
            }
#ifndef _WIN32
            real_time = MCF_get_wall_time() - real_time;
#endif
            
            if( !error )
                exec_command( EXEC_COMPUTE_COSTS, NULL, NULL, 
                              NULL, NULL, NULL );

            optimize = 1;
                        
            break;

        case GET_HELP: 
            exec_command( EXEC_PRINT_HELP, NULL, NULL, NULL, NULL, NULL );
            break;
            
        case GET_WRITE:
            if( !(command = strtok( NULL, " \t\n" )) )
            {
                printf( "Name of file to write: " );
                gets( outfile );
            }
            else
                strcpy( outfile, command );

            if( !exec_command( EXEC_WRITE_SOLUTION, NULL, NULL, NULL, NULL, NULL ) )
                problem_out = 1;
            break;

        case GET_DISPLAY: 
            exec_command( EXEC_DISPLAY_STATS, NULL, simplex, NULL, NULL, NULL );
            break;

        case GET_PIVOT: 
            if( !(command = strtok( NULL, " \t\n" )) )
            {
                printf( "Which strategy: " );
                gets( tmpstr );
            }

            switch( *tmpstr )
            {
            case 'a': case 'A':
                find_bea = MCF_primal_bea_all; 
                find_iminus = MCF_dual_iminus_cycle;
                break;
            case 'c': case 'C':
                find_bea = MCF_primal_bea_cycle; 
                find_iminus = MCF_dual_iminus_cycle;
                break;
            case 'm': case 'M':
                find_bea = MCF_primal_bea_mpp_30_5; 
                find_iminus = MCF_dual_iminus_mpp_30_5;
                break;
            default:
                printf( "Unknown pivot strategy\n" );
            }
            break;

        case GET_SYSTEM:
            if( (command = strtok( NULL, "\n" )) )
            {
                fflush( stdout );
                system( command );
            }
            break;
            
        case GET_STATUS:
            if( primal )
            {
                printf( "\nUsing the primal network simplex\n");
                printf( "Pivot strategy is " );
                if( find_bea == MCF_primal_bea_all )
                    printf( "most invalid\n" );
                else if( find_bea == MCF_primal_bea_cycle )
                    printf( "cycle-smallest index\n" );
                else if( find_bea == MCF_primal_bea_mpp_30_5 
                        || find_bea == MCF_primal_bea_mpp_50_10
                        || find_bea == MCF_primal_bea_mpp_200_20 )
                    printf( "multiple partial pricing\n" );
            }
            else
            {
                printf( "\nUsing the dual network simplex\n");
                printf( "Pivot strategy is " );
                if( find_iminus == MCF_dual_iminus_cycle )
                    printf( "cycle-smallest index\n" );
                else if( find_iminus == MCF_dual_iminus_mpp_30_5
                        || find_iminus == MCF_dual_iminus_mpp_50_10 )
                    printf( "multiple partial pricing\n" );
            }
            if( optimize )
                exec_command( EXEC_DISPLAY_STATS, NULL, simplex, 
                              NULL, NULL, NULL );
            else if( problem_in )
                printf( "Problem name %s\n", infile );
            else 
                printf( "No Problem exists\n" );
            break;

        case GET_DUAL_METHOD:
            simplex = MCF_dual_net_simplex;
            start = MCF_dual_start_artificial;
            primal = 0;
            
            break;
            
        case GET_PRIMAL_METHOD:
            simplex = MCF_primal_net_simplex;
            start = MCF_primal_start_artificial;
            primal = 1;
            
            break;
            
        default: 
            printf( "unknown command, type help\n" );
            break; 
        }
    }
} 








long exec_command( 
             long mode,
             long (*read)( char *, MCF_network_p ),
             long (*simplex)( MCF_network_p ),
             long (*start)( MCF_network_p ),
             MCF_node_p (*find_iminus)( MCF_DUAL_BLA_PARAMETERS ),
             MCF_arc_p (*find_bea)( MCF_PRIMAL_BEA_PARAMETERS )
             )
{
    long error = (long)NULL;
    MCF_node_p node;
    
    switch( mode )
    {
    case EXEC_READ: 
        error = read( infile, &net );
        break;

    case EXEC_RESET_AND_INIT:
        start_cpu_time = simplex_cpu_time = 0.0;
        net.primal_unbounded = 0;
        net.dual_unbounded = 0;
        net.feasible = 0;
        net.iterations = 0;
        for( node = net.nodes; node != net.stop_nodes; node++ )
        {
            node->basic_arc = NULL;
            node->pred = NULL;
            node->child = NULL;
            node->right_sibling = NULL;
            node->left_sibling = NULL;
            node->subtreesize = 0;
            node->orientation = 0;
            node->potential = MCF_ZERO;
            node->flow = MCF_ZERO;
        }
        
        if( (MCF_arc_p)find_bea == (MCF_arc_p)MCF_primal_bea_mpp_30_5 
           || (MCF_arc_p)find_bea == (MCF_arc_p)MCF_primal_bea_mpp_50_10 
           || (MCF_arc_p)find_bea == (MCF_arc_p)MCF_primal_bea_mpp_200_20 
           || (MCF_arc_p)find_bea == (MCF_arc_p)MCF_primal_bea_mpp_100_10 )
        {
            if( net.m < 10000 )
                find_bea = MCF_primal_bea_mpp_30_5;
            else if( net.m > 100000 )
                find_bea = MCF_primal_bea_mpp_200_20;
            else find_bea = MCF_primal_bea_mpp_50_10;
            /**
            find_bea = MCF_primal_bea_mpp_70_15;
            **/
        }
        
        if( (MCF_node_p)find_iminus == (MCF_node_p)MCF_dual_iminus_mpp_30_5
           || (MCF_node_p)find_iminus == (MCF_node_p)MCF_dual_iminus_mpp_50_10 )
        {
            if( net.n < 10000 )
                find_iminus = MCF_dual_iminus_mpp_30_5;
            else
                find_iminus = MCF_dual_iminus_mpp_50_10;
        }
        
        net.find_iminus = find_iminus;
        net.find_bea = find_bea;
        
        start_cpu_time = MCF_get_cpu_time();
                
        error = start( &net );

        start_cpu_time = MCF_get_cpu_time() - start_cpu_time;
        
        break;

    case EXEC_SIMPLEX:
        simplex_cpu_time = MCF_get_cpu_time();
        error = simplex( &net );
        simplex_cpu_time = MCF_get_cpu_time() - simplex_cpu_time;
        break;

    case EXEC_COMPUTE_COSTS:
        if( net.feasible )
            net.optcost = MCF_primal_obj( &net );
        break;

    case EXEC_WRITE_SOLUTION:
        error = MCF_write_solution( infile, outfile, &net, 
                           (long)(start_cpu_time+simplex_cpu_time) );
        break;

    case EXEC_FREE_NET:
        MCF_free( &net ); 
        problem_in = 0;
        net.iterations = 0;
        net.primal_unbounded = 0;
        net.dual_unbounded = 0;
        net.feasible = 0;
        net.optcost = 0;
        optimize = 0;
        infile[0] = outfile[0] = '\0';
        net.n = 0;
        net.m = 0;
        break;
        
    case EXEC_DISPLAY_STATS: 
        if( !problem_in )
        {
            printf( "no network in memory\n" );
            break;
        }
        printf( "\n%s: %ld nodes / %ld arcs\n",
               infile, (net.n), (net.m) );
        if( net.primal_unbounded )
        {
            printf( "\n   >>> problem primal unbounded <<<\n" );
            break;
        }
        
        if( net.dual_unbounded )
        {
            printf( "\n   >>> problem dual unbounded <<<\n" );
            break;
        }
        
        if( net.feasible == 0 )
        {
            printf( "\n   >>> problem infeasible or unbounded <<<\n" );
            break;
        }
        
        if( optimize )
        {
            printf( "Cpu time                          : " );
            printf( "%10.2f\n", start_cpu_time + simplex_cpu_time );
#ifndef _WIN32
            printf( "Real time                         : " );
            printf( "%10.2f\n", real_time );
#endif
            printf( "Iterations                        : " );
            printf( "%10ld\n", net.iterations );
                    
            if( MCF_ABS(MCF_dual_obj(&net)-MCF_primal_obj(&net))
                > (double)MCF_ZERO_EPS )
            {
                fprintf( stderr, "NETWORK SIMPLEX: primal-dual object" );
                fprintf( stderr, "ive mismatch!?\n" );
                printf( "Primal optimal objective         " );
#ifdef MCF_FLOAT
                printf( " : %10.6f\n", MCF_primal_obj(&net) );
#else
                printf( " : %10.0f\n", MCF_primal_obj(&net) );
#endif

                printf( "Dual optimal objective           " );
#ifdef MCF_FLOAT
                printf( " : %10.6f\n", MCF_dual_obj(&net) );
#else
                printf( " : %10.0f\n", MCF_dual_obj(&net) );
#endif
            }
            else
            {
                if( simplex == MCF_primal_net_simplex )
                    printf( "Primal optimal objective         " );
                else
                    printf( "Dual optimal objective           " );
#ifdef MCF_FLOAT
                printf( " : %10.6f\n", net.optcost );
#else
                printf( " : %10.0f\n", net.optcost );
#endif
            }
        }
        else
            printf( "No solution\n" );
        break;
        
    case EXEC_QUIT_MCF:
        exec_command( EXEC_FREE_NET, NULL, NULL, NULL, NULL, NULL );
        if( messages )
        {
            printf( "\n" );
        }
        exit(0);
        break;

    case EXEC_PRINT_HELP:
        fflush( stdout );
        printf( "Syntax is: MCF [ [options] inputfile ]\n" );
        printf( "Options:\n" );
        printf( "  -d         use the dual network simplex code\n" );
        printf( "  -px        pivot strategy\n" );
        printf( "             x = m >> multiple partial pricing (default)\n" );
        printf( "                 c >> cycle-smallest index\n" );
        printf( "                 a >> most invalid\n" );
        printf( "  -o         optimize problem\n" );
        printf( "  -v         display solution\n" );
        printf( "  -w [name]  write solution in DIMACS output format to\n");
        printf( "             file \"name\" or append it to the input file\n" );
        printf( "  -e         no messages\n" );
        printf( "  -q         exit at the end\n" );
        printf( "  -h         show this help text\n ");
        printf( "\nCommands:\n" );
        printf( "help (or ?)          display help text\n" );
        printf( "read [filename]      read a new problem from file\n" );
        printf( "load [filename]      read a new problem from file\n" );
        printf( "write [filename]     write solution to file\n" );
        printf( "optimize             optimize\n" );
        printf( "display              write solution to terminal\n" );
        printf( "status               display used routines by simplex\n" );
        printf( "pivot <m,c,a>        select pivot strategy\n" );
        printf( "                     m >> multiple partial pricing\n" );
        printf( "                     c >> cycle-smallest index\n" );
        printf( "                     a >> most invalid\n" );
        printf( "primal               switch to primal network simplex\n" );
        printf( "dual                 switch to dual network simplex\n" );
        printf( "system               issue a shell command\n" );
        printf( "quit                 terminate programm\n" );
        break;

    default: break;
    }

    return error;
}









#define CMP( str, x, i ) if( !strncmp( com, str, x ) ) return i;
#define MAX(a,b) ( (size_t)(a)>=(size_t)(b)?(a):(b) )

long get_command( char *com )
{
    CMP( "exit", MAX( 1, strlen(com) ), GET_QUIT );
    CMP( "quit", MAX( 1, strlen(com) ), GET_QUIT );

    CMP( "load", MAX( 1, strlen(com) ), GET_LOAD );
    CMP( "read", MAX( 1, strlen(com) ), GET_LOAD );

    CMP( "optimize", MAX( 1, strlen(com) ), GET_OPTIMIZE );

    CMP( "help", MAX( 1, strlen(com) ), GET_HELP );
    CMP( "?", MAX( 1, strlen(com) ), GET_HELP );

    CMP( "write", MAX( 1, strlen(com) ), GET_WRITE );

    CMP( "display", MAX( 2, strlen(com) ), GET_DISPLAY );

    CMP( "pivot", MAX( 2, strlen(com) ), GET_PIVOT );

    CMP( "system", MAX( 2, strlen(com) ), GET_SYSTEM );

    CMP( "status", MAX( 2, strlen(com) ), GET_STATUS );

    CMP( "dual", MAX( 2, strlen(com) ), GET_DUAL_METHOD );

    CMP( "primal", MAX( 2, strlen(com) ), GET_PRIMAL_METHOD );

    return -1; /* Unknown command. */
}


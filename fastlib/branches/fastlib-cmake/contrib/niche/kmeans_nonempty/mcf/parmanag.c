/**************************************************************************
Contains modul PARAMANAG.C of ZIB optimizer MCF

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
/*  LAST EDIT: Tue Jun  3 10:51:26 2003 by Andreas Loebel (opt0.zib.de)  */
#ident "$Id: parmanag.c,v 1.1.1.1 2003/06/03 12:06:54 bzfloebe Exp $"


#include "parmanag.h"


#define PRINT_UNKNOWN_PAR printf( "Unknown parameter %s\n", argv[i] )


long MCF_par_manager( 
                int argc,
                char *argv[],
                char *infile,
                char *outfile,
                long *exit_immediate,
                long *optimize,
                long *display,
                long *problem_out,
                long *pivot_selected, 
                long *messages,  
                long *help,
                long (**simplex)( MCF_network_p ),
                long (**start)( MCF_network_p ),
                MCF_arc_p (**findbea)( MCF_DECLARE_PRIMAL_BEA_PARAMETERS ),
                MCF_node_p (**findbla)( MCF_DECLARE_DUAL_BLA_PARAMETERS )
                )
{
    char c;
    long i;     
    long primal = 1; /* default: primal network simplex */
    


    if( *argv[argc-1] != '-' )
        strcpy( infile, argv[--argc] );
    else
        *infile = (char)0;
    
    for( i = 1; i < argc; i++ )
        if( !(strncmp(argv[i],"-d",2) && strncmp(argv[i],"-D",2)) )
            primal = 0;

    if( primal )
    {
        *simplex = MCF_primal_net_simplex;
        *start = MCF_primal_start_artificial;
    }
    else
    {
        *simplex = MCF_dual_net_simplex;
        *start = MCF_dual_start_artificial;
    }
        
    
    for( i = 1; i < argc; i++ )

        if( *argv[i] != '-' )
        /* Switches are introduced by '-' */
            PRINT_UNKNOWN_PAR;
    
        else if( (c=*(argv[i]+1)) == 'd' || c == 'D' )
            ;
        
        else if( (c=*(argv[i]+1)) == 'p' || c == 'P' )
        {
            *pivot_selected = 1; 
            if( primal )
            {
                switch( *(argv[i]+2) )
                {
                case 'a': case 'A': *findbea = MCF_primal_bea_all; 
                    break;
                case 'c': case 'C': *findbea = MCF_primal_bea_cycle;
                    break;
                case 'm': case 'M': *findbea = MCF_primal_bea_mpp_50_10; 
                    break;
                default: PRINT_UNKNOWN_PAR;
                }
            }
            else /* not primal */
            {
                switch( *(argv[i]+2) )
                {
                case 'm': case 'M': *findbla = MCF_dual_iminus_mpp_30_5;
                    break;
                case 'c': case 'C': *findbla = MCF_dual_iminus_cycle;
                    break;
                default: PRINT_UNKNOWN_PAR;
                }
            }
        }

        else if( (c=*(argv[i]+1)) == 'o' || c == 'O' )
            *optimize = 1;

        else if( (c=*(argv[i]+1)) == 'v' || c == 'V' )
            *display = 1;

        else if( (c=*(argv[i]+1)) == 'w' || c == 'W' )
        {
            *problem_out = 1;
            if( *(argv[i+1]) == '-' )
                strcpy( outfile, infile );
            else
                strcpy( outfile, argv[++i] );
        }

        else if( (c=*(argv[i]+1)) == 'e' || c == 'E' )
            *messages = 0;

        else if( (c=*(argv[i]+1)) == 'h' || c == 'H' )
            *help = 1;

        else if( (c=*(argv[i]+1)) == 'q' || c == 'Q' )
            *exit_immediate = 1;

        else
            PRINT_UNKNOWN_PAR;

    return 0;
}

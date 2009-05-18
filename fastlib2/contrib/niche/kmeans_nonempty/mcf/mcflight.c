/**************************************************************************
Contains modul MCFLIGHT.C of ZIB optimizer MCF

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
/*  LAST EDIT: Tue Jun  3 10:49:28 2003 by Andreas Loebel (opt0.zib.de)  */
#ident "$Id: mcflight.c,v 1.1.1.1 2003/06/03 12:06:54 bzfloebe Exp $"


#define PRIMAL


#include "mcf.h"


int main( int argc, char *argv[] )
{
    MCF_network_t net;

    char infile[81];
    char outfile[81];
    
    long stat = 0;
    

    if(    (argc != 2 && argc != 3)
        || (argc == 3 && *(argv[1]+1) != 'e' && *(argv[1]+1) != 'E') )
    {
        printf( "Usage: %s problem-file\n", argv[0] );
        return 0;
    }

    
    infile[0] = outfile[0] = '\0';

    if( !( argc == 3 && (*(argv[1]+1) == 'e' || *(argv[1]+1) == 'E') ) )
    {
        printf( "\nMCF Vers. 1.2.%s light\n", MCF_ARITHMETIC_TYPE );
        printf( "Written    by    Andreas   Loebel,\n" );
        printf( "Copyright (c) 1997-2000 ZIB Berlin\n" );
        printf( "All Rights Reserved.\n" );
        printf( "\n" );
    }

    if( argc == 3 )
    {
        strcpy( infile, argv[2] );
        sprintf( outfile, "%s.sol", argv[2] );
    }
    else
    {
        strcpy( infile, argv[1] );
        sprintf( outfile, "%s.sol", argv[1] );
    }

    memset( (void *)(&net), 0, (size_t)(sizeof(MCF_network_t)) );
    stat = MCF_read_dimacs_min( infile, &net );
    if( stat )
        return -1;


#ifdef PRIMAL

    if( net.m < 10000 )       net.find_bea = MCF_primal_bea_mpp_30_5;
    else if( net.m > 100000 ) net.find_bea = MCF_primal_bea_mpp_200_20;
    else                      net.find_bea = MCF_primal_bea_mpp_50_10;

    MCF_primal_start_artificial( &net );
    MCF_primal_net_simplex( &net );

#elif defined DUAL

    if( net.n < 10000 ) net.find_iminus = MCF_dual_iminus_mpp_30_5;
    else                net.find_iminus = MCF_dual_iminus_mpp_50_10;

    dual_start_artificial( &net );
    dual_net_simplex( &net );

#endif   
        
    
    printf( "\n%s: %ld nodes / %ld arcs\n", infile, (net.n), (net.m) );
    if( net.primal_unbounded )
    {
        printf( "\n   >>> problem primal unbounded <<<\n" );
        return -1;
    }
    
    if( net.dual_unbounded )
    {
        printf( "\n   >>> problem dual unbounded <<<\n" );
        return -1;
    }
    
    if( net.feasible == 0 )
    {
        printf( "\n   >>> problem infeasible or unbounded <<<\n" );
        return -1;
    }
    
    
    printf( "Iterations                        : %ld\n", net.iterations );
    net.optcost = MCF_primal_obj(&net);
    if( MCF_ABS(MCF_dual_obj(&net)-net.optcost)>(double)MCF_ZERO_EPS )
        printf( "NETWORK SIMPLEX: primal-dual objective mismatch!?\n" );
#ifdef MCF_FLOAT
    printf( "Primal optimal objective          : %10.6f\n",
            MCF_primal_obj(&net) );
    printf( "Dual optimal objective            : %10.6f\n", 
            MCF_dual_obj(&net) );
#else
    printf( "Primal optimal objective          : %10.0f\n", 
            MCF_primal_obj(&net) );
    printf( "Dual optimal objective            : %10.0f\n", 
            MCF_dual_obj(&net) );
#endif

    printf( "Write solution to file %s\n", outfile );
    MCF_write_solution( infile, outfile, &net, 0 );

    MCF_free( &net ); 

    return 0;
}


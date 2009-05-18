/**************************************************************************
Contains modul OUTPUT.C of ZIB optimizer MCF

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
/*  LAST EDIT: Tue Jun  3 10:50:52 2003 by Andreas Loebel (opt0.zib.de)  */
#ident "$Id: output.c,v 1.1.1.1 2003/06/03 12:06:54 bzfloebe Exp $"


#include "output.h"


#define RETURN_MSG2( message ) \
    { \
        if( out ) fclose( out ); \
        fprintf( stderr, "write_solution: %s", message ); \
        return( 1 ); \
    }


long MCF_write_solution( 
                   const char *infile,
                   const char *outfile,
                   MCF_network_p net,
                   time_t sec
                   )
{
    FILE *out = NULL;
    MCF_arc_p arc;




    if( !strcmp( infile, outfile ) ) 
    {
        if(( out = fopen( outfile, "at" )) == NULL )
            RETURN_MSG2( "can't open output file\n" );
    }
    else if(( out = fopen( outfile, "wt" )) == NULL )
        RETURN_MSG2( "can't open output file\n" );

    fprintf( out, "c Output to minimum-cost flow problem %s\n", infile );
    fprintf( out, "c The problem was solved with a network simplex " );
    fprintf( out, "code\nc\n" );

    if( net->primal_unbounded )
        fprintf( out, "c Problem is unbounded!\n" );
    else if( net->dual_unbounded )
        fprintf( out, "c Dual problem is unbounded!\n" );
    else if( !(net->feasible) )
        fprintf( out, "c Can't find a feasible flow\n" );
    else
    {
        fprintf( out, "c need %ld iteration(s) in ", net->iterations );
        fprintf( out, "%ld second(s).\n", sec );

        fprintf( out, "s %0.0f\n", net->optcost );

        
        /* Write flow values to output file. */
        for( arc = net->arcs; arc != net->stop_arcs; arc++ )
            if( MCF_ABS(arc->flow) > (MCF_flow_t)MCF_ZERO_EPS )
#ifdef MCF_FLOAT
                fprintf( out, "f %ld %ld %f\n", arc->tail->number,
                        arc->head->number, arc->flow );
#else
                fprintf( out, "f %ld %ld %ld\n", arc->tail->number,
                        arc->head->number, arc->flow );
#endif
    }

    fprintf( out, "c\nc All other variables are zero\n" );
    
    fclose(out);

    return 0;
}

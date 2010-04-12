/**************************************************************************
Contains modul PBEA.C of ZIB optimizer MCF

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
/*  LAST EDIT: Tue Jun  3 10:52:13 2003 by Andreas Loebel (opt0.zib.de)  */
#ident "$Id: pbea.c,v 1.1.1.1 2003/06/03 12:06:54 bzfloebe Exp $"


#include "pbea.h"


MCF_arc_p MCF_primal_bea_all( MCF_DECLARE_PRIMAL_BEA_PARAMETERS )
{
    MCF_arc_p arc;
    MCF_arc_p bea = NULL;
    MCF_cost_t red_cost;
    MCF_cost_t most_neg = MCF_ZERO;
    MCF_cost_t abs_most_neg = MCF_ZERO;

    stop_arcs = NULL; /* to prevent compiler warnings */
    
    for( arc = arcs; m; m--, arc++ )
    {
        if( arc->ident > MCF_BASIC ) 
        {
            MCF_COMPUTE_RED_COST( red_cost ); 
            MCF_CMP_ASSIGN_BEST( red_cost, most_neg, abs_most_neg, bea );
        }
    }
    
    *red_cost_of_bea = most_neg;
    return bea;
}










static long cycle_pos = 0; 

MCF_arc_p MCF_primal_bea_cycle( MCF_DECLARE_PRIMAL_BEA_PARAMETERS )
{
    MCF_arc_p arc, j, stop;        
    MCF_arc_p bea = NULL;     
    MCF_cost_t red_cost;   
    MCF_cost_t most_neg = MCF_ZERO;
    MCF_cost_t abs_most_neg = MCF_ZERO;
    


    m = 0; /* to prevent compiler warnings */

    stop = stop_arcs; 
    /* Search for the first nonbasic arc violating dual condition */
    for( arc = arcs + cycle_pos; arc < stop; arc++ )
        if( arc->ident > MCF_BASIC )
        {
            MCF_COMPUTE_RED_COST( most_neg );
            if( MCF_DUAL_INFEAS( most_neg ) )
            {   
                bea = arc;
                arc = stop;
            }
        }

    /* If no basis entering arc was found within the arc indices 
       { cycle_position, ..., |A| } restart at index 1.
       */
    if( !bea )
        for( arc = arcs, stop = arc + cycle_pos; arc < stop; arc++ )
            if( arc->ident > MCF_BASIC )
            {
                MCF_COMPUTE_RED_COST( most_neg );
                if( MCF_DUAL_INFEAS( most_neg ) )
                {   
                    bea = arc;
                    arc = stop;
                }
            }

    if( !bea )
    {
        /* Reset */
        cycle_pos = 0;
        return NULL; 
    }
    else
        j = bea;

    /* Set cycle_position for next iteration. */
    cycle_pos = bea - arcs;
    cycle_pos++;

    abs_most_neg = MCF_ABS( most_neg );
    MCF_IMPROVE( tail, firstin, nextin );
    MCF_IMPROVE( head, firstout, nextout );
    MCF_IMPROVE( tail, firstout, nextout );
    MCF_IMPROVE( head, firstin, nextin );
       
    *red_cost_of_bea = most_neg;
    return bea;
}

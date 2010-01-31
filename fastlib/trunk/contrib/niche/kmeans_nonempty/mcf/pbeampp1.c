/**************************************************************************
Contains modul PBEAMPP1.C of ZIB optimizer MCF

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
/*  LAST EDIT: Tue Jun  3 10:53:37 2003 by Andreas Loebel (opt0.zib.de)  */
#ident "$Id: pbeampp1.c,v 1.1.1.1 2003/06/03 12:06:54 bzfloebe Exp $"


/**************************************************************************
external parameters
K     basket size
B     old varibles
**************************************************************************/
#define K 30
#define B 5


#include "pbeadef.h"


static void sort_basket( long min, long max );


static long nr_group;
static long group_pos;


typedef struct basket
{
    MCF_arc_p a;
    MCF_cost_t cost;
    MCF_cost_t abs_cost;
} BASKET;


static long basket_size;
static BASKET basket[B+K+1];
static BASKET *perm[B+K+1];

static long initialize = 1;
static MCF_arc_p arc_base = (MCF_arc_p)NULL;
static MCF_arc_p arc_end  = (MCF_arc_p)NULL;



#define CATT(k,b) MCF_reset_mpp_module_ ## k ## _ ## b
#define RESET(x,y) CATT(x,y)
void RESET(K,B)( )
{
    initialize = 1;
}





#define CAT(k,b) MCF_primal_bea_mpp_ ## k ## _ ## b
#define BEAMPP(x,y) CAT(x,y)

MCF_arc_p BEAMPP(K,B)( MCF_DECLARE_PRIMAL_BEA_PARAMETERS )
{
    long i, next, old_group_pos;
    MCF_arc_p arc;
    MCF_cost_t red_cost;

    if( initialize )
    {
        for( i=1; i < K+B+1; i++ )
            perm[i] = &(basket[i]);
        nr_group = ( (m-1) / K ) + 1;
        group_pos = 0;
        basket_size = 0;
        arc_base = arcs;
        arc_end  = stop_arcs;
        initialize = 0;
    }
    else if( arc_base != arcs || arc_end > stop_arcs )
    {
        for( i=1; i < K+B+1; i++ )
            perm[i]->a = (MCF_arc_p)NULL;
        nr_group = ( (m-1) / K ) + 1;
        group_pos = 0;
        basket_size = 0;
    }
    else 
    {
        for( i = 2, next = 0; i <= B && i <= basket_size; i++ )
        {
            arc = perm[i]->a;
            MCF_COMPUTE_RED_COST( red_cost ); 
            if( MCF_DUAL_INFEAS( red_cost ) )
            {
                next++;
                perm[next]->a = arc;
                perm[next]->cost = red_cost;
                perm[next]->abs_cost = MCF_ABS(red_cost);
            }
        }   
        basket_size = next;
        }

    old_group_pos = group_pos;

NEXT:
    /* price next group */
    arc = arcs + group_pos;
    for( ; arc < stop_arcs; arc += nr_group )
    {
        if( arc->ident > MCF_BASIC )
        {
            MCF_COMPUTE_RED_COST( red_cost ); 
            if( MCF_DUAL_INFEAS( red_cost) )
            {
                basket_size++;
                perm[basket_size]->a = arc;
                perm[basket_size]->cost = red_cost;
                perm[basket_size]->abs_cost = MCF_ABS(red_cost);
            }
        }
        
    }

    if( ++group_pos == nr_group )
        group_pos = 0;

    if( basket_size < B && group_pos != old_group_pos )
        goto NEXT;

    if( basket_size == 0 )
    {
        initialize = 1;
        *red_cost_of_bea = 0; 
        return NULL;
    }
    
    sort_basket( 1, basket_size );
    
    *red_cost_of_bea = perm[1]->cost;
    return( perm[1]->a );
}


static void sort_basket( long min, long max )
{
    register long l, r;

    register MCF_cost_t cut;
    register BASKET *xchange;

    l = min; r = max;

    cut = perm[ (long)( (l+r) / 2 ) ]->abs_cost;

    do
    {
        while( perm[l]->abs_cost > cut )
            l++;
        while( cut > perm[r]->abs_cost )
            r--;
            
        if( l < r )
        {
            xchange = perm[l];
            perm[l] = perm[r];
            perm[r] = xchange;
        }
        if( l <= r )
        {
            l++; r--;
        }

    }
    while( l <= r );

    if( min < r )
        sort_basket( min, r );
    if( l < max && l <= B )
        sort_basket( l, max ); 
}

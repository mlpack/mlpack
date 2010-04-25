/**************************************************************************
Contains modul DBLAMPP1.C of ZIB optimizer MCF

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
/*  LAST EDIT: Tue Jun  3 10:44:45 2003 by Andreas Loebel (opt0.zib.de)  */
#ident "$Id: dblampp1.c,v 1.1.1.1 2003/06/03 12:06:54 bzfloebe Exp $"


/**************************************************************************
External definitions
K     basket size
B     old varibles
**************************************************************************/
#define K 30
#define B 5


#include "dbladef.h"


#define CAT(k,b) MCF_dual_iminus_mpp_ ## k ## _ ## b
#define BLAMPP(x,y) CAT(x,y)


static void sort_basket( long min, long max );

static long nr_group;
static long group_pos;


typedef struct
{
    MCF_node_p a;
    MCF_flow_t infeas;
    MCF_flow_t abs_infeas;
} BASKET;


static long basket_size;
static BASKET basket[B+K+1];
static BASKET *perm[B+K+1];

static long initialize = 1;

MCF_node_p BLAMPP(K,B)( MCF_DECLARE_DUAL_BLA_PARAMETERS )
{
    long i, next, old_group_pos;
    MCF_node_p node;
    MCF_arc_p arc;
    MCF_flow_t flow, infeas;






    if( initialize )
    {
        for( i=1; i < K+B+1; i++ )
            perm[i] = &(basket[i]);
        nr_group = ( (n-1) / K ) + 1;
        group_pos = 0;
        basket_size = 0;
        initialize = 0;
    }
    else
    {
        for( i = 2, next = 0; i <= B && i <= basket_size; i++ )
        {
            node = perm[i]->a;
            arc = node->basic_arc;
            flow = node->flow;
#ifdef MCF_LOWER_BOUNDS
            infeas = arc->lower - flow;
#else
            infeas = -flow;
#endif
            if( infeas > (MCF_flow_t)MCF_ZERO_EPS )
            {
                next++;
                perm[next]->a = node;
                perm[next]->infeas = infeas;
                perm[next]->abs_infeas = MCF_ABS(infeas);
            }
            else if( (infeas = (arc->upper - flow)) < (MCF_flow_t)(-MCF_ZERO_EPS) )
            {
                next++;
                perm[next]->a = node;
                perm[next]->infeas = infeas;
                perm[next]->abs_infeas = MCF_ABS(infeas);
            }
        }
    
        basket_size = next;
    }

    old_group_pos = group_pos;

NEXT:
    /* price next group */
    node = nodes + group_pos;
    for( node++; node < stop_nodes; node += nr_group )
    {
        arc = node->basic_arc;
        flow = node->flow;
#ifdef MCF_LOWER_BOUNDS
        infeas = arc->lower - flow;
#else
        infeas = -flow;
#endif
        if( infeas > (MCF_flow_t)MCF_ZERO_EPS )
        {
            basket_size++;
            perm[basket_size]->a = node;
            perm[basket_size]->infeas = infeas;
            perm[basket_size]->abs_infeas = MCF_ABS(infeas);
        }
        else if( (infeas = (arc->upper - flow)) < (MCF_flow_t)(-MCF_ZERO_EPS) )
        {
            basket_size++;
            perm[basket_size]->a = node;
            perm[basket_size]->infeas = infeas;
            perm[basket_size]->abs_infeas = MCF_ABS(infeas);
        }
    }

    if( ++group_pos == nr_group )
        group_pos = 0;

    if( basket_size < B && group_pos != old_group_pos )
        goto NEXT;

    if( basket_size == 0 )
    {
        initialize = 1;
        *delta = 0; 
        return NULL;
    }
    
    sort_basket( 1, basket_size );
    
    *delta = perm[1]->infeas;
    return( perm[1]->a );
}



static void sort_basket( long min, long max )
{
    long l, r;
    MCF_flow_t cut;
    BASKET *xchange;

    l = min; r = max;

    cut = perm[ (long)( (l+r) / 2 ) ]->abs_infeas;

    do
    {
        while( perm[l]->abs_infeas > cut )
            l++;
        while( cut > perm[r]->abs_infeas )
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

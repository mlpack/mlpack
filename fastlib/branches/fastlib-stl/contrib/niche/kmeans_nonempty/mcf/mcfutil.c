/**************************************************************************
Contains modul MCFUTIL.C of ZIB optimizer MCF

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
/*  LAST EDIT: Tue Jun  3 10:50:20 2003 by Andreas Loebel (opt0.zib.de)  */
#ident "$Id: mcfutil.c,v 1.1.1.1 2003/06/03 12:06:54 bzfloebe Exp $"


#undef MCF_DEBUG_NO


#include "mcfutil.h"


double MCF_primal_obj( MCF_network_p net )
{
    MCF_arc_p arc;
    MCF_node_p node;
    void *stop;
    double summ = 0.0;



    stop = net->stop_arcs;
    for( arc = net->arcs; arc != stop; arc++ )
    {
        if( arc->ident == MCF_AT_LOWER || arc->ident == MCF_FIXED )
#ifdef MCF_LOWER_BOUNDS
            arc->flow = arc->lower;
#else
            arc->flow = 0;
#endif
        else if( arc->ident == MCF_AT_UPPER )
            arc->flow = arc->upper;
        else
            arc->flow = (MCF_flow_t)MCF_ZERO;
    }

    stop = net->stop_nodes;
    for( node = net->nodes, node++; node != stop; node++ )
    {
        if( node->basic_arc >= net->dummy_arcs 
           && node->basic_arc < net->stop_dummy )
            summ += (double)(node->flow) * (double)(MCF_MAX_ART_COST);
        node->basic_arc->flow += node->flow;
    }
    

    stop = net->stop_arcs;
    for( arc = net->arcs; arc != stop; arc++ )
        summ += (double)(arc->flow) * (double)(arc->cost);


    return summ;
}









double MCF_dual_obj( MCF_network_p net )
{
    MCF_arc_p arc;
    MCF_node_p node;
    void *stop;
    double summ = 0.0;




    stop = net->stop_nodes;
    for( node = net->nodes + 1; node != stop; node++ )
        summ += (double)(node->potential) * (double)(node->balance);
    stop = net->stop_arcs;
    for( arc = net->arcs; arc != stop; arc++ )
    {
        if( arc->ident == MCF_AT_UPPER )
            summ += (double)(arc->upper) *
                    ((double)(arc->cost) - (double)(arc->tail->potential)
                             + (double)(arc->head->potential));
#ifdef MCF_LOWER_BOUNDS
        else if( arc->ident == MCF_AT_LOWER || arc->ident == MCF_FIXED )
            summ += (double)(arc->lower) *
                    ((double)(arc->cost) - (double)(arc->tail->potential) 
                             + (double)(arc->head->potential));
#endif
    }

    return summ;
}










long MCF_primal_feasible( MCF_network_p net )
{
    void *stop;
    MCF_node_p node;
    MCF_arc_p dummy = net->dummy_arcs;
    MCF_arc_p stop_dummy = net->stop_dummy;
    MCF_arc_p arc;
    MCF_flow_t flow;
    



    node = net->nodes;
    stop = net->stop_nodes;

    for( node++; node < (MCF_node_p )stop; node++ )
    {
        arc = node->basic_arc;
        flow = node->flow;
        if( arc >= dummy && arc < stop_dummy )
        {
            if( MCF_ABS(flow) > (MCF_flow_t)MCF_ZERO_EPS )
            {
                fprintf( stderr,"PRIMAL NETWORK SIMPLEX: " );
                fprintf( stderr,"artificial arc with nonzero flow\n" );
                return 1;
            }
        }
        else
        {
            if( 
#ifdef MCF_LOWER_BOUNDS
               flow - arc->lower < (MCF_flow_t)(-MCF_ZERO_EPS)
#else
               flow < (MCF_flow_t)(-MCF_ZERO_EPS)
#endif
               || flow - arc->upper > (MCF_flow_t)MCF_ZERO_EPS )
            {
                fprintf( stderr,"PRIMAL NETWORK SIMPLEX: " );
                fprintf( stderr,"basis primal infeasible\n" );
                net->feasible = 0;
                return 1;
            }
        }
    }
    
    net->feasible = 1;
    return 0;
}










long MCF_dual_feasible( MCF_network_p net )
{
    MCF_arc_t         *arc;
    MCF_arc_t         *stop     = net->stop_arcs;
    double        red_cost;
    

    for( arc = net->arcs; arc < stop; arc++ )
    {
        red_cost = (double)(arc->cost) - (double)(arc->tail->potential)
            + (double)(arc->head->potential);
        switch( arc->ident )
        {
        case MCF_BASIC:
        case MCF_AT_ZERO:
            if( fabs(red_cost) > 1.0E-6 )
            {
                net->feasible = 0;
#ifdef MCF_DEBUG
                printf("%ld %ld %ld %f\n", arc->tail->number, arc->head->number,
                       arc->ident, red_cost );
#else
                goto DUAL_INFEAS;
#endif
            }
            
            break;
        case MCF_AT_LOWER:
            if( red_cost < -1.0E-6 )
            {
                net->feasible = 0;
#ifdef MCF_DEBUG
                printf("%ld %ld %ld %f\n", arc->tail->number, arc->head->number,
                       arc->ident, red_cost );
#else
                goto DUAL_INFEAS;
#endif
            }

            break;
        case MCF_AT_UPPER:
            if( red_cost > 1.0E-6 )
            {
                net->feasible = 0;
#ifdef MCF_DEBUG
                printf("%ld %ld %ld %f\n", arc->tail->number, arc->head->number,
                       arc->ident, red_cost );
#else
                goto DUAL_INFEAS;
#endif
            }

            break;
        default:
            break;
        }
    }
    
    net->feasible = 1;
    return 0;
    
#ifndef MCF_DEBUG
DUAL_INFEAS:
    fprintf( stderr, "DUAL NETWORK SIMPLEX: " );
    fprintf( stderr, "basis dual infeasible\n" );
    return 1;
#endif
}










long MCF_is_basis( MCF_network_p net )
{
    MCF_node_p stop = net->stop_nodes;
    MCF_node_p node;
    MCF_node_p tmp;
    MCF_node_p root = net->nodes;
    long is = 0;
    





    for( node = root; node < stop; node++ )
        node->mark = 0;
    
        tmp = node = root;
        tmp->mark = 1;   
GO_DOWN: 
        tmp = node->child;
        if( tmp )
        {  
MARK: 
            tmp->mark = 1;   
            node = tmp; 
            goto GO_DOWN;
        } 
RIGHT: 
        if( node == root )
            goto CONTINUE;
        tmp = node->right_sibling;
        if( tmp ) 
            goto MARK; 
        node = node->pred; 
        goto RIGHT;
CONTINUE:
    
    for( node = root; node < stop; node++ )
        if( !node->mark )
            is++;

    return is;
}










long MCF_is_balanced( MCF_network_p net )
{
    MCF_arc_p arc;
    MCF_node_p node;
    void *stop;
    double summ = 0.0;
    long balanced = 0;
    





    stop = net->stop_arcs;
    for( arc = net->arcs; arc != stop; arc++ )
    {
        if( arc->ident == MCF_AT_LOWER || arc->ident == MCF_FIXED )
#ifdef MCF_LOWER_BOUNDS
            arc->flow = arc->lower;
#else
            arc->flow = (MCF_flow_t)0;
#endif
        else if( arc->ident == MCF_AT_UPPER )
            arc->flow = arc->upper;
        else if( arc->ident == MCF_BASIC )
            arc->flow = (MCF_flow_t)MCF_ZERO;
    }

    stop = net->stop_nodes;
    for( node = net->nodes, node++; node != stop; node++ )
        node->basic_arc->flow += node->flow;

    stop = net->stop_nodes;
    for( node = net->nodes; node != stop; node++ )
    {
        summ = (double)(node->balance);

        for( arc = node->firstout; arc; arc = arc->nextout )
            summ -= (double)(arc->flow);
        for( arc = node->firstin; arc; arc = arc->nextin )
            summ += (double)(arc->flow);

        arc = node->basic_arc;
        if( arc >= net->dummy_arcs && arc < net->stop_dummy )
        {
#ifdef MCF_DEBUG
            if( node->flow )
                fprintf( stderr, "WARINING: flow on artificial arc\n" );
#endif
            if( node->orientation == MCF_UP )
                summ -= (double)(node->flow);
            else 
                summ += (double)(node->flow);
        }
        
        
        if( MCF_ABS(summ) > 1.0E-6 )
        {
            balanced = 1;
            fprintf( stderr,"WARNING: node %ld not balanced,",node->number);
            fprintf( stderr,"imbalance = %f\n", summ );
        }
        
    }

    return balanced;
}










long MCF_free( MCF_network_p net )
{  
    MCF_FREE( net->nodes );
    MCF_FREE( net->arcs );
    MCF_FREE( net->dummy_arcs );
    net->nodes = net->stop_nodes = NULL;
    net->arcs = net->stop_arcs = NULL;
    net->dummy_arcs = net->stop_dummy = NULL;

    return 0;
}









#define TICKS_PER_SECONDS sysconf( _SC_CLK_TCK )
/* if this creates a compiling error try
   #define TICKS_PER_SECONDS CLK_TCK
*/

/* TIMES.C illustrates various time and date functions including:
 *      time            _ftime          ctime       asctime
 *      localtime       gmtime          mktime      _tzset
 *      _strtime        _strdate        strftime
 *
 * Also the global variable:
 *      _tzname
 */




double MCF_get_cpu_time( ) /* returns cpu-time in seconds */
{
#ifdef _WIN32
    struct _timeb now;

    _ftime(&now);
    return( (double)now.time + (double)now.millitm / 1000.0);
#else
    struct tms now;

    times( &now );
    return( (double)((double)now.tms_utime / (double)TICKS_PER_SECONDS) );
#endif
}


double MCF_get_system_time( ) /* returns system-time in seconds */
{
#ifdef _WIN32
    return 0.0;
#else
    struct tms now;

    times( &now );
    return( (double)((double)now.tms_stime / (double)TICKS_PER_SECONDS) );
#endif
}


double MCF_get_wall_time( ) /* returns wall-time in seconds */
{
#ifdef _WIN32
    time_t now;
    time(&now);
    return (double) now;
#else
    struct timeval tp;
    struct timezone tzp;
    
    gettimeofday( &tp, &tzp );
    return (double)(tp.tv_sec) + 0.000001 * (double)(tp.tv_usec);
#endif
}

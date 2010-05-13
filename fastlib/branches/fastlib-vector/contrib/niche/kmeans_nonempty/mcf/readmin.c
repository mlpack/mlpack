/**************************************************************************
Contains modul READMIN.C of ZIB optimizer MCF

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
/*  LAST EDIT: Tue Jun  3 10:57:34 2003 by Andreas Loebel (opt0.zib.de)  */
#ident "$Id: readmin.c,v 1.1.1.1 2003/06/03 12:06:54 bzfloebe Exp $"


#include "readmin.h"


#define RETURN_MSG( message ) \
{ \
    if( in ) fclose( in ); \
    MCF_free( net ); \
    fprintf( stderr, "read_dimacs_min: %s\n\n", message ); \
    return( 1 ); \
}






long MCF_read_dimacs_min( const char *filename, MCF_network_p net )
{                                       
    FILE *in = NULL;
    char instring[81];
    char *tmp1, *tmp2;
    char error_msg[81];
    MCF_flow_t balance_summ = MCF_ZERO;
    char ch;
    long t, h, l, u, c;
#ifdef MCF_FLOAT
    MCF_cost_t cc;
#endif
    register long i;
    MCF_arc_p arc;
    MCF_node_p node;
    long up, down;






    strcpy( error_msg, "unknown format in line " );

    if(( in = fopen( filename, "rt")) == NULL )
        RETURN_MSG( "can`t open input file" );

    for( ch = 'a'; ch != 'p' && ch; )
    {
        MCF_GET_NEXT_LINE;
        if( ch != 'p' && ch != 'c' )
        {
            strncpy( &error_msg[23], instring, 70 );
            RETURN_MSG( error_msg );
        }               
    }

    if( ch != 'p' )
        RETURN_MSG( "no problem line found" );

    for( i=1; isspace( (int)instring[i] ); i++ );

    if( sscanf( &(instring[i]), "min %ld %ld", &t, &h ) != 2 
        || t<1 || h<1 )
        RETURN_MSG( "no min cost flow problem" );
        
    /* The basis structure needs one artificial node, called the root. */
    net->n = (long)t; 
    net->m = (long)h; 

    /* Allocate memory for the node- and arc-vectors.
       In every artificial initial start basis structure every node
       will be connected to the root 0 by the artificial arc (0,j) or
       (j,0), therefor there must be additionally allocated memory
       for this |V| artificial arcs.
    */
    net->nodes      = (MCF_node_p) calloc( t+1, sizeof(MCF_node_t) );
    net->arcs       = (MCF_arc_p)  calloc( h,   sizeof(MCF_arc_t) );
    net->dummy_arcs = (MCF_arc_p)  calloc( t,   sizeof(MCF_arc_t) );
    
    net->stop_nodes = net->nodes + (t+1); /* one artificial node! */
    net->stop_arcs  = net->arcs + h;
    net->stop_dummy = net->dummy_arcs + t;

    if( !( net->nodes && net->arcs && net->dummy_arcs ) )
        RETURN_MSG( "not enough memory" );

    for( node = net->nodes, i = 0; i <= t; i++, node++ )
        node->number = i;

    /* reading nodes */
    node = net->nodes;
    MCF_GET_NEXT_LINE;
    while( ch != 'a' && ch )
    {
        if( ch == 'n' )
        {
            if( sscanf( instring,"n %ld %ld",&t,&c) != 2 
                || t<1 || t>net->n )
            {
                strncpy( &error_msg[23], instring, 70 );
                RETURN_MSG( error_msg );
            }    
            node[t].balance = (MCF_flow_t)c;           
            balance_summ += (MCF_flow_t)c;
        }
        else if( ch != 'c' )
        {
            strncpy( &error_msg[23], instring, 70 );
            RETURN_MSG( error_msg );
        }               
        MCF_GET_NEXT_LINE;
    }

    /* Summ of node balances = 0? */
    if( balance_summ > MCF_ZERO_EPS || balance_summ < -MCF_ZERO_EPS )
        RETURN_MSG( "excess/demand not balanced" );

    /* reading arcs */
    arc = net->arcs;;
    for( i = 0; i < net->m && ch; i++, arc++ )
    {
        while( ch == 'c' )
            MCF_GET_NEXT_LINE;
        
        if( ch == 'a' )
        {
            /* Arc descriptors have format 
               "a SRC DST LOW CAP COST".
               LOW and CAP can be ``free''!
             */
            up = down = 0;
            tmp1 = instring;
#ifdef MCF_FLOAT
            switch( sscanf(instring,"a %ld %ld %ld %ld %lf",
                &t, &h, &l, &u, &cc ) )
#else
            switch( sscanf(instring,"a %ld %ld %ld %ld %ld",
                &t, &h, &l, &u, &c ) )
#endif
            {
            case 2:
                if( (tmp2 = (char *)strstr(tmp1, "free")) != NULL)
                {
                    down = 1;
                    tmp1 = tmp2 + 4;
                }
                else if( (tmp2 = (char *)strstr(tmp1, "FREE")) != NULL )
                {
                    down = 1;
                    tmp1 = tmp2 + 4;
                }
                else
                    goto ARC_ERROR;
#ifdef MCF_FLOAT
                switch( sscanf(tmp1,"%ld %lf", &u, &cc) )
#else
                switch( sscanf(tmp1,"%ld %ld", &u, &c) )
#endif
                {
                case 0:
                    goto CASE_3;
                case 2:
                    break;
                default:
                    goto ARC_ERROR;
                }
                goto CONTINUE;
            case 3:
CASE_3:
                if( (tmp2 = (char *)strstr(tmp1, "free")) != NULL )
                {
                    up = 1;
                    tmp1 = tmp2 + 4;
                }
                else if( (tmp2 = (char *)strstr(tmp1, "FREE")) != NULL )
                {
                    up = 1;
                    tmp1 = tmp2 + 4;
                }
                else
                    goto ARC_ERROR;
#ifdef MCF_FLOAT
                switch( sscanf(tmp1,"%lf", &cc) )
#else
                switch( sscanf(tmp1,"%ld", &c) )
#endif
                {
                case 1:
                    break;
                default:
                    goto ARC_ERROR;
                }
            case 5:
CONTINUE:
                if( t>=1 && h>=1 && t<=net->n && h<=net->n )
                    break;
            default:
ARC_ERROR:
                strncpy( &error_msg[23],instring,70 );
                RETURN_MSG( error_msg );
                break;
            }
            
            arc->tail = &(node[t]);
            arc->head = &(node[h]);
#ifdef MCF_FLOAT
            arc->cost = (MCF_cost_t)cc;
#else
            arc->cost = (MCF_cost_t)c;
#endif
            if( up )
                arc->upper = (MCF_flow_t)MCF_UNBOUNDED;
            else
                arc->upper = (MCF_flow_t)u;
            if( down )
            {
#ifdef MCF_LOWER_BOUNDS
                arc->lower = (MCF_flow_t)-MCF_UNBOUNDED;
#else
                strncpy( &error_msg[23], instring, 70 );
                RETURN_MSG( error_msg );
#endif
            }
            else
            {
#ifdef MCF_LOWER_BOUNDS
                arc->lower = (MCF_flow_t)l;
#else
                if( l )
                {
                    strncpy( &error_msg[23], instring, 70 );
                    RETURN_MSG( error_msg );
                }
#endif
            }
            
            arc->nextout = arc->tail->firstout;
            arc->tail->firstout = arc;
            arc->nextin = arc->head->firstin;
            arc->head->firstin =  arc;
        }
        else if( ch != 'c' )
        {
            strncpy( &error_msg[23], instring, 70 );
            RETURN_MSG( error_msg );
        }               

        MCF_GET_NEXT_LINE;
    }
    if( net->stop_arcs != arc )
    {
        net->stop_arcs = arc;
        arc = net->arcs;
        for( net->m = 0; arc < net->stop_arcs; arc++ )
            (net->m)++;
        fprintf( stderr, "read_dimacs_min: Warning: EOF reached, " );
        fprintf( stderr, "have only scanned %ld arcs\n", net->m  );
    }
    
    

    fclose( in );
    return 0;
}

/**************************************************************************
Contains modul MCFDEFS.H of ZIB optimizer MCF

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
/*  LAST EDIT: Tue Jun  3 10:48:47 2003 by Andreas Loebel (opt0.zib.de)  */
#ident "$Id: mcfdefs.h,v 1.1.1.1 2003/06/03 12:06:54 bzfloebe Exp $"



/**@name MCF data structures.

  In the following, we give a description of the variable types and the data
  structures of MCF, which are defined in the file "mcfdefs.h".  For costs and
  flows, it is possible either to use the faster integer arithmetic restricted
  to (4-byte) integers or to use floating point arithmetic with double
  precision.

  \newpage
  For the network simplex algorithm, the input network is assumed to be
  connected, which is ensured by the following simple procedue: Having red a
  problem from file, we add to $V$ one artificial root node, denoted by
  "0". Each original node $i$ of $V$ is then connected to the root node $0$
  either by the artificially generated arc $(i,0)$ if $i$ is a supply or
  transshipment node or by the artificially generated arc $(0,i)$ if $i$ is a
  demand node.

  Node, arc, and network information are stored in the following data
  structures.

  \clearpage */
/*@{*/


#ifndef _DEFINES_H
#define _DEFINES_H


#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <limits.h>
#include <string.h>
#include <errno.h>
#include <ctype.h>
#include <time.h>

#ifndef _WIN32
#include <unistd.h>
#include <sys/time.h>
#include <sys/times.h>
#include <sys/types.h>
#else
#include <sys/timeb.h>
#endif

#include "pbeadef.h"
#include "dbladef.h"



#ifndef MCF_FLOAT
/** Default flow type definition. */
typedef long MCF_flow_t; 
/** Default flow pointer definition. */
typedef long *MCF_flow_p; 
/** Default cost type definition. */
typedef long MCF_cost_t;
/** Default cost pointer type definition. */
typedef long *MCF_cost_p;
#else
/** Flow type definition if MCF\_FLOAT is defined. */
typedef double MCF_flow_t;
/** Flow pointer definition if MCF\_FLOAT is defined. */
typedef double *MCF_flow_p;
/** Cost type definition if MCF\_FLOAT is defined. */
typedef double MCF_cost_t;
/** Cost pointer definition if MCF\_FLOAT is defined. */
typedef double *MCF_cost_p;
#endif

/** Node type definition. */
typedef struct MCF_node MCF_node_t;
/** Node pointer definition. */
typedef struct MCF_node *MCF_node_p;

/** Arc type definition. */
typedef struct MCF_arc MCF_arc_t;
/** Arc pointer definition. */
typedef struct MCF_arc *MCF_arc_p;

/** Network type definition. */
typedef struct MCF_network MCF_network_t;
/** Network pointer definition. */
typedef struct MCF_network *MCF_network_p;




/** Node description. 

  Let $T \subseteq A$ be a spanning tree in $D$, and consider some node $v \in V
  \setminus \{0\}$. There is an unique (undirected) path, denoted by $P(v)$,
  defined by $T$ from $v$ to the root node $0$. The arc in $P(v)$, which is
  incident to $v$, is called the {\bf basic arc} of $v$. The other terminal node
  $u$ of this basic arc is called the {\bf predecessor} ({\bf node}) of $v$. The
  basic arc of $v$ is called {\bf upward} ({\bf downward}) oriented if $v$ is
  the tail (head) node of its basic arc.  If $v$ is the predecessor of some
  other node $u$, we call $u$ a {\bf child} ({\bf node}) of $v$.  Given some
  order of all childs of $v$, and let $u$ and $w$ be two different childs of
  $v$. If $u$ is smaller than $w$ with respect to the given order, we call $u$
  the {\bf left sibling} of $w$ and $w$ the {\bf right sibling} of $u$.  If
  there is no child $u$ being smaller (greater) than a given child $w$, then $w$
  has no left (right) sibling.  Each node has at most one child reference, the
  other children of a node can be reached by traversing the sibling
  links. The number of nodes in $P(V)$ is called the {\bf subtree size} of $v$.

  The subtree size and predecessor variables are used by the ratio test.  The
  orientation, child, and sibling variables are used for the computation of the
  node potentials.  Figure \ref{fig-basis} shows a small example of a rooted
  basis tree for our data structures (the underlying network is a copy from
  \cite{AhujaMagnantiOrlin93}).


  \begin{figure}[!h]
  \begin{center}
  \epsfig{file=basis-13jun96.eps,width=.75\textwidth}
  \vskip 2\baselineskip
  \begin{tabular}{|l|c|c|c|c|c|c|c|c|c|}
  \hline
  node           & 0   &  1   &  2   & 3   &  4   &  5   &  6   & 7   & 8   \\
  \hline                 
  subtree size   & 9   &  8   &  5   & 2   &  1   &  1   &  1   & 2   & 1   \\
  predecessor    & nil &  0   &  1   & 2   &  3   &  3   &  2   & 1   & 7   \\
  child          & 1   &  2   &  3   & 4   &  nil &  nil &  nil & 8   & nil \\
  right sibling  & nil &  nil &  7   & 6   &  5   &  nil &  nil & nil & nil \\
  left sibling   & nil &  nil &  nil & nil &  nil &  4   &  3   & 2   & nil \\
  orientation    & -   & down & down &  up & down & down & down & up  & up  \\
  \hline
  \end{tabular}
  \end{center}
  \caption{Rooted basis tree.}
  \label{fig-basis}
  \end{figure}

  \clearpage */
struct MCF_node
{
    /** Node identifier.
     *
     * This variable is only used to assign some identification to each node.
     * Typically, as for the DIMACS format, nodes are indexed from 1 to n, where
     * n denotes the number of nodes.  */
    long number;
    

    /** predecessor node.
     *
     */
    MCF_node_p pred;     
    

    /** First child node.
     *
     */
    MCF_node_p child;     
    

    /** Next child of predecessor.
     *
     */
    MCF_node_p right_sibling;     
    

    /** Previous child of predecessor.
     *
     */
    MCF_node_p left_sibling;     
    

    /** Number of nodes (including this one) up to the root node.
     *
     */
    long subtreesize; 
    

    /** The node's basic arc.
     *
     */
    MCF_arc_p basic_arc; 
    

    /** Orientation of the node's basic arc.
     *
     * This variable stands for the orientation of the node's basic arc.  The
     * value UP (= 1) means that the arc points to the father, and the
     * value DOWN (= 0) means that the arc points from the father to this node.
     *
     */
    long orientation; 
    

    /** First arc of the neighbour list of arcs leaving this node.
     *
     */
    MCF_arc_p firstout;
    

    /** First arc of the neighbour list of arcs entering this node.
     *
     */
    MCF_arc_p firstin;
    

    /** Supply/Demand $b_i$ of this node.
     *
     * A node $i$ is called a supply node, a demand node, or a transshipment
     * node depending upon whether $b_i$ is larger than, smaller than, or equal
     * to zero, respectively.
     * */
    MCF_flow_t balance;  
    

    /** Dual node multipliers.
     *
     * This variable stands for the node potential corresponding with the flow
     * conservation constrait of this node.
     *
     */
    MCF_cost_t potential; 
    

    /** Flow value of the node's basic arc.
     *
     */
    MCF_flow_t flow;
    

    /** Temporary variable.
     *
     * This is a temporary variable, which you can use as you like.
     *
     */
    long mark;
};







/** Arc description. */
struct MCF_arc
{
    /** Tail node.
     *
     */
    MCF_node_p tail;


    /** Head node.
     *
     */
    MCF_node_p head;


    /** Next arc of the neighbour list of arcs leaving the tail node.
     *
     */
    MCF_arc_p nextout;


    /** Next arc of the neighbour list of arcs entering the head node.
     *
     */
    MCF_arc_p nextin;


    /** Arc costs.
     *
     * This variable stands for the arc cost (or weight).
     *
     * Our primal feasible starting basis consists just of artificial arcs
     * (corresponding to a slack basis), and all originally defined arcs are
     * first nonbasic at their lower bounds. The costs of the artificial arcs
     * are set to MAX\_ART\_COST, which is defined in the file mcfdefs.h. It is
     * easy to see that any feasible and optimal solution with artificial arcs
     * is also optimal and feasible for the original problem without artificials
     * iff no artifical arc yields a nonzero flow value. If, however, a solution
     * contains an artificial arc with positive flow, the original problem is
     * either indeed infeasible or the MAX\_ART\_COST is just too small compared
     * to the cost coefficients of the original arcs. If the latter is the case,
     * increase MAX\_ART\_COST, but we also strongly recommend to use then
     * floating point arithmetic!
     *  */
    MCF_cost_t cost;


    /** Arc upper bound.
     *
     * This variable stands for the arc upper bound value. Note that an
     * unbounded upper bound is set to UNBOUNDED, which is defined in the file
     * mcfdefs.h. Per default, UNBOUNDED is set to $10^9$.  Note, this value may
     * be too small for your purposes, and you should increase it appropriately.
     * However, we strongly recommend to use then floating point arithmetic
     * (define MCF\_FLOAT)!
     *
     */
    MCF_flow_t upper;


    /** Arc lower bound.
     *
     * This variable stands for the arc lower bound value. Note, this variable
     * is only active if the MCF\_LOWER\_BOUNDS variable is defined! An negative
     * unbounded lower bound is set to -UNBOUNDED, see also the arc upper bound.
     *
     */
#ifdef MCF_LOWER_BOUNDS
    MCF_flow_t lower;
#endif


    /** Arc flow value.
     *
     * This variable stands for the arc's flow value. Note that the flow value
     * is not set within the main (primal or dual) iteration loop; actually, it
     * can only be computed using the function primal\_obj().
     *
     */
    MCF_flow_t flow;


    /** Arc status.
     *
     * This variable shows the current arc status. Feasible is BASIC
     * (for basic arcs), MCF\_AT\_LOWER\_BOUND (nonbasic arcs set to lower 
     * bound), MCF\_AT\_UPPER\_BOUND (nonbasic arcs set to the upper bound), 
     * MCF\_AT\_ZERO (nonbasis arcs set to zero), or FIXED (arcs fixed to 
     * zero and being not considered by the optimization).
     *
     */
    long ident;
};








/** Network description. */
struct MCF_network
{
    /** Number of nodes.
     *
     * This variable stands for the number of originally defined nodes without
     * the artificial root node.
     * 
     */
    long n;


    /** Number of arcs.
     *
     * This variable stands for the number of arcs (without the artificial slack
     * arcs).
     * 
     */
    long m;


    /** Primal unbounded indicator.
     *
     * This variable is set to one iff the problem is determined to be primal
     * unbounded.
     * 
*/
    long primal_unbounded;


    /** Dual unbounded indiciator.
     *
     * This variable is set to one iff the problem is determined to be dual
     * unbounded.
     * 
     */
    long dual_unbounded;


    /** Feasible indicator.
     *
     * This variable is set to zero if the problem provides a feasible solution.
     * It can only be set by the function primal\_feasible() or dual\_feasible()
     * and is not set automatically by the optimization.
     * 
     */
    long feasible;


    /** Costs of current basis solution.
     *
     * This variable stands for the costs of the current (primal or dual) basis
     * solution. It is set by the return value of primal\_obj() or dual\_obj().
     * 
     */
    double optcost;  


    /** Vector of nodes.
     *
     * This variable points to the $n+1$ node structs (including the root node)
     * where the first node is indexed by zero and represents the artificial
     * root node.
     * 
     */
    MCF_node_p nodes;


    /** First infeasible node address.
     *
     * This variable is the address of the first infeasible node address,
     * i.\,e., it must be set to $nodes + n + 1$. 
     *
     */
    MCF_node_p stop_nodes;


    /** Vector of arcs.
     *
     * This variable points to the $m$ arc structs.
     *
     */
    MCF_arc_p arcs;


    /** First infeasible arc address.
     *
     * This variable is the address of the first infeasible arc address, i.\,e.,
     * it must be set to $nodes + m$. 
     *
     */
    MCF_arc_p stop_arcs;


    /** Vector of artificial slack arcs.
     *
     * This variable points to the artificial slack (or dummy) arc variables and
     * contains $n$ arc structs.  The artificial arcs are used to build (primal)
     * feasible starting bases. For each node $i$, there is exactly one dummy
     * arc defined to connect the node $i$ with the root node.
     * */
    MCF_arc_p dummy_arcs;


    /** First infeasible slack arc address.
     *
     * This variable is the address of the first infeasible slack arc address,
     * i.\,e., it must be set to $nodes + n$. 
     *
     */
    MCF_arc_p stop_dummy; 


    /** Iteration count.
     *
     * This variable contains the number of main simplex iterations performed to
     * solve the problem to optimality.
     * 
     */
    long iterations;


    /** Dual pricing rule.
     *
     * Pointer to the dual pricing rule function that is used by the dual
     * simplex code.
     * */
    MCF_node_p (*find_iminus) ( long n, MCF_node_p nodes, 
                                MCF_node_p stop_nodes, MCF_flow_p delta );


    /** Primal pricing rule.
     *
     * Pointer to the primal pricing rule function that is used by the primal
     * simplex code.
     * */
    MCF_arc_p (*find_bea) ( long m, MCF_arc_p arcs, MCF_arc_p stop_arcs, 
                            MCF_cost_p red_cost_of_bea );
};




/*@}*/


#ifndef NULL
#define NULL 0
#endif


#ifdef MCF_FLOAT

#define MCF_ARITHMETIC_TYPE "F"

#define MCF_UNBOUNDED        (double)1000000000.0
#define MCF_MAX_ART_COST     (double)1.0E8

#define MCF_ZERO             (double)0
#define MCF_ZERO_EPS         (double)1.0E-6

#else

#define MCF_ARITHMETIC_TYPE "I"

#define MCF_UNBOUNDED        1000000000
#define MCF_MAX_ART_COST     (long)(100000000L)

#define MCF_ZERO             0
#define MCF_ZERO_EPS         0

#endif


#define MCF_FIXED            -1
#define MCF_BASIC            0
#define MCF_AT_LOWER         1
#define MCF_AT_UPPER         2
#define MCF_AT_ZERO          3


#define MCF_UP    1
#define MCF_DOWN  0




#define MCF_GET_NEXT_LINE \
{ \
    if( !fgets( instring, 81, in ) ) \
        ch = 0; \
    else \
        ch = *instring; \
}



#ifdef MCF_FLOAT
#define MCF_ABS( x ) fabs( x )
#else
#define MCF_ABS( x ) ( ((x) >= 0) ? ( x ) : -( x ) )
#endif



#ifndef MCF_SET_ZERO
#define MCF_SET_ZERO( vec, n ) if( vec ) memset( (void *)(vec), 0, (size_t)(n) )
#endif



#ifndef MCF_FREE
#define MCF_FREE( vec ) \
{ \
    if( vec ) \
    { \
        free( (void *)vec ); \
        vec = NULL; \
    } \
}
#endif


#endif /* _DEFINES_H */

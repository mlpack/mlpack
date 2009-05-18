/**************************************************************************
Contains modul MCF.H of ZIB optimizer MCF

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
/*  LAST EDIT: Tue Jun  3 10:48:17 2003 by Andreas Loebel (opt0.zib.de)  */
#ident "$Id: mcf.h,v 1.1.1.1 2003/06/03 12:06:54 bzfloebe Exp $"


#ifndef _MCF_H
#define _MCF_H


/**@name MCF interface. */
/*@{*/


#include "mcfdefs.h"


/**@name Problem reading and writing. */
/*@{*/
//@Include: reading.dxx
extern long MCF_read_dimacs_min( char *filename, MCF_network_p net );

//@Include: writing.dxx
extern long MCF_write_solution( char *infile, char *outfile,
                               MCF_network_p net, time_t sec );
/*@}*/



/**@name Primal network simplex. */
/*@{*/

/**@name Slack basis. */
/*@{*/
//@Include: pstart.dxx
extern long MCF_primal_start_artificial ( MCF_network_p net );
/*@}*/

/**@name Main iteration loop. */
/*@{*/
//@Include: psimplex.dxx
extern long MCF_primal_net_simplex ( MCF_network_p net );
/*@}*/

//@Include: pprice.dxx
/*@{*/
//@Include: pbeampp.dxx
/*@{*/
/** K = 30, B = 5. */
extern MCF_arc_p MCF_primal_bea_mpp_30_5   
    ( long m, MCF_arc_p arcs, MCF_arc_p stop_arcs, MCF_cost_p red_cost_of_bea );

/** K = 50, B = 10. */
extern MCF_arc_p MCF_primal_bea_mpp_50_10  
    ( long m, MCF_arc_p arcs, MCF_arc_p stop_arcs, MCF_cost_p red_cost_of_bea );

/** K = 200, B = 20. */
extern MCF_arc_p MCF_primal_bea_mpp_200_20 
    ( long m, MCF_arc_p arcs, MCF_arc_p stop_arcs, MCF_cost_p red_cost_of_bea );

extern MCF_arc_p MCF_primal_bea_mpp_70_15
    ( long m, MCF_arc_p arcs, MCF_arc_p stop_arcs, MCF_cost_p red_cost_of_bea );
/*@}*/

/**@name First eligible arc rule. */
/*@{*/
/** First eligible arc pricing.
 * Searches for the basis entering arc in a wraparound fashion.
 */
extern MCF_arc_p MCF_primal_bea_cycle      
    ( long m, MCF_arc_p arcs, MCF_arc_p stop_arcs, MCF_cost_p red_cost_of_bea );
/*@}*/

/**@name Dantzig's rule. */
/*@{*/
/** Dantzig's rule.
 * Determins the arc violating the optimality condition at most.
 */
extern MCF_arc_p MCF_primal_bea_all        
    ( long m, MCF_arc_p arcs, MCF_arc_p stop_arcs, MCF_cost_p red_cost_of_bea );

/*@}*/
/*@}*/
/*@}*/



/**@name Dual network simplex. */
/*@{*/

/**@name Start basis. */
/*@{*/
//@Include: dstart.dxx
extern long MCF_dual_start_artificial ( MCF_network_p net );
/*@}*/

/**@name Main iteration loop. */
/*@{*/
//@Include: dsimplex.dxx
extern long MCF_dual_net_simplex ( MCF_network_p net );
/*@}*/

//@Include: dprice.dxx
/*@{*/
/**@name Multiple partial pricing */
/*@{*/
/** K = 30, B = 5. */
extern MCF_node_p MCF_dual_iminus_mpp_30_5  
          ( long n, MCF_node_p nodes, MCF_node_p stop_nodes, MCF_flow_p delta );

/** K = 50, B = 10. */
extern MCF_node_p MCF_dual_iminus_mpp_50_10
          ( long n, MCF_node_p nodes, MCF_node_p stop_nodes, MCF_flow_p delta );
/*@}*/

/**@name First eligible arc rule. */
/*@{*/
/** First eligible arc pricing.
 * Searches for the basis leaving arc in a wraparound fashion.
 */
extern MCF_node_p MCF_dual_iminus_cycle
          ( long n, MCF_node_p nodes, MCF_node_p stop_nodes, MCF_flow_p delta );
/*@}*/
/*@}*/
/*@}*/



/**@name MCF utilities. */
/*@{*/
/** Frees malloced data structures.
 *
 * @param  net  reference to network data structure.
 */ 
extern long MCF_free ( MCF_network_p net );

/** Primal objective $c^\trans\,x$.
 *
 * @param  net  reference to network data structure.
 */
extern double MCF_primal_obj   ( MCF_network_p net );

/** Dual objective $\pi^\trans b + \lambda^\trans l - \eta^\trans u$.
 *
 * @param  net  reference to network data structure.
 */
extern double MCF_dual_obj ( MCF_network_p net );

/** Primal basis checking.
 *
 * Checks whether a given basis is primal feasible. 
 *
 * @param  net  reference to network data structure.
 * @return value <> 0 indicates primal infeasible basis.
 */
extern long MCF_primal_feasible ( MCF_network_p net );

/** Dual basis checking.
 *
 * Checks whether a given basis is dual feasible. 
 *
 * @param  net  reference to network data structure.
 * @return value <> 0 indicates dual infeasible basis.
 */
extern long MCF_dual_feasible ( MCF_network_p net );

/** Basis checking.
 *
 * Checks whether the given basis structure is a spanning tree.
 *
 * @param  net  reference to network data structure.
 * @return value <> 0 indicates infeasible spanning tree.
 */
extern long MCF_is_basis ( MCF_network_p net );

/** Flow vector checking.
 *
 * Checks whether a given basis solution defines a balanced flow on each node.
 *
 * @param  net  reference to network data structure.
 * @return value <> 0 indicates unbalanced flow vector. 
 */
extern long MCF_is_balanced ( MCF_network_p net );
/*@}*/
/*@}*/

#endif






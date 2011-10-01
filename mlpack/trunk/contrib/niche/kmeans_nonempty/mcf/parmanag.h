/**************************************************************************
Contains modul PARMANAG.H of ZIB optimizer MCF

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
/*  LAST EDIT: Tue Jun  3 10:51:42 2003 by Andreas Loebel (opt0.zib.de)  */
#ident "$Id: parmanag.h,v 1.1.1.1 2003/06/03 12:06:54 bzfloebe Exp $"
     

#ifndef _PARMANAG_H
#define _PARMANAG_H


#include "mcfdefs.h"
#include "pbeadef.h"
#include "dbladef.h"
#include "pstart.h"
#include "dstart.h"
#include "psimplex.h"
#include "dsimplex.h"
#include "pbea.h"
#include "dbla.h"


extern long MCF_par_manager( 
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
                );


#endif

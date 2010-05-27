/**************************************************************************
Contains modul MCFUTIL.H of ZIB optimizer MCF

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
/*  LAST EDIT: Tue Jun  3 10:50:37 2003 by Andreas Loebel (opt0.zib.de)  */
#ident "$Id: mcfutil.h,v 1.1.1.1 2003/06/03 12:06:54 bzfloebe Exp $"


#ifndef _MCFUTIL_H
#define _MCFUTIL_H


#include "mcfdefs.h"


extern double MCF_primal_obj   ( MCF_network_p net );
extern double MCF_dual_obj     ( MCF_network_p net );
extern long MCF_primal_feasible( MCF_network_p net );
extern long MCF_dual_feasible  ( MCF_network_p net );
extern long MCF_is_basis       ( MCF_network_p net );
extern long MCF_is_balanced    ( MCF_network_p net );
extern long MCF_free           ( MCF_network_p net );

extern double MCF_get_cpu_time   ( void );
extern double MCF_get_system_time( void );
extern double MCF_get_wall_time  ( void );


#endif

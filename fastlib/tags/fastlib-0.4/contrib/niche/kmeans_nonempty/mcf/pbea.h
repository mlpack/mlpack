/**************************************************************************
Contains modul PBEA.H of ZIB optimizer MCF

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
/*  LAST EDIT: Tue Jun  3 10:52:30 2003 by Andreas Loebel (opt0.zib.de)  */
#ident "$Id: pbea.h,v 1.1.1.1 2003/06/03 12:06:54 bzfloebe Exp $"


#ifndef _PBEA_H
#define _PBEA_H


#include "mcfdefs.h"
#include "pbeadef.h"


extern MCF_arc_p MCF_primal_bea_all( MCF_DECLARE_PRIMAL_BEA_PARAMETERS );
extern MCF_arc_p MCF_primal_bea_cycle( MCF_DECLARE_PRIMAL_BEA_PARAMETERS );
extern void MCF_reset_mpp_module_30_5( void );
extern MCF_arc_p MCF_primal_bea_mpp_30_5( MCF_DECLARE_PRIMAL_BEA_PARAMETERS );
extern void MCF_reset_mpp_module_50_10( void );
extern MCF_arc_p MCF_primal_bea_mpp_50_10( MCF_DECLARE_PRIMAL_BEA_PARAMETERS );
extern void MCF_reset_mpp_module_200_20( void );
extern MCF_arc_p MCF_primal_bea_mpp_200_20( MCF_DECLARE_PRIMAL_BEA_PARAMETERS );
extern void MCF_reset_mpp_module_100_10( void );
extern MCF_arc_p MCF_primal_bea_mpp_100_10( MCF_DECLARE_PRIMAL_BEA_PARAMETERS );


#endif

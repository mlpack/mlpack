#ifndef KDTREE_H
#define KDTREE_H

#include "mrkd.h"

void mrkd_slow_nearest(dyv *metric,ivec *rows,dym *x,dyv *q,
					   int *r_row,double *r_dsqd,int not_me_row,ivec *visited);
void mrkd_knode_nearest(dyv *metric,knode *kn,dym *x,dyv *q,
						int *r_row,double *r_dsqd,double kn_q_dsqd,
						int not_me_row,ivec *visited);
int mrkd_nearest_neighbor(mrkd *mr,dym *x,dyv *q,int not_me_row,
						  ivec **r_visited_rows);

dym *mk_dym_from_filename_improved(char *filename,int argc,char *argv[]);

mrkd *mk_mrkd_from_args(int argc,char *argv[],dym **r_data);
void nn_kd_main(int argc,char *argv[]);

#endif /* #ifndef KDTREE_H */

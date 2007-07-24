double run_findbw(dym *q,dym *d,bwinfo *bws)
{
  int i; bool slow=FALSE, turn=FALSE;
  double score_e=0, score_u=0, score_l=0, score_err=0;
  double best_score=0, best_bw = -1, crit_bw = -1, bw, factor = 10;
  double max_bw = bws->bwmax, min_bw = bws->bwmin;
  dyv *the_bws = mk_dyv(0), *the_scores = mk_dyv(0);
  double guess_bw = bandwidth_guess(d); 

  fprintf(LOG,"\nFinding the optimum bandwidth.\n\n");

  if (BWSTART_VALUE) {
    if (max_bw > 0.0) 
      fprintf(LOG,"Using %g, user-specified value, as upper bound.\n",max_bw);
    else { fprintf(LOG,"No start value was specified\n"); 
    Bwstart = Oversmooth; }
  } 
  if (BWSTART_SUBSAMPLE) {
    int subsample_size = 10000;
    if (subsample_size < dym_rows(d)) {
      dym *subsample = mk_random_dym_subset(d, subsample_size);
      TREE Qtree_save = Qtree;
      Bwstart = Oversmooth;
      fprintf(LOG,"Using subsample of size %d.\n",subsample_size);
      make_trees(subsample, subsample, bws);
      max_bw = run_findbw(subsample,subsample,bws);
      Bwstart = Subsample;
      if (max_bw < guess_bw) 
        fprintf(LOG,"Using %g, from subsample, as upper bound.\n",max_bw);
      else Bwstart = Oversmooth;
      free_dym(subsample);
#ifdef USE_KD_TREE
      free_mrkd(Qtree);
#else
      free_batree(Qtree);
#endif
      Qtree = Qtree_save; Dtree = Qtree_save;
    }
    else { fprintf(LOG,"Not subsampling: tiny data\n"); Bwstart = Oversmooth; }
  }
  if (BWSTART_OVERSMOOTH) {
    max_bw = guess_bw;
    fprintf(LOG,"Using %g, theoretical guess, as upper bound.\n",max_bw);
  }

  if (min_bw == -1) { min_bw = .00001 * max_bw; }
  fprintf(LOG,"Using %g, 5 orders below max_bw, as lower bound.\n", min_bw); 
  
  if (LK_CV) best_score = -FLT_MAX; else if (LS_CV) best_score = FLT_MAX; 
  bw = max_bw; i = 0;
  while (bw >= min_bw) {
  
    { /* do this bandwidth */
      dym *l0 = mk_zero_dym(dym_rows(q),1); dyv *ll_l0 = mk_dyv(1);
      dym *e0 = mk_zero_dym(dym_rows(q),1); dyv *ll_e0 = mk_dyv(1);
      dym *u0 = mk_zero_dym(dym_rows(q),1); dyv *ll_u0 = mk_dyv(1);
      bwinfo *bws0 = mk_bwinfo(bw,bw,1); mk_set_bwinfo(bws0,d); 

      Qtree->bws = Dtree->bws = bws0;
      run_dualtree_kde(q,d,NULL,l0,e0,u0,ll_l0,ll_e0,ll_u0,bws0);

      if (LS_CV) { dyv *lscv_l0, *lscv_e0, *lscv_u0;
                   lscv_l0 = mk_lscv(l0,bws0); score_l = dyv_ref(lscv_l0,0);  
                   lscv_e0 = mk_lscv(e0,bws0); score_e = dyv_ref(lscv_e0,0); 
                   lscv_u0 = mk_lscv(u0,bws0); score_u = dyv_ref(lscv_u0,0); 
                   free_dyv(lscv_l0); free_dyv(lscv_e0); free_dyv(lscv_u0); }
      if (LK_CV) { score_l = dyv_ref(ll_l0,0); score_e = dyv_ref(ll_e0,0); 
                   score_u = dyv_ref(ll_u0,0); }
      free_dym(l0); free_dym(e0); free_dym(u0); 
      free_dyv(ll_l0); free_dyv(ll_e0); free_dyv(ll_u0); free_bwinfo(bws0);
    }
    add_to_dyv(the_bws, bw); add_to_dyv(the_scores, score_e);
    score_err = fabs((score_u-score_l)/score_l);
    fprintf(LOG,"  Score (datum-wise midpt): %g  Range: [%g, %g]\n", 
            score_e, score_l, score_u);
    fprintf(LOG,"  Maximum score error: %g\n", score_err);
  
    /* did we stop getting better? */
    if (LK_CV) turn = (score_e < best_score);
    if (LS_CV) turn = (score_e > best_score);
    if (turn) {
      if (slow==FALSE) { crit_bw = bw; slow = TRUE; 
                         if (LK_CV) best_score = -FLT_MAX; 
                         if (LS_CV) best_score = FLT_MAX; }
      else break;
    }
    else { best_score = score_e; best_bw = bw; }
  
    if (slow==TRUE) {
      dym *l0 = mk_zero_dym(dym_rows(q),10); dyv *ll_l0 = mk_dyv(10);
      dym *e0 = mk_zero_dym(dym_rows(q),10); dyv *ll_e0 = mk_dyv(10);
      dym *u0 = mk_zero_dym(dym_rows(q),10); dyv *ll_u0 = mk_dyv(10);
      dyv *lscv_l0=0, *lscv_e0=0, *lscv_u0=0;
      bwinfo *bws0 = mk_bwinfo(crit_bw * 1.1, real_min(crit_bw * 90, 
                               real_max(guess_bw, crit_bw * 1.1)), 10); 
      LOGSCALE = TRUE; mk_set_bwinfo(bws0,d);

      Qtree->bws = bws0; Dtree->bws = bws0;
      run_dualtree_kde(q,d,NULL,l0,e0,u0,ll_l0,ll_e0,ll_u0,bws0);

      if (LS_CV) { lscv_l0 = mk_lscv(l0,bws0); lscv_e0 = mk_lscv(e0,bws0); 
                   lscv_u0 = mk_lscv(u0,bws0); }
      for (i=0; i<10; i++) {
        bw = sqrt(dyv_ref((bws0->bwsqds).bwsqds_dyv,i));
        if (LS_CV) { score_l=dyv_ref(lscv_l0,i);score_e = dyv_ref(lscv_e0,i); 
                     score_u = dyv_ref(lscv_u0,i); }
        if (LK_CV) { score_l = dyv_ref(ll_l0,i); score_e = dyv_ref(ll_e0,i); 
                     score_u = dyv_ref(ll_u0,i); }
        add_to_dyv(the_bws, bw); add_to_dyv(the_scores, score_e);
        score_err = fabs((score_u-score_l)/score_l);
        fprintf(LOG,"  Score (datum-wise midpt): %g  Range: [%g, %g]\n", 
                score_e, score_l, score_u);
        fprintf(LOG,"  Maximum score error: %g\n", score_err);
      }
      if (LS_CV) { best_bw = dyv_ref(the_bws,dyv_argmin(the_scores)); }
      if (LK_CV) { best_bw = dyv_ref(the_bws,dyv_argmax(the_scores)); }

      if (LS_CV) { free_dyv(lscv_l0); free_dyv(lscv_e0); free_dyv(lscv_u0); }
      free_dym(l0); free_dym(e0); free_dym(u0); 
      free_dyv(ll_l0); free_dyv(ll_e0); free_dyv(ll_u0); free_bwinfo(bws0);
      break;
    }
    else bw = bw/factor;
  }
  if (crit_bw == -1) { // no critical bw was found
    fprintf(LOG,"\nPossible breakdown of leave-one-out cross-validation,\n");
    fprintf(LOG,"since no concavity was detected in the scores.\n");
    fprintf(LOG,"Using theoretical bandwidth guess.\n");
    best_bw = bandwidth_guess(d)/2.0; 
  }
  fprintf(LOG,"\n==> Optimum bandwidth is: %g\n\n",best_bw);
  fprintf(LOG,"Summary:\n");
  for (i=0; i<dyv_size(the_bws); i++)
    fprintf(LOG,"%g %g\n",dyv_ref(the_bws,i),dyv_ref(the_scores,i));
  fprintf(LOG,"\n");
  
  free_dyv(the_bws); free_dyv(the_scores);
  return best_bw;
}


cz = 11;

argvals = 1/240:1/240:1;

mybasis = create_bspline_basis([1/240 1], 30, 4);
basis_curves = eval_basis(argvals, mybasis);
basis_inner_products = full(eval_penalty(mybasis, int2Lfd(0)));

load /home/niche/neurofunk/bci_comp_III_Wads_2004/BCI_Comp_III_Wads_2004/responses.mat

myfdPar = fdPar(mybasis, 2, 0);

myfd = data2fd(responses, argvals, mybasis);

basis_curves = eval_basis(argvals, mybasis);
data_curves = basis_curves * getcoef(myfd);


myfdPar = fdPar(mybasis, 2, 1e-7);
pca_results = pca_fd(myfd, 30, myfdPar);


[ic_curves_pos, ic_coef_pos, Y_pos, h_Y_pos, pc_coef, pc_curves, pc_scores, mean_coef, W, whitening_transform] = ...
    funcica(argvals, 0, myfd, 30, basis_curves, myfdPar, ...
	    basis_inner_products);

pc_curves = basis_curves * pc_coef;
ic_curves = basis_curves * ic_coef_pos;


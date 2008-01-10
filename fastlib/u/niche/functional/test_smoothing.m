p = 30;
mybasis = create_bspline_basis([0 1], p, 4);
basis_curves = eval_basis(t, mybasis);
myfd_data = data2fd(data, t, mybasis);

lambda_set = zeros(1,5);
h_Y_set = zeros(2,5);

for j = 1
  switch(j)
   case 1,
    lambda = 0;
   case 2,
    lambda = 1e-4;
   case 3,
    lambda = 1e-3;
   case 4,
    lambda = 5e-3;
   case 5,
    lambda = 1e-2;
  end

  [ic_curves_pos, ic_coef_pos, Y_pos, h_Y_pos, pc_coef, pc_curves, mean_coef, ...
   W, whitening_transform] = funcica(t, s, myfd_data, p, basis_curves, ...
				     fdPar(mybasis, 2, lambda));
  
  figure(j);
  subplot(2, 2, 1); plot(ic_curves_pos(:,1));
  subplot(2, 2, 2); hist(Y_pos(1,:),  50);
  subplot(2, 2, 3); plot(ic_curves_pos(:,2));
  subplot(2, 2, 4); hist(Y_pos(2,:),  50);
  
  
  lambda_set(j) = lambda;
  h_Y_set(:,j) = h_Y_pos;
end

lambda_set
sum(h_Y_set)
h_Y_set


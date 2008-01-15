function [lambda_set, h_Y_set, correct_h_Y_set, sum_h_Y_set, ...
	  correct_sum_h_Y_set, pen, scores, scores2] = test_smoothing(data, t, s);


N = size(data, 2);
p = 30;
p_small = 2;
mybasis = create_bspline_basis([0 1], p, 4);
pen = full(eval_penalty(mybasis, int2Lfd(0)));

basis_curves = eval_basis(t, mybasis);
myfd_data = data2fd(data, t, mybasis);
data_coef = getcoef(myfd_data)';

lambda_set = zeros(1,5);
h_Y_set = zeros(2,5);
correct_h_Y_set = zeros(2,5);

for k = 1
  switch(k)
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

  tic
  [ic_curves_pos, ic_coef_pos, Y_pos, h_Y_pos, pc_coef, pc_curves, ...
   pc_scores, mean_coef, W, whitening_transform] = ...
      funcica(t, s, myfd_data, p, basis_curves, fdPar(mybasis, 2, lambda));
  toc

  
  data_coef = data_coef';
  pc_coef = pc_coef';
  scores = zeros(N,p_small);

  tic
  for j = 1:p_small
    pc_coef_j = pc_coef(j,:);
    for i = 1:N
      scores(i,j) = sum(sum((data_coef(:,i) * pc_coef_j) .* pen));
    end
  end
  toc
  
  
  scores = scores';
  Y_scores = W * scores(1:2,:);
  
  %{  
  figure(k);
  subplot(2, 2, 1); plot(ic_curves_pos(:,1));
  subplot(2, 2, 2); hist(Y_pos(1,:),  50);
  subplot(2, 2, 3); plot(ic_curves_pos(:,2));
  subplot(2, 2, 4); hist(Y_pos(2,:),  50);
  %}
  
  
  lambda_set(k) = lambda;
  h_Y_set(:,k) = h_Y_pos;
  correct_h_Y_set(:,k) = ...
      [get_vasicek_entropy_estimate_std(Y_scores(1,:)) ; ...
       get_vasicek_entropy_estimate_std(Y_scores(2,:))];
  
end

lambda_set
h_Y_set
correct_h_Y_set
sum_h_Y_set = sum(h_Y_set)
correct_sum_h_Y_set = sum(correct_h_Y_set)

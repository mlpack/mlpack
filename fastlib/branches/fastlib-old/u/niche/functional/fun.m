% initialize random number generator
rand('state', sum(100*clock))

clear;

D = 3;
N = 10000;
p = 30;

mu = 0;
sigma = 1;
b = sigma/2;



clear x px;
for i=1:D

  x(i,:) = laplacinv(rand(N, 1), mu, b);  % laplacian

  %x(i,:) = rand(N, 1);  % uniform random
  
  %x(i,:) = norminv(rand(N, 1), mu, sigma);  %gaussian
  
end

% center the sampling distribution
x = x - repmat(mean(x')', 1, N);


% generate b-spline basis curves
t = linspace(0,1,1000);
mybasis = create_bspline_basis([0 1], p, 4);
basis_curves = eval_basis(t, mybasis);


load s1s2;
s = [s1(t); s2(t)]';


data = s * x;



myfd_data = data2fd(data, t, mybasis);
coef = getcoef(myfd_data);
%data1 = basis_curves * coef(:,1);
pca_results = pca_fd(myfd_data, p);
pc_coef = getcoef(pca_results.harmfd);
pc_curves = basis_curves * pc_coef;
pc_scores = pca_results.harmscr;





% encode our source functions e1 and e2 using the pc basis
for i=1:p
  s1_weights(i) = ...
      diff(ppval(fnint(spline(t, s1(t) .* pc_curves(:,i)')), ...
		 [0 1]));
  s2_weights(i) = ...
      diff(ppval(fnint(spline(t, s2(t) .* pc_curves(:,i)')), ...
		 [0 1]));
end

for i=1:N
  s1_scores(i) = dot(s1_weights, pc_scores(i,:));
  s2_scores(i) = dot(s2_weights, pc_scores(i,:));
end


p_small = 2;

sub_pc_coef = pc_coef(:,1:p_small);
E = pc_scores(:,1:p_small)';

[Y_pos,Y_neg,W_pos,W_neg] = find_opt_unmixing_matrix(E);


for i=1:p_small
  h_E(i) = get_vasicek_entropy_estimate(E(i,:));
  h_Y_pos(i) = get_vasicek_entropy_estimate(Y_pos(i,:));
  h_Y_neg(i) = get_vasicek_entropy_estimate(Y_neg(i,:));
end



ic_coef_pos = (W_pos * sub_pc_coef')';
ic_coef_neg = (W_neg * sub_pc_coef')';


sub_pc_curves = basis_curves * sub_pc_coef;
ic_curves_pos = basis_curves * ic_coef_pos;
ic_curves_neg = basis_curves * ic_coef_neg;


figure(1);
clf;
hold on;
plot(s, 'b');
plot(sub_pc_curves, 'r');
plot(ic_curves_pos, 'g');
plot(ic_curves_neg, 'c');



% using sub_pc_coef', recover the data


  



% now we want to find a matrix W that unmixes well




% let f be some candidate solution

%f1_weights = rand(p,1);
%f1_weights = f1_weights / norm(f1_weights);


%f1 = pc_curves * f1_weights;





% we evaluate some f by considering projections P of the data onto f

% define the l2 norm for functional space:
%    given some vector a and another vector b, we dot multiply the
%    two vectors at the specified values, then approximate the
%    curves with splines, then use quadrature to evaluate the
%    integral in [0,1]

%f1_scores = zeros(N,1);

%for i=1:N
%  f1_scores(i) = dot(f1_weights, pc_scores(i,:));
%end







%given the f1_scores, what to do now?




% objective function
% min sigma H(X_i)
%  X    i
%for a given input variable X, we seek to minimize the sum of the ...
%      entropies of the marginal distributions we consider the sum
%      of the entropies of the marginal distributions

% in the case of one dimension, we are given a set of scalar values
% - we can study the distribution of these values

% in the case of two dimensions, we are given a set of 2-vector
% values
% we want to know the entropy of this distribution


% the m spacing estimator studies the spacing between the sample
% points


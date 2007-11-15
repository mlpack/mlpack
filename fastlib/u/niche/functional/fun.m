% Given
%   distribution F
%   observations X (dims N x T)

% construct random vector w by sampling from F

% sort w descending

% attribute highest values of w to observations with maximal variance


% initialize random number generator
rand('state', sum(100*clock))


% sample from laplacian

D = 2;
N = 10000;

mu = 0;
sigma = 1;
b = sigma/2;



% laplacian
clear l_x l_px;
for i=1:D
  l_x(i,:) = laplacinv(rand(N, 1), mu, b);
  l_px(i,:) = laplacpdf(l_x(i,:), mu, b);
end

% center the sampling distribution
l_x = l_x - repmat(mean(l_x')', 1, N);


% gaussian
%clear g_x g_px;
%for i=1:D
%  g_x(i,:) = norminv(rand(N, 1), mu, sigma);
%  g_px(i,:) = normpdf(g_x(i,:), mu, sigma);
%end

% center the sampling distribution
%g_x = g_x - repmat(mean(g_x')', 1, N);



% mix the source signals

%a = rand(D,D);
%X = a * l_x;

%shuffling = shuffle(1:N);
%X_shuffled = X(:,shuffling);


% generate b-spline basis curves
t = linspace(0,1,1000);
mybasis = create_bspline_basis([0 1], 30, 4);
basis_curves = eval_basis(t, mybasis);


load e1e2;
e = [e1(t); e2(t)]';


data = e * l_x;



myfd_data = data2fd(data, t, mybasis);
coef = getcoef(myfd_data);
%data1 = basis_curves * coef(:,1);
pca_results = pca_fd(myfd_data, 30);
pc_coef = getcoef(pca_results.harmfd);
pc_curves = basis_curves * pc_coef;
pc_scores = pca_results.harmscr;





% encode our source functions e1 and e2 using the pc basis
for i=1:30
  e1_weights(i) = ...
      diff(ppval(fnint(spline(t, e1(t) .* pc_curves(:,i)')), ...
		 [0 1]));
  e2_weights(i) = ...
      diff(ppval(fnint(spline(t, e2(t) .* pc_curves(:,i)')), ...
		 [0 1]));
end

for i=1:N
  e1_scores(i) = dot(e1_weights, pc_scores(i,:));
  e2_scores(i) = dot(e2_weights, pc_scores(i,:));
end




% let f be some candidate solution

let the data be encoded as your mother
f1_weights = rand(30,1);
f1_weights = f1_weights / norm(f1_weights);


f1 = pc_curves * f1_weights;





% we evaluate some f by considering projections P of the data onto f

% define the l2 norm for functional space:
%    given some vector a and another vector b, we dot multiply the
%    two vectors at the specified values, then approximate the
%    curves with splines, then use quadrature to evaluate the
%    integral in [0,1]

f1_scores = zeros(N,1);

for i=1:N
  f1_scores(i) = dot(f1_weights, pc_scores(i,:));
end



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


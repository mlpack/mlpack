function [] = SparseCensorship(X, publishers, K, rho, lambda, mu, gamma)
%function [] = SparseCensorship(X, publishers, K, rho, lambda, mu, gamma)
%
% X - word counts, stored as sparse (# vocab words) x (# docs)
%     since X is sparse, storing the transpose instead may not matter
% publishers - vector of length # docs where publishers_i indicates
%              publisher id for document i, for publisher ids in
%              [P] (each publisher id must be used at least once)
% rho, lambda, mu, and gamma are used as regularization parameters
% for l1-norm penalties. Their correspondence to regularized
% variables is:
%   rho     -  theta_bar
%   lambda  -  theta   (all theta_d are regularized by the same lambda)
%   mu      -  beta    (all beta_k are regularized by the same mu)
%   gamma   -  eta     (all eta_{k,p} are regularized by the same gamma)
%
%
% We use MAP to estimate theta_bar, theta, beta, and eta.
% [Note: Why MAP for theta? So that we get a sparse
% representation for each document, which can then be used in a
% discriminative estimator. Also, if we aren't MAP about theta,
% then it isn't clear that we will obtain a sparse code, but rather
% an average of sparse codes (which need not be sparse).]
%

% D is the number of documents
% V is the number of vocabulary words
[V, D] = size(X);
P = max(publishers);

% set up some variables for convenience
inds_by_doc = cell(D,1);
for d = 1:D
  inds_by_doc{d} = find(X(:,d)); 
end

counts_by_doc = sum(X);


% 1) Initialize parameters

% for now, just set up the sizes for the parameters. coding will happen later

% Initialize theta_bar
theta_bar = zeros(K, 1);

% Initialize theta for each document
theta = zeros(K, D);

% Initialize beta for each topic
beta = zeros(V, K);

% Initialize eta for each topic and publisher
%   - perhaps eta should be sparse?
eta = zeros(V, K, P);



% 2) Variational EM Loop

% 2 a) E-Step

% Given {theta_bar, theta, beta, eta}, Update phi (the variational parameters for z)

phi = ComputePhi(theta_bar, theta, beta, eta, publishers, inds_by_doc);


% Compute expectations needed for model parameter updates

% 2 b) M-step

% Update model parameters

% Update theta_bar
new_theta_d = ...
    L1GeneralProjection(@(theta_bar_var) ThetaBarObjective(theta_bar_var, theta, phi, X, ...
						  counts_by_doc), ...
			theta_bar, lambda); % also, can add an optional parameter for options

% Update theta
for d = 1:D
  fprintf('d = %d\n', d);

  new_theta_d = ...
      L1GeneralProjection(@(theta_d) ThetaObjective(theta_bar, theta_d, phi{d}, X(:,d), ...
                                                    counts_by_doc(d)), ...
                          theta(:,d), lambda); % also, can add an optional parameter for options
end

% Update beta



% Update eta


% 2 c) Go back to E-Step


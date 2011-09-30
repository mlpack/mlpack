function [theta_bar, theta, Beta, eta, phi] = SparseCensorship(X, publishers, K, rho, lambda, mu, nu)
%function [] = SparseCensorship(X, publishers, K, rho, lambda, mu, nu)
%
% X - word counts, stored as sparse (# vocab words) x (# docs)
%     since X is sparse, storing the transpose instead may not matter
% publishers - vector of length # docs where publishers_i indicates
%              publisher id for document i, for publisher ids in
%              [P] (each publisher id must be used at least once)
% rho, lambda, mu, and nu are used as regularization parameters
% for l1-norm penalties. Their correspondence to regularized
% variables is:
%   rho     -  theta_bar
%   lambda  -  theta   (all theta_d are regularized by the same lambda)
%   mu      -  Beta    (all beta_k are regularized by the same mu)
%   nu   -  eta     (all eta_{k,p} are regularized by the same nu)
%
%
% We use MAP to estimate theta_bar, theta, Beta, and eta.
% [Note: Why MAP for theta? So that we get a sparse
% representation for each document, which can then be used in a
% discriminative estimator. Also, if we aren't MAP about theta,
% then it isn't clear that we will obtain a sparse code, but rather
% an average of sparse codes (which need not be sparse).]
%

verbose = true;
skip_eta_updates = true;

if size(publishers, 1) > size(publishers, 2)
  publishers = publishers';
end


% set options for L1GeneralProjection
options.maxIter = 250; % Default iteration limit
options.adjustStep = 0; % ?
options.order = -1; % LBFGS
options.corrections = 10; % ?
options.verbose = false;


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

% Initialize theta_bar to uniform distribution
theta_bar = zeros(K, 1);
rho = rho * ones(K, 1);


% Initialize theta for each document to uniform distribution
theta = zeros(K, D);
lambda = lambda * ones(K, 1);


% Initialize Beta for each topic, randomly
Beta = zeros(V, K);
for j = 1:K
  Beta(:,j) = rand(V, 1);
  Beta(:,j) = log(Beta(:,j) / sum(Beta(:,j)));
end
mu = mu * ones(V, 1);


% Initialize eta for each topic and publisher to uniform distribution
%   - perhaps eta should be sparse?
eta = zeros(V, K, P);
nu = nu * ones(V, 1);


for iteration_num = 1:10
  fprintf('Main Iteration %d\n', iteration_num);
  % 2) Variational EM Loop
  
  % 2 a) E-Step
  
  % Given {theta_bar, theta, Beta, eta}, Update phi (the variational parameters for z)
  
  phi = ComputePhi(theta_bar, theta, Beta, eta, publishers, inds_by_doc);
  
  
  % Compute expectations needed for model parameter updates
  
  % 2 b) M-step

  % Update model parameters

  % Update theta_bar
  options.verbose = true;
  if verbose
    fprintf('Updating theta_bar\n');
  end
  new_theta_bar = ...
      L1GeneralProjection(@(theta_bar_var) ThetaBarObjective(theta_bar_var, theta, phi, X, ...
						  counts_by_doc), ...
			  theta_bar, rho, options);
  theta_bar = new_theta_bar;

  % Update theta
  options.verbose = false;
%  parfor(d = 1:D, 4)
  for d = 1:D
    if verbose
      fprintf('Updating theta_%d\n', d);
    end

    new_theta_d = ...
	L1GeneralProjection(@(theta_d) ThetaObjective(theta_d, theta_bar, phi{d}, X(:,d), ...
						      counts_by_doc(d)), ...
			    theta(:,d), lambda, options);
    theta(:,d) = new_theta_d;
  end

  % Update Beta
  options.verbose = true;
%  parfor(k = 1:K, 4)
  for k = 1:K
    if verbose
      fprintf('Updating beta_%d\n', k);
    end

    new_beta_k = ...
	L1GeneralProjection(@(beta_k) BetaObjective(beta_k, k, eta, phi, ...
						    X, publishers), ...
			    Beta(:,k), mu, options);
    Beta(:,k) = new_beta_k;
  end

  % Update eta
  if ~skip_eta_updates
    options.verbose = true;
    for k = 1:K
      for p = 1:P
	if verbose
	  fprintf('Updating eta_{%d,%d}\n', k, p);
	end
	
	new_eta_k_p = ...
	    L1GeneralProjection(@(eta_k_p) EtaObjective(eta_k_p, Beta(:,k), ...
							k, p, phi, X, ...
							publishers), ...
				eta(:,k,p), nu, options);
	eta(:,k,p) = new_eta_k_p;
      end
    end
  end
  
end

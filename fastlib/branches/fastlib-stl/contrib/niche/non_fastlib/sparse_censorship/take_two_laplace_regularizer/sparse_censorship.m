function [] = sparse_censorship(X, publishers, K)
%function [] = sparse_censorship(X)
%
% X - word counts, stored as sparse (# docs) x (# vocab words)
%     since X is sparse, storing the transpose instead may not matter
% publishers - vector of length # docs where publishers_i indicates
%              publisher id for document i, for publisher ids in
%              [P] (each publisher id must be used at least once)
%
% We use Empirical Bayes to estimate theta, beta, and eta.
% [Note: Why Empirical Bayes for theta? So that we get a sparse
% representation for each document, which can then be used in a
% discriminative estimator. Also, if we aren't Empirical Bayes
% about theta, then it isn't clear that we will obtain a sparse
% code, but rather an average of sparse codes (which need not be
% sparse).]
%
% Each of these 3 parameters is sparse. We use the Laplace prior,
% via a Gaussian prior with Exponentially distributed variance.
% Their respective variance parameters are nu, xi, and tau. The
% variance parameters' respective Exponential distribution
% parameters are sigma, iota, and gamma.
%

% D is the number of documents
% V is the number of vocabulary words
[D, V] = size(X);
P = max(publishers);

% 1) Initialize parameters

% for now, just set up the sizes for the parameters. coding will happen later

% Initialize theta for each document
theta = zeros(K, D);

% Initialize beta for each topic
beta = zeros(V, K);

% Initialize eta for each topic and publisher
eta = zeros(V, K, P);



% 2) Variational EM Loop

% 2 a) E-Step

%given theta, update variational parameters for z

% phi_dvk will be sparse
% may be useful to store non-zero set of word indices for each document

phi = cell(D, 1); % matlab doesn't do sparse tensors, so we use a
                  % cell-array of sparse matrices
for d = 1:D
  phi{d} = sparse(V, K);
end

inds_by_doc = cell(D,1);
for d = 1:D
  inds_by_doc{d} = find(X(d,:)); 
end


exp_beta = exp(beta);
exp_eta = exp(eta);
for d = 1:D
  p = publishers(d);
  for k = 1:K
    sum_exp_beta_k = sum(exp_beta(:,k));
    sum_exp_eta_k_p = sum(exp_eta(:,k,p));
    if isnan(sum_exp_beta_k)
      fprintf('nan at sum_exp_beta_k: d = %d, k = %d\n', d, k);
    end
    if isnan(sum_exp_eta_k_p)
      fprintf('nan at sum_exp_eta_k_p: d = %d, k = %d\n', d, k);
    end
    for v = inds_by_doc{d}
      phi{d}(v,k) = ...
          (exp_beta(v,k) / sum_exp_beta_k) ...
          * (exp_eta(v,k,p) / sum_exp_eta_k_p);
    end
  end
  phi{d}(inds_by_doc{d},:) = ...
      phi{d}(inds_by_doc{d},:) ...
      ./ repmat(sum(phi{d}(inds_by_doc{d},:), 2), 1, K);
end


% Update variational distributions

% Compute expectations needed for model parameter updates

% 2 b) M-step

% Update model parameters

% 2 c) Go back to E-Step


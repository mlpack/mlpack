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

% set up some variables for convenience
inds_by_doc = cell(D,1);
for d = 1:D
  inds_by_doc{d} = find(X(d,:)); 
end




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

% Given (theta, beta, eta), Update the variational parameters for z

phi = ComputePhi(theta, beta, eta, publishers, inds_by_doc);



% Update variational distributions

% Compute expectations needed for model parameter updates

% 2 b) M-step

% Update model parameters

% 2 c) Go back to E-Step


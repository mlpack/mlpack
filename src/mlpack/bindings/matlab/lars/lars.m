function beta = lars(X, Y, varargin)
%LARS
%
%  An implementation of LARS: Least Angle Regression (Stagewise/laSso).  This is
%  a stage-wise homotopy-based algorithm for L1-regularized linear regression
%  (LASSO) and L1+L2-regularized linear regression (Elastic Net).
%
%  Let X be a matrix where each row is a point and each column is a dimension,
%  and let y be a vector of targets.
%
%  The Elastic Net problem is to solve
%
%    min_beta 0.5 || X * beta - y ||_2^2 + lambda_1 ||beta||_1 +
%        0.5 lambda_2 ||beta||_2^2
%
%  If lambda_1 > 0 and lambda_2 = 0, the problem is the LASSO.
%  If lambda_1 > 0 and lambda_2 > 0, the problem is the Elastic Net.
%  If lambda_1 = 0 and lambda_2 > 0, the problem is Ridge Regression.
%  If lambda_1 = 0 and lambda_2 = 0, the problem is unregularized linear
%  regression.
%
%  For efficiency reasons, it is not recommended to use this algorithm with
%  lambda_1 = 0.
%
%Parameters
% X         		 - (required) Matrix containing covariates.
% Y              - (required) Matrix containing y.
% lambda1			   - (optional) Default value 0. l1-penalty regularization.
% lambda2				 - (optional) Default value 0. l2-penalty regularization.
% useCholesky    - (optional) Use Cholesky decomposition during computation
%                             rather than explicitly computing the full Gram
%                             matrix.

% a parser for the inputs
p = inputParser;
p.addParamValue('lambda1', @isscalar);
p.addParamValue('lambda2', @isscalar);
p.addParamValue('useCholesky', false, @(x) (x == true) || (x == false));

% parsing the varargin options
p.parse(varargin{:});
parsed = p.Results;

% interfacing with mlpack. Does not require transposing.
beta = mex_lars(X, Y, ...
	parsed.lambda1, parsed.lambda2, parsed.useCholesky);



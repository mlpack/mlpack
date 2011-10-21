function beta = lars(X, y, method, stop, useGram, Gram, trace)
% LARS  The LARS algorithm for performing LAR or LASSO.
%    BETA = LARS(X, Y) performs least angle regression on the variables in
%    X to approximate the response Y. Variables X are assumed to be
%    normalized (zero mean, unit length), the response Y is assumed to be
%    centered.
%    BETA = LARS(X, Y, METHOD), where METHOD is either 'LARS' or 'LARS'  
%    determines whether least angle regression or lasso regression should
%    be performed. 
%    BETA = LARS(X, Y, METHOD, STOP) with nonzero STOP will perform least
%    angle or lasso regression with early stopping. If STOP is negative,
%    STOP is an integer that determines the desired number of variables. If
%    STOP is positive, it corresponds to an upper bound on the L1-norm of
%    the BETA coefficients.
%
%    Jingu Kim: If STOP is negative, LARS() stops when the optimal solution for
%                 0.5*||X-y*BETA||_F^2 + lambda * ||BETA||_1
%
%    BETA = LARS(X, Y, METHOD, STOP, USEGRAM) specifies whether the Gram
%    matrix X'X should be calculated (USEGRAM = 1) or not (USEGRAM = 0).
%    Calculation of the Gram matrix is suitable for low-dimensional
%    problems. By default, the Gram matrix is calculated.
%    BETA = LARS(X, Y, METHOD, STOP, USEGRAM, GRAM) makes it possible to
%    supply a pre-computed Gram matrix. Set USEGRAM to 1 to enable. If no
%    Gram matrix is available, exclude argument or set GRAM = [].
%    BETA = LARS(X, Y, METHOD, STOP, USEGRAM, GRAM, TRACE) with nonzero
%    TRACE will print the adding and subtracting of variables as all
%    LARS/lasso solutions are found.
%    Returns BETA where each row contains the predictor coefficients of
%    one iteration. A suitable row is chosen using e.g. cross-validation,
%    possibly including interpolation to achieve sub-iteration accuracy.
%
% Author: Karl Skoglund, IMM, DTU, kas@imm.dtu.dk
% Reference: 'Least Angle Regression' by Bradley Efron et al, 2003.

%% Input checking
% Set default values.

if nargin < 7, trace = 0;, end
if nargin < 6, Gram = [];, end
if nargin < 5, useGram = 1;, end
if nargin < 4, stop = 0;, end
if nargin < 3, method = 'lars';, end
if strcmpi(method, 'lasso')
    lasso = 1;
else
    lasso = 0;
end

%% LARS variable setup
[n p] = size(X);
nvars = min(n-1,p); % 
% JINGU KIM
maxk = 20*nvars;
% maxk = 8*nvars; % Maximum number of iterations

% Jingu Kim
% if stop == 0, beta = zeros(2*nvars, p);
% elseif stop < 0, beta = zeros(2*round(-stop), p);
if stop <=0, beta = zeros(2*nvars, p);
else beta = zeros(100, p);
end
mu = zeros(n, 1); % current "position" as LARS travels towards lsq solution

% Calculate Gram matrix if necessary
if isempty(Gram) && useGram
    Gram = X'*X; % Precomputation of the Gram matrix. Fast but memory consuming.
    % Jingu Kim
    Xty = X'*y;
else
    Xty = X'*y;
end

if ~useGram
    R = []; % Cholesky factorization R'R = X'X where R is upper triangular
end

I = 1:p; % inactive set
A = []; % active set
vars = 0; % Current number of variables
lassocond = 0; % LASSO condition boolean
stopcond = 0; % Early stopping condition boolean
k = 0; % Iteration count

% Jingu Kim
prev_lambda  = max(abs(Xty));
if stop < 0 && (prev_lambda < (-stop))
    k = 1;, beta(k,:)=0;
    stopcond = 1;
end

if trace, disp(sprintf('Step\tAdded\tDropped\t\tActive set size'));, end

%% LARS main loop
while vars < nvars && ~stopcond && k < maxk
    k = k + 1;
    c = X'*(y - mu);
    [C j] = max(abs(c(I)));
    j = I(j);

    if ~lassocond % if a variable has been dropped, do one iteration with this configuration (don't add new one right away)
        if ~useGram
            R = cholinsert(R,X(:,j),X(:,A));
        end
        A = [A j];
        I(I == j) = [];
        vars = vars + 1;
        if trace, disp(sprintf('%d\t\t%d\t\t\t\t\t%d', k, j, vars));, end
    end

    s = sign(c(A)); % get the signs of the correlations

    if useGram
        S = s*ones(1,vars);
        %GA1 = inv(Gram(A,A).*S'.*S)*ones(vars,1); % numerically unstable!
	GA1 = (Gram(A,A).*S'.*S) \ ones(vars,1);
        AA = 1/sqrt(sum(GA1));
        w = AA*GA1.*s; % weights applied to each active variable to get equiangular direction
    else
        GA1 = R\(R'\s);
        AA = 1/sqrt(sum(GA1.*s));
        w = AA*GA1;
    end
    u = X(:,A)*w; % equiangular direction (unit vector)
  
    if vars == nvars % if all variables active, go all the way to the lsq solution
        gamma = C/AA;
    else
        a = X'*u; % correlation between each variable and eqiangular vector
        temp = [(C - c(I))./(AA - a(I)); (C + c(I))./(AA + a(I))];
        gamma = min([temp(temp > 0); C/AA]);
    end

    % LASSO modification
    if lasso
        lassocond = 0;
        temp = -beta(k,A)./w';
        [gamma_tilde] = min([temp(temp > 0) gamma]);
        j = find(temp == gamma_tilde);
        if gamma_tilde < gamma,
            gamma = gamma_tilde;
            lassocond = 1;
        end
    end

    mu = mu + gamma*u;
    if size(beta,1) < k+1
        beta = [beta; zeros(size(beta,1), p)];
    end
    beta(k+1,A) = beta(k,A) + gamma*w';
%    beta(k+1,:)'

  
    % If LASSO condition satisfied, drop variable from active set
    if lassocond == 1
        if ~useGram
            R = choldelete(R,j);
        end
        I = [I A(j)];
        A(j) = [];
        vars = vars - 1;
        if trace, disp(sprintf('%d\t\t\t\t%d\t\t\t%d', k, j, vars));, end
    end
  
    % Early stopping at specified number of variables
    % if stop < 0
    %     stopcond = vars >= -stop;
    % end
    %
    % Jingu Kim
    % Early stopping for specified lambda, 'useGram ==1' is assumed
    if stop < 0
        pLpb = -Xty+Gram*beta(k+1,:)';
        pLpb_nonzero = pLpb(A);
        current_lambda  = mean(abs(pLpb_nonzero));
        if current_lambda <= (-stop)
            interp = (prev_lambda - (-stop))/(prev_lambda - current_lambda);
            beta(k+1,:) = beta(k,:) + interp*(beta(k+1,:) - beta(k,:));
            stopcond = 1;
        else
            prev_lambda = current_lambda;
        end
    end
    
    % Early stopping at specified bound on L1 norm of beta
    % Jingu Kim: temporarily do not use   ...actually, niche now using
    if stop > 0
      t2 = sum(abs(beta(k+1,:)));
      if t2 >= stop
	t1 = sum(abs(beta(k,:)));
	s = (stop - t1)/(t2 - t1); % interpolation factor 0 < s < 1
	beta(k+1,:) = beta(k,:) + s*(beta(k+1,:) - beta(k,:));
	stopcond = 1;
      end
    end
end

% trim beta
if size(beta,1) > k+1, beta(k+2:end, :) = [];, end

if k == maxk
    disp('LARS warning: Forced exit. Maximum number of iteration reached.');
end

% ---------------------------------- Fast Cholesky insert and remove functions--------------------------------
% Updates R in a Cholesky factorization R'R = X'X of a data matrix X. R is
% the current R matrix to be updated. x is a column vector representing the
% variable to be added and X is the data matrix containing the currently
% active variables (not including x).
function R = cholinsert(R, x, X)
diag_k = x'*x; % diagonal element k in X'X matrix
if isempty(R)
    R = sqrt(diag_k);
else
    col_k = x'*X; % elements of column k in X'X matrix
    R_k = R'\col_k'; % R'R_k = (X'X)_k, solve for R_k
    R_kk = sqrt(diag_k - R_k'*R_k); % norm(x'x) = norm(R'*R), find last element by exclusion
    R = [R R_k; [zeros(1,size(R,2)) R_kk]]; % update R
end

% Deletes a variable from the X'X matrix in a Cholesky factorisation R'R =
% X'X. Returns the downdated R. This function is just a stripped version of
% Matlab's qrdelete.
function R = choldelete(R,j)
R(:,j) = []; % remove column j
n = size(R,2);
for k = j:n
    p = k:k+1;
    [G,R(p,k)] = planerot(R(p,k)); % remove extra element in column
    if k < n
        R(p,k+1:n) = G*R(p,k+1:n); % adjust rest of row
    end
end
R(end,:) = []; % remove zero'ed out row
    
%% To do
%
% There is a modification that turns least angle regression into stagewise
% (epsilon) regression. This has not been implemented. 

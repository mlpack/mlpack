function [W,D,costfunc] = train_alternating(trainx,trainy,task_indexes,gamma,Dini,iterations,...
    method,kernel_method,f_method,Dmin_method)

% Main algorithm for Multi-task Feature Learning (with a linear kernel)
% See [Argyriou,Evgeniou,Pontil, NIPS 2006, ML journal 2007]
% 
% task_indexes : sizes of samples per task (may be unbalanced)
% gamma        : regularization parameter
% method :  feat        = orthonormal feature learning, i.e. using trace(W' f(D) W)
%           independent = learning with no coupling across tasks (i.e.
%                         using ||W||_2 regularization)
%           diagonal    = variable (feature) selection (i.e. D is diagonal)
% kernel_method : method for kernel learning (e.g. SVM, least square
%                 regression etc.)
% f_method    : evaluates f(D) (acts on the singular values of D)
% Dmin_method : method for minimizing over D of the form 
%               min_d { sum_i f(d_i) b_i^2 } 
%               (b_i are the singular values of W, 
%               or in case of var. selection the L2 norms of the rows of W)

num_data = size(trainx,2);
dim = size(trainx,1);
T = length(task_indexes);

feat = 1; independent = 2; diagonal = 3;

if (max(max(abs(Dini-Dini'))) > eps)
    error('D should be symmetric');
end
if (min(eig(Dini)) < -eps)
    error('D should be positive semidefinite');
end
if (abs(trace(Dini)-1) > 100*eps)
    error('D should have trace  1');
end
D = Dini;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                            %
%               Feature Learning                             %
%                                                            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if (method == feat)
    costfunc = [];
    
    % Compute f(D)^{-1/2} for the next step
    [U,S,dummy] = svd(D);               % svd seems more robust than eig
    fS = feval(f_method,diag(S));
    temp = sqrt(fS);
    tempi = find(temp > eps);
    temp(tempi) = 1./temp(tempi);
    fD_isqrt = U * diag(temp) * U';

    for iter = 1:iterations
        % Use variable transform to solve the regularization problem for
        % fixed D
        new_trainx = fD_isqrt * trainx;
        [W,costf,err,reg] = train_kernel(new_trainx,trainy,task_indexes,gamma,kernel_method);
        W = fD_isqrt * W;
        
        costfunc = [costfunc; iter, costf, err, reg];

        % Update D
        [U,S,V] = svd(W);
        if (dim > T)
            S = [S, zeros(dim,dim-T)];
        end
        Smin = feval(Dmin_method, diag(S));
        D = U * diag(Smin) * U';
        
        % Compute f(D)^{-1/2} for the next step
        fS = feval(f_method,Smin);
        temp = sqrt(fS);
        tempi = find(temp > eps);
        temp(tempi) = 1./temp(tempi);
        fD_isqrt = U * diag(temp) * U';
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                          %
%           Independent Regularizations                    %
%                                                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if (method == independent)
    [W,costfunc,err,reg] = train_kernel(trainx,trainy,task_indexes,gamma,kernel_method);
    D = [];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                          %
%           Variable selection                             %
%                                                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if (method == diagonal)
    if (norm(D-diag(diag(D))) > eps)
        error('D should be diagonal');
    end
    costfunc = [];

    % Compute f(D)^{-1/2} for the next step
    fS = feval(f_method,diag(D));
    temp = sqrt(fS);
    tempi = find(temp > eps);
    temp(tempi) = 1./temp(tempi);
    fD_isqrt = diag(temp);
    
    for iter = 1:iterations
        new_trainx = fD_isqrt * trainx;
        [W,costf,err,reg] = train_kernel(new_trainx,trainy,task_indexes,gamma,kernel_method);
        W = fD_isqrt * W;
        
        costfunc = [costfunc; iter, costf, err, reg];

        % Update D
        Smin = feval(Dmin_method, sqrt(sum(W.^2,2)));
        D = diag(Smin);
        
        % Compute f(D)^{-1/2} for the next step
        fS = feval(f_method,Smin);
        temp = sqrt(fS);
        tempi = find(temp > eps);
        temp(tempi) = 1./temp(tempi);
        fD_isqrt = diag(temp);
    end
end



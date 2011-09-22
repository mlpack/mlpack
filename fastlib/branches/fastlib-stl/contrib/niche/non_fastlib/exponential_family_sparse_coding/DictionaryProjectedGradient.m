function D_opt = DictionaryProjectedGradient(D_0, S, T, alpha, beta, verbose)
%function D_opt = DictionaryProjectedGradient(D_0, S, T, alpha, beta, verbose)
%
% T are the sufficient statistics for the data, stored as
% num_features by num_points
%
% Nocedal and Wright suggested default alpha = 1e-4,
% but Boyd and Vandenberghe suggest using 0.01 to 0.3
%
% We use Armijo line search, with projection operator P, gradient g, and
% termination condition
%   f(P(x + t*g)) >= f(x) + alpha * g^T (P(x + t*g) - x)


if nargin < 4
  alpha = 1e-4;
end
if nargin < 5
  beta = 0.9;
end
if nargin < 6
  verbose = false;
end

%obj_tol = 1e-6;
obj_tol = 1e-1;

[d k] = size(D_0);
n = size(S, 2);


for main_iteration = 1:1000

  if verbose
    fprintf('Main Iteration %d\n', main_iteration);
  end

  % compute gradient
  %grad = zeros(size(D_0));
  %grad = grad - T * S';
  % easy to understand version of the above line
  %for i = 1:n
  %  grad = grad - T(:,i) * S(:,i)';
  %end

  %grad = grad + exp(D_0 * S) * S';
  % easy to understand version of the above line
  %for i = 1:n
  %  grad = grad + exp(D_0 * S(:,i)) * S(:,i)';
  %end

  grad = ComputePoissonDictionaryGradient(D_0, S, T);

  grad = grad / n; % not sure how well this works, maybe we should switch to the below again
  %grad = grad / sqrt(trace(grad' * grad)); % seems to work well
  
  % do line search along direction of negative gradient, using projected evaluation to find D_opt
  
  %start with step size t = 1, decreasing by beta until Armijo condition is satisfied
  %   f(P(x + t*g)) >= f(x) + alpha * g^T (P(x + t*g) - x)
  
  f_0 = ComputePoissonDictionaryObjective(D_0, S, T);
  
  if verbose == 2
    fprintf('\t\t\t\ttrace(grad^T * grad) = %f\n', trace(grad' * grad));
  end
    
  
  

  fro_norm_grad = norm(grad, 'fro');
  if fro_norm_grad > 2 * sqrt(k)
    % if t is any larger than this, then the resulting D_t would never be feasible
    t = sqrt(k) / fro_norm_grad; 
  else
    t = 1;
  end
  
  iteration_num = 0;
  done = false;
  prev_best_f = f_0;
  while ~done
    iteration_num = iteration_num + 1;
    if verbose == 2
      fprintf('Iteration %d, t = %f\n', iteration_num, t);
    end

    D_t = D_0 - t * grad;
    norms = sqrt(sum(D_t .^ 2));
    % project if necessary
    for i = find((norms > 1))
      D_t(:,i) = D_t(:,i) / norms(i);
    end
    
    f_t = ComputePoissonDictionaryObjective(D_t, S, T);
    if verbose == 2
      fprintf('f_0 = %f\tf_t = %f\n', f_0, f_t);
      fprintf('\t\t\t\ttrace(grad^T * (D_t - D_0)) = %f\n', trace(grad' * (D_t - D_0))); 
      fprintf('\t\t\t\tnorm(D_t - D_0) = %f\n', norm(D_t - D_0));
    end

    if f_t <= f_0 + alpha * trace(grad' * (D_t - D_0))
      done = true;
      if verbose
	fprintf('\t\tCompleted %d line search iterations\n', iteration_num);
	fprintf('\t\tObjective value: %f\n', f_t);
      end
      if f_t > prev_best_f
	error('Objective increased! Aborting...');
	return;
      end
      if prev_best_f - f_t < obj_tol
	if verbose
	  fprintf('Improvement to objective below tolerance. Finished.\n');
	end
	D_opt = D_0;
	return;
      end
      prev_best_f = f_t;
    end
    %pause;
    t = beta * t;
  end

  D_0 = D_t;
end

D_opt = D_0;

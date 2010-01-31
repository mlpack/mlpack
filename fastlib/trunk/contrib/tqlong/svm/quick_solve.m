%% Solve [diag(a) 1; 1' 0] [dx;w] = [b; 0]
function [dx,w] = quick_solve(a, b)
if (length(a) ~= length(b)) 
    error('length(a) must equal to length(b)');
end
n = length(a);
w = sum(b./a)/sum(1./a);
dx = (b-w)./a;
end
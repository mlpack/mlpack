% projection a'x+\gamma/2 \|x-bx\|^2
% equivalently \|x - (bx-a/gamma)\|^2
% on canonical simplex x_i >= 0 \sum x_i = 1

function [xout, I] = simplex_project(a, gamma, bx)
n = length(a);
I = false(n,1);
x = bx-a/gamma;
xout = x;

while (1)
    s = sum(x);
    n_I = n-sum(I);
    % projection on V_I
    xout(I) = 0;
    xout(~I) = x(~I)-(s-1)/n_I;
    
    % find new active set
    J = xout<0;
    
    if ~any(J)
        break;
    end
    I = I | J;
    
    % projection on X_I
    x = (xout > 0).*xout;
end

end
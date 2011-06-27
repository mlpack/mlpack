function [f,h] = kde(x,h,kernel)

if nargin < 3
    if nargin < 2
        h = [];
    end

    kernel = @gaussian_kernel;
end
if ~isempty(h) & h <= 0
    h = [];
end

my_options = optimset('GradObj','on');
my_options = optimset(my_options,'Display','off');
my_options = optimset(my_options,'TolFun',1e-4);
my_options = optimset(my_options,'TolX',1e-4);

iters = 5;

[N,D] = size(x);
dsqd = euclidean_metric(x);

if isempty(h)
    sorted_dsqd = sort(dsqd);
    best = Inf;
    for i = 1:iters
        h_init = median(sorted_dsqd(ceil(N/(5*i)),:));
        [h_temp,score] = fmincon(@(h) kde_loss(dsqd,D,h,kernel),h_init,[],[],[],[],0,[],[],my_options);
        if score < best
            h = h_temp;
            best = score;
        end
    end
end

f = mean(kernel(dsqd,D,h),2);



function [f,g] = kde_loss(dsqd,D,h,kernel)

N = size(dsqd,1);

[k,k2,d_k,d_k2] = kernel(dsqd,D,h);
[z,z2,d_z,d_z2] = kernel(0,D,h);

f = mean(mean(k2 - 2*k,1),2) + 2*z/N;
g = mean(mean(d_k2 - 2*d_k,1),2) + 2*d_z/N;



function [k,k2,d_k,d_k2] = gaussian_kernel(dsqd,D,h)

k    = exp(-dsqd/(2*h^2)) / (sqrt(2*pi)*h)^D;
k2   = exp(-dsqd/(8*h^2)) / (sqrt(8*pi)*h)^D;
d_k  = k .* (dsqd/h^3 - D/h);
d_k2 = k2 .* (dsqd/(4*h^3) - D/h);



function dsqd = euclidean_metric(x)

[N,D] = size(x);

if D < 130
    dsqd = sum((repmat(reshape(x,N,1,D),1,N) - repmat(reshape(x,1,N,D),N,1)).^2,3);
else
    dsqd = zeros(N);
    for i = 2:N
        for j = 1:i
            dsqd(i,j) = sum((x(i,:) - x(j,:)).^2,2);
        end
    end
    dsqd = dsqd + dsqd';
end
function[]=LeastSquareCV(bandwidths,data_file)
%This function does the least squares cv. I am not using the start kernel. 
% The code is based on eq 3.38 in Silverman 

data=csvread(data_file);
[n_rows,n_cols]=size(data);
num_of_points=n_rows;
num_of_dim=n_cols;
least_cv_score=bitmax;
least_index=1;
display('rows are');
display(n_rows);
display(n_cols);
for k=1:length(bandwidths)
    bw=bandwidths(k);
    norm_const=eval_norm_constant(num_of_dim);
    display(norm_const);
    int_f_hat_sqd=0;
    f_f_hat=0;

    for i=1:n_rows
        for j=1:n_rows
            diff=data(i:i,1:1)-data(j:j,1:1);
            distance=norm(diff);
            int_f_hat_sqd=int_f_hat_sqd+eval_gaussian_kernel(2*bw,distance);
            if(i~=j)
                f_f_hat=f_f_hat+eval_gaussian_kernel(bw,distance);
            else
                %f_f_hat=f_f_hat+eval_gaussian_kernel(bw,distance);
                %display('added');
                %display(eval_gaussian_kernel(bw,distance));
                %display(distance);
                %display(diff);
                %display('vector dist is');
                
            end
        end
    end
    inv_bandwidth_2=1.0/(bw^(2*num_of_dim-1));
    int_f_hat_sqd=(int_f_hat_sqd)/(num_of_points^2*norm_const);
    int_f_hat_sqd=int_f_hat_sqd*inv_bandwidth_2;
    f_f_hat=...
    f_f_hat/(num_of_points*(num_of_points-1)*bw^(num_of_dim)*norm_const^2);
    final_cv_score=int_f_hat_sqd-2*f_f_hat;
    display(bw);
    display(final_cv_score);
    display(f_f_hat);
    display(int_f_hat_sqd);
    if(final_cv_score<least_cv_score)
        least_cv_score=final_cv_score;
        least_index=k;
    end 
end
display(least_cv_score);
display(least_index);

function[norm_const]=eval_norm_constant(num_of_dim)
half_dim=num_of_dim/2;
norm_const=(2*pi)^half_dim;

function[val]=eval_gaussian_kernel(bw,distance)
dist_by_bw=-(distance/bw)^2;
val=exp(dist_by_bw);

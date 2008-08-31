function[hyperkernel_matrix]=FormGaussianHyperKernelMatrix(sigma_h_sqd,sigma_sqd)
%%This function fill sup th ehyper-kernel matrix. We shall use a gaussian
%%hyper-kernel matrix which takes in 2 parameters namely sigma_sqd and
%%sigma_h_sqd

data=dlmread('points_mog3_small_train.txt');
[num_of_points,dim]=size(data);
sqd_num_of_points=num_of_points^2;
display(num_of_points);

hyperkernel_matrix=zeros(sqd_num_of_points,sqd_num_of_points);
for i=1:num_of_points
    for j=1:num_of_points
        diff_between_x_i_and_x_j=data(j:j,1:end)-data(i:i,1:end);
        dist_between_x_i_and_x_j=norm(diff_between_x_i_and_x_j);
        sqd_dist_between_x_i_and_x_j=dist_between_x_i_and_x_j^2;
        mean_of_x_i_x_j=(data(j:j,1:end)+data(i:i,1:end))/2;
        gaussian_due_to_x_i_and_x_j=exp(-sqd_dist_between_x_i_and_x_j/4*sigma_sqd)/(4*pi*sigma_sqd)^(dim/2);
        row_num_in_hyperkernel_matrix=(i-1)*num_of_points+j;
        for p=1:num_of_points
            for q=1:num_of_points
                diff_between_x_p_and_x_q=data(p:p,1:end)-data(q:q,1:end);
                dist_between_x_p_and_x_q=norm(diff_between_x_p_and_x_q);
                sqd_dist_between_x_p_and_x_q=dist_between_x_p_and_x_q^2;
                gaussian_due_to_x_p_and_x_q=exp(-sqd_dist_between_x_p_and_x_q/4*sigma_sqd)/(4*pi*sigma_sqd)^(dim/2);
                mean_of_x_p_x_q=(data(p:p,1:end)+data(q:q,1:end))/2;
                dist_between_means=norm(mean_of_x_i_x_j-mean_of_x_p_x_q);
                sqd_dist_between_means=dist_between_means^2;
                gaussian_between_means=exp(-sqd_dist_between_means/(sigma_sqd+sigma_h_sqd))/((2*pi*(sigma_sqd+sigma_h_sqd))^(dim/2));
                kernel_value=gaussian_due_to_x_i_and_x_j*gaussian_due_to_x_p_and_x_q*gaussian_between_means;
                col_num_in_hyperkernel_matrix=(p-1)*num_of_points+q;
                hyperkernel_matrix(row_num_in_hyperkernel_matrix,col_num_in_hyperkernel_matrix)=kernel_value;
               
                %if(p==2 && q==2)
                 %   display(mean_of_x_p_x_q);
                  %  display(mean_of_x_i_x_j);
                   % display(sqd_dist_between_means);
                   %end
            end
        end
    end
end
display('gaussian kernel formed');
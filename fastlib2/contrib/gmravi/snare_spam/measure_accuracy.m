function[]=measure_accuracy()
% 1 is positive class
% 0 is negative class
% -1 is unknown class

data=csvread('aggr_all_features_scaled_test5.data');
[num_test_points,num_features]=size(data);
true_labels=data(1:end,end:end);

predicted_labels=csvread('out.csv');
display(predicted_labels(10));
display(true_labels(10));

confusion_matrix=zeros(3,2);
for i=1:num_test_points
    true_label=true_labels(i);
    predicted_label=predicted_labels(i);
    if(true_label==predicted_label)
        %This was correctly identified
        if(true_label==1)
            confusion_matrix(1,1)=confusion_matrix(1,1)+1;
        else
            confusion_matrix(2,2)=confusion_matrix(2,2)+1;
        end
    else
        if(predicted_label~=-1)
            %The predicted_label is not an unidentified label
            if(true_label==1)
                confusion_matrix(2,1)=confusion_matrix(2,1)+1;
            else
                confusion_matrix(1,2)=confusion_matrix(1,2)+1;
            end
        else
            %Predicted label is -1. That is it is unidentified label
            if(true_label==1)
                confusion_matrix(3,1)=confusion_matrix(3,1)+1;
            else
                confusion_matrix(3,2)=confusion_matrix(3,2)+1;
            end
        end
    end
end
display(confusion_matrix);
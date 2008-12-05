function[G_mat]=IncompleteCholesky(num_rows,epsilon)

Q_mat=GetQMatrix()
G_mat=[];
for i=1:num_rows
    display('i is');
    display(i);

    for j=i:num_rows
        G_diag(j)=GetDiagonalElements(Q_mat,j);
        for k=1:i-1
                G_diag(j)=G_diag(j)-(G_mat(j,k)*G_mat(j,k)); %u are affecting G_(j,j)
        end
    end
    new_vec=sparse(zeros(num_rows,1));
    new_vec(i)=G_diag(i);
    temp_vec=[];
    
    display('After calculations on G diagonal I have');
    display(G_diag);
    if(SumDiagonalElements(G_diag)>epsilon)
        j_star=FindIndexOfMaxDiagonalElement(G_diag,i,num_rows); 
        display(j_star);
        if(j_star<i)
            rukjaa
        end
        
        new_vec(i:i+num_rows-j_star+1)=Q_mat(j_star:num_rows,j_star:j_star);
        new_vec(i+num_rows-j_star+1:num_rows)=0;
        %for t=i+1:j_star-num_rows+1+i
        %    new_vec(t)=0;
        %end
        
        display('At the end of the first operation');
        display(full(new_vec));
      
        %G_(i,1:i) <-> G(j*,1:i)
        if(i>1)
            temp_vec=G_mat(j_star:j_star,1:i-1);
        end
        
        temp_vec(end+1)=new_vec(j_star);
        %Check if i=j_star
        if(j_star==i)
            temp_vec(end)=G_diag(j_star);
            display('It was a dirtied element');
        else
            %Do nothing
        end
        if(i>1)
            G_mat(j_star:j_star,1:i-1)=G_mat(i:i,1:i-1);
        end
        new_vec(j_star)=new_vec(i); %Because the diagonal elem is dirty
        if(i>1)
            G_mat(i:i,1:i-1)=temp_vec(1:i-1); 
        end
        new_vec(i)=temp_vec(i);
        display('After all swaps');
        display(full(new_vec));
        for j=1:i-1
            new_vec(i+1:num_rows)=new_vec(i+1:num_rows)-...
                                    (G_mat(i+1:num_rows,j)*G_mat(i,j));
        end
        display('Just before dividing');
        display(full(new_vec));
        new_vec(i)=sqrt(new_vec(i));
        new_vec(i+1:num_rows)=new_vec(i+1:num_rows)/new_vec(i);
        display('After dividing');
        display(full(new_vec));
        
        %At the end of all calculations append this new column to the
        %exisiting matrix
        G_mat=[G_mat,new_vec];
        display('adding new vector');
        display(full(new_vec));
    else
        k=i-1;
        %This ends the algorithm
        break;
    end
end
G_mat=full(G_mat);
      


%function[]=EvaluateAllDiagonalElements(num_rows)
%Q_diag=[];
%for i=1:num_rows
%    Q_diag(i)=
%end

function[Q_mat]=GetQMatrix()
Q_mat=[1,2,3;2,4,6;3,6,9];


function[val]=GetDiagonalElements(Q_mat,row_num) 
val=Q_mat(row_num,row_num);

function[val]=SumDiagonalElements(G_diag)
val=sum(G_diag);
display('Sum of diagonal elements');
display(val);

function[index]=FindIndexOfMaxDiagonalElement(G_diag,start,num_rows)
index=find(G_diag==max(G_diag));
index=index(end);


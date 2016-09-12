
function [model_LLSF] = LLSF( X, Y, optmParameter)
% This function is designed to learn the label specific features for each label 
%
%    Syntax
%
%       [model_LLSF] = LLSF( X, train_target,optmPara)
%
%    Input
%       X           - a n by d data matrix, n is the number of instances and d is the number of features 
%       Y           - a n by l label matrix, n is the number of instances and l is the number of labels
%       optmPara    - A struct variable with seven fields, the optimization parameters for LLSF
%                   1) alpha            - label correlation
%                   2) beta             - sparsity
%                   3) gama             - linear regression
%                   4) maxIter          - number of maximum iterations
%                   5) miniLossMargin   - minmum loss margin between two iterations
%                   6) outputtempresult - whether outputing the temporal resutls, {1 - yes, 0 - not}
%    Output
%
%       model_LLSF  - a d by l Coefficient matrix
%
%[1] J. Huang, G.-R Li, Q.-M. Huang and X.-D. Wu. Learning Label Specific Features for Multi-Label Classifcation. 
%    In: Proceedings of the International Conference on Data Mining, 2015.

    
   %% optimization parameters
    alpha            = optmParameter.alpha;
    beta             = optmParameter.beta;
    gamma            = optmParameter.gamma;
    maxIter          = optmParameter.maxIter;
    miniLossMargin   = optmParameter.minimumLossMargin;

   %% initializtion
    num_dim = size(X,2);
    XTX = X'*X;
    XTY = X'*Y;
    W_s   = (XTX + gamma*eye(num_dim)) \ (XTY);
    W_s_1 = W_s;
    R     = pdist2( Y'+eps, Y'+eps, 'cosine' );

    iter    = 1;
    oldloss = 0;
    
    Lip = sqrt(2*(norm(XTX)^2 + norm(alpha*R)^2));

    bk = 1;
    bk_1 = 1; 
    
   %% proximal gradient
    while iter <= maxIter

       W_s_k  = W_s + (bk_1 - 1)/bk * (W_s - W_s_1);
       Gw_s_k = W_s_k - 1/Lip * ((XTX*W_s_k - XTY) + alpha * W_s_k*R);
       bk_1   = bk;
       bk     = (1 + sqrt(4*bk^2 + 1))/2;
       W_s_1  = W_s;
       W_s    = softthres(Gw_s_k,beta/Lip);
       
       predictionLoss = trace((X*W_s - Y)'*(X*W_s - Y));
       correlation     = trace(R*W_s'*W_s);
       sparsity    = sum(sum(W_s~=0));
       totalloss = predictionLoss + alpha*correlation + beta*sparsity;
      
       if abs(oldloss - totalloss) <= miniLossMargin
           break;
       elseif totalloss <=0
           break;
       else
           oldloss = totalloss;
       end
       
       iter=iter+1;
    end
    model_LLSF = W_s;

end


%% soft thresholding operator
function W = softthres(W_t,lambda)
    W = max(W_t-lambda,0) - max(-W_t-lambda,0); 
end

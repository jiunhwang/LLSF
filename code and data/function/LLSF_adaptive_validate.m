
function [ BestParameter, BestResult ] = LLSF_adaptive_validate( train_data, train_target, oldOptmParameter)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% tune the best parameters for LLSF by crossvalidation
% Input
%   - train_data            : n by d data matrix
%   - trian_target          : n by l lable matrix
%   - oldOptmParameter      : initilization parameter
%
% Output
%   - BestParameter         : a structral variable with searched paramters,
%                             ie. alpha, beta, and gamma
%   - BestResult            : best result on the training data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    num_train             = size(train_data,1);
    randorder             = randperm(num_train);
    
    optmParameter         = oldOptmParameter;
    
    alpha_searchrange     = oldOptmParameter.alpha_searchrange;
    beta_searchrange      = oldOptmParameter.beta_searchrange;
    gamma_searchrange     = oldOptmParameter.gamma_searchrange;
    
    BestResult = zeros(15,1);
    num_cv = 5;
    index = 1;
    total = length(alpha_searchrange)*length(beta_searchrange)*length(gamma_searchrange);
    for i=1:length(alpha_searchrange) % alpha
        for j=1:length(beta_searchrange) % beta
            for k = 1:length(gamma_searchrange) % gamma
                fprintf('\n-   %d-th/%d: search parameter alpha and beta for LLSF, alpha = %f, beta = %f, and gamma = %f',index, total, alpha_searchrange(i), beta_searchrange(j), gamma_searchrange(k));
                index = index + 1;
                optmParameter.alpha   = alpha_searchrange(i); % label correlation
                optmParameter.beta    = beta_searchrange(j);  % sparsity
                optmParameter.gamma   = gamma_searchrange(k); % {0.01, 0.1}

                optmParameter.maxIter           = 100;
                optmParameter.minimumLossMargin = 0.01;
                optmParameter.outputtempresult  = 0;
                optmParameter.drawConvergence   = 0;

                Result = zeros(15,1);
                for cv = 1:num_cv
                    [cv_train_data,cv_train_target,cv_test_data,cv_test_target ] = generateCVSet( train_data,train_target',randorder,cv,num_cv);
                    [model_LLSF]  = LLSF( cv_train_data, cv_train_target,optmParameter);
                    Outputs     = (cv_test_data*model_LLSF)';
                    Pre_Labels  = round(Outputs);Pre_Labels  = (Pre_Labels >= 1); Pre_Labels  = double(Pre_Labels);
                    Result      = Result + EvaluationAll(Pre_Labels,Outputs,cv_test_target');
                end
                Result = Result./num_cv;
                if optmParameter.bQuiet == 0
                    PrintResults(Result)
                end
                r = IsBetterThanBefore(BestResult,Result);
                if r == 1
                    BestResult = Result;
                    PrintResults(Result);
                    BestParameter = optmParameter;
                end
            end
        end
    end
end


function r = IsBetterThanBefore(Result,CurrentResult)
% 1 HammingLoss
% 2 ExampleBasedAPCCuracy
% 3 ExampleBasedPrecision
% 4 ExampleBasedRecall
% 5 ExampleBasedFmeasure
% 6 SubsetAPCCuracy
% 7 LabelBasedAPCCuracy
% 8 LabelBasedPrecision
% 9 LabelBasedRecall
% 10 LabelBasedFmeasure
% 11 MicroF1Measure
% 12 Average_Precision
% 13 OneError
% 14 RankingLoss
% 15 Coverage
% 
%  the combination of Accuracy, F1, Macro F1 and Micro F1. Of course, any evaluation metrics or the combination of them can be used.

    a = CurrentResult(2,1) + CurrentResult(5,1)  + CurrentResult(10,1) + CurrentResult(11,1);
    b = Result(2,1) + Result(5,1) + Result(10,1) + Result(11,1);

    if a > b
        r =1;
    else
        r = 0;
    end
end

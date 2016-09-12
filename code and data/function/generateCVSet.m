function [ train_x,train_y,test_x,test_y ] = generateCVSet( X,Y,kk,index,totalCV )
    assert(index <= 10);
    assert(totalCV <= 10);
    m = size(X,1);
    slice = ceil(m/totalCV);
    test_x = X(kk((index - 1) * slice + 1: min( index * slice , m ) ) ,:);
    test_y = Y(kk((index - 1) * slice + 1: min( index * slice , m ) ) ,:);
    
    train_x = X(setdiff(kk,kk((index - 1) * slice + 1: min( index * slice , m ) )),:);
    train_y = Y(setdiff(kk,kk((index - 1) * slice + 1: min( index * slice , m ) )),:);
end
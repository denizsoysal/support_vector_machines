


gam = 100;  
sig2 = 0.1;


load iris.mat;
% Train the classification model.
[alpha , b] = trainlssvm ({ Xtrain , Ytrain , 'c', gam , sig2 , 'RBF_kernel'});
% Classification of the test data.
[Yest , Ylatent ] = simlssvm ({ Xtrain , Ytrain , 'c', gam , sig2 , 'RBF_kernel'}, {alpha , b}, Xtest );
% Generating the ROC curve.
roc( Ylatent , Ytest );
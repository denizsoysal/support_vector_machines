clc, clf, clear
close all;

load diabetes.mat;

%count occurence in dataset
[GC_train,GR_train] = groupcounts(labels_train) 
[GC_test,GR_test] = groupcounts(labels_test) 

Xtrain = trainset;
Xtest = testset;
Ytrain = labels_train;
Ytest = labels_test;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%tuning of parameters%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%tune parameter with a rbf kernel using 10 fold cross validation and grid search 
disp('RBF kernel'),

[gam_rbf ,sig2_rbf , cost_rbf ] = tunelssvm ({ Xtrain , Ytrain , 'c', [], [],'RBF_kernel'}, 'gridsearch', 'crossvalidatelssvm',{10, 'misclass'});

%tune parameter with a polynomial kernel using 10 fold cross validation and grid search 

disp('poly kernel'),

[gam_poly , degree_poly ] = tunelssvm ({ Xtrain , Ytrain , 'c', [], [],'poly_kernel'}, 'gridsearch', 'crossvalidatelssvm',{10, 'misclass'});
degree_poly = degree_poly(2)
t = degree_poly(1);

%tune parameter with a linear kernel using 10 fold cross validation and grid search 

disp('linear kernel'),

[gam_linear , cost_linear ] = tunelssvm ({ Xtrain , Ytrain , 'c', [], [],'lin_kernel'}, 'gridsearch', 'crossvalidatelssvm',{10, 'misclass'});

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%use tuned parameter to train a LS SVM %%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

type='c'; 


%train rbf kernel 
%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('RBF kernel'),

[alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam_rbf,sig2_rbf,'RBF_kernel'});


% Obtain the output of the trained classifier
[Yht, Zt] = simlssvm({Xtrain,Ytrain,type,gam_rbf,sig2_rbf,'RBF_kernel'}, {alpha,b}, Xtest);
err = sum(Yht~=Ytest); 
disp('RBF kernel result'),
fprintf('\n on test: #misclass = %d, error rate = %.2f%% \n', err, err/length(Ytest)*100)

%plot ROC
roc( Zt , Ytest );

%train poly kernel 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('Polynomial kernel'),

[alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam_poly,[t; degree_poly],'poly_kernel'});


[Yht, Zt] = simlssvm({Xtrain,Ytrain,type,gam_poly,[t; degree_poly],'poly_kernel'}, {alpha,b}, Xtest);

err = sum(Yht~=Ytest); 
disp('Polynomial kernel result'),
fprintf('\n on test: #misclass = %d, error rate = %.2f%%\n', err, err/length(Ytest)*100)

%plot ROC
roc( Zt , Ytest );

%train linear kernel 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('Linear kernel'),

[alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam_linear,[],'lin_kernel'});


[Yht, Zt] = simlssvm({Xtrain,Ytrain,type,gam_linear,[],'lin_kernel'}, {alpha,b}, Xtest);

err = sum(Yht~=Ytest); 
disp('Linear kernel results'),
fprintf('\n on test: #misclass = %d, error rate = %.2f%%\n', err, err/length(Ytest)*100)

%plot ROC
roc( Zt , Ytest );










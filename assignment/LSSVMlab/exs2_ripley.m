clc, clf, clear
close all;

load ripley.mat;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%visualization%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%visualize training data

%first 125 datapoint are class -1, next 125 datapoint are class 1
X_train_1 = Xtrain(1:125,:);
X_train_2 = Xtrain(126:250,:);
figure ;
hold on;
%class 1 : 
plot (X_train_1 (: ,1) , X_train_1 (: ,2) , 'ro',LineWidth=10);
%class 2 :
plot (X_train_2 (: ,1) , X_train_2 (: ,2) , 'bo',LineWidth=10);
title('Distribution of the training data - Ripley', fontsize=28);
legend('Class - 1', 'Class +1', fontsize=20);
hold off;


%visualize test data

%first 500 datapoint are class -1 , next 500 datapoints are class 1 
X_test_1 = Xtest(1:500,:);
X_test_2 = Xtest(501:1000,:);
figure ;
hold on;
%class 1 : 
plot (X_test_1 (: ,1) , X_test_1 (: ,2) , 'ro',LineWidth=10);
%class 2 :
plot (X_test_2 (: ,1) , X_test_2 (: ,2) , 'bo',LineWidth=10);
title('Distribution of the test data - Ripley', fontsize=28);
legend('Class - 1', 'Class +1', fontsize=20);
hold off;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%tuning of parameters%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%tune parameter with a rbf kernel using 10 fold cross validation and grid search 

[gam_rbf ,sig2_rbf , cost_rbf ] = tunelssvm ({ Xtrain , Ytrain , 'c', [], [],'RBF_kernel'}, 'gridsearch', 'crossvalidatelssvm',{10, 'misclass'});

%tune parameter with a polynomial kernel using 10 fold cross validation and grid search 

[gam_poly , degree_poly ] = tunelssvm ({ Xtrain , Ytrain , 'c', [], [],'poly_kernel'}, 'gridsearch', 'crossvalidatelssvm',{10, 'misclass'});
degree_poly = degree_poly(2)


%tune parameter with a linear kernel using 10 fold cross validation and grid search 

[gam_linear , cost_linear ] = tunelssvm ({ Xtrain , Ytrain , 'c', [], [],'lin_kernel'}, 'gridsearch', 'crossvalidatelssvm',{10, 'misclass'});

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%use tuned parameter to train a LS SVM %%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

t = 1; 
type='c'; 


%train rbf kernel 
%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('RBF kernel'),

[alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam_rbf,sig2_rbf,'RBF_kernel'});

% Plot the decision boundary of a 2-d LS-SVM classifier
plotlssvm({Xtrain,Ytrain,type,gam_rbf,sig2_rbf,'RBF_kernel','preprocess'},{alpha,b});

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

figure; plotlssvm({Xtrain,Ytrain,type,gam_poly,[t; degree_poly],'poly_kernel','preprocess'},{alpha,b});

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

figure; plotlssvm({Xtrain,Ytrain,type,gam_linear,[],'lin_kernel','preprocess'},{alpha,b});

[Yht, Zt] = simlssvm({Xtrain,Ytrain,type,gam_linear,[],'lin_kernel'}, {alpha,b}, Xtest);

err = sum(Yht~=Ytest); 
disp('Linear kernel results'),
fprintf('\n on test: #misclass = %d, error rate = %.2f%%\n', err, err/length(Ytest)*100)

%plot ROC
roc( Zt , Ytest );










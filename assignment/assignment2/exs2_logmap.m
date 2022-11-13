%Let's now apply time series prediction on the logmap dataset.


clear,clc,clf
close all;



load logmap.mat;

% Two variables are loaded into the workspace: Z (training data) and Ztest (test data).
% First, we have to map our sequence Z into a regression problem. This can be done using the
% command windowize:
% Then, a model can be built using these data points. We will tune it using
% cross validation. The parameter to tune are : gam,sig2,order
%We can use MAE to report the performance

%what we will do is to test for a range of order, and for each order value
%do cross validation, and find at the end the lowest value


costs = [];
gams = [];
sigs= [];
for i=1:50
    order = i
    X = windowize (Z, 1:( order + 1));
    Y = X(:, end);
    X = X(:, 1: order );
    [ gam , sig2 ] = tunelssvm({ X , Y , 'f', [] , [] ,'RBF_kernel'} , 'simplex', 'crossvalidatelssvm',{10 , 'mae'}) ;
    [alpha , b] = trainlssvm ({X, Y, 'f', gam , sig2 });
    Xs = Z(end - order +1: end , 1);
    nb = 50;
    prediction = predict ({X, Y, 'f', gam , sig2 }, Xs , nb);
    mae = sum(abs(Ztest(:)-prediction(:)))/numel(Ztest)
    costs(end+1) = mae;
    gams(end+1) = gam;
    sigs(end+1) = sig2;
end

%take the indice with the minimal cost
[M,I] = min(costs)
%the indice correspond also to the order because we start at 1
best_order = I
%find corresponding gams and sig2 based on indice
best_gam = gams(I)
best_sig2 = sigs(I)

%now train the LSSVM
X = windowize (Z, 1:( best_order + 1));
Y = X(:, end);
X = X(:, 1: best_order );
[alpha , b] = trainlssvm ({X, Y, 'f', best_gam , best_sig2 });


% It is straightforward to predict the next data points using the predict function of the LS-
% SVMlab toolbox. In order to call the function, we first have to define the starting point of the 
% prediction:

Xs = Z(end - best_order +1: end , 1);

% Naturally, this is the last point of the training set. The test set Ztest presents data points
% after this point, which we will try to predict. This can be implemented as follows:

nb = 50;
prediction = predict ({X, Y, 'f', best_gam , best_sig2 }, Xs , nb);

% where nb indicates how many time points we want to predict. Here, we
% define this number  % equal to the number of data points in the test set. 
% Finally, the performance of the predictor can be checked visually:

figure ;
hold on;
plot (Ztest , 'k');
plot ( prediction , 'r');
mae = sum(abs(Ztest(:)-prediction(:)))/numel(Ztest)
immse(Ztest,prediction)
hold off;

% In this figure, the data points of the test set, that is, the actual data points that we want
% to predict are depicted in black, while the prediction is presented in red.
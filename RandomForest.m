%{
%Pseudocode
1. Load table into matlab as dataCleanedtable.
2. Define input and target.
3. Separate data into training data (80%).
4. Separate data into testing data (20%).
5. Train the data using fitcensemble and Bag.
6. Compute model accuracy.
7. Compute AUC, ROC and confusion matrix.
%}

%%
%save table into workspace by using 'readtable'
dataCleanedtable = readtable("datacleanforanntraining.xlsx");
%%
%%define inputs and targets columns
% variable 1 to 27 as inputs
% variable 28 as target
x = dataCleanedtable(:,1:27);
y = dataCleanedtable(:,28);
%%
%xtr and ytr are training data sets
%80% = training
%random permutation of every row
rand = randperm(70434);
xtr = x(rand(1:56347),:);
ytr = y(rand(1:56347),:);
%%
%testing dataset
% 20% = rest
xt = x(rand(56348:70434),:);
yt = y(rand(56348:70434),:);

%%
%Train Classification Ensemble
RFModel = fitcensemble(xtr,ytr, 'Method', 'bag')

%%
%test model accuracy
result = predict(RFModel, xt);
yt = table2array(yt);
accuracy = sum(result == yt)/length(yt)*100;
sp = sprintf("Test accuracy = %.2f", accuracy);
disp(sp)
%%
%confusion matrix
cm = confusionchart(yt,result);
cm
%%
%auc
[X,Y,T,AUC] = perfcurve(yt,result,1);
AUC
%%
%roc
figure,plot(X,Y)
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC for Classification by Random Forest')
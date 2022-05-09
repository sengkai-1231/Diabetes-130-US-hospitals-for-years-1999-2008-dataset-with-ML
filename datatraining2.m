%{
%Pseudocode
1. Load table into matlab as dataCleaned.
2. Review summary of table to check if all variables are cleaned correctly.
3. Transpose the table.
4. Define input and output variables.
5. Create a Neural Network with hidden layer size 20.
6. Set up division of data.
7. Train and test network.
8. Compute AUROC, ROC and confusin matrix
%}

%save table into workspace by using 'readtable'
dataCleaned = readtable("datacleanforanntraining.xlsx");

%get summary of table (make sure everything is in double)
summary(dataCleaned);

%transpose
dataclean = table2array(dataCleaned);
transposedata = dataclean';

%define inputs and targets columns
% variable 1 to 27 as inputs
% variable 28 as target
inputs = transposedata([1:27],:);
targets = transposedata(28,:);

%Create a Neural Network
hiddenLayerSize = 5;
net = patternnet(hiddenLayerSize, 'trainrp');

%Set up Division
%training 80
% testing 10,validation 10
net.divideParam.trainRatio = 80/100;
net.divideParam.testRatio = 20/100;

%train network
[net,tr] = train(net,inputs,targets);

%Test the network
% calculate the error and performance
outputs = net(inputs);
errors = gsubtract(targets,outputs);
performance = perform(net,targets,outputs);

%View the network
view(net);

% Compute AUROC 
[X,Y,T,AUC] = perfcurve(targets,outputs,1);
AUC

% plot ROC
figure, plotroc(targets,outputs);
% plot performance
figure, plotperform(tr)
% plot confusion matrix
figure, plotconfusion(targets, outputs)







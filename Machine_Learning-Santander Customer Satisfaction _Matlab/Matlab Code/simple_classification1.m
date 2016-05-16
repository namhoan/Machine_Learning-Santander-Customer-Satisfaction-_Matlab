clc
clear all
close all

% =========================================================================
%                   Classification using Decision Tree
% =========================================================================

% Training.

% Read Training data.

fprintf('Reading Training Data . . .\n')
trdata = csvread('train.csv',1,0);
fprintf('\nTraining data loaded.\n')

% Separate training data.

datanum = trdata(:,1);
inputs = trdata(:,2:end-1);
target = trdata(:,end);

fprintf('\nTraining decision tree . . . \n')

% Train a decision tree.

dTree = ClassificationTree.fit(inputs,num2str(target));

dTree
view(dTree)

fprintf('Training Complete. \n')

% Testing.

% Load testing data.

fprintf('\nReading test data . . . \n')
tedata = csvread('test.csv',1,0);
fprintf('\nTest data loaded. \n')

testdatanum = tedata(:,1);
testinputs = tedata(:,2:end);

dataclass = dTree.predict(testinputs);

% Save Test results.

dclass_dt = str2num(dataclass);

result = uint32([testdatanum,dclass_dt]);

csvwrite('dectree_result.csv',result)









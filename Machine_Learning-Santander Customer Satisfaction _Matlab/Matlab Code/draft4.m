clc
clear all
close all

% Load dataset

fprintf('Reading Training Data . . .\n')
trdata = csvread('train.csv',1,0);
fprintf('\nTraining data loaded.\n')

% Separate training data.

datanum = trdata(:,1);
inputs = trdata(:,2:end-1);
target = trdata(:,end);

% Normalize training data.

% uniques = zeros(1,size(inputs,2));
% norm_id = zeros(1,size(inputs,2));
% for i = 1:size(uniques,2)
%     uniques(i) = size(unique(inputs(:,i)),1);
%     if(uniques(i) == 1)
%         norm_id(i) = 0;                       % Null, do nothing
%     elseif(uniques(i) == 2)
%         if(sort(unique(inputs(:,i))) == [-1;1])
%             norm_id(i) = 0; % Already normalized
%         else
%             norm_id(i) = 1;                   % Binary
%         end
%     elseif(std(inputs(:,i)) < 1) %temp value
%         %norm_id(i) = 2;                      % Categorical
%         norm_id(i) = 3; % Currently avoiding trying to recognize discrete
%     else
%         norm_id(i) = 3;                       % Continuous/Numeric
%     end
% end

%Normalization of inputs:

%normalized = normc(inputs);

% normalized = inputs;
% for i = 1:size(inputs,2)
%     switch norm_id(i)
%         case 0
%             % Do nothing
%         case 1
%             % Replace the lesser value with -1, the greater with 1
%             pos = max(inputs(:,i));
%             normalized(:,i) = (inputs(:,i)==pos) + -1*(inputs(:,i)~=pos);
%         case 2
%             % Apply 1-of-(C-1) categorization to case
%             cats = zeros(size(unique(inputs(:,i))));
%             cats(end) = []; %Prune one dimension
%             normalized(:,i) = zeros(size(normalized(:,i)));
%             temp_dim = 1;
%             for j = sort(unique(inputs(:,i)))
%                 if(inputs(j,i)==median(inputs(:,i)))
%                     cats = ones(size(cats)) * -1;
%                 else
%                     cats(temp_dim) = 1;
%                     temp_dim = temp_dim + 1;
%                 end
%                 normalized(:,i) = normalized + (cats * (input(:,i) == j));
%                 cats = zeros(size(unique(inputs(:,i))));
%             end
%         case 3
%             % Apply median-and-MAD normalization
%             p = median(inputs(:,i)) * ones(size(inputs(:,i)));
%             q = mad(inputs(:,i));
%             normalized(:,i) = (inputs(:,i)-p)/q;
%     end
% end

% Train network
% TODO?: Determine ideal hidden neurons (3rd value of newpr)
% 'useParallel' possibly?
net = configure(fitnet([36 36]),'outputs',1);
% net.numLayers = 3;
%net.layers{1}.transferFcn='tansig';
net.biases{1}.learnFcn='trainscg';
net.biases{2}.learnFcn='trainrp';
% net.layers{2}.size=36;
net.biases{3}.learnFcn='purelin';
net.layers{3}.size=1;
net.trainParam.epochs = 100;                      % (default 1000)
net.trainParam.showWindow = false;                % no GUI, for server use
net.trainParam.showCommandLine = true;            % display in command line
net.trainParam.show = 1;                          % display every iteration
net = train(net, inputs', double(target)', 'showResources', 'yes');

% Save trained network for future possible test use

save net;

% Performance on self
% TODO: Replace linear classification with maybe a real predictor
output = sim(net, inputs');

prune = output';
i = size(prune,1);
while i > 0
    if(target(i) == 0)
        prune(i) = [];
    end
    i = i-1;
end

E = sum(prune)/size(prune,1);

output2 = double(output' >= E);
%[c,cm] = confusion(target,output);
off = sum(target == output2)/size(target, 1);
fprintf('\n%d\n', off);

% Histogram of neural network results + classification of training set
% Turn off for use in console, no GUI

graph = 1;
if graph
    n_bins = 1000;
    start = floor(min(output')*n_bins)/n_bins;
    ending = floor(max(output')*n_bins)/n_bins;
    bins = start:(ending-start)/n_bins:ending;
    hist(output',bins);
    h1 = findobj(gca,'Type','patch');
    set(h1,'FaceColor','r','EdgeColor','k'); % Alt blue: [.4,.7,1]
    hold on;
    hist(prune,bins);
    h2 = findobj(gca,'Type','patch');
    %set(h2,'FaceColor',[.4,.7,1],'EdgeColor','k');
    hold off;
end

fprintf('\nReading test data . . . \n')
tedata = csvread('test.csv',1,0);
fprintf('\nTest data loaded. \n')

testdatanum = tedata(:,1);
testinputs = tedata(:,2:end);

res = sim(net,testinputs');
res = double(res' >= E);

result = uint32([testdatanum,res]);

csvwrite('ann_result.csv',result)
function modelParameters = positionEstimatorTraining(trainingData)
% Train linear regression using a spike-count buffer

step = 20;          % must match test script
bufferBins = 10;    % 10 bins = 200ms (if each bin is 20ms)
initTime = 300;     % ignore first 300ms

numTrials = size(trainingData,1);

X = [];
Y = [];

for direction = 1:8
    for i = 1:numTrials
        spikes = trainingData(i,direction).spikes;        % 98 x T
        pos    = trainingData(i,direction).handPos(1:2,:);% 2 x T
        T      = size(spikes,2);

        for t = (initTime+step):step:T
            startBin = t - bufferBins + 1;
            if startBin < 1
                startBin = 1;
            end

            feat = sum(spikes(:, startBin:t), 2)'; % 1 x 98
            X = [X; feat];
            Y = [Y; pos(:,t)'];
        end
    end
end

Xb = [X ones(size(X,1),1)];

modelParameters.linearModelX = Xb \ Y(:,1);
modelParameters.linearModelY = Xb \ Y(:,2);

modelParameters.step = step;
modelParameters.bufferSize = bufferBins;

end
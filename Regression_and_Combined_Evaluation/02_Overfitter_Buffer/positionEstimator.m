function [decodedPosX, decodedPosY] = positionEstimator(testData, modelParameters)
% Predict position using buffered spike counts + linear regression

bufferBins = modelParameters.bufferSize;

T = size(testData.spikes, 2);
startBin = T - bufferBins + 1;
if startBin < 1
    startBin = 1;
end

feat = sum(testData.spikes(:, startBin:T), 2)';  % 1 x 98
featB = [feat 1];

decodedPosX = featB * modelParameters.linearModelX;
decodedPosY = featB * modelParameters.linearModelY;

end
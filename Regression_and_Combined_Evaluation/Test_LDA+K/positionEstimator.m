function [decodedPosX, decodedPosY, modelParameters] = positionEstimator(testData, modelParameters)
% Self-contained estimator:
% 1) manual LDA to classify direction once per trial
% 2) buffered ridge regression for (x,y) using that direction's model

trialId = double(testData.trialId);

% ---------- classify direction once per trial ----------
if isKey(modelParameters.trialDirMap, trialId)
    d = modelParameters.trialDirMap(trialId);
else
    spk = testData.spikes;                      % 98 x T
    Tend = min(size(spk,2), modelParameters.dirWindowEnd);
    featEarly = sum(spk(:,1:Tend), 2)';         % 1 x 98 (counts)

    % z-score with training stats
    Xn = (featEarly - modelParameters.mu) ./ modelParameters.sigma;

    % LDA scores
    scores = Xn * modelParameters.Wlda + modelParameters.clda';  % 1 x 8
    [~, d] = max(scores);

    modelParameters.trialDirMap(trialId) = d;
end

% ---------- regression feature (buffer) ----------
bufferBins = modelParameters.bufferBins;
T = size(testData.spikes, 2);
startBin = T - bufferBins + 1;
if startBin < 1
    startBin = 1;
end

feat = sum(testData.spikes(:, startBin:T), 2)';   % 1 x 98
featB = [feat 1];

decodedPosX = featB * modelParameters.wX{d};
decodedPosY = featB * modelParameters.wY{d};
end
#include "LossFunctions.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cmath>

// ============================================
// LOSS CONFIG IMPLEMENTATION
// ============================================

void LossConfig::validate() const {
    if (focalAlpha < 0.0f || focalAlpha > 1.0f) {
        throw std::invalid_argument("Focal alpha must be between 0 and 1");
    }
    
    if (focalGamma < 0.0f) {
        throw std::invalid_argument("Focal gamma must be non-negative");
    }
    
    if (labelSmoothing < 0.0f || labelSmoothing > 1.0f) {
        throw std::invalid_argument("Label smoothing must be between 0 and 1");
    }
    
    if (reduction != "mean" && reduction != "sum" && reduction != "none") {
        throw std::invalid_argument("Reduction must be 'mean', 'sum', or 'none'");
    }
}

// ============================================
// METRICS IMPLEMENTATION
// ============================================

void Metrics::calculateFromConfusionMatrix(const std::vector<std::vector<int>>& confusionMatrix, 
                                          int numClasses) {
    this->confusionMatrix = confusionMatrix;
    
    if (confusionMatrix.empty()) return;
    
    // Calculate overall accuracy
    int total = 0;
    int correct = 0;
    
    for (int i = 0; i < numClasses; ++i) {
        for (int j = 0; j < numClasses; ++j) {
            total += confusionMatrix[i][j];
            if (i == j) {
                correct += confusionMatrix[i][j];
            }
        }
    }
    
    if (total > 0) {
        accuracy = static_cast<float>(correct) / total;
    }
    
    // Calculate per-class metrics for multi-class
    if (numClasses > 2) {
        perClassPrecision.clear();
        perClassRecall.clear();
        perClassF1.clear();
        
        for (int i = 0; i < numClasses; ++i) {
            int tp = confusionMatrix[i][i];
            int fp = 0;
            int fn = 0;
            
            for (int j = 0; j < numClasses; ++j) {
                if (j != i) {
                    fp += confusionMatrix[j][i];  // Column i, row j
                    fn += confusionMatrix[i][j];  // Row i, column j
                }
            }
            
            float prec = (tp + fp > 0) ? static_cast<float>(tp) / (tp + fp) : 0.0f;
            float rec = (tp + fn > 0) ? static_cast<float>(tp) / (tp + fn) : 0.0f;
            float f1 = (prec + rec > 0) ? 2 * prec * rec / (prec + rec) : 0.0f;
            
            perClassPrecision[i] = prec;
            perClassRecall[i] = rec;
            perClassF1[i] = f1;
        }
        
        // Calculate macro-averaged metrics
        float sumPrecision = 0.0f, sumRecall = 0.0f, sumF1 = 0.0f;
        for (int i = 0; i < numClasses; ++i) {
            sumPrecision += perClassPrecision[i];
            sumRecall += perClassRecall[i];
            sumF1 += perClassF1[i];
        }
        
        precision = sumPrecision / numClasses;
        recall = sumRecall / numClasses;
        f1Score = sumF1 / numClasses;
    } 
    else {
        // Binary classification
        if (confusionMatrix.size() >= 2 && confusionMatrix[0].size() >= 2) {
            int tp = confusionMatrix[1][1];  // DEFECT predicted as DEFECT
            int tn = confusionMatrix[0][0];  // OK predicted as OK
            int fp = confusionMatrix[0][1];  // OK predicted as DEFECT
            int fn = confusionMatrix[1][0];  // DEFECT predicted as OK
            
            precision = (tp + fp > 0) ? static_cast<float>(tp) / (tp + fp) : 0.0f;
            recall = (tp + fn > 0) ? static_cast<float>(tp) / (tp + fn) : 0.0f;
            f1Score = (precision + recall > 0) ? 2 * precision * recall / (precision + recall) : 0.0f;
            specificity = (tn + fp > 0) ? static_cast<float>(tn) / (tn + fp) : 0.0f;
            npv = (tn + fn > 0) ? static_cast<float>(tn) / (tn + fn) : 0.0f;
            mcc = MetricsCalculator::calculateMCC(tp, tn, fp, fn);
        }
    }
}

void Metrics::reset() {
    loss = 0.0f;
    accuracy = 0.0f;
    precision = 0.0f;
    recall = 0.0f;
    f1Score = 0.0f;
    auc = 0.0f;
    specificity = 0.0f;
    npv = 0.0f;
    mcc = 0.0f;
    perClassPrecision.clear();
    perClassRecall.clear();
    perClassF1.clear();
    confusionMatrix.clear();
}

void Metrics::print(const std::string& prefix) const {
    if (!prefix.empty()) {
        std::cout << prefix << "\n";
    }
    
    std::cout << "  Loss: " << loss << "\n";
    std::cout << "  Accuracy: " << accuracy << "\n";
    std::cout << "  Precision: " << precision << "\n";
    std::cout << "  Recall: " << recall << "\n";
    std::cout << "  F1 Score: " << f1Score << "\n";
    std::cout << "  Specificity: " << specificity << "\n";
    std::cout << "  MCC: " << mcc << "\n";
    
    if (!perClassPrecision.empty()) {
        std::cout << "  Per-class metrics:\n";
        for (const auto& pair : perClassPrecision) {
            std::cout << "    Class " << pair.first << ": "
                      << "P=" << perClassPrecision.at(pair.first) << ", "
                      << "R=" << perClassRecall.at(pair.first) << ", "
                      << "F1=" << perClassF1.at(pair.first) << "\n";
        }
    }
}

std::map<std::string, float> Metrics::toMap() const {
    std::map<std::string, float> map;
    map["loss"] = loss;
    map["accuracy"] = accuracy;
    map["precision"] = precision;
    map["recall"] = recall;
    map["f1_score"] = f1Score;
    map["auc"] = auc;
    map["specificity"] = specificity;
    map["npv"] = npv;
    map["mcc"] = mcc;
    
    for (const auto& pair : perClassPrecision) {
        map["precision_class_" + std::to_string(pair.first)] = pair.second;
    }
    
    for (const auto& pair : perClassRecall) {
        map["recall_class_" + std::to_string(pair.first)] = pair.second;
    }
    
    for (const auto& pair : perClassF1) {
        map["f1_class_" + std::to_string(pair.first)] = pair.second;
    }
    
    return map;
}

void Metrics::update(const Metrics& other, float weight) {
    float alpha = 1.0f / (weight + 1.0f);
    loss = loss * (1.0f - alpha) + other.loss * alpha;
    accuracy = accuracy * (1.0f - alpha) + other.accuracy * alpha;
    precision = precision * (1.0f - alpha) + other.precision * alpha;
    recall = recall * (1.0f - alpha) + other.recall * alpha;
    f1Score = f1Score * (1.0f - alpha) + other.f1Score * alpha;
    auc = auc * (1.0f - alpha) + other.auc * alpha;
    specificity = specificity * (1.0f - alpha) + other.specificity * alpha;
    npv = npv * (1.0f - alpha) + other.npv * alpha;
    mcc = mcc * (1.0f - alpha) + other.mcc * alpha;
}

// ============================================
// BASE LOSS FUNCTION IMPLEMENTATION
// ============================================

BaseLossFunction::BaseLossFunction(const LossConfig& config) 
    : config_(config) {
    config_.validate();
}

const LossConfig& BaseLossFunction::getConfig() const {
    return config_;
}

LossType BaseLossFunction::getType() const {
    return config_.lossType;
}

torch::Tensor BaseLossFunction::applyClassWeights(const torch::Tensor& loss, 
                                                 const torch::Tensor& targets) const {
    if (!config_.useClassWeights || config_.classWeights.empty()) {
        return loss;
    }
    
    // Create weight tensor based on class weights
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(loss.device());
    torch::Tensor weights = torch::full_like(loss, 1.0f, options);
    
    for (size_t i = 0; i < config_.classWeights.size(); ++i) {
        torch::Tensor mask = (targets == static_cast<int64_t>(i));
        weights.masked_fill_(mask, config_.classWeights[i]);
    }
    
    return loss * weights;
}

torch::Tensor BaseLossFunction::applyLabelSmoothing(const torch::Tensor& targets, 
                                                   int numClasses) const {
    if (config_.labelSmoothing <= 0.0f) {
        return targets;
    }
    
    // Convert targets to one-hot encoding
    auto oneHot = torch::nn::functional::one_hot(targets, numClasses)
        .to(torch::kFloat32);
    
    // Apply label smoothing
    float smoothValue = config_.labelSmoothing / numClasses;
    float keepValue = 1.0f - config_.labelSmoothing + smoothValue;
    
    return oneHot * keepValue + smoothValue;
}

// ============================================
// CROSS ENTROPY LOSS IMPLEMENTATION
// ============================================

CrossEntropyLoss::CrossEntropyLoss(const LossConfig& config, int numClasses)
    : BaseLossFunction(config), numClasses_(numClasses) {}

torch::Tensor CrossEntropyLoss::forward(const torch::Tensor& predictions, 
                                       const torch::Tensor& targets) {
    // Apply label smoothing if enabled
    torch::Tensor smoothedTargets = targets;
    if (config_.labelSmoothing > 0.0f) {
        smoothedTargets = applyLabelSmoothing(targets, numClasses_);
    }
    
    // Calculate cross-entropy loss
    auto loss = torch::nn::functional::cross_entropy(
        predictions, 
        targets, 
        torch::nn::CrossEntropyLossOptions()
            .reduction(torch::kNone)
    );
    
    // Apply class weights if enabled
    loss = applyClassWeights(loss, targets);
    
    // Apply reduction
    if (config_.reduction == "mean") {
        return loss.mean();
    } else if (config_.reduction == "sum") {
        return loss.sum();
    } else {
        return loss;
    }
}

// ============================================
// BINARY CROSS ENTROPY LOSS IMPLEMENTATION
// ============================================

BinaryCrossEntropyLoss::BinaryCrossEntropyLoss(const LossConfig& config)
    : BaseLossFunction(config) {}

torch::Tensor BinaryCrossEntropyLoss::forward(const torch::Tensor& predictions, 
                                             const torch::Tensor& targets) {
    // Apply label smoothing if enabled
    torch::Tensor smoothedTargets = targets;
    if (config_.labelSmoothing > 0.0f) {
        // For binary classification, numClasses = 2
        smoothedTargets = applyLabelSmoothing(targets, 2);
    }
    
    // Calculate binary cross-entropy loss
    auto loss = torch::nn::functional::binary_cross_entropy_with_logits(
        predictions, 
        smoothedTargets.to(predictions.dtype()),
        torch::nn::BCEWithLogitsLossOptions()
            .reduction(torch::kNone)
    );
    
    // Apply class weights if enabled
    loss = applyClassWeights(loss, targets);
    
    // Apply reduction
    if (config_.reduction == "mean") {
        return loss.mean();
    } else if (config_.reduction == "sum") {
        return loss.sum();
    } else {
        return loss;
    }
}

// ============================================
// FOCAL LOSS IMPLEMENTATION
// ============================================

FocalLoss::FocalLoss(const LossConfig& config, int numClasses)
    : BaseLossFunction(config), numClasses_(numClasses) {}

torch::Tensor FocalLoss::forward(const torch::Tensor& predictions, 
                                const torch::Tensor& targets) {
    // Get class probabilities using softmax
    auto probs = torch::softmax(predictions, -1);
    
    // Compute focal loss
    auto loss = computeFocalLoss(probs, targets);
    
    // Apply class weights if enabled
    loss = applyClassWeights(loss, targets);
    
    // Apply reduction
    if (config_.reduction == "mean") {
        return loss.mean();
    } else if (config_.reduction == "sum") {
        return loss.sum();
    } else {
        return loss;
    }
}

torch::Tensor FocalLoss::computeFocalLoss(const torch::Tensor& probs, 
                                         const torch::Tensor& targets) const {
    // Gather probabilities for the target classes
    auto batchSize = targets.size(0);
    auto indices = targets.unsqueeze(1);
    auto targetProbs = probs.gather(1, indices).squeeze(1);
    
    // Compute modulating factor
    auto modulatingFactor = torch::pow(1.0 - targetProbs, config_.focalGamma);
    
    // Compute alpha factor
    torch::Tensor alphaFactor = torch::ones_like(targetProbs);
    if (config_.focalAlpha > 0.0f) {
        auto alpha = torch::full_like(targetProbs, config_.focalAlpha);
        auto oneMinusAlpha = torch::full_like(targetProbs, 1.0f - config_.focalAlpha);
        alphaFactor = torch::where(targets == 1, alpha, oneMinusAlpha);
    }
    
    // Compute focal loss
    auto logProbs = torch::log(targetProbs + 1e-8f);
    auto focalLoss = -alphaFactor * modulatingFactor * logProbs;
    
    return focalLoss;
}

// ============================================
// DICE LOSS IMPLEMENTATION
// ============================================

DiceLoss::DiceLoss(const LossConfig& config)
    : BaseLossFunction(config) {}

torch::Tensor DiceLoss::forward(const torch::Tensor& predictions,
                               const torch::Tensor& targets) {
    // For binary segmentation, use sigmoid; for multi-class, use softmax
    torch::Tensor probs;
    if (predictions.sizes().size() > 1 && predictions.size(1) > 1) {
        probs = torch::softmax(predictions, 1);
    } else {
        probs = torch::sigmoid(predictions);
    }
    
    // Compute Dice coefficient
    auto loss = 1.0 - diceCoefficient(probs, targets.to(probs.dtype()));
    
    // Apply class weights if enabled
    if (config_.useClassWeights && !config_.classWeights.empty()) {
        loss = applyClassWeights(loss, targets);
    }
    
    // Apply reduction
    if (config_.reduction == "mean") {
        return loss.mean();
    } else if (config_.reduction == "sum") {
        return loss.sum();
    } else {
        return loss;
    }
}

torch::Tensor DiceLoss::diceCoefficient(const torch::Tensor& predictions,
                                       const torch::Tensor& targets,
                                       float smooth) const {
    // Compute intersection and union
    auto intersection = (predictions * targets).sum();
    auto unionTensor = (predictions + targets).sum();
    
    // Compute Dice coefficient
    auto dice = (2.0 * intersection + smooth) / (unionTensor + smooth);
    
    return dice;
}

// ============================================
// LOSS FUNCTION FACTORY IMPLEMENTATION
// ============================================

std::shared_ptr<BaseLossFunction> LossFunctionFactory::create(const LossConfig& config, 
                                                             int numClasses) {
    switch (config.lossType) {
        case LossType::CROSS_ENTROPY:
            return std::make_shared<CrossEntropyLoss>(config, numClasses);
        
        case LossType::BINARY_CROSS_ENTROPY:
            return std::make_shared<BinaryCrossEntropyLoss>(config);
        
        case LossType::FOCAL_LOSS:
            return std::make_shared<FocalLoss>(config, numClasses);
        
        case LossType::DICE_LOSS:
            return std::make_shared<DiceLoss>(config);
        
        case LossType::MSE:
            // For MSE, we use PyTorch's built-in MSELoss wrapped
            return std::make_shared<CrossEntropyLoss>(config, numClasses); // Fallback
        
        case LossType::NLL_LOSS:
            // For NLL, use CrossEntropyLoss which includes log softmax
            return std::make_shared<CrossEntropyLoss>(config, numClasses);
        
        default:
            throw std::invalid_argument("Unsupported loss type");
    }
}

LossConfig LossFunctionFactory::getDefaultConfig(LossType lossType) {
    LossConfig config;
    config.lossType = lossType;
    
    switch (lossType) {
        case LossType::FOCAL_LOSS:
            config.focalAlpha = 0.25f;
            config.focalGamma = 2.0f;
            break;
            
        case LossType::CROSS_ENTROPY:
        case LossType::BINARY_CROSS_ENTROPY:
        case LossType::NLL_LOSS:
            config.labelSmoothing = 0.1f; // Small label smoothing by default
            break;
        
        case LossType::MSE:
            config.labelSmoothing = 0.0f; // No label smoothing for MSE
            break;
            
        default:
            break;
    }
    
    return config;
}

// ============================================
// METRICS CALCULATOR IMPLEMENTATION
// ============================================

MetricsCalculator::MetricsCalculator(int numClasses) 
    : numClasses_(numClasses) {}

Metrics MetricsCalculator::calculate(const torch::Tensor& predictions, 
                                    const torch::Tensor& targets,
                                    float loss) {
    Metrics metrics;
    metrics.loss = loss;
    
    // Get predicted classes
    torch::Tensor predClasses;
    if (predictions.sizes().size() > 1 && predictions.size(1) > 1) {
        // Multi-class: use argmax
        predClasses = torch::argmax(predictions, 1);
    } else {
        // Binary: use sigmoid and threshold 0.5
        auto probs = torch::sigmoid(predictions).squeeze();
        predClasses = (probs > 0.5).to(torch::kLong);
    }
    
    // Calculate confusion matrix
    auto confusionMatrix = calculateConfusionMatrix(predClasses, targets);
    
    // Calculate all metrics from confusion matrix
    metrics.calculateFromConfusionMatrix(confusionMatrix, numClasses_);
    
    // Calculate AUC for binary classification
    if (numClasses_ == 2) {
        try {
            metrics.auc = calculateAUC(predictions, targets);
        } catch (const std::exception& e) {
            std::cerr << "[Warning] Failed to calculate AUC: " << e.what() << std::endl;
            metrics.auc = 0.0f;
        }
    }
    
    return metrics;
}

std::vector<std::vector<int>> MetricsCalculator::calculateConfusionMatrix(
    const torch::Tensor& predictions,
    const torch::Tensor& targets) const {
    
    std::vector<std::vector<int>> confusionMatrix(numClasses_, 
                                                 std::vector<int>(numClasses_, 0));
    
    // Ensure tensors are on CPU and are 1D
    auto predCpu = predictions.to(torch::kCPU);
    auto targetCpu = targets.to(torch::kCPU);
    
    // Convert to vectors for easier processing
    auto predAccessor = predCpu.accessor<int64_t, 1>();
    auto targetAccessor = targetCpu.accessor<int64_t, 1>();
    
    size_t n = predCpu.size(0);
    
    for (size_t i = 0; i < n; ++i) {
        int pred = static_cast<int>(predAccessor[i]);
        int target = static_cast<int>(targetAccessor[i]);
        
        if (pred >= 0 && pred < numClasses_ && 
            target >= 0 && target < numClasses_) {
            confusionMatrix[target][pred]++; // Note: [true][predicted]
        }
    }
    
    return confusionMatrix;
}

float MetricsCalculator::calculatePrecision(int tp, int fp) {
    return (tp + fp > 0) ? static_cast<float>(tp) / (tp + fp) : 0.0f;
}

float MetricsCalculator::calculateRecall(int tp, int fn) {
    return (tp + fn > 0) ? static_cast<float>(tp) / (tp + fn) : 0.0f;
}

float MetricsCalculator::calculateF1Score(float precision, float recall) {
    return (precision + recall > 0) ? 2 * precision * recall / (precision + recall) : 0.0f;
}

float MetricsCalculator::calculateAccuracy(int correct, int total) {
    return (total > 0) ? static_cast<float>(correct) / total : 0.0f;
}

float MetricsCalculator::calculateAUC(const torch::Tensor& predictions,
                                     const torch::Tensor& targets) {
    // Simple AUC calculation using trapezoidal rule
    // This is a simplified implementation
    
    auto predCpu = predictions.to(torch::kCPU).contiguous();
    auto targetCpu = targets.to(torch::kCPU).contiguous();
    
    // Get probabilities for positive class
    torch::Tensor probs;
    if (predCpu.sizes().size() > 1 && predCpu.size(1) > 1) {
        // Multi-class: use softmax
        probs = torch::softmax(predCpu, 1);
        probs = probs.select(1, 1); // Probability of class 1 (DEFECT)
    } else {
        // Binary: use sigmoid
        probs = torch::sigmoid(predCpu).squeeze();
    }
    
    // Sort by probability
    auto sortedIndices = torch::argsort(probs, -1, true); // Descending
    auto sortedProbs = probs.index_select(0, sortedIndices);
    auto sortedTargets = targetCpu.index_select(0, sortedIndices);
    
    auto probsAccessor = sortedProbs.accessor<float, 1>();
    auto targetsAccessor = sortedTargets.accessor<int64_t, 1>();
    
    size_t n = sortedProbs.size(0);
    if (n == 0) return 0.5f;
    
    // Count positive and negative samples
    int totalPos = 0, totalNeg = 0;
    for (size_t i = 0; i < n; ++i) {
        if (targetsAccessor[i] == 1) totalPos++;
        else totalNeg++;
    }
    
    if (totalPos == 0 || totalNeg == 0) return 0.5f;
    
    // Calculate AUC using trapezoidal rule
    float auc = 0.0f;
    int currentPos = 0, currentNeg = 0;
    float prevFPR = 0.0f, prevTPR = 0.0f;
    
    for (size_t i = 0; i < n; ++i) {
        if (targetsAccessor[i] == 1) {
            currentPos++;
        } else {
            currentNeg++;
        }
        
        float tpr = static_cast<float>(currentPos) / totalPos;
        float fpr = static_cast<float>(currentNeg) / totalNeg;
        
        // Add trapezoid area
        auc += (fpr - prevFPR) * (tpr + prevTPR) / 2.0f;
        
        prevFPR = fpr;
        prevTPR = tpr;
    }
    
    // Add remaining area if needed
    if (currentNeg < totalNeg) {
        float tpr = 1.0f;
        float fpr = 1.0f;
        auc += (fpr - prevFPR) * (tpr + prevTPR) / 2.0f;
    }
    
    return auc;
}

float MetricsCalculator::calculateMCC(int tp, int tn, int fp, int fn) {
    float numerator = static_cast<float>(tp * tn - fp * fn);
    float denominator = std::sqrt(static_cast<float>((tp + fp) * (tp + fn) * 
                                                    (tn + fp) * (tn + fn)));
    
    if (denominator == 0.0f) return 0.0f;
    return numerator / denominator;
}

void MetricsCalculator::reset() {
    // Nothing to reset for this implementation
}

int MetricsCalculator::getNumClasses() const {
    return numClasses_;
}

// ============================================
// TRAINING HISTORY IMPLEMENTATION
// ============================================

TrainingHistory::TrainingHistory(size_t maxEntries) 
    : maxEntries_(maxEntries) {}

void TrainingHistory::addEpoch(int epoch, const Metrics& trainMetrics, 
                              const Metrics& valMetrics) {
    history_[epoch] = std::make_pair(trainMetrics, valMetrics);
    trimHistory();
}

std::pair<Metrics, Metrics> TrainingHistory::getEpoch(int epoch) const {
    auto it = history_.find(epoch);
    if (it != history_.end()) {
        return it->second;
    }
    return std::make_pair(Metrics(), Metrics());
}

const std::map<int, std::pair<Metrics, Metrics>>& TrainingHistory::getAllEpochs() const {
    return history_;
}

int TrainingHistory::getLatestEpoch() const {
    if (history_.empty()) return -1;
    return history_.rbegin()->first;
}

int TrainingHistory::getBestEpoch(const std::string& metricName, 
                                 bool maximize) const {
    if (history_.empty()) return -1;
    
    int bestEpoch = history_.begin()->first;
    float bestValue = getMetricValue(history_.begin()->second.second, metricName);
    
    for (const auto& pair : history_) {
        float currentValue = getMetricValue(pair.second.second, metricName);
        
        if ((maximize && currentValue > bestValue) ||
            (!maximize && currentValue < bestValue)) {
            bestValue = currentValue;
            bestEpoch = pair.first;
        }
    }
    
    return bestEpoch;
}

bool TrainingHistory::hasImproved(const Metrics& currentMetrics, 
                                 const Metrics& previousMetrics,
                                 const std::string& metricName,
                                 bool maximize,
                                 float minDelta) const {
    float currentValue = getMetricValue(currentMetrics, metricName);
    float previousValue = getMetricValue(previousMetrics, metricName);
    
    if (maximize) {
        return currentValue > previousValue + minDelta;
    } else {
        return currentValue < previousValue - minDelta;
    }
}

void TrainingHistory::saveToCSV(const std::string& filePath) const {
    std::ofstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filePath);
    }
    
    // Write header
    file << "epoch,train_loss,train_accuracy,train_precision,train_recall,train_f1,"
         << "val_loss,val_accuracy,val_precision,val_recall,val_f1,val_auc\n";
    
    // Write data
    for (const auto& pair : history_) {
        int epoch = pair.first;
        const auto& train = pair.second.first;
        const auto& val = pair.second.second;
        
        file << epoch << ","
             << train.loss << ","
             << train.accuracy << ","
             << train.precision << ","
             << train.recall << ","
             << train.f1Score << ","
             << val.loss << ","
             << val.accuracy << ","
             << val.precision << ","
             << val.recall << ","
             << val.f1Score << ","
             << val.auc << "\n";
    }
    
    std::cout << "[TrainingHistory] Saved " << history_.size() 
              << " epochs to: " << filePath << std::endl;
}

void TrainingHistory::loadFromCSV(const std::string& filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filePath);
    }
    
    history_.clear();
    
    std::string line;
    std::getline(file, line); // Skip header
    
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        
        std::vector<std::string> tokens;
        while (std::getline(ss, token, ',')) {
            tokens.push_back(token);
        }
        
        if (tokens.size() >= 12) {
            int epoch = std::stoi(tokens[0]);
            
            Metrics trainMetrics;
            trainMetrics.loss = std::stof(tokens[1]);
            trainMetrics.accuracy = std::stof(tokens[2]);
            trainMetrics.precision = std::stof(tokens[3]);
            trainMetrics.recall = std::stof(tokens[4]);
            trainMetrics.f1Score = std::stof(tokens[5]);
            
            Metrics valMetrics;
            valMetrics.loss = std::stof(tokens[6]);
            valMetrics.accuracy = std::stof(tokens[7]);
            valMetrics.precision = std::stof(tokens[8]);
            valMetrics.recall = std::stof(tokens[9]);
            valMetrics.f1Score = std::stof(tokens[10]);
            valMetrics.auc = std::stof(tokens[11]);
            
            history_[epoch] = std::make_pair(trainMetrics, valMetrics);
        }
    }
    
    std::cout << "[TrainingHistory] Loaded " << history_.size() 
              << " epochs from: " << filePath << std::endl;
}

void TrainingHistory::clear() {
    history_.clear();
}

size_t TrainingHistory::size() const {
    return history_.size();
}

bool TrainingHistory::empty() const {
    return history_.empty();
}

void TrainingHistory::printSummary() const {
    if (history_.empty()) {
        std::cout << "Training history is empty" << std::endl;
        return;
    }
    
    std::cout << "\n=== TRAINING HISTORY SUMMARY ===\n";
    std::cout << "Total epochs: " << history_.size() << "\n";
    
    int bestEpoch = getBestEpoch("val_accuracy", true);
    if (bestEpoch != -1) {
        auto bestMetrics = getEpoch(bestEpoch);
        std::cout << "Best epoch: " << bestEpoch 
                  << " (Val Accuracy: " << bestMetrics.second.accuracy 
                  << ", Val Loss: " << bestMetrics.second.loss << ")\n";
    }
    
    std::cout << "Latest epoch: " << getLatestEpoch() << "\n";
    
    // Print last epoch metrics
    auto latest = getEpoch(getLatestEpoch());
    std::cout << "\nLatest epoch metrics:\n";
    latest.first.print("  Training:");
    latest.second.print("  Validation:");
    
    std::cout << "================================\n" << std::endl;
}

std::pair<Metrics, Metrics> TrainingHistory::getAverageMetrics() const {
    Metrics avgTrain, avgVal;
    int count = 0;
    
    for (const auto& pair : history_) {
        avgTrain.update(pair.second.first, count);
        avgVal.update(pair.second.second, count);
        count++;
    }
    
    return std::make_pair(avgTrain, avgVal);
}

std::pair<std::vector<float>, std::vector<float>> TrainingHistory::getLearningCurves(
    const std::string& metricName) const {
    
    std::vector<float> trainCurve, valCurve;
    
    for (const auto& pair : history_) {
        trainCurve.push_back(getMetricValue(pair.second.first, metricName));
        valCurve.push_back(getMetricValue(pair.second.second, metricName));
    }
    
    return std::make_pair(trainCurve, valCurve);
}

void TrainingHistory::trimHistory() {
    if (maxEntries_ > 0 && history_.size() > maxEntries_) {
        // Remove oldest entries
        while (history_.size() > maxEntries_) {
            history_.erase(history_.begin());
        }
    }
}

float TrainingHistory::getMetricValue(const Metrics& metrics, 
                                     const std::string& metricName) const {
    if (metricName == "loss") return metrics.loss;
    if (metricName == "accuracy") return metrics.accuracy;
    if (metricName == "precision") return metrics.precision;
    if (metricName == "recall") return metrics.recall;
    if (metricName == "f1_score") return metrics.f1Score;
    if (metricName == "auc") return metrics.auc;
    if (metricName == "specificity") return metrics.specificity;
    if (metricName == "mcc") return metrics.mcc;
    
    // Default to accuracy
    return metrics.accuracy;
}
#include "../include/Evaluator.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>

const float Z_95 = 1.96f; // Z-score for 95% confidence interval

//helper struct implementation 

void EvaluationMetrics::print() const {
    std::cout << "\n========== EVALUATION METRICS ==========\n";
    std::cout << "Total Samples: " << totalSamples << "\n";
    std::cout << "\n--- Confusion Matrix Counts ---\n";
    std::cout << "True Positives  (TP): " << truePositives 
              << " (Defect → Defect)\n";
    std::cout << "False Positives (FP): " << falsePositives 
              << " (OK → Defect) - FALSE ALARM!\n";
    std::cout << "True Negatives  (TN): " << trueNegatives 
              << " (OK → OK)\n";
    std::cout << "False Negatives (FN): " << falseNegatives 
              << " (Defect → OK) - MISSED DEFECT!\n";
    
    std::cout << "\n--- Performance Metrics ---\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Accuracy:    " << accuracy << "\n";
    std::cout << "Precision:   " << precision 
              << " (Reliability of defect alarms)\n";
    std::cout << "Recall:      " << recall 
              << " ← MOST IMPORTANT! (Defect catch rate)\n";
    std::cout << "F1-Score:    " << f1Score << "\n";
    std::cout << "Specificity: " << specificity 
              << " (OK identification rate)\n";
    
    std::cout << "\n--- Error Rates ---\n";
    std::cout << "False Positive Rate: " << falsePositiveRate 
              << " (False alarm rate)\n";
    std::cout << "False Negative Rate: " << falseNegativeRate 
              << " ← CRITICAL! (Defect escape rate)\n";
    
    if (auc > 0) {
        std::cout << "AUC: " << auc << " (Area Under ROC Curve)\n";
    }
    
    std::cout << "========================================\n";
}


std::map<std::string, std::string> EvaluationMetrics::toMap() const {
    std::map<std::string, std::string> metricsMap;
    
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);
    
    metricsMap["total_samples"] = std::to_string(totalSamples);
    metricsMap["true_positives"] = std::to_string(truePositives);
    metricsMap["false_positives"] = std::to_string(falsePositives);
    metricsMap["true_negatives"] = std::to_string(trueNegatives);
    metricsMap["false_negatives"] = std::to_string(falseNegatives);
    
    oss.str(""); oss << accuracy;
    metricsMap["accuracy"] = oss.str();
    
    oss.str(""); oss << precision;
    metricsMap["precision"] = oss.str();
    
    oss.str(""); oss << recall;
    metricsMap["recall"] = oss.str();
    
    oss.str(""); oss << f1Score;
    metricsMap["f1_score"] = oss.str();
    
    oss.str(""); oss << specificity;
    metricsMap["specificity"] = oss.str();
    
    oss.str(""); oss << falsePositiveRate;
    metricsMap["false_positive_rate"] = oss.str();
    
    oss.str(""); oss << falseNegativeRate;
    metricsMap["false_negative_rate"] = oss.str();
    
    if (auc > 0) {
        oss.str(""); oss << auc;
        metricsMap["auc"] = oss.str();
    }
    
    return metricsMap;
}

std::string ConfusionMatrix::toString() const {
    std::ostringstream oss;
    
    oss << "\n        Predicted\n";
    oss << "        OK     DEFECT\n";
    oss << "      ┌───────┬───────┐\n";
    oss << "  OK  │ " << std::setw(5) << matrix[0][0] 
         << " │ " << std::setw(5) << matrix[0][1] << " │\n";
    oss << "      ├───────┼───────┤\n";
    oss << "DEFECT│ " << std::setw(5) << matrix[1][0] 
         << " │ " << std::setw(5) << matrix[1][1] << " │\n";
    oss << "      └───────┴───────┘\n";
    
    oss << "\nInterpretation:\n";
    oss << "  Top-Left: " << matrix[0][0] << " OK correctly identified\n";
    oss << "  Top-Right: " << matrix[0][1] << " OK called defect (False Alarm)\n";
    oss << "  Bottom-Left: " << matrix[1][0] << " Defect missed (CRITICAL ERROR!)\n";
    oss << "  Bottom-Right: " << matrix[1][1] << " Defect correctly caught\n";
    
    return oss.str();
}

int ConfusionMatrix::total() const {
    return matrix[0][0] + matrix[0][1] + matrix[1][0] + matrix[1][1];
}

//Evaluator class implementation
Evaluator::Evaluator(float defectThreshold) 
    : defectThreshold_(defectThreshold) {
    std::cout << "[Evaluator] Initialized with threshold: " 
              << defectThreshold_ << std::endl;
}

Evaluator::~Evaluator() {
    std::cout << "[Evaluator] Destroyed with " << results_.size() 
              << " results collected" << std::endl;
}

//data collection methods
void Evaluator::addResult(int trueLabel, int predictedLabel, 
                          float confidence, const std::string& imagePath) {
    // Validate labels
    if (trueLabel < 0 || trueLabel > 1 || predictedLabel < 0 || predictedLabel > 1) {
        std::cerr << "[Warning] Invalid labels: true=" << trueLabel 
                  << ", pred=" << predictedLabel << std::endl;
        return;
    }
    
    results_.emplace_back(trueLabel, predictedLabel, confidence, imagePath);
}
void Evaluator::addResultWithProbability(int trueLabel, float defectProbability,
                                         const std::string& imagePath) {
    if (defectProbability < 0.0f || defectProbability > 1.0f) {
        std::cerr << "[Warning] Invalid probability: " << defectProbability << std::endl;
        return;
    }
    
    int predictedLabel = probabilityToLabel(defectProbability);
    results_.emplace_back(trueLabel, predictedLabel, defectProbability, imagePath);
}

void Evaluator::addBatchResults(const std::vector<int>& trueLabels,
                                const std::vector<int>& predictedLabels) {
    if (trueLabels.size() != predictedLabels.size()) {
        std::cerr << "[Error] Mismatched batch sizes: " 
                  << trueLabels.size() << " vs " << predictedLabels.size() << std::endl;
        return;
    }
    
    for (size_t i = 0; i < trueLabels.size(); ++i) {
        addResult(trueLabels[i], predictedLabels[i], 0.0f);
    }
}

void Evaluator::clearResults() {
    results_.clear();
    std::cout << "[Evaluator] All results cleared" << std::endl;
}

//metrics computation helper
EvaluationMetrics Evaluator::computeMetrics() const {
    if (results_.empty()) {
        std::cerr << "[Warning] No results to evaluate" << std::endl;
        return EvaluationMetrics();
    }
    
    // Count confusion matrix elements
    int tp = 0, fp = 0, tn = 0, fn = 0;
    
    for (const auto& result : results_) {
        if (result.trueLabel == 1 && result.predictedLabel == 1) tp++;
        else if (result.trueLabel == 0 && result.predictedLabel == 1) fp++;
        else if (result.trueLabel == 0 && result.predictedLabel == 0) tn++;
        else if (result.trueLabel == 1 && result.predictedLabel == 0) fn++;
    }
    
    return computeMetricsFromCounts(tp, fp, tn, fn);
}

ConfusionMatrix Evaluator::computeConfusionMatrix() const {
    ConfusionMatrix cm;
    
    for (const auto& result : results_) {
        if (result.trueLabel >= 0 && result.trueLabel < 2 &&
            result.predictedLabel >= 0 && result.predictedLabel < 2) {
            cm.matrix[result.trueLabel][result.predictedLabel]++;
        }
    }
    
    return cm;
}

EvaluationMetrics Evaluator::computeMetricsAtThreshold(float threshold) const {
    if (results_.empty()) {
        return EvaluationMetrics();
    }
    
    std::vector<ClassificationResult> thresholdedResults;
    
    for (const auto& result : results_) {
        ClassificationResult newResult = result;
        // Re-classify using new threshold
        newResult.predictedLabel = (result.confidence >= threshold) ? 1 : 0;
        thresholdedResults.push_back(newResult);
    }
    
    ConfusionMatrix cm = computeConfusionMatrixFromResults(thresholdedResults);
    
    int tp = cm.matrix[1][1];
    int fp = cm.matrix[0][1];
    int tn = cm.matrix[0][0];
    int fn = cm.matrix[1][0];
    
    EvaluationMetrics metrics = computeMetricsFromCounts(tp, fp, tn, fn);
    metrics.totalSamples = cm.total();
    
    return metrics;
}

// ============================================
// ADVANCED ANALYSIS
// ============================================

std::vector<std::pair<float, float>> Evaluator::computeROCCurve(int numThresholds) const {
    std::vector<std::pair<float, float>> rocPoints;
    
    if (results_.empty()) {
        return rocPoints;
    }
    
    // Generate thresholds from 0.0 to 1.0
    for (int i = 0; i <= numThresholds; ++i) {
        float threshold = static_cast<float>(i) / numThresholds;
        EvaluationMetrics metrics = computeMetricsAtThreshold(threshold);
        
        // Add point (False Positive Rate, True Positive Rate)
        rocPoints.emplace_back(metrics.falsePositiveRate, metrics.recall);
    }
    
    return rocPoints;
}

float Evaluator::computeAUC() const {
    auto rocPoints = computeROCCurve(20);
    if (rocPoints.size() < 2) {
        return 0.0f;
    }
    
    // Compute AUC using trapezoidal rule
    float auc = 0.0f;
    for (size_t i = 1; i < rocPoints.size(); ++i) {
        float width = rocPoints[i].first - rocPoints[i-1].first;
        float avgHeight = (rocPoints[i].second + rocPoints[i-1].second) / 2.0f;
        auc += width * avgHeight;
    }
    
    return auc;
}

float Evaluator::findOptimalThreshold(float start, float end, float step) const {
    if (results_.empty()) {
        return defectThreshold_;
    }
    
    float bestThreshold = defectThreshold_;
    float bestF1 = -1.0f;
    
    for (float t = start; t <= end; t += step) {
        EvaluationMetrics metrics = computeMetricsAtThreshold(t);
        if (metrics.f1Score > bestF1) {
            bestF1 = metrics.f1Score;
            bestThreshold = t;
        }
    }
    
    std::cout << "[Evaluator] Optimal threshold: " << bestThreshold 
              << " (F1-score: " << bestF1 << ")" << std::endl;
    return bestThreshold;
}

float Evaluator::findThresholdForRecall(float targetRecall) const {
    if (results_.empty() || targetRecall <= 0.0f || targetRecall >= 1.0f) {
        return defectThreshold_;
    }
    
    float bestThreshold = defectThreshold_;
    float minDiff = std::numeric_limits<float>::max();
    
    // Search thresholds from high to low (higher recall at lower thresholds)
    for (float t = 0.9f; t >= 0.1f; t -= 0.05f) {
        EvaluationMetrics metrics = computeMetricsAtThreshold(t);
        float diff = std::abs(metrics.recall - targetRecall);
        
        if (diff < minDiff) {
            minDiff = diff;
            bestThreshold = t;
        }
    }
    
    EvaluationMetrics finalMetrics = computeMetricsAtThreshold(bestThreshold);
    std::cout << "[Evaluator] Threshold for recall " << targetRecall 
              << ": " << bestThreshold << " (actual recall: " 
              << finalMetrics.recall << ")" << std::endl;
    
    return bestThreshold;
}

// ============================================
// REPORT GENERATION
// ============================================

std::string Evaluator::generateReport(const std::string& title) const {
    std::ostringstream oss;
    
    oss << "========================================\n";
    oss << "    " << title << "\n";
    oss << "========================================\n\n";
    
    // Basic info
    oss << "Evaluation Summary:\n";
    oss << "  Total samples: " << results_.size() << "\n";
    oss << "  Current threshold: " << defectThreshold_ << "\n\n";
    
    // Confusion Matrix
    ConfusionMatrix cm = computeConfusionMatrix();
    oss << cm.toString() << "\n";
    
    // Metrics
    EvaluationMetrics metrics = computeMetrics();
    oss << "Performance Metrics:\n";
    oss << std::fixed << std::setprecision(4);
    oss << "  Accuracy:    " << metrics.accuracy << "\n";
    oss << "  Precision:   " << metrics.precision << "\n";
    oss << "  Recall:      " << metrics.recall 
        << " ← Defect catch rate\n";
    oss << "  F1-Score:    " << metrics.f1Score << "\n";
    oss << "  Specificity: " << metrics.specificity << "\n";
    
    // Critical metrics for defect detection
    oss << "\n--- DEFECT DETECTION CRITICAL METRICS ---\n";
    oss << "  False Negative Rate: " << metrics.falseNegativeRate * 100 
        << "% (Defect escape rate)\n";
    oss << "  False Positive Rate: " << metrics.falsePositiveRate * 100 
        << "% (False alarm rate)\n";
    
    if (metrics.falseNegatives > 0) {
        oss << "\n⚠️  WARNING: " << metrics.falseNegatives 
            << " DEFECTS WERE MISSED! ⚠️\n";
    }
    
    // ROC/AUC if enough data
    if (results_.size() > 10) {
        float auc = computeAUC();
        oss << "\nModel Discriminative Power:\n";
        oss << "  AUC: " << auc << " ";
        if (auc >= 0.9) oss << "(Excellent)";
        else if (auc >= 0.8) oss << "(Good)";
        else if (auc >= 0.7) oss << "(Fair)";
        else oss << "(Poor)";
        oss << "\n";
    }
    
    oss << "\n========================================\n";
    
    return oss.str();
}

std::string Evaluator::generateConfusionMatrixString() const {
    ConfusionMatrix cm = computeConfusionMatrix();
    return cm.toString();
}

std::string Evaluator::generateMetricsSummary() const {
    EvaluationMetrics metrics = computeMetrics();
    
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    oss << "ACC: " << metrics.accuracy * 100 << "% | ";
    oss << "REC: " << metrics.recall * 100 << "% | ";
    oss << "PRE: " << metrics.precision * 100 << "% | ";
    oss << "F1: " << metrics.f1Score * 100 << "%";
    
    return oss.str();
}

bool Evaluator::saveToCSV(const std::string& filePath) const {
    std::ofstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "[Error] Failed to open file: " << filePath << std::endl;
        return false;
    }
    
    // Write header
    file << "true_label,predicted_label,confidence,image_path\n";
    
    // Write data
    for (const auto& result : results_) {
        file << result.trueLabel << "," 
             << result.predictedLabel << ","
             << result.confidence << ","
             << "\"" << result.imagePath << "\"\n";
    }
    
    file.close();
    std::cout << "[Evaluator] Saved " << results_.size() 
              << " results to: " << filePath << std::endl;
    
    return true;
}

bool Evaluator::saveDetailedResults(const std::string& filePath) const {
    std::ofstream file(filePath);
    if (!file.is_open()) {
        return false;
    }
    
    // Write metrics summary
    EvaluationMetrics metrics = computeMetrics();
    file << "Evaluation Metrics:\n";
    file << "===================\n";
    
    for (const auto& [key, value] : metrics.toMap()) {
        file << key << ": " << value << "\n";
    }
    
    file << "\n\nDetailed Results:\n";
    file << "================\n";
    file << "Index,True,Predicted,Confidence,Correct,ImagePath\n";
    
    for (size_t i = 0; i < results_.size(); ++i) {
        const auto& result = results_[i];
        bool correct = (result.trueLabel == result.predictedLabel);
        
        file << i << ","
             << result.trueLabel << ","
             << result.predictedLabel << ","
             << result.confidence << ","
             << (correct ? "YES" : "NO") << ","
             << result.imagePath << "\n";
    }
    
    file.close();
    return true;
}

// ============================================
// GETTERS & SETTERS
// ============================================

float Evaluator::getThreshold() const {
    return defectThreshold_;
}

void Evaluator::setThreshold(float threshold) {
    if (threshold < 0.0f || threshold > 1.0f) {
        std::cerr << "[Error] Invalid threshold: " << threshold 
                  << ". Must be between 0 and 1." << std::endl;
        return;
    }
    
    defectThreshold_ = threshold;
    std::cout << "[Evaluator] Threshold set to: " << defectThreshold_ << std::endl;
}

int Evaluator::getResultCount() const {
    return static_cast<int>(results_.size());
}

const std::vector<ClassificationResult>& Evaluator::getResults() const {
    return results_;
}

std::vector<ClassificationResult> Evaluator::getMisclassifiedResults() const {
    std::vector<ClassificationResult> misclassified;
    
    for (const auto& result : results_) {
        if (result.trueLabel != result.predictedLabel) {
            misclassified.push_back(result);
        }
    }
    
    return misclassified;
}

std::vector<ClassificationResult> Evaluator::getFalseNegatives() const {
    std::vector<ClassificationResult> fn;
    
    for (const auto& result : results_) {
        if (result.trueLabel == 1 && result.predictedLabel == 0) {
            fn.push_back(result);
        }
    }
    
    return fn;
}

std::vector<ClassificationResult> Evaluator::getFalsePositives() const {
    std::vector<ClassificationResult> fp;
    
    for (const auto& result : results_) {
        if (result.trueLabel == 0 && result.predictedLabel == 1) {
            fp.push_back(result);
        }
    }
    
    return fp;
}

// ============================================
// STATISTICAL METHODS
// ============================================

std::pair<float, float> Evaluator::computeConfidenceInterval(
    const std::string& metric, float confidenceLevel) const {
    
    if (results_.empty()) {
        return {0.0f, 0.0f};
    }
    
    EvaluationMetrics metrics = computeMetrics();
    float p = 0.0f;
    float n = static_cast<float>(results_.size());
    
    if (metric == "accuracy") p = metrics.accuracy;
    else if (metric == "recall") p = metrics.recall;
    else if (metric == "precision") p = metrics.precision;
    else if (metric == "f1_score") p = metrics.f1Score;
    else {
        std::cerr << "[Error] Unknown metric: " << metric << std::endl;
        return {0.0f, 0.0f};
    }
    
    // Z-score for confidence level (simplified - using 95% Z=1.96)
    float z = Z_95;
    
    // Standard error
    float se = std::sqrt(p * (1 - p) / n);
    
    // Margin of error
    float moe = z * se;
    
    return {std::max(0.0f, p - moe), std::min(1.0f, p + moe)};
}

float Evaluator::performMcNemarTest(const Evaluator& other) const {
    // Simplified McNemar's test for comparing two models
    // This would need paired results - for now, return placeholder
    std::cout << "[Evaluator] McNemar test requires paired predictions" << std::endl;
    return 1.0f;  // Placeholder
}

// ============================================
// PRIVATE HELPER METHODS
// ============================================

int Evaluator::probabilityToLabel(float probability) const {
    return (probability >= defectThreshold_) ? 1 : 0;
}

EvaluationMetrics Evaluator::computeMetricsFromCounts(int tp, int fp, int tn, int fn) const {
    EvaluationMetrics metrics;
    
    metrics.truePositives = tp;
    metrics.falsePositives = fp;
    metrics.trueNegatives = tn;
    metrics.falseNegatives = fn;
    metrics.totalSamples = tp + fp + tn + fn;
    
    if (metrics.totalSamples > 0) {
        // Basic metrics
        metrics.accuracy = static_cast<float>(tp + tn) / metrics.totalSamples;
        
        // Precision = TP / (TP + FP)
        if (tp + fp > 0) {
            metrics.precision = static_cast<float>(tp) / (tp + fp);
        }
        
        // Recall = TP / (TP + FN) - MOST IMPORTANT!
        if (tp + fn > 0) {
            metrics.recall = static_cast<float>(tp) / (tp + fn);
        }
        
        // F1-Score = 2 * (precision * recall) / (precision + recall)
        if (metrics.precision + metrics.recall > 0) {
            metrics.f1Score = 2.0f * (metrics.precision * metrics.recall) 
                            / (metrics.precision + metrics.recall);
        }
        
        // Specificity = TN / (TN + FP)
        if (tn + fp > 0) {
            metrics.specificity = static_cast<float>(tn) / (tn + fp);
        }
        
        // Error rates
        if (fp + tn > 0) {
            metrics.falsePositiveRate = static_cast<float>(fp) / (fp + tn);
        }
        
        if (fn + tp > 0) {
            metrics.falseNegativeRate = static_cast<float>(fn) / (fn + tp);
        }
    }
    
    return metrics;
}

ConfusionMatrix Evaluator::computeConfusionMatrixFromResults(
    const std::vector<ClassificationResult>& results) const {
    
    ConfusionMatrix cm;
    
    for (const auto& result : results) {
        if (result.trueLabel >= 0 && result.trueLabel < 2 &&
            result.predictedLabel >= 0 && result.predictedLabel < 2) {
            cm.matrix[result.trueLabel][result.predictedLabel]++;
        }
    }
    
    return cm;
}

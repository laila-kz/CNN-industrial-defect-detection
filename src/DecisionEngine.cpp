#include "../include/DecisionEngine.h"
#include <iostream>
#include <cmath>
#include <iomanip>
#include <sstream>

using namespace std;

// ============================================
// DecisionResult IMPLEMENTATION
// ============================================

DecisionResult::DecisionResult() 
    : finalDecision(Decision::UNCERTAIN),
      label("PENDING"),
      defectProbabilities(0.0f),
      confidence(0.0f),
      overThreshold(false),
      explanation("Decision not yet made") {
}

string DecisionResult::toString() const {
    stringstream ss;
    ss << fixed << setprecision(2);
    
    ss << "Decision: ";
    switch(finalDecision) {
        case Decision::OK: ss << "OK"; break;
        case Decision::DEFECT: ss << "DEFECT"; break;
        case Decision::UNCERTAIN: ss << "UNCERTAIN (Needs Review)"; break;
    }
    
    ss << " | Label: " << label;
    ss << " | Defect Probability: " << (defectProbabilities * 100) << "%";
    ss << " | Confidence: " << (confidence * 100) << "%";
    ss << " | Over Threshold: " << (overThreshold ? "YES" : "NO");
    
    if (!explanation.empty()) {
        ss << " | Explanation: " << explanation;
    }
    
    return ss.str();
}

vector<int> DecisionResult::getDisplayColor() const {
    // Returns BGR colors (OpenCV format)
    switch(finalDecision) {
        case Decision::OK:
            return {0, 255, 0};      // Green
        case Decision::DEFECT:
            return {0, 0, 255};      // Red
        case Decision::UNCERTAIN:
            return {0, 255, 255};    // Yellow
        default:
            return {255, 255, 255};  // White
    }
}

// ============================================
// DecisionConfig IMPLEMENTATION
// ============================================

DecisionConfig::DecisionConfig() 
    : defectThreshold(0.75f),
      uncertaintyMargin(0.10f),
      useUncertaintyThreshold(true),
      requireManualReview(false),
      manualReviewThreshold(0.60f),
      costFalseNegative(100.0f),
      costFalsePositive(20.0f),
      useCostOptimization(false) {
}

void DecisionConfig::validate() const {
    if (defectThreshold < 0.0f || defectThreshold > 1.0f) {
        throw invalid_argument("Defect threshold must be between 0.0 and 1.0");
    }
    
    if (uncertaintyMargin < 0.0f || uncertaintyMargin > 0.5f) {
        throw invalid_argument("Uncertainty margin must be between 0.0 and 0.5");
    }
    
    if (manualReviewThreshold < 0.0f || manualReviewThreshold > 1.0f) {
        throw invalid_argument("Manual review threshold must be between 0.0 and 1.0");
    }
    
    if (costFalseNegative <= 0.0f || costFalsePositive <= 0.0f) {
        throw invalid_argument("Costs must be positive values");
    }
    
    // Ensure uncertainty zone makes sense
    if (useUncertaintyThreshold) {
        float uncertaintyMin = defectThreshold - uncertaintyMargin;
        float uncertaintyMax = defectThreshold + uncertaintyMargin;
        
        if (uncertaintyMin < 0.0f || uncertaintyMax > 1.0f) {
            throw invalid_argument("Uncertainty zone extends beyond valid probability range");
        }
    }
}

float DecisionConfig::calculateOptimalThreshold() const {
    if (!useCostOptimization) {
        return defectThreshold;
    }
    
    // Optimal threshold formula: 
    // threshold = cost_FP / (cost_FN + cost_FP)
    float optimal = costFalsePositive / (costFalseNegative + costFalsePositive);
    
    cout << "[DecisionEngine] Cost-optimized threshold calculated: " << optimal << endl;
    cout << "  - False Negative Cost: " << costFalseNegative << endl;
    cout << "  - False Positive Cost: " << costFalsePositive << endl;
    cout << "  - Recommended Threshold: " << optimal << endl;
    
    return optimal;
}

// ============================================
// DecisionEngine IMPLEMENTATION
// ============================================

// Constructor 1: Default
DecisionEngine::DecisionEngine() 
    : config_(),
      totalDecisions_(0) {
    
    // Initialize decision counters
    decisionCounts_[Decision::OK] = 0;
    decisionCounts_[Decision::DEFECT] = 0;
    decisionCounts_[Decision::UNCERTAIN] = 0;
    
    cout << "[DecisionEngine] Initialized with default configuration" << endl;
}

// Constructor 2: With custom config
DecisionEngine::DecisionEngine(const DecisionConfig& config)
    : config_(config),
      totalDecisions_(0) {
    
    // Validate configuration
    config_.validate();
    
    // Initialize decision counters
    decisionCounts_[Decision::OK] = 0;
    decisionCounts_[Decision::DEFECT] = 0;
    decisionCounts_[Decision::UNCERTAIN] = 0;
    
    cout << "[DecisionEngine] Initialized with custom configuration" << endl;
    printConfig();
}

// Destructor
DecisionEngine::~DecisionEngine() {
    cout << "[DecisionEngine] Destroyed. Final statistics:" << endl;
    auto stats = getStatics();
    for (const auto& [key, value] : stats) {
        cout << "  - " << key << ": " << value << endl;
    }
}

// ============================================
// CONFIGURATION METHODS
// ============================================

void DecisionEngine::setConfig(const DecisionConfig& config) {
    config_.validate();
    config_ = config;
    cout << "[DecisionEngine] Configuration updated" << endl;
}

const DecisionConfig& DecisionEngine::getConfig() const {
    return config_;
}

void DecisionEngine::setDefectThreshold(float newThreshold) {
    if (newThreshold < 0.0f || newThreshold > 1.0f) {
        throw invalid_argument("Threshold must be between 0.0 and 1.0");
    }
    
    config_.defectThreshold = newThreshold;
    cout << "[DecisionEngine] Defect threshold set to: " << newThreshold << endl;
}

void DecisionEngine::setUncertaintyZone(bool enabled) {
    config_.useUncertaintyThreshold = enabled;
    cout << "[DecisionEngine] Uncertainty zone " 
         << (enabled ? "ENABLED" : "DISABLED") << endl;
}

// ============================================
// MAIN DECISION METHODS
// ============================================

DecisionResult DecisionEngine::makeDecision(float defectProbability) {
    // Validate input
    validateProbability(defectProbability);
    
    // Create result object
    DecisionResult result;
    result.defectProbabilities = defectProbability;
    result.confidence = calculateConfidence(defectProbability);
    result.overThreshold = isDefect(defectProbability);
    
    // Make preliminary decision
    Decision preliminaryDecision;
    if (result.overThreshold) {
        preliminaryDecision = Decision::DEFECT;
    } else {
        preliminaryDecision = Decision::OK;
    }
    
    // Apply business rules (no metadata in this simple version)
    Decision finalDecision = applyBusinessRules(defectProbability, preliminaryDecision);
    result.finalDecision = finalDecision;
    
    // Set label
    switch(finalDecision) {
        case Decision::OK:
            result.label = "OK";
            break;
        case Decision::DEFECT:
            result.label = "DEFECT";
            break;
        case Decision::UNCERTAIN:
            result.label = "REVIEW NEEDED";
            break;
    }
    
    // Generate explanation
    result.explanation = generateExplanation(defectProbability, finalDecision);
    
    // Update statistics
    updateStatistics(finalDecision);
    
    return result;
}

DecisionResult DecisionEngine::makeDecisionWithcontext(
    float defectProbability,
    const string& imageId,
    const map<string, string>& metadata) {
    
    // Start with basic decision
    DecisionResult result = makeDecision(defectProbability);
    
    // Add context information if available
    if (!imageId.empty()) {
        result.explanation += " | Image: " + imageId;
    }
    
    // Process metadata if any
    if (!metadata.empty()) {
        result.explanation += " | Context: ";
        for (const auto& [key, value] : metadata) {
            // Example: Check if this is a high-value product
            if (key == "product_value" && value == "high") {
                result.explanation += "(High-Value Product) ";
            }
        }
    }
    
    return result;
}

vector<DecisionResult> DecisionEngine::makeBatchDecision(
    const vector<float>& probabilities) {
    
    vector<DecisionResult> results;
    results.reserve(probabilities.size());
    
    cout << "[DecisionEngine] Processing batch of " << probabilities.size() << " items" << endl;
    
    for (size_t i = 0; i < probabilities.size(); i++) {
        try {
            DecisionResult result = makeDecision(probabilities[i]);
            
            // Add sequence number for tracking
            result.explanation += " | Batch Item #" + to_string(i + 1);
            
            results.push_back(result);
            
        } catch (const exception& e) {
            cout << "[Warning] Error processing item " << i << ": " << e.what() << endl;
            
            // Create error result
            DecisionResult errorResult;
            errorResult.finalDecision = Decision::UNCERTAIN;
            errorResult.label = "ERROR";
            errorResult.defectProbabilities = probabilities[i];
            errorResult.confidence = 0.0f;
            errorResult.overThreshold = false;
            errorResult.explanation = "Processing error: " + string(e.what());
            
            results.push_back(errorResult);
        }
    }
    
    return results;
}

// ============================================
// BUSINESS RULE METHODS
// ============================================

bool DecisionEngine::isDefect(float probability) const {
    validateProbability(probability);
    
    // Use cost-optimized threshold if enabled
    float threshold = config_.useCostOptimization 
                    ? config_.calculateOptimalThreshold() 
                    : config_.defectThreshold;
    
    return probability >= threshold;
}

bool DecisionEngine::requiresReview(float probability) const {
    validateProbability(probability);
    
    if (!config_.useUncertaintyThreshold) {
        return false;
    }
    
    float lowerBound = config_.defectThreshold - config_.uncertaintyMargin;
    float upperBound = config_.defectThreshold + config_.uncertaintyMargin;
    
    // Check if probability is in the uncertainty zone
    return (probability >= lowerBound && probability < upperBound);
}

float DecisionEngine::calculateConfidence(float probability) const {
    validateProbability(probability);
    
    // Confidence is how far we are from the decision boundary
    float distanceFromThreshold = abs(probability - config_.defectThreshold);
    
    // Normalize to [0, 1] range
    // Maximum distance is 0.5 (from 0.0 or 1.0 to threshold at 0.5)
    float maxDistance = 0.5f;
    float normalizedDistance = distanceFromThreshold / maxDistance;
    
    // Clamp to [0, 1]
    if (normalizedDistance > 1.0f) normalizedDistance = 1.0f;
    if (normalizedDistance < 0.0f) normalizedDistance = 0.0f;
    
    return normalizedDistance;
}

string DecisionEngine::generateExplanation(float probability, Decision decision) const {
    stringstream ss;
    ss << fixed << setprecision(2);
    
    ss << "Probability: " << (probability * 100) << "%. ";
    
    switch(decision) {
        case Decision::OK:
            ss << "Below defect threshold (" << (config_.defectThreshold * 100) 
               << "%). Product is ACCEPTED.";
            break;
            
        case Decision::DEFECT:
            ss << "Above defect threshold (" << (config_.defectThreshold * 100) 
               << "%). Product is REJECTED.";
            
            if (config_.useCostOptimization) {
                ss << " Cost-optimized decision.";
            }
            break;
            
        case Decision::UNCERTAIN:
            ss << "Within uncertainty zone [" 
               << ((config_.defectThreshold - config_.uncertaintyMargin) * 100) << "%-"
               << ((config_.defectThreshold + config_.uncertaintyMargin) * 100) 
               << "%]. Manual review recommended.";
            break;
    }
    
    return ss.str();
}

// ============================================
// STATISTICS & MONITORING
// ============================================

map<string, int> DecisionEngine::getStatics() const {
    map<string, int> stats;
    
    stats["Total Decisions"] = totalDecisions_;
    stats["OK Decisions"] = decisionCounts_.at(Decision::OK);
    stats["DEFECT Decisions"] = decisionCounts_.at(Decision::DEFECT);
    stats["UNCERTAIN Decisions"] = decisionCounts_.at(Decision::UNCERTAIN);
    
    // Calculate percentages
    if (totalDecisions_ > 0) {
        stats["OK Percentage"] = static_cast<int>(
            (decisionCounts_.at(Decision::OK) * 100.0f) / totalDecisions_);
        stats["DEFECT Percentage"] = static_cast<int>(
            (decisionCounts_.at(Decision::DEFECT) * 100.0f) / totalDecisions_);
        stats["UNCERTAIN Percentage"] = static_cast<int>(
            (decisionCounts_.at(Decision::UNCERTAIN) * 100.0f) / totalDecisions_);
    } else {
        stats["OK Percentage"] = 0;
        stats["DEFECT Percentage"] = 0;
        stats["UNCERTAIN Percentage"] = 0;
    }
    
    return stats;
}

void DecisionEngine::resetStatics() {
    decisionCounts_[Decision::OK] = 0;
    decisionCounts_[Decision::DEFECT] = 0;
    decisionCounts_[Decision::UNCERTAIN] = 0;
    totalDecisions_ = 0;
    
    cout << "[DecisionEngine] Statistics reset" << endl;
}

void DecisionEngine::printConfig() const {
    cout << "\n=== DecisionEngine Configuration ===" << endl;
    cout << fixed << setprecision(3);
    cout << "Defect Threshold: " << config_.defectThreshold << endl;
    cout << "Uncertainty Zone: " << (config_.useUncertaintyThreshold ? "ON" : "OFF") << endl;
    
    if (config_.useUncertaintyThreshold) {
        cout << "Uncertainty Margin: ±" << config_.uncertaintyMargin << endl;
        cout << "Uncertainty Range: [" 
             << (config_.defectThreshold - config_.uncertaintyMargin) << ", "
             << (config_.defectThreshold + config_.uncertaintyMargin) << "]" << endl;
    }
    
    cout << "Manual Review: " << (config_.requireManualReview ? "REQUIRED" : "NOT REQUIRED") << endl;
    if (config_.requireManualReview) {
        cout << "Manual Review Threshold: " << config_.manualReviewThreshold << endl;
    }
    
    cout << "Cost Optimization: " << (config_.useCostOptimization ? "ENABLED" : "DISABLED") << endl;
    if (config_.useCostOptimization) {
        cout << "  - False Negative Cost: " << config_.costFalseNegative << endl;
        cout << "  - False Positive Cost: " << config_.costFalsePositive << endl;
        cout << "  - Optimal Threshold: " << config_.calculateOptimalThreshold() << endl;
    }
    cout << "==================================\n" << endl;
}

// ============================================
// STATIC METHODS
// ============================================

float DecisionEngine::calculateOptimalThreshold(
    float falseNegativeCost,
    float falsePositiveCost,
    float defectPrevalence) {
    
    if (falseNegativeCost <= 0 || falsePositiveCost <= 0) {
        throw invalid_argument("Costs must be positive");
    }
    
    if (defectPrevalence < 0.0f || defectPrevalence > 1.0f) {
        throw invalid_argument("Defect prevalence must be between 0.0 and 1.0");
    }
    
    // Bayes-optimal threshold formula:
    // threshold = (cost_FP * (1 - prevalence)) / 
    //             (cost_FN * prevalence + cost_FP * (1 - prevalence))
    
    float numerator = falsePositiveCost * (1.0f - defectPrevalence);
    float denominator = (falseNegativeCost * defectPrevalence) + 
                       (falsePositiveCost * (1.0f - defectPrevalence));
    
    if (denominator == 0.0f) {
        return 0.5f;  // Default if denominator is zero
    }
    
    float optimalThreshold = numerator / denominator;
    
    // Clamp to reasonable range
    if (optimalThreshold < 0.1f) optimalThreshold = 0.1f;
    if (optimalThreshold > 0.9f) optimalThreshold = 0.9f;
    
    return optimalThreshold;
}

// ============================================
// PRIVATE HELPER METHODS
// ============================================

void DecisionEngine::validateProbability(float probability) const {
    if (probability < 0.0f || probability > 1.0f) {
        stringstream ss;
        ss << "Probability must be between 0.0 and 1.0, got: " << probability;
        throw invalid_argument(ss.str());
    }
}

void DecisionEngine::updateStatistics(Decision decision) {
    decisionCounts_[decision]++;
    totalDecisions_++;
}

Decision DecisionEngine::applyBusinessRules(
    float probability, 
    Decision preliminaryDecision,
    const map<string, string>& metadata) {
    
    Decision finalDecision = preliminaryDecision;
    
    // Rule 1: Check uncertainty zone
    if (config_.useUncertaintyThreshold && requiresReview(probability)) {
        finalDecision = Decision::UNCERTAIN;
    }
    
    // Rule 2: Check if manual review is required
    if (config_.requireManualReview && requireManualReview(probability)) {
        finalDecision = Decision::UNCERTAIN;
    }
    
    // Rule 3: Apply cost-based adjustment if enabled
    if (config_.useCostOptimization) {
        float optimalThreshold = config_.calculateOptimalThreshold();
        bool isDefectOptimal = probability >= optimalThreshold;
        
        // Only override if different from current decision
        if (isDefectOptimal && finalDecision == Decision::OK) {
            finalDecision = Decision::DEFECT;
        } else if (!isDefectOptimal && finalDecision == Decision::DEFECT) {
            finalDecision = Decision::OK;
        }
    }
    
    // Rule 4: Apply any metadata-based rules
    if (!metadata.empty()) {
        // Example: If product is "high_value", be more conservative
        auto it = metadata.find("product_value");
        if (it != metadata.end() && it->second == "high") {
            // Lower threshold for high-value products
            float conservativeThreshold = config_.defectThreshold * 0.8f;
            if (probability >= conservativeThreshold && finalDecision == Decision::OK) {
                finalDecision = Decision::UNCERTAIN;  // Review high-value items
            }
        }
    }
    
    return finalDecision;
}

bool DecisionEngine::requireManualReview(float probability) const {
    if (!config_.requireManualReview) {
        return false;
    }
    
    // Manual review required if probability is in suspicious range
    return (probability >= config_.manualReviewThreshold && 
            probability < config_.defectThreshold);
}
//Define custom loss functions and metrics

#ifndef LOSS_FUNCTIONS_H
#define LOSS_FUNCTIONS_H

#include <torch/torch.h>
#include <vector>
#include <string>
#include <map>
#include <cmath>

enum class LossType{
    BINARY_CROSS_ENTROPY,  ///< Binary Cross-Entropy Loss
    CROSS_ENTROPY,   ///< Standard cross-entropy loss
    FOCAL_LOSS,      ///< Focal loss for imbalanced datasets
    BCE_WITH_LOGITS, ///< Binary cross-entropy with logits
    MSE,             ///< Mean squared error (for regression)
    DICE_LOSS,       ///< Dice Loss for segmentation tasks
    NLL_LOSS         ///< Negative Log Likelihood Loss
};
struct LossConfig{
    LossType lossType = LossType::CROSS_ENTROPY;  ///< Type of loss function
    float focalAlpha = 0.25f;                     ///< Alpha parameter for Focal Loss
    float focalGamma = 2.0f;                      ///< Gamma parameter for Focal Loss
    //class weights for imbalanced data
    std::vector<float> classWeights;              ///< Weights for each class
    bool useClassWeights = false;              ///< Whether to use class weights

    //reduction type 
    std::string reduction = "mean";               ///< Reduction method: "none", "mean", "sum"

    //label smoothing 
    float labelSmoothing = 0.0f;                  ///< Label smoothing factor (0.0 = no smoothing)

    void validate() const;

};
struct Metrics
{
    float loss = 0.0f;
    float accuracy = 0.0f;
    float precision = 0.0f;
    float recall = 0.0f;
    float f1Score = 0.0f;
    float auc = 0.0f;            ///< Area under ROC curve

    //pre-class metrics
    std::map<int, float> perClassPrecision;
    std::map<int, float> perClassRecall;
    std::map<int, float> perClassF1;

    std::vector<std::vector<int>> confusionMatrix;  ///< Confusion matrix

    //additional metrics 
    float specificity = 0.0f;    ///< True negative rate
    float npv = 0.0f;           ///< Negative predictive value
    float mcc = 0.0f;           ///< Matthews correlation coefficient

    void calculateFromConfusionMatrix(const std::vector<std::vector<int>>& confusionMatrix, 
                                     int numClasses);
    void reset();
    void print(const std::string& prefix = "") const;
    std::map<std::string, float> toMap() const;
    void update(const Metrics& other, float weight = 1.0f);

};

class BaseLossFunction : public torch::nn::Module {
public:
    BaseLossFunction(const LossConfig& config);
    virtual ~BaseLossFunction() = default;
    
    /**
     * @brief Compute loss between predictions and targets
     * 
     * @param predictions Model predictions (logits)
     * @param targets Ground truth labels
     * @return torch::Tensor Loss value
     */
    virtual torch::Tensor forward(const torch::Tensor& predictions, 
                                 const torch::Tensor& targets) = 0;
    
    /**
     * @brief Get loss configuration
     * @return const LossConfig& Configuration
     */
    const LossConfig& getConfig() const;
    
    /**
     * @brief Get loss type
     * @return LossType Type of loss function
     */
    LossType getType() const;
    
protected:
    LossConfig config_;
    
    /**
     * @brief Apply class weights to loss
     * 
     * @param loss Individual loss values
     * @param targets Target labels
     * @return torch::Tensor Weighted loss
     */
    torch::Tensor applyClassWeights(const torch::Tensor& loss, 
                                   const torch::Tensor& targets) const;
    
    /**
     * @brief Apply label smoothing
     * 
     * @param targets Original target labels
     * @param numClasses Number of classes
     * @return torch::Tensor Smoothed targets
     */
    torch::Tensor applyLabelSmoothing(const torch::Tensor& targets, 
                                     int numClasses) const;
};
class CrossEntropyLoss : public BaseLossFunction {
public:
    CrossEntropyLoss(const LossConfig& config, int numClasses);
    
    torch::Tensor forward(const torch::Tensor& predictions, 
                         const torch::Tensor& targets) override;
    
private:
    int numClasses_;
};

/**
 * @class BinaryCrossEntropyLoss
 * @brief Binary cross-entropy loss implementation
 */
class BinaryCrossEntropyLoss : public BaseLossFunction {
public:
    BinaryCrossEntropyLoss(const LossConfig& config);
    
    torch::Tensor forward(const torch::Tensor& predictions, 
                         const torch::Tensor& targets) override;
};

/**
 * @class FocalLoss
 * @brief Focal loss implementation for handling class imbalance
 * 
 * Based on: "Focal Loss for Dense Object Detection" by Lin et al.
 */
class FocalLoss : public BaseLossFunction {
public:
    FocalLoss(const LossConfig& config, int numClasses);
    
    torch::Tensor forward(const torch::Tensor& predictions, 
                         const torch::Tensor& targets) override;
    
private:
    int numClasses_;
    
    /**
     * @brief Compute focal loss
     * 
     * @param probs Class probabilities
     * @param targets Target labels
     * @return torch::Tensor Focal loss
     */
    torch::Tensor computeFocalLoss(const torch::Tensor& probs, 
                                  const torch::Tensor& targets) const;
};

/**
 * @class DiceLoss
 * @brief Dice loss implementation for segmentation tasks
 */
class DiceLoss : public BaseLossFunction {
public:
    DiceLoss(const LossConfig& config);
    
    torch::Tensor forward(const torch::Tensor& predictions, 
                         const torch::Tensor& targets) override;
    
private:
    /**
     * @brief Compute dice coefficient
     * 
     * @param predictions Predictions
     * @param targets Targets
     * @param smooth Smoothing factor
     * @return torch::Tensor Dice coefficient
     */
    torch::Tensor diceCoefficient(const torch::Tensor& predictions,
                                 const torch::Tensor& targets,
                                 float smooth = 1.0f) const;
};

/**
 * @class LossFunctionFactory
 * @brief Factory class for creating loss functions
 */
class LossFunctionFactory {
public:
    /**
     * @brief Create a loss function based on configuration
     * 
     * @param config Loss configuration
     * @param numClasses Number of classes
     * @return std::shared_ptr<BaseLossFunction> Created loss function
     */
    static std::shared_ptr<BaseLossFunction> create(const LossConfig& config, 
                                                   int numClasses);
    
    /**
     * @brief Get default loss configuration
     * 
     * @param lossType Type of loss function
     * @return LossConfig Default configuration
     */
    static LossConfig getDefaultConfig(LossType lossType);
};

/**
 * @class MetricsCalculator
 * @brief Calculate various evaluation metrics
 */
class MetricsCalculator {
public:
    /**
     * @brief Construct a new Metrics Calculator
     * 
     * @param numClasses Number of classes
     */
    explicit MetricsCalculator(int numClasses);
    
    /**
     * @brief Calculate metrics from predictions and targets
     * 
     * @param predictions Model predictions (logits)
     * @param targets Ground truth labels
     * @param loss Computed loss value
     * @return Metrics Calculated metrics
     */
    Metrics calculate(const torch::Tensor& predictions, 
                     const torch::Tensor& targets,
                     float loss);
    
    /**
     * @brief Calculate confusion matrix
     * 
     * @param predictions Predicted class indices
     * @param targets True class indices
     * @return std::vector<std::vector<int>> Confusion matrix
     */
    std::vector<std::vector<int>> calculateConfusionMatrix(
        const torch::Tensor& predictions,
        const torch::Tensor& targets) const;
    
    /**
     * @brief Calculate precision
     * 
     * @param tp True positives
     * @param fp False positives
     * @return float Precision (0-1)
     */
    static float calculatePrecision(int tp, int fp);
    
    /**
     * @brief Calculate recall
     * 
     * @param tp True positives
     * @param fn False negatives
     * @return float Recall (0-1)
     */
    static float calculateRecall(int tp, int fn);
    
    /**
     * @brief Calculate F1 score
     * 
     * @param precision Precision
     * @param recall Recall
     * @return float F1 score (0-1)
     */
    static float calculateF1Score(float precision, float recall);
    
    /**
     * @brief Calculate accuracy
     * 
     * @param correct Number of correct predictions
     * @param total Total number of predictions
     * @return float Accuracy (0-1)
     */
    static float calculateAccuracy(int correct, int total);
    
    /**
     * @brief Calculate AUC (Area Under ROC Curve)
     * 
     * @param predictions Probability predictions
     * @param targets Binary targets
     * @return float AUC score (0-1)
     */
    static float calculateAUC(const torch::Tensor& predictions,
                            const torch::Tensor& targets);
    
    /**
     * @brief Calculate Matthews Correlation Coefficient
     * 
     * @param tp True positives
     * @param tn True negatives
     * @param fp False positives
     * @param fn False negatives
     * @return float MCC (-1 to 1)
     */
    static float calculateMCC(int tp, int tn, int fp, int fn);
    
    /**
     * @brief Reset internal state
     */
    void reset();
    
    /**
     * @brief Get number of classes
     * @return int Number of classes
     */
    int getNumClasses() const;
    
private:
    int numClasses_;
    
    /**
     * @brief Calculate per-class metrics
     * 
     * @param confusionMatrix Confusion matrix
     * @param metrics Metrics structure to fill
     */
    void calculatePerClassMetrics(const std::vector<std::vector<int>>& confusionMatrix,
                                 Metrics& metrics) const;
    
    /**
     * @brief Calculate binary classification metrics
     * 
     * @param confusionMatrix 2x2 confusion matrix
     * @param metrics Metrics structure to fill
     */
    void calculateBinaryMetrics(const std::vector<std::vector<int>>& confusionMatrix,
                               Metrics& metrics) const;
};

/**
 * @class TrainingHistory
 * @brief Track and manage training history
 */
class TrainingHistory {
public:
    /**
     * @brief Construct a new Training History
     * 
     * @param maxEntries Maximum number of entries to keep (0 = unlimited)
     */
    explicit TrainingHistory(size_t maxEntries = 0);
    
    /**
     * @brief Add epoch metrics to history
     * 
     * @param epoch Epoch number
     * @param trainMetrics Training metrics
     * @param valMetrics Validation metrics (optional)
     */
    void addEpoch(int epoch, const Metrics& trainMetrics, 
                  const Metrics& valMetrics = Metrics());
    
    /**
     * @brief Get metrics for a specific epoch
     * 
     * @param epoch Epoch number
     * @return std::pair<Metrics, Metrics> Train and validation metrics
     */
    std::pair<Metrics, Metrics> getEpoch(int epoch) const;
    
    /**
     * @brief Get all epochs
     * 
     * @return const std::map<int, std::pair<Metrics, Metrics>>& All epochs
     */
    const std::map<int, std::pair<Metrics, Metrics>>& getAllEpochs() const;
    
    /**
     * @brief Get latest epoch number
     * 
     * @return int Latest epoch number (-1 if empty)
     */
    int getLatestEpoch() const;
    
    /**
     * @brief Get best epoch based on a metric
     * 
     * @param metricName Metric name ("loss", "accuracy", "f1", etc.)
     * @param maximize Whether to maximize the metric (true for accuracy, false for loss)
     * @return int Best epoch number
     */
    int getBestEpoch(const std::string& metricName = "val_accuracy", 
                    bool maximize = true) const;
    
    /**
     * @brief Check if metrics improved
     * 
     * @param currentMetrics Current metrics
     * @param previousMetrics Previous metrics
     * @param metricName Metric to compare
     * @param maximize Whether to maximize the metric
     * @param minDelta Minimum improvement delta
     * @return true if metrics improved
     */
    bool hasImproved(const Metrics& currentMetrics, 
                    const Metrics& previousMetrics,
                    const std::string& metricName = "accuracy",
                    bool maximize = true,
                    float minDelta = 0.0f) const;
    
    /**
     * @brief Save history to CSV file
     * 
     * @param filePath Path to CSV file
     */
    void saveToCSV(const std::string& filePath) const;
    
    /**
     * @brief Load history from CSV file
     * 
     * @param filePath Path to CSV file
     */
    void loadFromCSV(const std::string& filePath);
    
    /**
     * @brief Clear history
     */
    void clear();
    
    /**
     * @brief Get number of epochs in history
     * 
     * @return size_t Number of epochs
     */
    size_t size() const;
    
    /**
     * @brief Check if history is empty
     * 
     * @return true if empty
     */
    bool empty() const;
    
    /**
     * @brief Print history summary
     */
    void printSummary() const;
    
    /**
     * @brief Get average metrics over all epochs
     * 
     * @return std::pair<Metrics, Metrics> Average train and validation metrics
     */
    std::pair<Metrics, Metrics> getAverageMetrics() const;
    
    /**
     * @brief Get learning curves data for plotting
     * 
     * @param metricName Metric to extract
     * @return std::pair<std::vector<float>, std::vector<float>> Train and validation curves
     */
    std::pair<std::vector<float>, std::vector<float>> getLearningCurves(
        const std::string& metricName = "loss") const;
    
private:
    std::map<int, std::pair<Metrics, Metrics>> history_;
    size_t maxEntries_;
    
    /**
     * @brief Trim history if maximum entries exceeded
     */
    void trimHistory();
    
    /**
     * @brief Get metric value by name
     * 
     * @param metrics Metrics structure
     * @param metricName Metric name
     * @return float Metric value
     */
    float getMetricValue(const Metrics& metrics, 
                        const std::string& metricName) const;
};

#endif // LOSS_FUNCTIONS_H

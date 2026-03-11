#ifndef EVALUATOR_H
#define EVALUATOR_H
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <utility>

//strcutrue classification result
struct ClassificationResult {
    int trueLabel;           ///< Ground truth (0=OK, 1=DEFECT)
    int predictedLabel;      ///< Model prediction (0=OK, 1=DEFECT)
    float confidence;        ///< Confidence score [0, 1]
    std::string imagePath;   ///< Optional: path to image for debugging
    
    ClassificationResult(int trueLbl = -1, int predLbl = -1, 
                         float conf = 0.0f, const std::string& path = "")
        : trueLabel(trueLbl), predictedLabel(predLbl), 
          confidence(conf), imagePath(path) {}

};

//structure evaluation metrics
struct EvaluationMetrics{
    // Basic counts
    int totalSamples = 0;
    int truePositives = 0;   // Defect correctly identified
    int falsePositives = 0;  // OK incorrectly called defect (False Alarm)
    int trueNegatives = 0;   // OK correctly identified
    int falseNegatives = 0;  // Defect missed (MISSED DEFECT - CRITICAL ERROR!)
    
    // Derived metrics
    float accuracy = 0.0f;           // (TP + TN) / Total
    float precision = 0.0f;          // TP / (TP + FP) - How reliable are defect alarms?
    float recall = 0.0f;             // TP / (TP + FN) - How many defects did we catch?
    float f1Score = 0.0f;            // 2 * (precision * recall) / (precision + recall)
    float specificity = 0.0f;        // TN / (TN + FP) - How good at identifying OK products?

    // Additional metrics
    float falsePositiveRate = 0.0f;  // FP / (FP + TN)
    float falseNegativeRate = 0.0f;  // FN / (FN + TP) - DEFECT ESCAPE RATE!
    
    // For threshold analysis
    std::vector<std::pair<float, EvaluationMetrics>> rocCurve;
    float auc = 0.0f;  // Area Under ROC Curve

    void print() const;
    std::map<std::string, std::string> toMap() const;

};

//structure confusion matrix
struct ConfusionMatrix{
    int matrix[2][2] = {{0, 0}, {0, 0}}; // [true][predicted]

    std::string classNames[2] = {"OK", "DEFECT"};
    std::string toString() const; //get confusion matrix as string
    int total() const; //get total number of samples

};

//Evaluator class for computing evaluation metrics
class Evaluator {
    public:
        explicit Evaluator(float defectThreshold = 0.5f); 
        virtual ~Evaluator();

        //data collection methods
        //add a single classification result
        void addResult(int trueLabel, int predictedLabel, 
                       float confidence, const std::string& imagePath = "");

        //add a result using defect proba 
        void addResultWithProbability(int trueLabel, float defectProbability, 
                                      const std::string& imagePath = "");

        //add multiple results at once
        void addBatchResults(const std::vector<int>& trueLabels,
        const std::vector<int>& predictedLabels);
        void clearResults();

        //evaluation metrics 
        EvaluationMetrics computeMetrics() const;
        ConfusionMatrix computeConfusionMatrix() const;
        EvaluationMetrics computeMetricsAtThreshold(float threshold) const;

        //advanced analysis 
        std::vector<std::pair<float, float>> computeROCCurve(int numThresholds = 100) const;
        float computeAUC() const;

        float findOptimalThreshold(float start = 0.3f, float end = 0.9f, 
                       float step = 0.05f) const;

        float findThresholdForRecall(float targetRecall) const ;

        //report generation
        std::string generateConfusionMatrixString() const;

        std::string generateMetricsSummary() const;

        bool saveToCSV(const std::string& filePath) const;

        bool saveDetailedResults(const std::string& filePath) const;
    std::string generateReport(const std::string& title) const;

        //getters and setters

        float getThreshold() const;
        void setThreshold(float threshold);
        int getResultCount() const;
        const std::vector<ClassificationResult>& getResults() const; //get all results
        std::vector<ClassificationResult> getMisclassifiedResults() const; //get misclassified results
        std::vector<ClassificationResult> getFalsePositives() const; //get false positives
        std::vector<ClassificationResult> getFalseNegatives() const; //get false negatives

        //statistical methods 
        std::pair<float, float> computeConfidenceInterval(
        const std::string& metric, float confidenceLevel = 0.95f) const;
    
   
        float performMcNemarTest(const Evaluator& other) const;
  private :
    std::vector<ClassificationResult> results_;  ///< All collected results
    float defectThreshold_;                      ///< Current classification threshold
    
    
    int probabilityToLabel(float probability) const;
    
   
    EvaluationMetrics computeMetricsFromCounts(int tp, int fp, int tn, int fn) const;
    
    ConfusionMatrix computeConfusionMatrixFromResults(
        const std::vector<ClassificationResult>& results) const;
};

#endif // EVALUATOR_H

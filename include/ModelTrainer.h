#ifndef ModelTrainer_H
#define ModelTrainer_H

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <chrono>
#include <functional>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

// Forward declarations for custom classes
class ImageLoader;
class Preprocessor;
class DataLoader;
class DefectDataset;

//training metrics structure
struct TrainingMetrics {
    int epoch = 0;
    float trainLoss = 0.0f;
    float trainAccuracy = 0.0f;
    float trainPrecision = 0.0f;
    float trainRecall = 0.0f;
    float trainF1Score = 0.0f;
    float valLoss = 0.0f;
    float valAccuracy = 0.0f;
    float valPrecision = 0.0f;
    float valRecall = 0.0f;
    float valF1Score = 0.0f;
    float learningRate = 0.0f;
    std::chrono::milliseconds epochTime{0};
    std::chrono::steady_clock::time_point timestamp;
    
    void toMap(std::map<std::string, float>& metricsMap) const;
    void fromMap(const std::map<std::string, float>& metricsMap);
};

enum class LossFunctionType {
    CROSS_ENTROPY,    ///< Standard cross-entropy loss
    FOCAL_LOSS,       ///< Focal loss for imbalanced datasets
    BCE_WITH_LOGITS,  ///< Binary cross-entropy with logits
    MSE               ///< Mean squared error (for regression)
};

enum class OptimizerType {
    SGD,          ///< Stochastic Gradient Descent
    ADAM,         ///< Adam optimizer
    RMSPROP,      ///< RMSProp optimizer
    ADAGRAD       ///< Adagrad optimizer
};

enum class LearningRateSchedulerType {
    STEP_LR,         ///< Step learning rate scheduler
    EXPONENTIAL_LR,  ///< Exponential decay learning rate scheduler
    COSINE_ANNEALING_LR ///< Cosine annealing learning rate scheduler
};

//training structures
struct TrainingConfig {
    // ============ DATA CONFIGURATION ============
    std::string trainDataPath;      ///< Path to training data directory
    std::string valDataPath;        ///< Path to validation data directory (if separate)
    std::string testDataPath;       ///< Path to test data directory
    float trainValSplit = 0.8f;     ///< Train/validation split ratio (if valDataPath empty)
    int numClasses = 2;             ///< Number of output classes (binary: 2)
    
    // ============ TRAINING HYPERPARAMETERS ============
    int numEpochs = 50;             ///< Total number of training epochs
    int batchSize = 32;             ///< Batch size for training
    float learningRate = 0.001f;    ///< Initial learning rate
    float weightDecay = 0.0001f;    ///< Weight decay for regularization
    bool useGPU = true;             ///< Use GPU if available
    
    // ============ LOSS FUNCTION ============
    LossFunctionType lossFunction = LossFunctionType::CROSS_ENTROPY;
    
    // Focal loss parameters (if using focal loss)
    float focalLossGamma = 2.0f;
    float focalLossAlpha = 0.25f;
    
    // ============ OPTIMIZER ============
    OptimizerType optimizerType = OptimizerType::ADAM;
    
    // SGD specific parameters
    float sgdMomentum = 0.9f;
    bool sgdNesterov = true;
    
    // Adam specific parameters
    float adamBeta1 = 0.9f;
    float adamBeta2 = 0.999f;
    
    // ============ LEARNING RATE SCHEDULER ============
    LearningRateSchedulerType lrSchedulerType = LearningRateSchedulerType::STEP_LR;
    
    // StepLR parameters
    int lrStepSize = 10;            ///< Step size for StepLR
    float lrGamma = 0.1f;           ///< Multiplicative factor for StepLR/ExponentialLR
    
    // ExponentialLR parameters
    float lrDecayRate = 0.96f;      ///< Decay rate for ExponentialLR
    
    // CosineAnnealingLR parameters
    int lrTMax = 50;                ///< T_max for CosineAnnealingLR
    float lrEtaMin = 1e-6f;         ///< Minimum learning rate
    
    // ReduceLROnPlateau parameters
    int lrPatience = 5;             ///< Patience for ReduceLROnPlateau
    float lrFactor = 0.1f;          ///< Factor for ReduceLROnPlateau
    float lrThreshold = 0.0001f;    ///< Threshold for ReduceLROnPlateau
    
    // ============ EARLY STOPPING ============
    bool earlyStoppingEnabled = true;
    int earlyStoppingPatience = 10;
    float earlyStoppingMinDelta = 0.001f;
    std::string earlyStoppingMetric = "val_loss"; ///< Metric to monitor
    
    // ============ MODEL CHECKPOINTING ============
    std::string checkpointDir = "./checkpoints";
    int checkpointFrequency = 5;    ///< Save checkpoint every N epochs
    bool saveBestOnly = true;       ///< Save only best model
    std::string metricToMonitor = "val_accuracy"; ///< Metric to track for best model
    
    // ============ DATA AUGMENTATION ============
    bool useDataAugmentation = true;
    float rotationRange = 15.0f;    ///< Rotation range in degrees
    float zoomRange = 0.2f;         ///< Zoom range (0.8-1.2)
    float horizontalFlipProb = 0.5f;
    float verticalFlipProb = 0.0f;  ///< Usually 0 for defect detection
    float brightnessRange = 0.2f;
    float contrastRange = 0.2f;
    float saturationRange = 0.2f;
    float hueRange = 0.1f;
    
    // ============ CLASS WEIGHTS (for imbalanced data) ============
    std::vector<float> classWeights = {1.0f, 1.0f};
    bool useClassWeights = false;
    
    // ============ LOGGING & VISUALIZATION ============
    bool verbose = true;
    int logFrequency = 10;          ///< Log every N batches
    bool saveTrainingHistory = true;
    std::string historyFile = "./training_history.csv";
    bool plotMetrics = true;
    
    // ============ VALIDATION ============
    int valFrequency = 1;           ///< Validate every N epochs
    bool shuffleData = true;
    
    // ============ DATASET SPECIFIC ============
    std::vector<std::string> classNames = {"OK", "DEFECT"};

    //load config from the yaml file 
    void loadFromYAML(const std::string& filePath);

    //save config to yaml file
    void saveToYAML(const std::string& filePath) const;
    void printSummary() const;

};

//callbacks for training
//class trainingCallbacks 
class trainingCallbacks {
    public: 
        virtual ~trainingCallbacks() = default;
        virtual void onEpochBegin(int epoch) {}
        virtual void onEpochEnd(int epoch, const std::map<std::string, float>& logs) {}
        virtual void onTrainBatchBegin(int batch) {}
        virtual void onTrainBatchEnd(int batch, const std::map<std::string, float>& logs) {}
        virtual void onBatchBegin(int batch) {}
        virtual void onBatchEnd(int batch, const std::map<std::string, float>& logs) {}

};

//class ModelTrainer: handles the training process
class ModelTrainer {
    public:
        explicit ModelTrainer(const TrainingConfig& config);
        ModelTrainer(const TrainingConfig& config, std::shared_ptr<Preprocessor> preprocessor);
        virtual ~ModelTrainer();

        //train the model
        std::vector<TrainingMetrics> train(std::shared_ptr<torch::nn::Module> model);

        //train model with tranxsfer learning
        std::vector<TrainingMetrics> trainWithTransferLearning(std::shared_ptr<torch::nn::Module> model, const std::string& baseModelPath);

        //resume training from a checkpoint
        std::vector<TrainingMetrics> resumeTraining(std::shared_ptr<torch::nn::Module> model, const std::string& checkpointPath);

        //validate the model on validation set 
        TrainingMetrics validate(std::shared_ptr<torch::nn::Module> model);

        //test the model on test set
        TrainingMetrics test(std::shared_ptr<torch::nn::Module> model);

        // data preparation 
        //prepare training and validation datasets
        void prepareDatasets();  //Load and prepare training/validation data

        //prepare test dataset 
        void prepareTestDataset();

        //model managemnat methods
        void saveModelCheckpoint(std::shared_ptr<torch::nn::Module> model, int epoch, float metricValue);
        void loadModelCheckpoint(std::shared_ptr<torch::nn::Module> model, const
            std::string& checkpointPath);
        void saveFinalModel(std::shared_ptr<torch::nn::Module> model, const std::string& filePath);

        //callback management
        void addCallback(std::shared_ptr<trainingCallbacks> callback);
        void clearCallbacks();

        //utility methods
        //get training history
        const std::vector<TrainingMetrics>& getTrainingHistory() const;

        //get current config 
        const TrainingConfig& getConfig() const;

        //update config before training 
        void updateConfig(const TrainingConfig& newConfig);

        //export training history to csv
        void exportTrainingHistoryToCSV(const std::string& filePath) const;

        //plot training metrics 
        void plotTrainingMetrics(const std::string& savePath = "./training_metrics.png") const;

        //get the best model path
        std::string getBestModelPath() const;

    private:
    TrainingConfig config_;  ///< Training configuration
    std::vector<TrainingMetrics> trainingHistory_;  ///< Training metrics history
    std::vector<std::shared_ptr<trainingCallbacks>> callbacks_;  ///< Training callbacks
    std::shared_ptr<Preprocessor> preprocessor_;  ///< Data preprocessor
    std::shared_ptr<ImageLoader> imageLoader_;  ///< Image loader for dataset
    std::shared_ptr<DataLoader> dataLoader_;  ///< Data loader for datasets
    std::string bestModelPath_;  ///< Path to the best model checkpoint
    float bestMetricValue_ = -1.0f;  ///< Best metric value found during training
    int epochsWithoutImprovement_ = 0;  ///< Counter for early stopping
    bool trainingInProgress_ = false;  ///< Flag indicating if training is active

    //private mthodes
    /**
     * @brief Execute one training epoch
     * 
     * @param model Model to train
     * @param trainLoader Training data loader
     * @param epoch Current epoch number
     * @return TrainingMetrics Training metrics for this epoch
     */
    TrainingMetrics trainEpoch(std::shared_ptr<torch::nn::Module> model,
                              int epoch);
    
    /**
     * @brief Create optimizer for the model
     * 
     * @param model Model parameters
     * @return std::shared_ptr<torch::optim::Optimizer> Optimizer instance
     */
    std::shared_ptr<torch::optim::Optimizer> createOptimizer(
        std::shared_ptr<torch::nn::Module> model);
    
    /**
     * @brief Create learning rate scheduler
     * 
     * @param optimizer Optimizer to schedule
     * @return std::shared_ptr<void> Type-erased scheduler
     */
    std::shared_ptr<void> createLRScheduler(
        std::shared_ptr<torch::optim::Optimizer> optimizer);
    
    /**
     * @brief Create loss function based on configuration
     * 
     * @return auto Loss function module
     */
    auto createLossFunction();
    
    /**
     * @brief Apply data augmentation to batch
     * 
     * @param batch Input batch
     * @param seed Random seed for reproducibility
     * @return torch::Tensor Augmented batch
     */
    torch::Tensor augmentBatch(const torch::Tensor& batch, unsigned int seed = 0);
    
    /**
     * @brief Compute metrics for predictions
     * 
     * @param predictions Model predictions
     * @param targets Ground truth labels
     * @param loss Loss value
     * @return TrainingMetrics Computed metrics
     */
    TrainingMetrics computeMetrics(const torch::Tensor& predictions,
                                  const torch::Tensor& targets,
                                  float loss) const;
    
    /**
     * @brief Check early stopping condition
     * 
     * @param currentMetrics Current epoch metrics
     * @return true if training should stop
     * @return false otherwise
     */
    bool checkEarlyStopping(const TrainingMetrics& currentMetrics);
    
    /**
     * @brief Update learning rate scheduler
     * 
     * @param scheduler Learning rate scheduler
     * @param currentMetrics Current epoch metrics
     * @param epoch Current epoch
     */
    void updateLRScheduler(std::shared_ptr<void> scheduler,
                          const TrainingMetrics& currentMetrics,
                          int epoch);
    
    /**
     * @brief Log training progress
     * 
     * @param epoch Current epoch
     * @param metrics Current metrics
     * @param isValidation Whether logging validation metrics
     */
    void logProgress(int epoch, const TrainingMetrics& metrics, bool isValidation = false);
};



#endif // ModelTrainer_H    
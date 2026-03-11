/**
 * @file train_main.cpp
 * @brief Main training executable for Industrial Defect Detection System
 * 
 * This program trains a CNN model for defect detection using:
 * 1. Configuration from YAML file
 * 2. DataLoader for image loading and preprocessing
 * 3. ModelTrainer for training loop and optimization
 * 4. LossFunctions for loss computation and metrics
 * 
 * Outputs trained model to models/ folder
 */

#include "ModelTrainer.h"
#include "DataLoader.h"
#include "LossFunctions.h"
#include <iostream>
#include <filesystem>
#include <memory>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <cstdlib>

namespace fs = std::filesystem;
using namespace std::chrono;

// Function declarations
void printBanner();
void parseArguments(int argc, char* argv[], std::string& configPath);
void createDirectories();
void setupLogging();
void validateEnvironment();
void trainModel(const std::string& configPath);
void printTrainingSummary(const std::vector<TrainingMetrics>& history);

/**
 * @brief Main training function
 * 
 * @param argc Argument count
 * @param argv Argument values
 * @return int Exit code (0 = success, >0 = error)
 */
int main(int argc, char* argv[]) {
    try {
        // Print banner
        printBanner();
        
        // Parse command line arguments
        std::string configPath;
        parseArguments(argc, argv, configPath);
        
        // Validate environment
        validateEnvironment();
        
        // Create necessary directories
        createDirectories();
        
        // Setup logging
        setupLogging();
        
        // Start training
        trainModel(configPath);
        
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "\n❌ TRAINING FAILED: " << e.what() << std::endl;
        return 1;
    }
    catch (...) {
        std::cerr << "\n❌ TRAINING FAILED: Unknown error" << std::endl;
        return 1;
    }
}

// ============================================
// HELPER FUNCTION IMPLEMENTATIONS
// ============================================

/**
 * @brief Print program banner
 */
void printBanner() {
    std::cout << R"(
╔══════════════════════════════════════════════════════════╗
║   Industrial Defect Detection System - Training Module   ║
╠══════════════════════════════════════════════════════════╣
║  • CNN-based defect detection                           ║
║  • Binary classification (OK/DEFECT)                   ║
║  • Real-time inference capable                         ║
║  • Designed for manufacturing environments             ║
╚══════════════════════════════════════════════════════════╝
)" << std::endl;
}

/**
 * @brief Parse command line arguments
 * 
 * @param argc Argument count
 * @param argv Argument values
 * @param configPath Output: Path to configuration file
 */
void parseArguments(int argc, char* argv[], std::string& configPath) {
    // Default configuration path
    configPath = "config/training_config.yaml";
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--config" || arg == "-c") {
            if (i + 1 < argc) {
                configPath = argv[++i];
            } else {
                throw std::runtime_error("--config requires a file path");
            }
        }
        else if (arg == "--help" || arg == "-h") {
            std::cout << R"(
Usage: train [OPTIONS]

Options:
  -c, --config PATH   Path to training configuration file
                      (default: config/training_config.yaml)
  -h, --help          Show this help message

Examples:
  train                           # Use default config
  train --config my_config.yaml   # Use custom config
)" << std::endl;
            exit(0);
        }
        else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }
    
    std::cout << "📋 Using configuration file: " << configPath << std::endl;
}

/**
 * @brief Create necessary directories
 */
void createDirectories() {
    std::vector<std::string> dirs = {
        "models",
        "checkpoints",
        "logs",
        "output",
        "config"
    };
    
    for (const auto& dir : dirs) {
        if (!fs::exists(dir)) {
            if (fs::create_directories(dir)) {
                std::cout << "📁 Created directory: " << dir << std::endl;
            }
        }
    }
}

/**
 * @brief Setup logging system
 */
void setupLogging() {
    // Create timestamp for log file
    auto now = system_clock::now();
    auto time = system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&time);
    
    std::stringstream ss;
    ss << "logs/training_" 
       << std::put_time(&tm, "%Y%m%d_%H%M%S") 
       << ".log";
    
    std::string logFile = ss.str();
    
    // Redirect std::cout to both console and file
    std::cout << "📝 Log file: " << logFile << std::endl;
    
    // Note: For proper logging, you might want to use a logging library
    // This is a simplified implementation
}

/**
 * @brief Validate training environment
 */
void validateEnvironment() {
    std::cout << "\n🔍 Validating environment..." << std::endl;
    
    // Check for required directories
    if (!fs::exists("data")) {
        throw std::runtime_error("'data' directory not found. Please create it with train/ and val/ subdirectories.");
    }
    
    // Check for training data
    if (!fs::exists("data/train") && !fs::exists("data/train_images")) {
        throw std::runtime_error("Training data not found. Expected 'data/train/' or 'data/train_images/'.");
    }
    
    // Check OpenCV availability
    try {
        cv::Mat test(10, 10, CV_8UC3, cv::Scalar(0, 0, 255));
        if (test.empty()) {
            throw std::runtime_error("OpenCV failed to create test image");
        }
    } catch (const cv::Exception& e) {
        throw std::runtime_error("OpenCV not properly configured: " + std::string(e.what()));
    }
    
    // Check LibTorch availability
    try {
        torch::Tensor test = torch::rand({2, 3, 224, 224});
        if (!test.defined()) {
            throw std::runtime_error("LibTorch failed to create test tensor");
        }
        
        // Check CUDA availability
        if (torch::cuda::is_available()) {
            std::cout << "✅ CUDA available: GPU acceleration enabled" << std::endl;
        } else {
            std::cout << "⚠️  CUDA not available: Using CPU (training will be slower)" << std::endl;
        }
    } catch (const std::exception& e) {
        throw std::runtime_error("LibTorch not properly configured: " + std::string(e.what()));
    }
    
    std::cout << "✅ Environment validation passed" << std::endl;
}

/**
 * @brief Load configuration from YAML file
 * 
 * @param configPath Path to configuration file
 * @return TrainingConfig Loaded configuration
 */
TrainingConfig loadTrainingConfig(const std::string& configPath) {
    std::cout << "\n📖 Loading training configuration..." << std::endl;
    
    TrainingConfig config;
    
    if (fs::exists(configPath)) {
        // Load from YAML file
        config.loadFromYAML(configPath);
        std::cout << "✅ Configuration loaded from: " << configPath << std::endl;
    } else {
        // Use default configuration
        std::cout << "⚠️  Config file not found, using defaults" << std::endl;
        
        // Set default paths
        config.trainDataPath = "data/train";
        config.valDataPath = "data/val";
        config.testDataPath = "data/test";
        
        // Set default hyperparameters
        config.numEpochs = 50;
        config.batchSize = 32;
        config.learningRate = 0.001f;
        config.optimizerType = OptimizerType::ADAM;
        config.lossFunction = LossFunctionType::CROSS_ENTROPY;
        
        // Create default config file for future use
        config.saveToYAML(configPath);
        std::cout << "📝 Created default config file: " << configPath << std::endl;
    }
    
    config.printSummary();
    return config;
}

/**
 * @brief Create CNN model for defect detection
 * 
 * @param numClasses Number of output classes
 * @return std::shared_ptr<torch::nn::Module> CNN model
 */
std::shared_ptr<torch::nn::Module> createDefectDetectionModel(int numClasses) {
    std::cout << "\n🧠 Creating CNN model..." << std::endl;
    
    // Simple CNN model for defect detection
    struct DefectNetImpl : torch::nn::Module {
        DefectNetImpl(int num_classes = 2) {
            // Feature extraction layers
            conv1 = register_module("conv1", 
                torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 32, 3).padding(1)));
            conv2 = register_module("conv2", 
                torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).padding(1)));
            conv3 = register_module("conv3", 
                torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).padding(1)));
            
            // Batch normalization
            bn1 = register_module("bn1", torch::nn::BatchNorm2d(32));
            bn2 = register_module("bn2", torch::nn::BatchNorm2d(64));
            bn3 = register_module("bn3", torch::nn::BatchNorm2d(128));
            
            // Pooling layers
            pool = register_module("pool", 
                torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));
            
            // Global average pooling
            global_pool = register_module("global_pool", 
                torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({1, 1})));
            
            // Fully connected layers
            fc1 = register_module("fc1", torch::nn::Linear(128, 64));
            fc2 = register_module("fc2", torch::nn::Linear(64, num_classes));
            
            // Dropout for regularization
            dropout = register_module("dropout", torch::nn::Dropout(0.5));
            
            // Initialize weights
            initializeWeights();
        }
        
        torch::Tensor forward(torch::Tensor x) {
            // Normalize input to [-1, 1]
            x = (x - 0.5) * 2.0;
            
            // Feature extraction
            x = torch::relu(bn1(conv1(x)));
            x = pool(x);
            
            x = torch::relu(bn2(conv2(x)));
            x = pool(x);
            
            x = torch::relu(bn3(conv3(x)));
            x = pool(x);
            
            // Global pooling
            x = global_pool(x);
            x = x.view({x.size(0), -1});
            
            // Classification head
            x = torch::relu(fc1(x));
            x = dropout(x);
            x = fc2(x);
            
            return x;
        }
        
        void initializeWeights() {
            // Initialize weights using Kaiming initialization
            for (auto& param : named_parameters()) {
                if (param.value().dim() > 1) {
                    torch::nn::init::kaiming_normal_(param.value(), 0.0, torch::kFanOut, torch::kReLU);
                } else if (param.key().find("bias") != std::string::npos) {
                    torch::nn::init::constant_(param.value(), 0);
                } else if (param.key().find("weight") != std::string::npos) {
                    torch::nn::init::constant_(param.value(), 1);
                }
            }
        }
        
        torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
        torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr}, bn3{nullptr};
        torch::nn::MaxPool2d pool{nullptr};
        torch::nn::AdaptiveAvgPool2d global_pool{nullptr};
        torch::nn::Linear fc1{nullptr}, fc2{nullptr};
        torch::nn::Dropout dropout{nullptr};
    };
    
    TORCH_MODULE(DefectNet);
    
    auto model = std::make_shared<DefectNet>(numClasses);
    
    // Print model architecture
    std::cout << "Model Architecture:" << std::endl;
    std::cout << "  Input: 3x224x224 (RGB image)" << std::endl;
    std::cout << "  Conv1: 32 channels, 3x3, ReLU, BatchNorm" << std::endl;
    std::cout << "  Conv2: 64 channels, 3x3, ReLU, BatchNorm" << std::endl;
    std::cout << "  Conv3: 128 channels, 3x3, ReLU, BatchNorm" << std::endl;
    std::cout << "  Global Average Pooling" << std::endl;
    std::cout << "  FC1: 128 -> 64, ReLU, Dropout(0.5)" << std::endl;
    std::cout << "  FC2: 64 -> " << numClasses << " (output)" << std::endl;
    
    // Count parameters
    size_t totalParams = 0;
    for (const auto& param : model->named_parameters()) {
        totalParams += param.value().numel();
    }
    
    std::cout << "✅ Model created with " << totalParams << " parameters" << std::endl;
    
    return model;
}

/**
 * @brief Main training function
 * 
 * @param configPath Path to configuration file
 */
void trainModel(const std::string& configPath) {
    auto trainingStart = steady_clock::now();
    
    try {
        // 1. Load training configuration
        TrainingConfig trainConfig = loadTrainingConfig(configPath);
        
        // 2. Setup DataLoader
        std::cout << "\n📊 Setting up DataLoader..." << std::endl;
        
        DataLoaderConfig dataConfig;
        dataConfig.dataRootPath = "data";
        dataConfig.trainFolder = "train";
        dataConfig.valFolder = "val";
        dataConfig.testFolder = "test";
        dataConfig.batchSize = trainConfig.batchSize;
        dataConfig.targetSize = cv::Size(224, 224);
        dataConfig.useAugmentation = trainConfig.useDataAugmentation;
        dataConfig.rotationRange = trainConfig.rotationRange;
        dataConfig.horizontalFlipProb = trainConfig.horizontalFlipProb;
        
        DataLoader dataLoader(dataConfig);
        
        // Create datasets
        auto trainDataset = dataLoader.createTrainDataset();
        auto valDataset = dataLoader.createValidationDataset();
        
        if (!trainDataset || trainDataset->size().value() == 0) {
            throw std::runtime_error("No training data found. Check your data/train/ directory.");
        }
        
        std::cout << "✅ DataLoader ready" << std::endl;
        std::cout << "   Training samples: " << trainDataset->size().value() << std::endl;
        std::cout << "   Validation samples: " << (valDataset ? valDataset->size().value() : 0) << std::endl;
        
        // 3. Create model
        auto model = createDefectDetectionModel(trainConfig.numClasses);
        
        // 4. Setup ModelTrainer
        std::cout << "\n🎯 Setting up ModelTrainer..." << std::endl;
        ModelTrainer trainer(trainConfig);
        
        // 5. Start training
        std::cout << "\n🚀 Starting training..." << std::endl;
        std::cout << "   Epochs: " << trainConfig.numEpochs << std::endl;
        std::cout << "   Batch size: " << trainConfig.batchSize << std::endl;
        std::cout << "   Learning rate: " << trainConfig.learningRate << std::endl;
        
        auto history = trainer.train(model);
        
        // 6. Save final model
        std::string modelPath = "models/defect_model.pt";
        trainer.saveFinalModel(model, modelPath);
        
        // 7. Export training history
        std::string historyPath = "output/training_history.csv";
        trainer.exportTrainingHistoryToCSV(historyPath);
        
        // 8. Calculate training time
        auto trainingEnd = steady_clock::now();
        auto duration = duration_cast<minutes>(trainingEnd - trainingStart);
        
        // 9. Print summary
        printTrainingSummary(history);
        
        std::cout << "\n✅ Training completed successfully!" << std::endl;
        std::cout << "   Total time: " << duration.count() << " minutes" << std::endl;
        std::cout << "   Model saved: " << modelPath << std::endl;
        std::cout << "   History saved: " << historyPath << std::endl;
        std::cout << "\n✨ Ready for inference!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Training failed: " << e.what() << std::endl;
        throw;
    }
}

/**
 * @brief Print training summary
 * 
 * @param history Training history
 */
void printTrainingSummary(const std::vector<TrainingMetrics>& history) {
    if (history.empty()) {
        std::cout << "\nNo training history available" << std::endl;
        return;
    }
    
    // Find best epoch
    auto bestEpoch = std::max_element(history.begin(), history.end(),
        [](const TrainingMetrics& a, const TrainingMetrics& b) {
            return a.valAccuracy < b.valAccuracy;
        });
    
    // Calculate averages
    float avgTrainLoss = 0.0f, avgTrainAcc = 0.0f;
    float avgValLoss = 0.0f, avgValAcc = 0.0f;
    
    for (const auto& metrics : history) {
        avgTrainLoss += metrics.trainLoss;
        avgTrainAcc += metrics.trainAccuracy;
        avgValLoss += metrics.valLoss;
        avgValAcc += metrics.valAccuracy;
    }
    
    avgTrainLoss /= history.size();
    avgTrainAcc /= history.size();
    avgValLoss /= history.size();
    avgValAcc /= history.size();
    
    std::cout << "\n📈 TRAINING SUMMARY" << std::endl;
    std::cout << "══════════════════════════════════════════════════════════" << std::endl;
    std::cout << "Total epochs: " << history.size() << std::endl;
    std::cout << "Best epoch: " << (bestEpoch - history.begin() + 1) << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Best validation accuracy: " << bestEpoch->valAccuracy << std::endl;
    std::cout << "Best validation loss: " << bestEpoch->valLoss << std::endl;
    std::cout << "\nAverages:" << std::endl;
    std::cout << "  Training loss: " << avgTrainLoss << std::endl;
    std::cout << "  Training accuracy: " << avgTrainAcc << std::endl;
    std::cout << "  Validation loss: " << avgValLoss << std::endl;
    std::cout << "  Validation accuracy: " << avgValAcc << std::endl;
    
    // Print last epoch metrics
    const auto& last = history.back();
    std::cout << "\nLast epoch (" << history.size() << "):" << std::endl;
    std::cout << "  Training - Loss: " << last.trainLoss 
              << ", Acc: " << last.trainAccuracy << std::endl;
    std::cout << "  Validation - Loss: " << last.valLoss 
              << ", Acc: " << last.valAccuracy << std::endl;
    
    // Print learning progress
    if (history.size() > 1) {
        const auto& first = history[0];
        float lossImprovement = ((first.valLoss - last.valLoss) / first.valLoss) * 100;
        float accImprovement = ((last.valAccuracy - first.valAccuracy) / first.valAccuracy) * 100;
        
        std::cout << "\n📊 Learning progress:" << std::endl;
        std::cout << "  Loss improved by: " << std::setprecision(1) 
                  << lossImprovement << "%" << std::endl;
        std::cout << "  Accuracy improved by: " << std::setprecision(1) 
                  << accImprovement << "%" << std::endl;
    }
    
    std::cout << "══════════════════════════════════════════════════════════" << std::endl;
}

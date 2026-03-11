#include "../include/CNNModel.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <stdexcept>
#include <iomanip>

using namespace std;

// model config implementation 

void ModelConfig::validate() const{
    /* this fct validates :
    input dimensions , number of classes , confidence threshold ,  batch size ,class names ,threads , backend
    */
    if(inputWidth <= 0 || inputHeight <= 0 ){
        throw std::invalid_argument("Input width and height must be positive");
    }
    if (inputChannels != 1 && inputChannels != 3){
        throw std::invalid_argument("Input width and height must be 1 (grayscale) or 3 (RGB)");
    }
    if(numClasses <= 0){
        throw std::invalid_argument("Number of classese must be postive ");
    }
    if(confidenceThreshold < 0.0f || confidenceThreshold > 1.0f){
        throw std::invalid_argument("Confidence threshold must be between 0.0 and 1.0");
    }
    if(batchSize <= 0){
        throw std::invalid_argument("Batch size must be postive ");

    }
    if (classNames.size() != static_cast<size_t>(numClasses)) {
        throw std::invalid_argument("Number of class names must match numClasses");
    }
    
    if (numThreads <= 0) {
        throw std::invalid_argument("Number of threads must be positive");
    }
    
    std::cout << "[ModelConfig] Configuration validated successfully" << std::endl;
}

//model output implementation 
void ModelOutput::clear(){
    rawScores.clear();
    probabilities.clear();
    predictedClass = -1;
    confidence = 0.0f;
    inferenceTimeMs = 0.0;
    featureMaps.clear();
    heatmaps.clear();
}

bool ModelOutput::isValid() const {
    //check if we have any output  
    if(probabilities.empty() && rawScores.empty()){
        return false;
    } 
    //check if the predicted class is valid 
    if(predictedClass < 0){
        return false;
    }
    //check if confidence is valid 
    if(confidence < 0.0f || confidence > 1.0f){
        return false;
    }
    //check if sum of proba is 1 
    if(!probabilities.empty()){
        float sum =0.0f ;
        for(float prob : probabilities ){
            sum += prob;
        }
        if(std::abs(sum -1.0f) > 0.01f){
            return false;
        }
    }
    return true ;
}

std::string ModelOutput::getClassName(const std::vector<std::string>& classNames) const {
    if (predictedClass < 0 || predictedClass >= static_cast<int>(classNames.size())) {
        return "UNKNOWN";
    }
    return classNames[predictedClass];
}

void ModelOutput::print() const {
    std::cout << "\n=== Model Output ===" << std::endl;
    std::cout << "Predicted Class: " << predictedClass << std::endl;
    std::cout << "Confidence: " << std::fixed << std::setprecision(4) 
              << confidence * 100.0f << "%" << std::endl;
    std::cout << "Inference Time: " << inferenceTimeMs << " ms" << std::endl;
    
    if (!probabilities.empty()) {
        std::cout << "Class Probabilities:" << std::endl;
        for (size_t i = 0; i < probabilities.size(); ++i) {
            std::cout << "  Class " << i << ": " 
                      << std::fixed << std::setprecision(4) 
                      << probabilities[i] * 100.0f << "%" << std::endl;
        }
    }
    
    std::cout << "===================\n" << std::endl;
}

// ============================================
// CNNMODEL BASE CLASS IMPLEMENTATION
// ============================================

CNNModel::CNNModel() : isInitialized_(false), lastInferenceTime_(0.0) {
    std::cout << "[CNNModel] Base constructor called" << std::endl;
}

CNNModel::CNNModel(const ModelConfig& config) 
    : config_(config), isInitialized_(false), lastInferenceTime_(0.0) {
    std::cout << "[CNNModel] Constructor with config called" << std::endl;
    config_.validate();
}

CNNModel::~CNNModel() {
    std::cout << "[CNNModel] Base destructor called" << std::endl;
}

const ModelConfig& CNNModel::getConfig() const {
    return config_;
}

std::vector<float> CNNModel::applySoftmax(const std::vector<float>& scores) const {
    if (scores.empty()) {
        return {};
    }
    
    std::vector<float> probabilities(scores.size());
    
    // Find maximum score for numerical stability
    float maxScore = *std::max_element(scores.begin(), scores.end());
    
    // Compute exponentials
    float sum = 0.0f;
    for (size_t i = 0; i < scores.size(); ++i) {
        probabilities[i] = std::exp(scores[i] - maxScore);
        sum += probabilities[i];
    }
    
    // Normalize to get probabilities
    if (sum > 0.0f) {
        for (size_t i = 0; i < probabilities.size(); ++i) {
            probabilities[i] /= sum;
        }
    }
    
    return probabilities;
}

std::string CNNModel::deviceTypeToString(DeviceType device) const {
    switch (device) {
        case DeviceType::CPU:    return "CPU";
        case DeviceType::CUDA:   return "CUDA";
        case DeviceType::OPENCL: return "OPENCL";
        case DeviceType::AUTO:   return "AUTO";
        default:                 return "UNKNOWN";
    }
}

std::string CNNModel::modelTypeToString(ModelType model) const {
    switch (model) {
        case ModelType::RESNET18:        return "ResNet18";
        case ModelType::RESNET34:        return "ResNet34";
        case ModelType::RESNET50:        return "ResNet50";
        case ModelType::EFFICIENTNET_B0: return "EfficientNet-B0";
        case ModelType::EFFICIENTNET_B4: return "EfficientNet-B4";
        case ModelType::CUSTOM_CNN:      return "Custom-CNN";
        case ModelType::MOBILENET_V2:    return "MobileNet-V2";
        case ModelType::VGG16:           return "VGG16";
        default:                         return "UNKNOWN";
    }
}

// ============================================
// STUB IMPLEMENTATIONS (placeholder backend)
// ============================================

bool CNNModel::initialize(const ModelConfig& config) {
    config.validate();
    config_ = config;
    isInitialized_ = true;
    std::cout << "[CNNModel] Initialized with config" << std::endl;
    return true;
}

bool CNNModel::LoadModel(const std::string& modelPath) {
    try {
        std::cout << "[CNNModel] Loading model from: " << modelPath << std::endl;
        
        // 1. Validate model file exists
        if (!ValidateModelFile(modelPath)) {
            std::cerr << "[CNNModel] ERROR: Invalid model file: " << modelPath << std::endl;
            return false;
        }
        
        // 2. Check file extension
        std::string extension = fs::path(modelPath).extension().string();
        if (extension != ".pt" && extension != ".pth") {
            std::cerr << "[CNNModel] ERROR: Unsupported model format: " << extension 
                     << " (expected .pt or .pth)" << std::endl;
            return false;
        }
        
        // 3. Configure device (CPU/GPU)
        bool useGPU = false;  // Change to true if you have CUDA version
        ConfigureDevice(useGPU);
        
        // 4. Load TorchScript model
        try {
            std::cout << "[CNNModel] Loading TorchScript model..." << std::endl;
            
            // Deserialize the ScriptModule from file
            model_ = torch::jit::load(modelPath, device_);
            
            // Set model to evaluation mode (important for inference)
            model_.eval();
            
            // Enable gradient calculation if needed (set to false for inference)
            torch::NoGradGuard no_grad;
            
            model_loaded_ = true;
            config_.modelPath = modelPath;
            
            std::cout << "[CNNModel] ✓ Model loaded successfully!" << std::endl;
            std::cout << "[CNNModel]   Device: " << (device_.is_cuda() ? "CUDA" : "CPU") << std::endl;
            std::cout << "[CNNModel]   Model type: " << extension << std::endl;
            
            // 5. Determine input shape by doing a test forward pass
            try {
                // Create a dummy input to get input shape
                torch::Tensor dummy_input = torch::randn({1, 3, 224, 224}).to(device_);
                auto output = model_.forward({dummy_input}).toTensor();
                
                input_shape_ = dummy_input.sizes().vec();
                
                std::cout << "[CNNModel]   Input shape: [";
                for (size_t i = 0; i < input_shape_.size(); ++i) {
                    std::cout << input_shape_[i];
                    if (i < input_shape_.size() - 1) std::cout << ", ";
                }
                std::cout << "]" << std::endl;
                
                std::cout << "[CNNModel]   Output shape: [";
                auto output_shape = output.sizes().vec();
                for (size_t i = 0; i < output_shape.size(); ++i) {
                    std::cout << output_shape[i];
                    if (i < output_shape.size() - 1) std::cout << ", ";
                }
                std::cout << "]" << std::endl;
                
                // If output has 2 elements, it's binary classification
                if (output.sizes().back() == 2) {
                    std::cout << "[CNNModel]   Model type: Binary classifier (2 classes)" << std::endl;
                }
                
            } catch (const std::exception& e) {
                std::cout << "[CNNModel] Warning: Could not determine input shape: " 
                         << e.what() << std::endl;
                // Set default input shape
                input_shape_ = {1, 3, 224, 224};  // Default for most CNNs
            }
            
            return true;
            
        } catch (const c10::Error& e) {
            std::cerr << "[CNNModel] ERROR loading TorchScript model: " << e.what() << std::endl;
            std::cerr << "[CNNModel] Make sure the .pt file is a TorchScript model (traced or scripted)" << std::endl;
            return false;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "[CNNModel] ERROR in LoadModel: " << e.what() << std::endl;
        return false;
    }
}

bool CNNModel::LoadTorchScriptModel(const std::string& modelPath , bool useGPU) {
    try{
        std::cout << "[CNNModel] Loading TorchScript model with explicit device..." << std::endl;
        
        // Configure device
        ConfigureDevice(useGPU);
        
        // Load model
        model_ = torch::jit::load(modelPath, device_);
        model_.eval();
        
        model_loaded_ = true;
        config_.modelPath = modelPath;
        
        std::cout << "[CNNModel] ✓ TorchScript model loaded on " 
                  << (device_.is_cuda() ? "GPU" : "CPU") << std::endl;
        
        return true;
    }catch(const std::exception& e){
        std::cerr << "[CNNModel] ERROR loading TorchScript: " << e.what() << std::endl;
        return false;
    }
}

//forward implementation
torch::Tensor CNNModel::Forward(const torch::Tensor& inputTensor) {
    if (!model_loaded_) {
        throw std::runtime_error("[CNNModel] ERROR: Model not loaded. Call LoadModel() first.");
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Forward pass
    torch::NoGradGuard no_grad;  // Disable gradient calculation for inference
    torch::Tensor output = model_.forward({inputTensor}).toTensor();
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> inference_duration = end - start;
    lastInferenceTime_ = inference_duration.count();
    
    return output;
}

//helper methods 

bool CNNModel::ValidateModelFile(const std::string& modelPath) {
    // Check if file exists and is readable
    if (!fs::exists(modelPath)) {
        std::cerr << "[CNNModel] Model file does not exist: " << modelPath << std::endl;
        return false;
    }
    
    // Check file size (should be > 0)
    uintmax_t fileSize = fs::file_size(modelPath);
    if (fileSize == 0) {
        std::cerr << "[CNNModel] Model file is empty: " << modelPath << std::endl;
        return false;
    }
    
    std::cout << "[CNNModel] Model file size: " << fileSize << " bytes" << std::endl;
    return true;
}

void CNNModel::ConfigureDevice(bool useGPU) {
    // Set device (CPU or CUDA)
    if (useGPU && torch::cuda::is_available()) {
        device_ = torch::Device(torch::kCUDA);
        std::cout << "[CNNModel] Using CUDA device" << std::endl;
    } else {
        device_ = torch::Device(torch::kCPU);
        if (useGPU && !torch::cuda::is_available()) {
            std::cout << "[CNNModel] CUDA requested but not available. Using CPU." << std::endl;
        } else {
            std::cout << "[CNNModel] Using CPU device" << std::endl;
        }
    }
}

std::vector<int64_t> CNNModel::GetInputShape() const {
    if (!model_loaded_) {
        throw std::runtime_error("[CNNModel] Model not loaded");
    }
    return input_shape_;
}

bool CNNModel::isLoaded() const {
    return model_loaded_;
}


bool CNNModel::SaveModel(const std::string& modelPath) const {
    try{
        std::cout << "[CNNModel] SaveModel stub called for " << modelPath << std::endl;
        // Check if model is loaded
        if (!model_loaded_) {
            std::cerr << "[CNNModel] ERROR: No model loaded. Cannot save." << std::endl;
            return false;
        }
        
        // Create directory if it doesn't exist
        std::filesystem::path fsPath(modelPath);
        std::filesystem::path dir = fsPath.parent_path();
        if (!dir.empty() && !std::filesystem::exists(dir)) {
            std::filesystem::create_directories(dir);
            std::cout << "[CNNModel] Created directory: " << dir << std::endl;
        }
        
        // Save based on requested format
        std::string upperFormat;
        std::transform(format.begin(), format.end(), 
                      std::back_inserter(upperFormat), ::toupper);
        
        bool success = false;
        
        if (upperFormat == "TORCHSCRIPT" || upperFormat == "PT") {
            success = SaveAsTorchScript(modelPath);
        } 
        else if (upperFormat == "STATELIST" || upperFormat == "PTH") {
            success = SaveStateDict(modelPath);
        }
        else if (upperFormat == "ONNX") {
            success = ExportToONNX(modelPath);
        }
        else {
            std::cerr << "[CNNModel] ERROR: Unknown format: " << format 
                     << " (supported: TORCHSCRIPT, STATELIST, ONNX)" << std::endl;
            return false;
        }
        
        if (success) {
            std::cout << "[CNNModel] ✓ Model saved successfully!" << std::endl;
            std::cout << "[CNNModel]   Format: " << format << std::endl;
            std::cout << "[CNNModel]   Path: " << modelPath << std::endl;
            std::cout << "[CNNModel]   Size: " 
                     << std::filesystem::file_size(modelPath) << " bytes" << std::endl;
        }
        
        return success;

    }catch(const std::exception& e){
        std::cerr << "[CNNModel] ERROR in SaveModel: " << e.what() << std::endl;
        return false;
    }
    
}

bool CNNModel::isReady() const {
    return isInitialized_;
}

ModelOutput CNNModel::predict(const cv::Mat& inputImage) {
    ModelOutput output;
    
    try {
        // 1. Start timing
        auto startTime = std::chrono::high_resolution_clock::now();
        
        // 2. Validate input
        if (inputImage.empty()) {
            throw std::runtime_error("[CNNModel] ERROR: Input image is empty");
        }
        
        if (!model_loaded_) {
            throw std::runtime_error("[CNNModel] ERROR: Model not loaded. Call LoadModel() first.");
        }
        
        std::cout << "[CNNModel] Performing inference on image: " 
                  << inputImage.cols << "x" << inputImage.rows 
                  << " channels: " << inputImage.channels() << std::endl;
        
        // 3. Preprocess image for model input
        torch::Tensor inputTensor;
        try {
            inputTensor = preprocessImage(inputImage);
            std::cout << "[CNNModel] Input tensor shape: [";
            auto shape = inputTensor.sizes().vec();
            for (size_t i = 0; i < shape.size(); ++i) {
                std::cout << shape[i];
                if (i < shape.size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("[CNNModel] Preprocessing failed: ") + e.what());
        }
        
        // 4. Perform inference
        torch::Tensor outputTensor;
        try {
            // Move tensor to correct device
            inputTensor = inputTensor.to(device_);
            
            // Ensure correct dimensions [batch, channels, height, width]
            if (inputTensor.dim() == 3) {
                inputTensor = inputTensor.unsqueeze(0);  // Add batch dimension
            }
            
            // Disable gradient calculation for inference
            torch::NoGradGuard no_grad;
            
            // Forward pass
            outputTensor = model_.forward({inputTensor}).toTensor();
            
            // Move to CPU for processing
            outputTensor = outputTensor.to(torch::kCPU);
            
            std::cout << "[CNNModel] Output tensor shape: [";
            auto out_shape = outputTensor.sizes().vec();
            for (size_t i = 0; i < out_shape.size(); ++i) {
                std::cout << out_shape[i];
                if (i < out_shape.size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
            
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("[CNNModel] Inference failed: ") + e.what());
        }
        
        // 5. Calculate inference time
        auto endTime = std::chrono::high_resolution_clock::now();
        double inferenceTimeMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
        lastInferenceTime_ = inferenceTimeMs;
        totalInferences_++;
        
        // 6. Convert tensor to ModelOutput
        output = tensorToOutput(outputTensor, inferenceTimeMs);
        
        // 7. Add class name
        if (output.predictedClass >= 0 && output.predictedClass < static_cast<int>(classNames_.size())) {
            output.className = classNames_[output.predictedClass];
        } else {
            output.className = "Unknown";
        }
        
        // 8. Log results
        std::cout << "[CNNModel] ✓ Inference successful!" << std::endl;
        std::cout << "[CNNModel]   Predicted class: " << output.predictedClass 
                  << " (" << output.className << ")" << std::endl;
        std::cout << "[CNNModel]   Confidence: " << output.confidence * 100 << "%" << std::endl;
        std::cout << "[CNNModel]   Inference time: " << inferenceTimeMs << " ms" << std::endl;
        
        // Print probabilities for debugging
        if (!output.probabilities.empty()) {
            std::cout << "[CNNModel]   Class probabilities:" << std::endl;
            for (size_t i = 0; i < output.probabilities.size(); ++i) {
                std::string className = (i < classNames_.size()) ? classNames_[i] : "Class " + std::to_string(i);
                std::cout << "    " << className << ": " 
                         << output.probabilities[i] * 100 << "%" << std::endl;
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "[CNNModel] ERROR in predict: " << e.what() << std::endl;
        
        // Return error output
        output.predictedClass = -1;
        output.confidence = 0.0f;
        output.className = "Error";
        output.inferenceTimeMs = 0.0;
        output.probabilities.assign(config_.numClasses, 0.0f);
        output.rawScores.assign(config_.numClasses, 0.0f);
    }
    
    return output;
}

std::vector<ModelOutput> CNNModel::predictBatch(const std::vector<cv::Mat>& inputImages) {
    std::vector<ModelOutput> outputs;
    
    if (inputImages.empty()) {
        std::cerr << "[CNNModel] ERROR: No input images provided for batch prediction" << std::endl;
        return outputs;
    }
    
    try {
        auto startTime = std::chrono::high_resolution_clock::now();
        
        std::cout << "[CNNModel] Starting batch inference on " << inputImages.size() << " images" << std::endl;
        
        // 1. Preprocess all images into a batch tensor
        std::vector<torch::Tensor> tensors;
        for (const auto& img : inputImages) {
            if (img.empty()) {
                std::cerr << "[CNNModel] Warning: Empty image in batch, skipping" << std::endl;
                continue;
            }
            tensors.push_back(preprocessImage(img));
        }
        
        if (tensors.empty()) {
            throw std::runtime_error("[CNNModel] No valid images to process");
        }
        
        // 2. Stack tensors into a batch
        torch::Tensor batchTensor = torch::stack(tensors);
        batchTensor = batchTensor.to(device_);
        
        std::cout << "[CNNModel] Batch tensor shape: [";
        auto shape = batchTensor.sizes().vec();
        for (size_t i = 0; i < shape.size(); ++i) {
            std::cout << shape[i];
            if (i < shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        // 3. Perform batch inference
        torch::NoGradGuard no_grad;
        torch::Tensor batchOutput = model_.forward({batchTensor}).toTensor();
        batchOutput = batchOutput.to(torch::kCPU);
        
        // 4. Calculate total inference time
        auto endTime = std::chrono::high_resolution_clock::now();
        double totalTimeMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
        double avgTimeMs = totalTimeMs / inputImages.size();
        
        std::cout << "[CNNModel] Batch inference completed in " << totalTimeMs << " ms" << std::endl;
        std::cout << "[CNNModel] Average time per image: " << avgTimeMs << " ms" << std::endl;
        
        // 5. Convert each output to ModelOutput
        for (int i = 0; i < batchOutput.size(0); ++i) {
            torch::Tensor singleOutput = batchOutput[i];
            ModelOutput out = tensorToOutput(singleOutput, avgTimeMs);
            
            if (out.predictedClass >= 0 && out.predictedClass < static_cast<int>(classNames_.size())) {
                out.className = classNames_[out.predictedClass];
            }
            
            outputs.push_back(out);
        }
        
    } catch (const std::exception& e) {
        std::cerr << "[CNNModel] ERROR in batch prediction: " << e.what() << std::endl;
    }
    
    return outputs;
}

ModelOutput CNNModel::predictRaw(const std::vector<float>& inputData, 
                                const std::vector<int>& inputShape) {
    ModelOutput output;
    
    try {
        // 1. Start timing
        auto startTime = std::chrono::high_resolution_clock::now();
        
        // 2. Validate input
        if (inputData.empty()) {
            throw std::runtime_error("[CNNModel] ERROR: Input data is empty");
        }
        
        if (inputShape.empty()) {
            throw std::runtime_error("[CNNModel] ERROR: Input shape is empty");
        }
        
        if (!model_loaded_) {
            throw std::runtime_error("[CNNModel] ERROR: Model not loaded. Call LoadModel() first.");
        }
        
        // 3. Validate expected input shape
        std::vector<int> expectedShape = getExpectedInputShape();
        if (!expectedShape.empty()) {
            // Check if input shape matches expected (ignore batch dimension)
            if (inputShape.size() >= 3) {
                // Compare [C, H, W] dimensions
                int inputChannels = inputShape[inputShape.size() - 3];
                int inputHeight = inputShape[inputShape.size() - 2];
                int inputWidth = inputShape[inputShape.size() - 1];
                
                int expectedChannels = expectedShape[expectedShape.size() - 3];
                int expectedHeight = expectedShape[expectedShape.size() - 2];
                int expectedWidth = expectedShape[expectedShape.size() - 1];
                
                if (inputChannels != expectedChannels || 
                    inputHeight != expectedHeight || 
                    inputWidth != expectedWidth) {
                    
                    std::cout << "[CNNModel] WARNING: Input shape mismatch" << std::endl;
                    std::cout << "[CNNModel]   Expected: [";
                    for (size_t i = 0; i < expectedShape.size(); ++i) {
                        std::cout << expectedShape[i];
                        if (i < expectedShape.size() - 1) std::cout << ", ";
                    }
                    std::cout << "]" << std::endl;
                    
                    std::cout << "[CNNModel]   Received: [";
                    for (size_t i = 0; i < inputShape.size(); ++i) {
                        std::cout << inputShape[i];
                        if (i < inputShape.size() - 1) std::cout << ", ";
                    }
                    std::cout << "]" << std::endl;
                    
                    // Continue anyway - model might handle resizing internally
                }
            }
        }
        
        // 4. Calculate expected data size
        size_t expectedSize = 1;
        for (int dim : inputShape) {
            expectedSize *= dim;
        }
        
        if (inputData.size() != expectedSize) {
            std::stringstream ss;
            ss << "[CNNModel] ERROR: Input data size mismatch. "
               << "Expected " << expectedSize << " elements based on shape [";
            for (size_t i = 0; i < inputShape.size(); ++i) {
                ss << inputShape[i];
                if (i < inputShape.size() - 1) ss << ", ";
            }
            ss << "], but got " << inputData.size() << " elements";
            throw std::runtime_error(ss.str());
        }
        
        std::cout << "[CNNModel] predictRaw called with " 
                  << inputData.size() << " elements, shape: [";
        for (size_t i = 0; i < inputShape.size(); ++i) {
            std::cout << inputShape[i];
            if (i < inputShape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        // 5. Create tensor from raw data
        torch::Tensor inputTensor = createTensorFromData(inputData, inputShape);
        
        std::cout << "[CNNModel] Created tensor with shape: [";
        auto tensorShape = inputTensor.sizes().vec();
        for (size_t i = 0; i < tensorShape.size(); ++i) {
            std::cout << tensorShape[i];
            if (i < tensorShape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        // 6. Perform inference using the tensor method
        output = predictFromTensor(inputTensor);
        
        // 7. Update inference time
        auto endTime = std::chrono::high_resolution_clock::now();
        output.inferenceTimeMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
        lastInferenceTime_ = output.inferenceTimeMs;
        
        // 8. Add class name
        if (output.predictedClass >= 0 && output.predictedClass < static_cast<int>(classNames_.size())) {
            output.className = classNames_[output.predictedClass];
        }
        
        std::cout << "[CNNModel] ✓ predictRaw completed in " 
                  << output.inferenceTimeMs << " ms" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "[CNNModel] ERROR in predictRaw: " << e.what() << std::endl;
        
        // Return error output
        output.predictedClass = -1;
        output.confidence = 0.0f;
        output.className = "Error";
        output.inferenceTimeMs = 0.0;
        output.probabilities.assign(config_.numClasses, 0.0f);
        output.rawScores.assign(config_.numClasses, 0.0f);
    }
    
    return output;
}

std::vector<int> CNNModel::getInputShape() const {
    return {config_.inputChannels, config_.inputHeight, config_.inputWidth};
}

std::vector<int> CNNModel::getOutputShape() const {
    return {config_.numClasses};
}

std::string CNNModel::getArchitecture() const {
    return modelTypeToString(config_.modelType);
}

size_t CNNModel::getNumParameters() const {
    return 0;
}

size_t CNNModel::getModelSize() const {
    return 0;
}

void CNNModel::warmUp(int iterations) {
    (void)iterations;
}

float CNNModel::benchmark(int iterations) {
    // Placeholder benchmark: return zero latency
    (void)iterations;
    return 0.0f;
}

double CNNModel::getLastInferenceTime() const {
    return lastInferenceTime_;
}

bool CNNModel::setGPUEnabled(bool enabled) {
    config_.useGPU = enabled;
    return true;
}

bool CNNModel::setPrecision(const std::string& precision) {
    config_.useHalfPrecision = (precision == "FP16" || precision == "half");
    return true;
}

cv::Mat CNNModel::generateHeatmap(const cv::Mat& inputImage, int targetClass) {
    (void)targetClass;
    return inputImage.clone();
}

std::vector<cv::Mat> CNNModel::getFeatureMaps(const cv::Mat& inputImage, const std::string& layerName) {
    (void)inputImage;
    (void)layerName;
    return {};
}

void CNNModel::printSummary() const {
    std::cout << "=== CNNModel Summary ===" << std::endl;
    std::cout << "Architecture: " << getArchitecture() << std::endl;
    std::cout << "Input shape: [" << config_.inputChannels << ", "
              << config_.inputHeight << ", " << config_.inputWidth << "]" << std::endl;
    std::cout << "Num classes: " << config_.numClasses << std::endl;
    std::cout << "Model path: " << config_.modelPath << std::endl;
}

bool CNNModel::exportToONNX(const std::string& outputPath) {
    std::cout << "[CNNModel] exportToONNX stub called for " << outputPath << std::endl;
    return true;
}

void* CNNModel::preprocessInput(const cv::Mat& inputImage) {
    (void)inputImage;
    return nullptr;
}

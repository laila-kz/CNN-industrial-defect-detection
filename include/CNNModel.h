#ifndef CNNMODEL_H
#define CNNMODEL_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <memory>

namespace cv {
    class Mat;
}
//forward declarationfor torch model class
#ifdef USE_TORCH
namespace torch {
    namespace nn {
        class Module;
    }   
}
#endif

//use an enum for the cnn architecture types
enum class ModelType {
    RESNET18,
    RESNET34,
    RESNET50,
    EFFICIENTNET_B0,
    EFFICIENTNET_B4,
    CUSTOM_CNN,      // Your custom architecture
    MOBILENET_V2,
    VGG16
};
//another enum for device type
enum class DeviceType {
    CPU,
    CUDA, 
    OPENCL,
    AUTO
};

//STRUCTURE TO HOLD MODEL CONFIGURATION PARAMETERS
struct ModelConfig{
    //model arch 
    ModelType modelType = ModelType::RESNET18; //default model

    //model file path
    std::string modelPath = "./models/defect_model.pt";  //pytroch model file
    std::string onnxPath = "./models/defect_model.onnx"; //onnx model file

    //input image size
    int inputWidth = 224;
    int inputHeight = 224;
    int inputChannels = 3; //RGB

    //output classes
    int numClasses = 2; //binary classification by default
    std::vector<std::string> classNames = {"defect", "no_defect"};

    //device type
    std::string device = "CPU"; //default to CPU
    bool useGPU = false;
    int gpuID = 0; //default GPU ID


    //inference settings 
    float confidenceThreshold = 0.5f; //default confidence threshold
    int batchSize = 1; //default batch size
    bool useHalfPrecision = false; //default to full precision

    //perference tuning
    bool enableOptimizations = true; //enable optimizations by default
    int numThreads = 1; //default number of threads for cpu inference 

    //model - specific parameters
    bool isPretrained = true; //use pretrained weights by default
    bool freezeLayers = false; //do not freeze layers by default

    //constructor
    ModelConfig() = default;
    void validate() const;
   

};


//model output structure : contain model inference output results 
// Contains raw model outputs, class probabilities, and predictions.
// Decision logic is applied separately based on these outputs.

struct ModelOutput {
    //logits output from the model
    std::vector<float> rawScores; //raw output scores from the model

    //prob after softmax 
    std::vector<float> probabilities; //class probabilities after softmax

    //predicted class index
    int predictedClass = -1; //index of the predicted class

    //predicted confidence score
    float confidence = 0.0f; //confidence score of the prediction

    //inference time in milliseconds
    double inferenceTimeMs = 0.0f; //time taken for inference

    //feature maps from intermediate layers (if needed)
    std::vector<cv::Mat> featureMaps;

    //heatmaps for visualization (if applicable)
    std::vector<cv::Mat> heatmaps;

    //constructor
    ModelOutput() = default;

    void clear();
    bool isValid() const ;

    std::string getClassName(const std::vector<std::string>& classNames) const;

    void print() const;
};
/*    
* This class provides a unified interface for different CNN backends:
 * - PyTorch C++ (LibTorch)
 * - ONNX Runtime
 * - TensorFlow C++
 * - Custom C++ implementation
 * 
 * Key responsibilities:
 * 1. Model initialization and loading
 * 2. Forward pass inference
 * 3. Memory management
 * 
 * Key design principle: This class ONLY handles model inference.
 * Decision logic, preprocessing, and visualization are separate.
 */
class CNNModel {
    public:
    CNNModel();

    explicit CNNModel(const ModelConfig& config);

    virtual ~CNNModel();
    virtual bool initialize(const ModelConfig& config);

    torch::Tensor Forward(const torch::Tensor& inputTensor);
    std::vector<int64_t> GetInputShape() const ;
    virtual bool LoadModel(const std::string& modelPath); //Convert and load PyTorch model for inference
    virtual bool LoadTorchScriptModel(const std::string& modelPath , bool useGPU = false); //load torchscript model

    //save model to file 
    virtual bool SaveModel(const std::string& modelPath) const;

    virtual bool isReady() const;
    
    virtual const ModelConfig& getConfig() const;

    //inference methods
    virtual ModelOutput predict(const cv::Mat& inputImage);
    virtual std::vector<ModelOutput> predictBatch(const std::vector<cv::Mat>& inputImages);

    virtual ModelOutput predictRaw(const std::vector<float>& inputData , const std::vector<int>& inputShape);
    //model iinfos methods 
    virtual std::vector<int> getInputShape() const;
    virtual std::vector<int> getOutputShape() const;
    virtual std::string getArchitecture() const;
    virtual size_t getNumParameters() const;

    virtual size_t getModelSize() const;

    //performance methods 
    //run dummy inference to warm up the model
    virtual void warmUp(int iterations = 5);

    //metric used to evaluate how well a model performs
    virtual float benchmark(int iterations = 50);

    //get last inference time in milliseconds
    virtual double getLastInferenceTime() const;

    //advanced features 
    virtual bool setGPUEnabled(bool enabled);
    virtual bool setPrecision(const std::string&  precision);
    //generatae grad cam heatmap for visualization
    virtual cv::Mat generateHeatmap(const cv::Mat& inputImage , int targetClass =-1);

    //get feature maps from specified layer
    virtual std::vector<cv::Mat> getFeatureMaps(const cv::Mat& inputImage , const std::string& layerName);

    //utility methods 
    virtual void printSummary() const;

    //Export model to ONNX format
    virtual bool exportToONNX(const std::string& outputPath);

    //convert opencv mat tp model input tensor 
    virtual void* preprocessInput(const cv::Mat& inputImage);

protected:
    //protected member vars 
    ModelConfig config_;
    bool isInitialized_ = false ;
    double lastInferenceTime_ = 0.0 ;
    //protected helper methods 

    virtual std::vector<float> applySoftmax(const std::vector<float>& scores )const ;
    std::string deviceTypeToString(DeviceType device) const;
    std::string modelTypeToString(ModelType model) const;

private:
    // Private copy constructor and assignment operator to prevent copying
    CNNModel(const CNNModel&) = delete;
    CNNModel& operator=(const CNNModel&) = delete;
    torch::jit::script::Module model_;  // TorchScript module
    torch::Device device_;              // CPU or CUDA device
    bool model_loaded_ = false;         // Model loading flag
    std::vector<int64_t> input_shape_;  // Expected input shape
    
    // Private helper methods
    bool ValidateModelFile(const std::string& modelPath);
    void ConfigureDevice(bool useGPU);

    
};






#endif // CNNMODEL_H
#include "Preprocessor.h"
#include <iostream>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <cmath>


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

//helper macro for throwing exceptions with file line info 
#define THROW_IF(condition , message) \
    do { \
        if (condition) { \
            throw std::runtime_error(std::string("[Preprocessor] Error: ") + message + \
                                     " (File: " + __FILE__ + ", Line: " + std::to_string(__LINE__) + ")"); \
        } \
    } while (0) 


//implementation of Preprocessor class

void PreprocessorConfig::validate() const{
    //check dim
    THROW_IF(targetWidth <=0 || targetHeight <=0, "Target dimensions must be positive integers");
    //check normalization vectors
    THROW_IF(mean.size() !=3, "Mean vector must have exactly 3 elements");
    THROW_IF(stdDev.size() !=3, "StdDev vector must have exactly 3 elements");

    for (double s : stdDev) {
        THROW_IF(s <= 0.0, "StdDev values must be positive");

    }

    //check augmentation flags

}

//constructor implementations
Preprocessor::Preprocessor(){
    config_ = PreprocessorConfig();
    config_.validate();
    std::cout << "[Preprocessor] Initialized with default config" << std::endl;
}
//deconstructor with config
Preprocessor::Preprocessor(const PreprocessorConfig& config){
    config_ = config;
    config_.validate();
    std::cout << "[Preprocessor] Initialized with custom config" << std::endl;
}

Preprocessor::~Preprocessor(){
    std::cout << "[Preprocessor] Destroyed" << std::endl;
}

//configuration methods
void Preprocessor::setConfig(const PreprocessorConfig& config){
    config.validate();
    config_ = config;
    std::cout << "[Preprocessor] Configuration updated" << std::endl;
}

const PreprocessorConfig& Preprocessor::getConfig() const{
    return config_;
}

//the main preprocessing method
cv::Mat Preprocessor::preprocess(const cv::Mat& inputImage) const{
    //validate input
    validateImage(inputImage);
    //apply the preprocessing pipeline
    cv::Mat preprocessedImage = applyPipeline(inputImage);
    return preprocessedImage;
}

std::vector<cv::Mat> Preprocessor::preprocessBatch(const std::vector<cv::Mat>& inputImages) const{
    std::vector<cv::Mat> preprocessedImages;
    preprocessedImages.reserve(inputImages.size());
    int preprocessedCount = 0;
    int failedCount = 0;
    for (const auto& image : inputImages){
        try{
            cv::Mat preprocessedImage = preprocess(image);
            preprocessedImages.push_back(preprocessedImage);
            preprocessedCount++;
        }catch(const std::runtime_error& e){
            std::cerr << "[Preprocessor] Warning: Failed to preprocess an image: " << e.what() << std::endl;
            failedCount++;
        }
    }
    std::cout << "[Preprocessor] Preprocessed " << preprocessedCount << " images, " << failedCount << " failures." << std::endl;
    return preprocessedImages;
}

cv::Mat Preprocessor::preprocessWithAugmentation(const cv::Mat& inputImage, unsigned int seed){
    //validate input
    validateImage(inputImage);
    cv::Mat augmentedImage = inputImage.clone();
    //apply augmentations based on config
    if(config_.enableRandomFlip){
        augmentedImage = applyRandomFlip(augmentedImage, seed);
    }
    if(config_.enableRandomBrightnessContrast){
        augmentedImage = applyRandomBrightnessContrast(augmentedImage, seed);
    }
    //finally apply standard pipeline
    cv::Mat preprocessedImage = applyPipeline(augmentedImage);
    return preprocessedImage;
}


//individual preprocessing steps implementations
cv::Mat Preprocessor::resize(const cv::Mat& image) const{
    if(!config_.resize){
        return image.clone(); //return copy if resize disabled
    }
    cv::Mat resizedImage;
    cv::resize(image, resizedImage, cv::Size(config_.targetWidth, config_.targetHeight));
    return resizedImage;
}

cv::Mat Preprocessor::convertColorSpace(const cv::Mat& image) const{
    if(!config_.convertColor){
        return image.clone(); //return copy if conversion disabled
    }
    cv::Mat convertedImage;
    if(config_.convertToRGB){
        cv::cvtColor(image, convertedImage, cv::COLOR_BGR2RGB);
    }else{
        cv::cvtColor(image, convertedImage, cv::COLOR_RGB2BGR);
    }
    return convertedImage;
}

cv::Mat Preprocessor::normalize(const cv::Mat& image) const{
    if(!config_.normalize){
        return image.clone(); //return copy if normalization disabled
    }
    cv::Mat floatImage;
    image.convertTo(floatImage, CV_32F, 1.0 / 255.0); //scale to [0,1]
    std::vector<cv::Mat> channels(3);
    cv::split(floatImage, channels);
    for(int i =0; i <3; ++i){
        channels[i] = (channels[i] - config_.mean[i]) / config_.stdDev[i];
    }
    cv::Mat normalizedImage;
    cv::merge(channels, normalizedImage);
    return normalizedImage;
}
cv::Mat Preprocessor::denoiseImage(const cv::Mat& image) const{
    if(!config_.denoising){
        return image.clone(); //return copy if denoising disabled
    }
    cv::Mat denoisedImage;
    cv::fastNlMeansDenoisingColored(image, denoisedImage, 10, 10, 7, 21);
    return denoisedImage;
}

cv::Mat Preprocessor::equalizeHist(const cv::Mat& image) const{
    if(!config_.histogramEqualization){
        return image.clone(); //return copy if equalization disabled
    }
    cv::Mat ycrcbImage;
    cv::cvtColor(image, ycrcbImage, cv::COLOR_BGR2YCrCb);
    std::vector<cv::Mat> channels(3);
    cv::split(ycrcbImage, channels);
    cv::equalizeHist(channels[0], channels[0]); //equalize Y channel
    cv::Mat equalizedImage;
    cv::merge(channels, ycrcbImage);
    cv::cvtColor(ycrcbImage, equalizedImage, cv::COLOR_YCrCb2BGR);
    return equalizedImage;
}

//data augmentation methods implementations
cv::Mat Preprocessor::applyRandomRotation(const cv::Mat& image, unsigned int seed) const {
    if(config_.rotationRange <= 0.0){
        return image.clone(); //return copy if rotation disabled
    }
    //generate random angle
    float angle = randomFloat(-config_.rotationRange, config_.rotationRange, seed);
    //get the img center
    cv::Point2f center(static_cast<float>(image.cols) / 2.0f,
                      static_cast<float>(image.rows) / 2.0f);

    //get rotation matrix
    cv::Mat rotMatrix = cv::getRotationMatrix2D(center, angle, 1.0);

    //calculate bounding box
    cv::Rect bbox = cv::RotatedRect(center, image.size(), angle).boundingRect();
    //adjust transformation matrix
    rotMatrix.at<double>(0,2) += bbox.width /2.0 -center.x;
    rotMatrix.at<double>(1,2) += bbox.height /2.0 -center.y;

    //apply the rotation 
    cv::Mat rotatedImage;
    cv::warpAffine(image, rotatedImage, rotMatrix, bbox.size());
    return rotatedImage;

}

cv::Mat Preprocessor::applyRandomFlip(const cv::Mat& image , unsigned int seed) const {
    if(!config_.horizontalFlip && !config_.verticalFlip){
        return image.clone(); //return copy if flipping disabled
    }
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0,2); //0: no flip, 1: horizontal, 2: vertical
    cv::Mat flipped = image.clone();
    int flipCode = dist(rng);
    int actualFlip = -2;
    if(flipCode ==0 && config_.verticalFlip){
        actualFlip =0;
    }else if(flipCode ==1 && config_.horizontalFlip){
        actualFlip =1;
    }else if(flipCode == 2 && config_.horizontalFlip && config_.verticalFlip){
        actualFlip = -1; //both flips
    }
    if(actualFlip >= -1){
        cv::flip(image, flipped, actualFlip);
    }
    return flipped;
}

cv::Mat Preprocessor::applyRandomBrightnessContrast(const cv::Mat& image, unsigned int seed) const {
    if(config_.brightnessRange <=0.0f && config_.contrastRange <=0.0f){
        return image.clone(); //return copy if adjustment disabled
    }
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> brightnessDist(-config_.brightnessRange, config_.brightnessRange);
    std::uniform_real_distribution<float> contrastDist(-config_.contrastRange, config_.contrastRange);

    float brightnessShift = brightnessDist(rng);
    float contrastFactor = 1.0f + contrastDist(rng);

    cv::Mat adjustedImage;
    image.convertTo(adjustedImage, -1, contrastFactor, brightnessShift *255.0f);
    return adjustedImage;
}


//utility methods implementations
// ============================================
// UTILITY METHODS
// ============================================

std::vector<float> Preprocessor::matToVector(const cv::Mat& image) const {
    // Expects normalized float image
    THROW_IF(image.type() != CV_32FC3, 
             "Input image must be CV_32FC3 for vector conversion");
    
    // Get image dimensions
    int channels = image.channels();
    int height = image.rows;
    int width = image.cols;
    
    // Create vector with correct size
    std::vector<float> flatData;
    flatData.reserve(channels * height * width);
    
    // Convert to CHW format (channels first, row-major)
    std::vector<cv::Mat> channelMats;
    cv::split(image, channelMats);
    
    for (int c = 0; c < channels; c++) {
        for (int h = 0; h < height; h++) {
            const float* rowPtr = channelMats[c].ptr<float>(h);
            for (int w = 0; w < width; w++) {
                flatData.push_back(rowPtr[w]);
            }
        }
    }
    
    return flatData;
}

std::vector<int> Preprocessor::getInputShape() const {
    // Returns CNN input shape: {channels, height, width}
    return {3, config_.targetHeight, config_.targetWidth};
}

void Preprocessor::printConfig() const {
    std::cout << "\n=== Preprocessor Configuration ===" << std::endl;
    std::cout << "Target size: " << config_.targetWidth 
              << "x" << config_.targetHeight << std::endl;
    std::cout << "Normalization mean: [" 
              << config_.mean[0] << ", " 
              << config_.mean[1] << ", " 
              << config_.mean[2] << "]" << std::endl;
    std::cout << "Normalization std: [" 
              << config_.stdDev[0] << ", " 
              << config_.stdDev[1] << ", " 
              << config_.stdDev[2] << "]" << std::endl;
    std::cout << "Convert BGR->RGB: " 
              << (config_.convertToRGB ? "Yes" : "No") << std::endl;
    std::cout << "Denoising: " 
              << (config_.denoising ? "Yes" : "No") << std::endl;
    std::cout << "Augmentation: " 
              << (config_.enableRandomFlip || config_.enableRandomBrightnessContrast ? "Yes" : "No") << std::endl;
    if (config_.enableRandomFlip || config_.enableRandomBrightnessContrast) {
        std::cout << "  Rotation range: ±" << config_.rotationRange << "°" << std::endl;
        std::cout << "  Horizontal flip: " << (config_.horizontalFlip ? "Yes" : "No") << std::endl;
        std::cout << "  Brightness range: ±" << config_.brightnessRange << std::endl;
    }
    std::cout << "==================================\n" << std::endl;
}

// ============================================
// PRIVATE HELPER METHODS
// ============================================

cv::Mat Preprocessor::applyPipeline(const cv::Mat& image) const {
    cv::Mat processed = image.clone();
    
    // Apply preprocessing steps in correct order
    if (config_.denoising) {
        processed = denoiseImage(processed);
    }
    
    if (config_.histogramEqualization) {
        processed = equalizeHist(processed);
    }
    
    if (config_.resize) {
        processed = resize(processed);
    }
    
    if (config_.convertColor && config_.convertToRGB) {
        processed = convertColorSpace(processed);
    }
    
    if (config_.normalize) {
        processed = normalize(processed);
    }
    
    return processed;
}

void Preprocessor::validateImage(const cv::Mat& image) const {
    THROW_IF(image.empty(), "Input image is empty");
    THROW_IF(image.cols <= 0 || image.rows <= 0, 
             "Image dimensions must be positive");
    
    // Check if image has valid number of channels
    int channels = image.channels();
    THROW_IF(channels != 1 && channels != 3 && channels != 4,
             "Image must have 1, 3, or 4 channels. Got: " + std::to_string(channels));
    
    // Check depth (8-bit, 16-bit, or 32-bit float)
    int depth = image.depth();
    bool validDepth = (depth == CV_8U || depth == CV_16U || 
                       depth == CV_32F || depth == CV_64F);
    THROW_IF(!validDepth, "Unsupported image depth");
}

float Preprocessor::randomFloat(float min, float max, unsigned int seed) const {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(min, max);
    return dist(rng);
}
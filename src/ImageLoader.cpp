//Implements: ImageLoader

#include "../include/ImageLoader.h"
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <stdexcept>

//for directory operations
namespace fs = std::filesystem;

// SUPPORTED IMG FORMATS
const std::vector<std::string> SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"};

// Constructor
ImageLoader::ImageLoader()
    : frameWidth_(0), frameHeight_(0), 
      totalFrames_(0), currentFrameIndex_(0) {
    // Initialize to default values
    std::cout << "[ImageLoader] Initialized" << std::endl;
}

ImageLoader::~ImageLoader(){
    if(videoCapture_.isOpened()){
        videoCapture_.release();
    }
    std::cout << "[ImageLoader] Destroyed" << std::endl;
}

// Load a single image from file
cv::Mat ImageLoader::loadImage(const std::string& imagePath){
    //1 validate path 
    if(!validateImagePath(imagePath)){
        throw std::runtime_error("invalid image path: " + imagePath);
    }
    //2 check format
    if(!isImageFormatSupported(imagePath)){
        throw std::runtime_error("unsupported image format: " + imagePath);
    }
    //3 load using open cv
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    //4check if loaded
    if(image.empty()){
        throw std::runtime_error("failed to load image: " + imagePath);
    }
    //5 update internal state if needed
    updateFrameInfo(image);
    std::cout << "[ImageLoader] Loaded image: " << imagePath << std::endl;
    return image;

}
// Load multiple images from a directory
std::vector<cv::Mat> ImageLoader::loadImagesFromDirectory(const std::string& directoryPath){
    std::vector<cv::Mat> images;
    //1 validate directory path
    if(!fs::exists(directoryPath) || !fs::is_directory(directoryPath)){
        throw std::runtime_error("invalid directory path: " + directoryPath);
    }
    //2 iterate over files in directory
    for(const auto& entry : fs::directory_iterator(directoryPath)){
        if(entry.is_regular_file()){
            const std::string filePath = entry.path().string();
            //3 check if format is supported
            if(isImageFormatSupported(filePath)){
                try{
                    cv::Mat img = loadImage(filePath);
                    images.push_back(img);
                }catch(const std::runtime_error& e){
                    std::cerr << "[ImageLoader] Warning: " << e.what() << std::endl;
                }
            }
        }
    }
    std::cout << "[ImageLoader] Loaded " << images.size() << " images from directory: " << directoryPath << std::endl;
    return images;
}

//webcam and video methods would go here


//Utility methods
std::vector<std::string>ImageLoader::getSupportedFormats() const{
    return SUPPORTED_EXTENSIONS;
}
bool ImageLoader::isImageFile(const std::string& filePath) const{
    //convert to lower case for comparison
    std::string pathLower =filePath;
    std::transform(pathLower.begin(), pathLower.end(), pathLower.begin(), ::tolower);

    //check for all extensions
    for(const auto& ext : SUPPORTED_EXTENSIONS){
        if(pathLower.size() >= ext.size() &&
           pathLower.compare(pathLower.size() - ext.size(), ext.size(), ext) == 0){
            return true;
        }
    }
    return false;
}

bool ImageLoader::isImageFormatSupported(const std::string& filePath) const{
    return isImageFile(filePath);
}

bool ImageLoader::isFormatSupported(const std::string& fileExtension) const{
    std::string extLower = fileExtension;
    std::transform(extLower.begin(), extLower.end(), extLower.begin(), ::tolower);
    
    // Add leading dot if not present
    if(!extLower.empty() && extLower[0] != '.'){
        extLower = "." + extLower;
    }
    
    for(const auto& ext : SUPPORTED_EXTENSIONS){
        if(ext == extLower){
            return true;
        }
    }
    return false;
}

//getter methods for frame info
int ImageLoader::getCurrentWidth() const {
    return frameWidth_;
}
int ImageLoader::getCurrentHeight() const {
    return frameHeight_;
}
int ImageLoader::getTotalFrames() const {
    return totalFrames_;
}
int ImageLoader::getCurrentFrameIndex() const {
    return currentFrameIndex_;
}

//Private heloer methods

bool ImageLoader::validateImagePath(const std::string& imagePath) const{
    return fs::exists(imagePath) && fs::is_regular_file(imagePath);
}
void ImageLoader::updateFrameInfo(const cv::Mat& frame){
    if(!frame.empty()){
        currentFrame_ = frame.clone();
        frameWidth_ = frame.cols;
        frameHeight_ = frame.rows;
    }
}

std::vector<std::string>ImageLoader::scanImageFiles(const std::string& folderPath) const{
    std::vector<std::string> imagePaths;
    try{
        //iterare through directory
        for(const auto& entry : fs::directory_iterator(folderPath)){
            if(entry.is_regular_file()){
                const std::string filePath = entry.path().string();
                if(isImageFormatSupported(filePath)){
                    imagePaths.push_back(filePath);
                }
            }
        }
        std::sort(imagePaths.begin(), imagePaths.end());
    }catch(const fs::filesystem_error& e){
        std::cerr << "[ImageLoader] Filesystem error: " << e.what() << std::endl;
    }
    return imagePaths;
}
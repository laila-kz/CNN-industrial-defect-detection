// Image acquisition interface
//Create a clean C++ interface (header file) that defines WHAT the ImageLoader can do, not HOW it does it.
#ifndef IMAGELOADER_H
#define IMAGELOADER_H

// Add necessary includes
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

namespace cv {
    class Mat; // Forward declaration of OpenCV Mat class
}
/**
 * @class ImageLoader
 * @brief Universal image acquisition interface for defect detection system
 * 
 * This class provides a unified interface to load images from various sources:
 * - Single image files
 * - Folders of images (batch loading)
 * - Webcam streams x not yet implemented
 * - Video files x not yet implemented
 * 
 * @note All images are returned in OpenCV's standard BGR format
 * @note The class maintains internal state for video/webcam streams
 */
// Image loader class 
class ImageLoader {
public:
    // Constructor
    /**
     * @brief Default constructor
     */
    ImageLoader();

    // Destructor
    /**
     * @brief Virtual destructor for proper inheritance
     */
    virtual ~ImageLoader();
    
    /**
     * @brief Loads a single image from the specified file path
     * 
     * @param imagePath Full path to the image file
     * @return cv::Mat Loaded image in BGR format
     * @throws std::runtime_error if file doesn't exist or cannot be read
     */
    cv::Mat loadImage(const std::string& filePath);
    
    /**
     * @brief Loads multiple images from a folder for batch processing
     * 
     * @param folderPath Path to folder containing images
     * @param batchSize Number of images to load (-1 for all images)
     * @return std::vector<cv::Mat> Vector of loaded images
     * @note Images are loaded in alphabetical order
     */
    std::vector<cv::Mat> loadImagesFromDirectory(const std::string& directoryPath);

    /**
     * @brief Check if the image file format is supported
     * @param fileExtension File extension to check
     * @return true if supported, false otherwise
     */
    bool isFormatSupported(const std::string& fileExtension) const;
    
    /**
     * @brief Get list of supported image formats
     * @return Vector of supported file extensions
     */
    std::vector<std::string> getSupportedFormats() const;
    
    /**
     * @brief Check if a file is an image file
     * @param filePath Path to check
     * @return true if it's a supported image file
     */
    bool isImageFile(const std::string& filePath) const;
    
    // Getters for frame information
    int getCurrentWidth() const;
    int getCurrentHeight() const;
    int getTotalFrames() const;
    int getCurrentFrameIndex() const;

private:
    // Member variables
    cv::Mat currentFrame_;
    cv::VideoCapture videoCapture_;
    int frameWidth_;
    int frameHeight_;
    int totalFrames_;
    int currentFrameIndex_;
    
    // Helper methods
    bool validateImagePath(const std::string& imagePath) const;
    bool isImageFormatSupported(const std::string& filePath) const;
    void updateFrameInfo(const cv::Mat& frame);
    std::vector<std::string> scanImageFiles(const std::string& folderPath) const;
};




#endif
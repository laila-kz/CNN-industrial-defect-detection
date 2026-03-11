#ifndef PREPROCESSOR_H
#define PREPROCESSOR_H

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

namespace cv {
    class Mat;
}

//config structure for the img preprocessing 
//contains all params needed for preprocessing
//populated from a config file or user input

struct PreprocessorConfig
{
    //target img size 
    int targetWidth = 224;
    int targetHeight = 224;

    //normalization params
    std::vector<double> mean = {0.485, 0.456, 0.406};
    std::vector<double> stdDev = {0.229, 0.224, 0.225};

    //color space conversion
    bool convertToRGB = true; //model expects RGB input

    //preprocessing steps
    bool resize = true;
    bool normalize = true;
    bool convertColor = false;

    //denoising
    bool denoising = false;

    //histogram equalization
    bool histogramEqualization = false;

    //augmentation params (for training)
    bool enableRandomFlip = false;
    bool enableRandomBrightnessContrast = false;
    bool horizontalFlip = false;
    bool verticalFlip = false;
    float rotationRange = 0.0f;
    float brightnessRange = 0.0f;
    float contrastRange = 0.0f;

    //constructeur 
    PreprocessorConfig() = default;
    void validate() const;

};
//class Preprocessor declaration
// This class prepares images for CNN input by:
//  * 1. Resizing to consistent dimensions
//  * 2. Converting color space (BGR to RGB)
//  * 3. Normalizing pixel values

class Preprocessor {
    public:
        Preprocessor();
        explicit Preprocessor(const PreprocessorConfig& config);

        virtual ~Preprocessor();
        //set preprocessing config
        //config new configuration to use 
        void setConfig(const PreprocessorConfig& config);

        //get current config
        //returns the current preprocessing configuration
        const PreprocessorConfig& getConfig() const;


        //Main preprocessing method
        //Preprocess the input image according to the config
        //input: cv::Mat image to preprocess
        //returns: preprocessed cv::Mat

        cv::Mat preprocess(const cv::Mat& inputImage) const;

        std::vector<cv::Mat> preprocessBatch(const std::vector<cv::Mat>& inputImages) const;

        cv::Mat preprocessWithAugmentation(const cv::Mat& inputImage, unsigned int seed = 0);
    
    // ============================================
    // INDIVIDUAL PREPROCESSING STEPS
    // ============================================
    
    /**
     * @brief Resize image to target dimensions
     * 
     * @param image Input image
     * @return cv::Mat Resized image
     */
    cv::Mat resize(const cv::Mat& image) const;
    
    /**
     * @brief Convert color space (BGR to RGB or vice versa)
     * 
     * @param image Input image
     * @return cv::Mat Color-converted image
     */
    cv::Mat convertColorSpace(const cv::Mat& image) const;
    
    /**
     * @brief Normalize pixel values using mean and std
     * 
     * @param image Input image (typically after color conversion)
     * @return cv::Mat Normalized image
     */
    cv::Mat normalize(const cv::Mat& image) const;
    
    /**
     * @brief Apply denoising filter
     * 
     * @param image Input image
     * @return cv::Mat Denoised image
     */
    cv::Mat denoiseImage(const cv::Mat& image) const;
    
    /**
     * @brief Apply histogram equalization
     * 
     * @param image Input image
     * @return cv::Mat Image with enhanced contrast
     */
    cv::Mat equalizeHist(const cv::Mat& image) const;
    
    // ============================================
    // DATA AUGMENTATION METHODS (TRAINING ONLY)
    // ============================================
    
    /**
     * @brief Apply random rotation
     * 
     * @param image Input image
     * @param seed Random seed
     * @return cv::Mat Rotated image
     */
    cv::Mat applyRandomRotation(const cv::Mat& image, unsigned int seed = 0) const;
    
    /**
     * @brief Apply random flip (horizontal/vertical)
     * 
     * @param image Input image
     * @param seed Random seed
     * @return cv::Mat Flipped image
     */
    cv::Mat applyRandomFlip(const cv::Mat& image, unsigned int seed = 0) const;
    
    /**
     * @brief Apply random brightness/contrast adjustment
     * 
     * @param image Input image
     * @param seed Random seed
     * @return cv::Mat Adjusted image
     */
    cv::Mat applyRandomBrightnessContrast(const cv::Mat& image, unsigned int seed = 0) const;
    
    // ============================================
    // UTILITY METHODS
    // ============================================
    
    /**
     * @brief Convert OpenCV Mat to vector<float> (for CNN input)
     * 
     * @param image Preprocessed image
     * @return std::vector<float> Flattened image data
     */
    std::vector<float> matToVector(const cv::Mat& image) const;
    
    /**
     * @brief Get expected input tensor shape for the CNN
     * 
     * @return std::vector<int> Tensor shape: {channels, height, width}
     */
    std::vector<int> getInputShape() const;
    
    /**
     * @brief Print current preprocessing configuration
     */
    void printConfig() const;
    
private:
    // ============================================
    // PRIVATE MEMBER VARIABLES
    // ============================================
    PreprocessorConfig config_;  ///< Preprocessing configuration
    
    // ============================================
    // PRIVATE HELPER METHODS
    // ============================================
    
    /**
     * @brief Apply standard preprocessing pipeline (no augmentation)
     * 
     * @param image Input image
     * @return cv::Mat Fully preprocessed image
     */
    cv::Mat applyPipeline(const cv::Mat& image) const;
    
    /**
     * @brief Validate that image meets preprocessing requirements
     * 
     * @param image Image to validate
     * @throws std::invalid_argument if image is invalid
     */
    void validateImage(const cv::Mat& image) const;
    
    /**
     * @brief Generate random float in range [min, max]
     * 
     * @param min Minimum value
     * @param max Maximum value
     * @param seed Random seed
     * @return float Random value
     */
    float randomFloat(float min, float max, unsigned int seed = 0) const;
};

#endif // PREPROCESSOR_H
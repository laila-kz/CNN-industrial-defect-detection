#include "DataLoader.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <algorithm>
#include <random>
#include <thread>
#include <mutex>
#include <chrono>
#include <iomanip>

namespace fs = std::filesystem;
using namespace torch;

//data Loader implementation and config 
void DataLoaderConfig::validate() const {
    // Check if data root path exists
    if (!fs::exists(dataRootPath)) {
        throw std::runtime_error("Data root path does not exist: " + dataRootPath);
    }
    
    // Check class names
    if (classNames.empty()) {
        throw std::runtime_error("Class names list cannot be empty");
    }
    
    // Check batch size
    if (batchSize <= 0) {
        throw std::runtime_error("Batch size must be positive");
    }
    
    // Check target size
    if (targetSize.width <= 0 || targetSize.height <= 0) {
        throw std::runtime_error("Target image size must be positive");
    }
    
    // Check normalization parameters
    if (normalize) {
        if (mean.size() != 3 || stdDev.size() != 3) {
            throw std::runtime_error("Normalization parameters must have 3 values each");
        }
    }
    
    // Check augmentation parameters
    if (rotationRange < 0 || rotationRange > 360) {
        throw std::runtime_error("Rotation range must be between 0 and 360 degrees");
    }
    
    if (zoomRange < 0 || zoomRange > 1.0f) {
        throw std::runtime_error("Zoom range must be between 0 and 1");
    }
    
    if (horizontalFlipProb < 0 || horizontalFlipProb > 1.0f) {
        throw std::runtime_error("Horizontal flip probability must be between 0 and 1");
    }
    
    if (verticalFlipProb < 0 || verticalFlipProb > 1.0f) {
        throw std::runtime_error("Vertical flip probability must be between 0 and 1");
    }
    
    std::cout << "[DataLoaderConfig] Configuration validated successfully" << std::endl;
}

void DataLoaderConfig::loadFromYAML(const std::string& filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open config file: " + filePath);
    }
    
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string key;
        if (std::getline(iss, key, ':')) {
            std::string value;
            if (std::getline(iss, value)) {
                // Trim whitespace
                key.erase(key.find_last_not_of(" \n\r\t")+1);
                value.erase(0, value.find_first_not_of(" \n\r\t"));
                
                // Parse known keys
                if (key == "dataRootPath") {
                    dataRootPath = value;
                } else if (key == "trainFolder") {
                    trainFolder = value;
                } else if (key == "valFolder") {
                    valFolder = value;
                } else if (key == "testFolder") {
                    testFolder = value;
                } else if (key == "batchSize") {
                    batchSize = std::stoi(value);
                } else if (key == "shuffle") {
                    shuffle = (value == "true");
                } else if (key == "numWorkers") {
                    numWorkers = std::stoi(value);
                }
                // Add more keys as needed
            }
        }
    }
    
    file.close();
    std::cout << "[DataLoaderConfig] Loaded configuration from: " << filePath << std::endl;
}

// Print summary of the configuration
void DataLoaderConfig::printSummary() const {
    std::cout << "\n=== DATA LOADER CONFIGURATION ===\n";
    std::cout << "Data Paths:\n";
    std::cout << "  Root: " << dataRootPath << "\n";
    std::cout << "  Train Folder: " << trainFolder << "\n";
    std::cout << "  Val Folder: " << valFolder << "\n";
    std::cout << "  Test Folder: " << testFolder << "\n";
    
    std::cout << "\nClasses (" << classNames.size() << "):\n";
    for (size_t i = 0; i < classNames.size(); ++i) {
        std::cout << "  " << i << ": " << classNames[i] << "\n";
    }
    
    std::cout << "\nBatch Configuration:\n";
    std::cout << "  Batch Size: " << batchSize << "\n";
    std::cout << "  Shuffle: " << (shuffle ? "Yes" : "No") << "\n";
    std::cout << "  Workers: " << numWorkers << "\n";
    
    std::cout << "\nImage Preprocessing:\n";
    std::cout << "  Target Size: " << targetSize.width << "x" << targetSize.height << "\n";
    std::cout << "  Normalize: " << (normalize ? "Yes" : "No") << "\n";
    if (normalize) {
        std::cout << "  Mean: [" << mean[0] << ", " << mean[1] << ", " << mean[2] << "]\n";
        std::cout << "  Std: [" << stdDev[0] << ", " << stdDev[1] << ", " << stdDev[2] << "]\n";
    }
    
    std::cout << "\nData Augmentation:\n";
    std::cout << "  Enabled: " << (useAugmentation ? "Yes" : "No") << "\n";
    if (useAugmentation) {
        std::cout << "  Rotation Range: " << rotationRange << " degrees\n";
        std::cout << "  Zoom Range: " << zoomRange << "\n";
        std::cout << "  Horizontal Flip: " << (horizontalFlip ? "Yes" : "No") 
                  << " (Prob: " << horizontalFlipProb << ")\n";
        std::cout << "  Vertical Flip: " << (verticalFlip ? "Yes" : "No") 
                  << " (Prob: " << verticalFlipProb << ")\n";
        std::cout << "  Brightness Range: " << brightnessRange << "\n";
        std::cout << "  Contrast Range: " << contrastRange << "\n";
    }
    
    std::cout << "\nCache:\n";
    std::cout << "  Cache Images: " << (cacheImages ? "Yes" : "No") << "\n";
    if (cacheImages) {
        std::cout << "  Cache Limit: " << cacheSizeLimit << " images\n";
    }
    
    std::cout << "===============================\n" << std::endl;
}

//img infos implementation of defect dataset
bool ImageInfo::isValid() const {
    if (imagePath.empty()) return false;
    if (labelIndex < 0) return false;
    
    // Check if file exists
    if (!fs::exists(imagePath)) {
        return false;
    }
    
    // Check file extension
    std::string ext = fs::path(imagePath).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    
    std::vector<std::string> validExtensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"};
    return std::find(validExtensions.begin(), validExtensions.end(), ext) != validExtensions.end();
}

//DEFECT DATASET IMPLEMENTATION
DefectDataset::DefectDataset(const DataLoaderConfig& config, DatasetMode mode)
    : config_(config), mode_(mode), rng_(std::random_device{}()) {
    
    // Initialize class distribution
    for (const auto& className : config_.classNames) {
        classDistribution_[className] = 0;
    }
    
    // Initialize class-to-index mapping
    for (size_t i = 0; i < config_.classNames.size(); ++i) {
        config_.classToIndex[config_.classNames[i]] = static_cast<int>(i);
    }
    
    // Load data based on source
    std::string dataPath;
    switch (mode) {
        case DatasetMode::TRAIN:
            dataPath = config_.dataRootPath + "/" + config_.trainFolder;
            break;
        case DatasetMode::VALIDATION:
            dataPath = config_.dataRootPath + "/" + config_.valFolder;
            break;
        case DatasetMode::TEST:
            dataPath = config_.dataRootPath + "/" + config_.testFolder;
            break;
    }
    
    if (!fs::exists(dataPath)) {
        std::cerr << "[Warning] Dataset path does not exist: " << dataPath << std::endl;
        return;
    }
    
    loadDataFromFolder(dataPath);
}

DefectDataset::DefectDataset(const DataLoaderConfig& config, 
                           const std::string& imagesDir,
                           const std::string& csvPath,
                           DatasetMode mode)
    : config_(config), mode_(mode), rng_(std::random_device{}()) {
    
    // Initialize class distribution
    for (const auto& className : config_.classNames) {
        classDistribution_[className] = 0;
    }
    
    // Initialize class-to-index mapping
    for (size_t i = 0; i < config_.classNames.size(); ++i) {
        config_.classToIndex[config_.classNames[i]] = static_cast<int>(i);
    }
    
    loadDataFromKaggleCSV(imagesDir, csvPath);
}

void DefectDataset::loadDataFromFolder(const std::string& imagesDir) {
    std::cout << "[DefectDataset] Loading data from folder: " << imagesDir << std::endl;
    
    // Clear existing data
    imageInfos_.clear();
    
    // For each class folder
    for (const auto& className : config_.classNames) {
        std::string classDir = imagesDir + "/" + className;
        
        if (!fs::exists(classDir)) {
            std::cerr << "[Warning] Class directory does not exist: " << classDir << std::endl;
            continue;
        }
        
        if (!fs::is_directory(classDir)) {
            std::cerr << "[Warning] Not a directory: " << classDir << std::endl;
            continue;
        }
        
        // Scan for image files
        for (const auto& entry : fs::directory_iterator(classDir)) {
            if (entry.is_regular_file()) {
                std::string filePath = entry.path().string();
                std::string ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                
                // Check if it's an image file
                if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || 
                    ext == ".bmp" || ext == ".tiff" || ext == ".tif") {
                    
                    ImageInfo info;
                    info.imagePath = filePath;
                    info.imageName = entry.path().filename().string();
                    info.className = className;
                    info.labelIndex = config_.classToIndex[className];
                    info.hasAnnotation = false;
                    
                    // Get image dimensions
                    cv::Mat image = cv::imread(filePath, cv::IMREAD_UNCHANGED);
                    if (image.empty()) {
                        std::cerr << "[Warning] Failed to read image: " << filePath << std::endl;
                        continue;
                    }
                    
                    info.originalSize = cv::Size(image.cols, image.rows);
                    
                    imageInfos_.push_back(info);
                    classDistribution_[className]++;
                }
            }
        }
    }
    
    config_.totalSamples = static_cast<int>(imageInfos_.size());
    
    std::cout << "[DefectDataset] Loaded " << imageInfos_.size() 
              << " images from " << imagesDir << std::endl;
}


//load data from kaggle csv
void DefectDataset::loadDataFromKaggleCSV(const std::string& imagesDir, const std::string& csvPath) {
    std::cout << "[DefectDataset] Loading Kaggle data from CSV: " << csvPath << std::endl;
    
    // Clear existing data
    imageInfos_.clear();
    
    std::ifstream csvFile(csvPath);
    if (!csvFile.is_open()) {
        throw std::runtime_error("Failed to open CSV file: " + csvPath);
    }
    
    std::string line;
    std::map<std::string, ImageInfo> imageMap; // Map image name to ImageInfo
    
    // Skip header line
    std::getline(csvFile, line);
    
    while (std::getline(csvFile, line)) {
        std::stringstream ss(line);
        std::string imageId, classIdStr, encodedPixels;
        
        std::getline(ss, imageId, ',');
        std::getline(ss, classIdStr, ',');
        std::getline(ss, encodedPixels, ',');
        
        int classId = std::stoi(classIdStr);
        
        // Determine class name based on classId
        std::string className;
        if (classId == 0) {
            className = "OK";
        } else {
            className = "DEFECT"; // Class 1-4 are all defects
        }
        
        std::string imagePath = imagesDir + "/" + imageId;
        
        // Create or update ImageInfo
        if (imageMap.find(imageId) == imageMap.end()) {
            ImageInfo info;
            info.imagePath = imagePath;
            info.imageName = imageId;
            info.className = className;
            info.labelIndex = config_.classToIndex[className];
            info.hasAnnotation = true;
            
            // Get image dimensions
            cv::Mat image = cv::imread(imagePath, cv::IMREAD_UNCHANGED);
            if (!image.empty()) {
                info.originalSize = cv::Size(image.cols, image.rows);
            }
            
            imageMap[imageId] = info;
        }
        
        // Add RLE data if it's a defect
        if (classId > 0) {
            // Parse RLE string
            std::vector<int> rleValues;
            std::stringstream rleStream(encodedPixels);
            std::string value;
            
            while (std::getline(rleStream, value, ' ')) {
                if (!value.empty()) {
                    rleValues.push_back(std::stoi(value));
                }
            }
            
            if (rleValues.size() % 2 == 0) {
                imageMap[imageId].maskRLE.push_back(rleValues);
            }
        }
    }
    
    // Convert map to vector
    for (auto& pair : imageMap) {
        imageInfos_.push_back(pair.second);
        classDistribution_[pair.second.className]++;
    }
    
    config_.totalSamples = static_cast<int>(imageInfos_.size());
    
    std::cout << "[DefectDataset] Loaded " << imageInfos_.size() 
              << " images from Kaggle CSV" << std::endl;
}



cv::Mat DefectDataset::parseRLEToMask(const std::string& rleString, const cv::Size& imageSize) const {
    cv::Mat mask = cv::Mat::zeros(imageSize, CV_8UC1);
    
    std::vector<int> rleValues;
    std::stringstream ss(rleString);
    std::string value;
    
    while (std::getline(ss, value, ' ')) {
        if (!value.empty()) {
            rleValues.push_back(std::stoi(value));
        }
    }
    
    // RLE format: [start1, length1, start2, length2, ...]
    for (size_t i = 0; i < rleValues.size(); i += 2) {
        if (i + 1 < rleValues.size()) {
            int start = rleValues[i];
            int length = rleValues[i + 1];
            
            for (int j = 0; j < length; ++j) {
                int pixelPos = start + j - 1; // RLE is 1-indexed
                if (pixelPos < imageSize.width * imageSize.height) {
                    int row = pixelPos / imageSize.width;
                    int col = pixelPos % imageSize.width;
                    mask.at<uchar>(row, col) = 255;
                }
            }
        }
    }
    
    return mask;
}

cv::Mat DefectDataset::loadAndPreprocessImage(const std::string& imagePath) const {
    // Check cache first
    if (config_.cacheImages) {
        std::lock_guard<std::mutex> lock(cacheMutex_);
        auto it = imageCache_.find(imagePath);
        if (it != imageCache_.end()) {
            return it->second.clone();
        }
    }
    
    // Load image
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("Failed to load image: " + imagePath);
    }
    
    // Convert BGR to RGB (OpenCV loads as BGR, but models expect RGB)
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    
    // Resize to target size
    cv::resize(image, image, config_.targetSize);
    
    // Apply augmentation for training mode
    if (mode_ == DatasetMode::TRAIN && config_.useAugmentation) {
        image = applyAugmentation(image);
    }
    
    // Cache the image if enabled
    if (config_.cacheImages) {
        std::lock_guard<std::mutex> lock(cacheMutex_);
        
        // Check cache size limit
        if (imageCache_.size() >= config_.cacheSizeLimit) {
            // Remove oldest entry (simplified FIFO)
            if (!imageCache_.empty()) {
                imageCache_.erase(imageCache_.begin());
            }
        }
        
        imageCache_[imagePath] = image.clone();
    }
    
    return image;
}

cv::Mat DefectDataset::applyAugmentation(const cv::Mat& image) const {
    cv::Mat augmented = image.clone();
    std::uniform_real_distribution<float> probDist(0.0f, 1.0f);
    std::uniform_real_distribution<float> rotationDist(-config_.rotationRange, config_.rotationRange);
    std::uniform_real_distribution<float> zoomDist(1.0f - config_.zoomRange, 1.0f + config_.zoomRange);
    
    // Rotation
    float angle = rotationDist(rng_);
    cv::Point2f center(augmented.cols / 2.0f, augmented.rows / 2.0f);
    cv::Mat rotMat = cv::getRotationMatrix2D(center, angle, 1.0);
    cv::warpAffine(augmented, augmented, rotMat, augmented.size());
    
    // Zoom
    float zoomFactor = zoomDist(rng_);
    cv::resize(augmented, augmented, cv::Size(), zoomFactor, zoomFactor);
    
    // Crop or pad to original size
    if (zoomFactor > 1.0f) {
        // Crop center
        int x = (augmented.cols - image.cols) / 2;
        int y = (augmented.rows - image.rows) / 2;
        cv::Rect roi(x, y, image.cols, image.rows);
        augmented = augmented(roi);
    } else if (zoomFactor < 1.0f) {
        // Pad
        int top = (image.rows - augmented.rows) / 2;
        int bottom = image.rows - augmented.rows - top;
        int left = (image.cols - augmented.cols) / 2;
        int right = image.cols - augmented.cols - left;
        cv::copyMakeBorder(augmented, augmented, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
    }
    
    // Horizontal Flip
    if (config_.horizontalFlip && probDist(rng_) < config_.horizontalFlipProb) {
        cv::flip(augmented, augmented, 1);
    }
    
    // Vertical Flip
    if (config_.verticalFlip && probDist(rng_) < config_.verticalFlipProb) {
        cv::flip(augmented, augmented, 0);
    }
    
    return augmented;
}

torch::Tensor DefectDataset::convertToTensor(const cv::Mat& image) const {
    cv::Mat floatImage;
    image.convertTo(floatImage, CV_32FC3, 1.0 / 255.0);
    
    // Convert HWC to CHW
    std::vector<int64_t> sizes = {1, floatImage.channels(), floatImage.rows, floatImage.cols};
    torch::Tensor tensorImage = torch::from_blob(floatImage.data, sizes, torch::kFloat32).clone();
    
    // Normalize if required
    if (config_.normalize) {
        for (int c = 0; c < 3; ++c) {
            tensorImage[0][c] = tensorImage[0][c].sub(config_.mean[c]).div(config_.stdDev[c]);
        }
    }
    
    return tensorImage;
}

torch::Tensor DefectDataset::createLabelTensor(int labelIndex) const {
    return torch::full({1}, static_cast<int64_t>(labelIndex), torch::kInt64);
}

bool DefectDataset::validateImageFile(const std::string& imagePath) const {
    if (!fs::exists(imagePath)) {
        return false;
    }
    
    std::string ext = fs::path(imagePath).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    
    std::vector<std::string> validExtensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"};
    return std::find(validExtensions.begin(), validExtensions.end(), ext) != validExtensions.end();
}

void DefectDataset::scanDirectoryForImages(const std::string& directoryPath) {
    // This is a simplified version - the main loading is in loadDataFromFolder
    for (const auto& entry : fs::directory_iterator(directoryPath)) {
        if (entry.is_regular_file()) {
            std::string filePath = entry.path().string();
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            
            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || 
                ext == ".bmp" || ext == ".tiff" || ext == ".tif") {
                // File is valid
            }
        }
    }
}

torch::data::Example<> DefectDataset::get(size_t index) {
    try {
        const ImageInfo& info = imageInfos_.at(index);
        
        cv::Mat image = loadAndPreprocessImage(info.imagePath);
        int label = info.labelIndex;
        
        // Convert to tensor
        torch::Tensor tensor_image = convertToTensor(image);
        torch::Tensor tensor_label = createLabelTensor(label);
        
        return {tensor_image, tensor_label};
    }
    catch (const std::exception& e) {
        std::cerr << "[Error] Failed to load image: " << e.what() << std::endl;
        
        // Return a dummy example
        torch::Tensor dummyImage = torch::zeros({3, config_.targetSize.height, 
                                                 config_.targetSize.width}, torch::kFloat32);
        torch::Tensor dummyLabel = torch::zeros({1}, torch::kInt64);
        return {dummyImage, dummyLabel};
    }
}
    
torch::optional<size_t> DefectDataset::size() const {
    return imageInfos_.size();
}

ImageInfo DefectDataset::getImageInfo(size_t index) const {
    if (index >= imageInfos_.size()) {
        throw std::out_of_range("Index out of range in getImageInfo");
    }
    return imageInfos_.at(index);
}

const std::vector<ImageInfo>& DefectDataset::getAllImageInfos() const {
    return imageInfos_;
}

const std::map<std::string, int>& DefectDataset::getClassDistribution() const {
    return classDistribution_;
}

std::vector<std::shared_ptr<DefectDataset>> DefectDataset::random_split(const std::vector<double>& splitRatios) {
    if (splitRatios.empty()) {
        throw std::invalid_argument("Split ratios cannot be empty");
    }
    
    double totalRatio = 0.0;
    for (double ratio : splitRatios) {
        if (ratio <= 0.0) {
            throw std::invalid_argument("Split ratios must be positive");
        }
        totalRatio += ratio;
    }
    
    // Normalize ratios
    std::vector<double> normalizedRatios;
    for (double ratio : splitRatios) {
        normalizedRatios.push_back(ratio / totalRatio);
    }
    
    // Shuffle indices
    std::vector<size_t> indices(imageInfos_.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng_);
    
    // Determine split sizes
    std::vector<size_t> splitSizes;
    size_t totalSize = imageInfos_.size();
    size_t accumulated = 0;
    
    for (size_t i = 0; i < normalizedRatios.size(); ++i) {
        size_t splitSize = static_cast<size_t>(normalizedRatios[i] * totalSize);
        if (i == normalizedRatios.size() - 1) {
            splitSize = totalSize - accumulated; // Ensure all samples are used
        }
        splitSizes.push_back(splitSize);
        accumulated += splitSize;
    }
    
    // Create subsets
    std::vector<std::shared_ptr<DefectDataset>> subsets;
    size_t currentIndex = 0;
    
    for (size_t splitSize : splitSizes) {
        auto subset = std::make_shared<DefectDataset>(config_, mode_);
        
        for (size_t i = 0; i < splitSize; ++i) {
            size_t idx = indices[currentIndex++];
            subset->imageInfos_.push_back(imageInfos_[idx]);
            subset->classDistribution_[imageInfos_[idx].className]++;
        }
        
        subsets.push_back(subset);
    }
    
    return subsets;
}

const DataLoaderConfig& DefectDataset::getConfig() const {
    return config_;
}

void DefectDataset::printStats() const {
    std::cout << "\n=== DATASET STATISTICS ===\n";
    std::cout << "Mode: ";
    switch (mode_) {
        case DatasetMode::TRAIN: std::cout << "TRAIN"; break;
        case DatasetMode::VALIDATION: std::cout << "VALIDATION"; break;
        case DatasetMode::TEST: std::cout << "TEST"; break;
    }
    std::cout << "\n";
    
    std::cout << "Total Samples: " << imageInfos_.size() << "\n";
    std::cout << "Class Distribution:\n";
    
    for (const auto& pair : classDistribution_) {
        std::cout << "  " << pair.first << ": " << pair.second << " ("
                  << std::fixed << std::setprecision(1)
                  << (100.0 * pair.second / imageInfos_.size()) << "%)\n";
    }
    
    std::cout << "Image Size: " << config_.targetSize.width << "x" 
              << config_.targetSize.height << "\n";
    std::cout << "Cache Enabled: " << (config_.cacheImages ? "Yes" : "No") << "\n";
    std::cout << "Augmentation: " << (config_.useAugmentation ? "Yes" : "No") << "\n";
    std::cout << "==========================\n" << std::endl;
}

//DATA LOADER IMPLEMENTATION
DataLoader::DataLoader(const DataLoaderConfig& config): config_(config), rng_(std::random_device{}()){
    config_.validate();
    initializeClassMapping();
    validatePaths();
    std::cout << "[DataLoader] Initialized with config." << std::endl;
}

DataLoader::~DataLoader() {
    std::cout << "[DataLoader] Destructor called. Cleaning up resources." << std::endl;
}

void DataLoader::initializeClassMapping() {
    for (size_t i = 0; i < config_.classNames.size(); ++i) {
        config_.classToIndex[config_.classNames[i]] = static_cast<int>(i);
    }
    std::cout << "[DataLoader] Class mapping initialized." << std::endl;
}

void DataLoader::validatePaths() const {
    std::vector<std::string> paths = {
        config_.dataRootPath + "/" + config_.trainFolder,
        config_.dataRootPath + "/" + config_.valFolder,
        config_.dataRootPath + "/" + config_.testFolder
    };
    
    for (const auto& path : paths) {
        if (!fs::exists(path)) {
            std::cerr << "[Warning] Data path does not exist: " << path << std::endl;
        } else {
            std::cout << "[DataLoader] Validated data path: " << path << std::endl;
        }
    }
}

std::shared_ptr<DefectDataset> DataLoader::createDatasetFromSource(
    const std::string& folderName, DatasetMode mode) {
    std::string dataPath = config_.dataRootPath + "/" + folderName;
    if (!fs::exists(dataPath)) {
        std::cerr << "Dataset path does not exist: " << dataPath << std::endl;
        return nullptr;
    }
    return std::make_shared<DefectDataset>(config_, mode);
}

std::shared_ptr<DefectDataset> DataLoader::createTrainDataset() {
    if (!trainDataset_) {
        trainDataset_ = createDatasetFromSource(config_.trainFolder, DatasetMode::TRAIN);
        if (trainDataset_) {
            updateStatistics();
            trainDataset_->printStats();
        }
    }
    return trainDataset_;
}

std::shared_ptr<DefectDataset> DataLoader::createValidationDataset() {
    if (!valDataset_) {
        valDataset_ = createDatasetFromSource(config_.valFolder, DatasetMode::VALIDATION);
        if (valDataset_) {
            valDataset_->printStats();
        }
    }
    return valDataset_;
}

std::shared_ptr<DefectDataset> DataLoader::createTestDataset() {
    if (!testDataset_) {
        testDataset_ = createDatasetFromSource(config_.testFolder, DatasetMode::TEST);
        if (testDataset_) {
            testDataset_->printStats();
        }
    }
    return testDataset_;
}

std::tuple<std::shared_ptr<DefectDataset>,
           std::shared_ptr<DefectDataset>,
           std::shared_ptr<DefectDataset>> DataLoader::createAllDatasets() {
    return std::make_tuple(
        createTrainDataset(),
        createValidationDataset(),
        createTestDataset()
    );
}

auto
DataLoader::createTrainDataLoader(std::shared_ptr<DefectDataset> dataset) {
    if (!dataset) {
        throw std::invalid_argument("Dataset is null");
    }
    
    return torch::data::make_data_loader(
        *dataset,
        torch::data::DataLoaderOptions()
            .batch_size(config_.batchSize)
            .workers(config_.numWorkers)
    );
}

auto
DataLoader::createValidationDataLoader(std::shared_ptr<DefectDataset> dataset) {
    if (!dataset) {
        throw std::invalid_argument("Dataset is null");
    }
    
    return torch::data::make_data_loader(
        *dataset,
        torch::data::DataLoaderOptions()
            .batch_size(config_.batchSize)
            .workers(config_.numWorkers)
    );
}


auto
DataLoader::createTestDataLoader(std::shared_ptr<DefectDataset> dataset) {
    if (!dataset) {
        throw std::invalid_argument("Dataset is null");
    }
    
    return torch::data::make_data_loader(
        *dataset,
        torch::data::DataLoaderOptions()
            .batch_size(config_.batchSize)
            .workers(config_.numWorkers)
    );
}


std::pair<torch::Tensor, torch::Tensor> DataLoader::getBatch(
    std::shared_ptr<DefectDataset> dataset,
    int batchSize,
    bool shuffle) {
    
    if (!dataset) {
        throw std::invalid_argument("Dataset is null");
    }
    
    if (batchSize <= 0) {
        batchSize = config_.batchSize;
    }
    
    size_t datasetSize = dataset->size().value();
    if (datasetSize == 0) {
        throw std::runtime_error("Dataset is empty");
    }
    
    // Create indices
    std::vector<size_t> indices(datasetSize);
    std::iota(indices.begin(), indices.end(), 0);
    
    if (shuffle) {
        std::shuffle(indices.begin(), indices.end(), rng_);
    }
    
    // Get batch size (handle case where batch size > dataset size)
    size_t actualBatchSize = std::min(static_cast<size_t>(batchSize), datasetSize);
    
    // Collect data
    std::vector<torch::Tensor> imageTensors;
    std::vector<torch::Tensor> labelTensors;
    
    for (size_t i = 0; i < actualBatchSize; ++i) {
        auto example = dataset->get(indices[i]);
        imageTensors.push_back(example.data);
        labelTensors.push_back(example.target);
    }
    
    // Stack tensors
    torch::Tensor batchImages = torch::stack(imageTensors);
    torch::Tensor batchLabels = torch::cat(labelTensors);
    
    return {batchImages, batchLabels};
}

std::pair<torch::Tensor, torch::Tensor> DataLoader::getAllData(
    std::shared_ptr<DefectDataset> dataset) {
    return getBatch(dataset, dataset->size().value(), false);
}

const DataLoaderConfig& DataLoader::getConfig() const {
    return config_;
}

const std::vector<std::string>& DataLoader::getClassNames() const {
    return config_.classNames;
}

int DataLoader::getClassIndex(const std::string& className) const {
    auto it = config_.classToIndex.find(className);
    if (it != config_.classToIndex.end()) {
        return it->second;
    } else {
        return -1; // Class not found
    }
}

std::string DataLoader::getClassName(int classIndex) const {
    if (classIndex >= 0 && classIndex < static_cast<int>(config_.classNames.size())) {
        return config_.classNames[classIndex];
    }
    return "";
}

std::map<std::string, int> DataLoader::getClassDistribution() const {
    if (trainDataset_) {
        return trainDataset_->getClassDistribution();
    }
    return {};
}

void DataLoader::printDatasetInfo() const {
    std::cout << "\n=== DATA LOADER INFORMATION ===\n";
    
    std::cout << "Datasets:\n";
    std::cout << "  Train: " << (trainDataset_ ? std::to_string(trainDataset_->size().value()) : "0") << " samples\n";
    std::cout << "  Validation: " << (valDataset_ ? std::to_string(valDataset_->size().value()) : "0") << " samples\n";
    std::cout << "  Test: " << (testDataset_ ? std::to_string(testDataset_->size().value()) : "0") << " samples\n";
    
    std::cout << "\nConfiguration:\n";
    config_.printSummary();
    std::cout << "===============================\n" << std::endl;
}

std::pair<std::shared_ptr<DefectDataset>, std::shared_ptr<DefectDataset>>
DataLoader::splitTrainValidation(double validationRatio) {
    if (!trainDataset_) {
        throw std::runtime_error("Train dataset is not initialized");
    }
    
    if (validationRatio <= 0.0 || validationRatio >= 1.0) {
        throw std::invalid_argument("Validation ratio must be between 0 and 1");
    }
    
    double trainRatio = 1.0 - validationRatio;
    std::vector<std::shared_ptr<DefectDataset>> splits = 
        trainDataset_->random_split({trainRatio, validationRatio});
    
    if (splits.size() != 2) {
        throw std::runtime_error("Failed to split dataset into train and validation");
    }
    
    trainDataset_ = splits[0];
    valDataset_ = splits[1];
    
    std::cout << "[DataLoader] Split train dataset into " 
              << trainDataset_->size().value() << " training samples and " 
              << valDataset_->size().value() << " validation samples." << std::endl;
    
    return {trainDataset_, valDataset_};
}

void DataLoader::updateStatistics() {
    // Update overall statistics
    config_.classCounts.clear();
    config_.totalSamples = 0;
    
    auto updateFromDataset = [&](const std::shared_ptr<DefectDataset>& dataset) {
        if (dataset) {
            auto distribution = dataset->getClassDistribution();
            for (const auto& pair : distribution) {
                config_.classCounts[pair.first] += pair.second;
                config_.totalSamples += pair.second;
            }
        }
    };
    
    updateFromDataset(trainDataset_);
    updateFromDataset(valDataset_);
    updateFromDataset(testDataset_);
}


//BATCH ITERATOR IMPLEMENTATION
BatchIterator::BatchIterator(std::shared_ptr<DefectDataset> dataset,
                             int batchSize,
                             bool shuffle)
    : dataset_(dataset), batchSize_(batchSize), shuffle_(shuffle), currentIndex_(0), 
      datasetSize_(0), totalBatches_(0), rng_(std::random_device{}()) {
    
    if (!dataset_) {
        throw std::invalid_argument("Dataset is null");
    }
    
    datasetSize_ = dataset_->size().value();
    totalBatches_ = static_cast<int>(std::ceil(static_cast<double>(datasetSize_) / batchSize_));
    indices_.resize(datasetSize_);
    std::iota(indices_.begin(), indices_.end(), 0);
    
    if (shuffle_) {
        std::shuffle(indices_.begin(), indices_.end(), rng_);
    }
}

bool BatchIterator::hasNext() const {
    return currentIndex_ < datasetSize_;
}

std::pair<torch::Tensor, torch::Tensor> BatchIterator::next() {
    if (!hasNext()) {
        throw std::out_of_range("No more batches available");
    }
    
    size_t actualBatchSize = std::min(static_cast<size_t>(batchSize_), datasetSize_ - currentIndex_);
    
    std::vector<torch::Tensor> imageTensors;
    std::vector<torch::Tensor> labelTensors;
    
    for (size_t i = 0; i < actualBatchSize; ++i) {
        size_t dataIndex = indices_[currentIndex_ + i];
        auto example = dataset_->get(dataIndex);
        imageTensors.push_back(example.data);
        labelTensors.push_back(example.target);
    }
    
    currentIndex_ += actualBatchSize;
    
    torch::Tensor batchImages = torch::stack(imageTensors);
    torch::Tensor batchLabels = torch::cat(labelTensors);
    
    return {batchImages, batchLabels};
}

void BatchIterator::reset() {
    currentIndex_ = 0;
    if (shuffle_) {
        std::shuffle(indices_.begin(), indices_.end(), rng_);
    }
}

int BatchIterator::getTotalBatches() const {
    return static_cast<int>(std::ceil(static_cast<double>(datasetSize_) / batchSize_));
}

int BatchIterator::getCurrentBatchIndex() const {
    return static_cast<int>(currentIndex_ / batchSize_);
}

void BatchIterator::shuffleIndices() {
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices_.begin(), indices_.end(), g);
}

std::vector<size_t> BatchIterator::getBatchIndices(size_t start) const {
    size_t end = std::min(start + batchSize_, indices_.size());
    return std::vector<size_t>(indices_.begin() + start, indices_.begin() + end);
}

//stub methods 
std::shared_ptr<DefectDataset> DataLoader::balanceDataset(
    std::shared_ptr<DefectDataset> dataset) {
    std::cout << "[DataLoader] Balancing dataset (stub)" << std::endl;
    return dataset; // TODO: Implement
}

void DataLoader::saveDatasetInfo(std::shared_ptr<DefectDataset> dataset,
                               const std::string& filePath) const {
    std::ofstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filePath);
    }
    
    file << "image_path,image_name,class_name,label_index,width,height\n";
    
    const auto& infos = dataset->getAllImageInfos();
    for (const auto& info : infos) {
        file << info.imagePath << ","
             << info.imageName << ","
             << info.className << ","
             << info.labelIndex << ","
             << info.originalSize.width << ","
             << info.originalSize.height << "\n";
    }
    
    std::cout << "[DataLoader] Dataset info saved to: " << filePath << std::endl;
}

std::shared_ptr<DefectDataset> DataLoader::loadDatasetInfo(const std::string& filePath) {
    std::cout << "[DataLoader] Loading dataset info from: " << filePath << std::endl;
    return nullptr; // TODO: Implement
}

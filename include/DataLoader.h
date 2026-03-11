#ifndef DATA_LOADER_H
#define DATA_LOADER_H
#include <string>
#include <vector>
#include <memory>
#include <random>
#include <map>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

class Preprocessor;  // Forward declaration

enum class DatasetMode{
    TRAIN,
    VALIDATION,
    TEST
};

enum class DataSource {
    FOLDER,
    CSV,
    KAGGLE
};

struct DataLoaderConfig {
    //dataset paths 
    std::string dataRootPath;           ///< Root path to dataset
    std::string trainFolder = "train";  ///< Training folder name
    std::string valFolder = "val";      ///< Validation folder name
    std::string testFolder = "test";    ///< Test folder name

    //class infos 
    std::vector<std::string> classNames ={"OK" , "DEFECT"};  ///< List of class names
    std::map<std::string, int> classToIndex;  ///< Mapping from class names to indices

    //data source config 
    std::string annotationsFile;        ///< CSV file for Kaggle format
    
    // Batch configuration
    int batchSize = 32;
    bool shuffle = true;
    int numWorkers = 4;                 ///< Number of worker threads

    // Image preprocessing
    cv::Size targetSize = cv::Size(224, 224);
    bool normalize = true;
    std::vector<float> mean = {0.485f, 0.456f, 0.406f};
    std::vector<float> stdDev = {0.229f, 0.224f, 0.225f};

    //data sugmentation
    bool useAugmentation = true;
    float rotationRange = 15.0f;
    float zoomRange = 0.2f;
    bool horizontalFlip = true;
    float horizontalFlipProb = 0.5f;
    bool verticalFlip = false;
    float verticalFlipProb = 0.0f;
    float brightnessRange = 0.2f;
    float contrastRange = 0.2f;
    float saturationRange = 0.2f;
    float hueRange = 0.1f;

    //data statistics
    int totalSamples = 0;
    std::map<std::string, int> classCounts;
    
    // Cache configuration
    bool cacheImages = false;           ///< Cache images in memory
    int cacheSizeLimit = 1000;          ///< Maximum images to cache

    void validate() const;
    void loadFromYAML(const std::string& filePath);
    void printSummary() const;
    
};

struct ImageInfo
{
    std::string imagePath;      ///< Full path to image file
    std::string imageName;      ///< Image filename without path
    int labelIndex;             ///< Class label as integer index
    std::string className;      ///< Class name as string
    cv::Size originalSize;      ///< Original image dimensions
    bool hasAnnotation = false; ///< Whether image has annotation data
    
    // For Kaggle RLE format
    std::vector<std::vector<int>> maskRLE; ///< Run-length encoding for defects
    std::vector<cv::Rect> defectBoxes;     ///< Bounding boxes for defects

    bool isValid() const;
};

class DefectDataset: public torch::data::datasets::Dataset<DefectDataset> {
    public:
    DefectDataset(const DataLoaderConfig& config, DatasetMode mode);
    DefectDataset(const DataLoaderConfig& config, 
                 const std::string& imagesDir,
                 const std::string& csvPath,
                 DatasetMode mode = DatasetMode::TRAIN);

    //get a single sample
    torch::data::Example<> get(size_t index) override;

    //get dataset size 
    torch::optional<size_t> size() const override;

    //get infos about a specific image
    ImageInfo getImageInfo(size_t index) const;

    //get infos about all imgs 
    const std::vector<ImageInfo>& getAllImageInfos() const;

    //get class distrubution
    const std::map<std::string, int>& getClassDistribution() const;

    //split dataset into train/val subsets
    std::vector<std::shared_ptr<DefectDataset>> random_split(const std::vector<double>& splitRatios);
    //get datset config
    const DataLoaderConfig& getConfig() const;
    void printStats() const;
    private:
    DataLoaderConfig config_;
    DatasetMode mode_;
    std::vector<ImageInfo> imageInfos_;
    std::map<std::string, int> classDistribution_;
    std::mt19937 rng_;

    // Image cache for faster loading
    mutable std::map<std::string, cv::Mat> imageCache_;
    mutable std::mutex cacheMutex_;

    void loadDataFromFolder(const std::string& imagesDir);

    void loadDataFromKaggleCSV(const std::string& imagesDir, const std::string& csvPath);

    cv::Mat parseRLEToMask(const std::string& rleString, const cv::Size& imageSize) const;

    //load and preprocess image
    cv::Mat loadAndPreprocessImage(const std::string& imagePath) const;

    //apply data augmentation
    cv::Mat applyAugmentation(const cv::Mat& image) const;

    //convert openCV Mat to Torch Tensor
    torch::Tensor convertToTensor(const cv::Mat& image) const;

    //create label tensor 
    torch::Tensor createLabelTensor(int labelIndex) const;

    bool validateImageFile(const std::string& imagePath) const;

    void scanDirectoryForImages(const std::string& directoryPath);

};

class DataLoader {
public:
    /**
     * @brief Construct a new DataLoader
     * 
     * @param config DataLoader configuration
     */
    explicit DataLoader(const DataLoaderConfig& config);
    
    /**
     * @brief Destructor
     */
    virtual ~DataLoader();
    
    // ============================================
    // DATASET CREATION METHODS
    // ============================================
    
    /**
     * @brief Create training dataset
     * 
     * @return std::shared_ptr<DefectDataset> Training dataset
     */
    std::shared_ptr<DefectDataset> createTrainDataset();
    
    /**
     * @brief Create validation dataset
     * 
     * @return std::shared_ptr<DefectDataset> Validation dataset
     */
    std::shared_ptr<DefectDataset> createValidationDataset();
    
    /**
     * @brief Create test dataset
     * 
     * @return std::shared_ptr<DefectDataset> Test dataset
     */
    std::shared_ptr<DefectDataset> createTestDataset();
    
    /**
     * @brief Create datasets for all modes
     * 
     * @return std::tuple of train, validation, test datasets
     */
    std::tuple<std::shared_ptr<DefectDataset>,
               std::shared_ptr<DefectDataset>,
               std::shared_ptr<DefectDataset>> createAllDatasets();
    
    // ============================================
    // DATA LOADER CREATION (LibTorch DataLoaders)
    // ============================================
    
    /**
     * @brief Create LibTorch DataLoader for training
     * 
     * @param dataset Dataset to load
     * @return auto DataLoader
     */
    auto
    createTrainDataLoader(std::shared_ptr<DefectDataset> dataset);
    
    /**
     * @brief Create LibTorch DataLoader for validation
     * 
     * @param dataset Dataset to load
     * @return auto DataLoader
     */
    auto
    createValidationDataLoader(std::shared_ptr<DefectDataset> dataset);
    
    /**
     * @brief Create LibTorch DataLoader for testing
     * 
     * @param dataset Dataset to load
     * @return auto DataLoader
     */
    auto
    createTestDataLoader(std::shared_ptr<DefectDataset> dataset);
    
    // ============================================
    // BATCH PROCESSING METHODS
    // ============================================
    
    /**
     * @brief Get a single batch of data
     * 
     * @param dataset Dataset to sample from
     * @param batchSize Batch size
     * @param shuffle Whether to shuffle data
     * @return std::pair<torch::Tensor, torch::Tensor> Batch of images and labels
     */
    std::pair<torch::Tensor, torch::Tensor> getBatch(
        std::shared_ptr<DefectDataset> dataset,
        int batchSize = -1,
        bool shuffle = true);
    
    /**
     * @brief Get all data as a single batch
     * 
     * @param dataset Dataset to load
     * @return std::pair<torch::Tensor, torch::Tensor> All images and labels
     */
    std::pair<torch::Tensor, torch::Tensor> getAllData(
        std::shared_ptr<DefectDataset> dataset);
    
    // ============================================
    // DATASET INFORMATION
    // ============================================
    
    /**
     * @brief Get dataset configuration
     * 
     * @return const DataLoaderConfig& Configuration
     */
    const DataLoaderConfig& getConfig() const;
    
    /**
     * @brief Get class names
     * 
     * @return const std::vector<std::string>& Class names
     */
    const std::vector<std::string>& getClassNames() const;
    
    /**
     * @brief Get class index from name
     * 
     * @param className Class name
     * @return int Class index (-1 if not found)
     */
    int getClassIndex(const std::string& className) const;
    
    /**
     * @brief Get class name from index
     * 
     * @param classIndex Class index
     * @return std::string Class name (empty if invalid)
     */
    std::string getClassName(int classIndex) const;
    
    /**
     * @brief Get class distribution for a dataset
     * 
     * @param dataset Dataset to analyze
     * @return std::map<std::string, int> Class distribution
     */
    std::map<std::string, int> getClassDistribution(
        std::shared_ptr<DefectDataset> dataset) const;
    
    /**
     * @brief Print dataset information
     */
    void printDatasetInfo() const;
    
    // ============================================
    // UTILITY METHODS
    // ============================================
    
    /**
     * @brief Split dataset into train/validation sets
     * 
     * @param dataset Full dataset
     * @param trainRatio Ratio for training (0.0-1.0)
     * @return std::pair of train and validation datasets
     */
    std::pair<std::shared_ptr<DefectDataset>, std::shared_ptr<DefectDataset>>
    splitDataset(std::shared_ptr<DefectDataset> dataset, float trainRatio = 0.8f);
    
    /**
     * @brief Balance dataset by oversampling minority classes
     * 
     * @param dataset Dataset to balance
     * @return std::shared_ptr<DefectDataset> Balanced dataset
     */
    std::shared_ptr<DefectDataset> balanceDataset(
        std::shared_ptr<DefectDataset> dataset);
    
    /**
     * @brief Save dataset information to CSV
     * 
     * @param dataset Dataset to save
     * @param filePath Path to CSV file
     */
    void saveDatasetInfo(std::shared_ptr<DefectDataset> dataset,
                        const std::string& filePath) const;
    
    /**
     * @brief Load dataset information from CSV
     * 
     * @param filePath Path to CSV file
     * @return std::shared_ptr<DefectDataset> Loaded dataset
     */
    std::shared_ptr<DefectDataset> loadDatasetInfo(const std::string& filePath);
    
private:
    DataLoaderConfig config_;
    std::shared_ptr<DefectDataset> trainDataset_;
    std::shared_ptr<DefectDataset> valDataset_;
    std::shared_ptr<DefectDataset> testDataset_;
    
    // Random number generator for shuffling
    std::mt19937 rng_;
    
    /**
     * @brief Initialize class index mapping
     */
    void initializeClassMapping();
    
    /**
     * @brief Create dataset based on source type
     * 
     * @param folderName Subfolder name (train/val/test)
     * @param mode Dataset mode
     * @return std::shared_ptr<DefectDataset> Created dataset
     */
    std::shared_ptr<DefectDataset> createDatasetFromSource(
        const std::string& folderName, DatasetMode mode);
    
    /**
     * @brief Validate dataset paths
     * 
     * @throws std::runtime_error if paths are invalid
     */
    void validatePaths() const;
    
    /**
     * @brief Update dataset statistics
     */
    void updateStatistics();
};

class BatchIterator {
public:
    /**
     * @brief Construct a new BatchIterator
     * 
     * @param dataset Dataset to iterate over
     * @param batchSize Batch size
     * @param shuffle Whether to shuffle data
     */
    BatchIterator(std::shared_ptr<DefectDataset> dataset,
                  int batchSize = 32,
                  bool shuffle = true);
    
    /**
     * @brief Check if more batches are available
     * 
     * @return true if more batches exist
     */
    bool hasNext() const;
    
    /**
     * @brief Get next batch
     * 
     * @return std::pair<torch::Tensor, torch::Tensor> Batch of images and labels
     */
    std::pair<torch::Tensor, torch::Tensor> next();
    
    /**
     * @brief Reset iterator to beginning
     */
    void reset();
    
    /**
     * @brief Get total number of batches
     * 
     * @return int Number of batches
     */
    int getTotalBatches() const;
    
    /**
     * @brief Get current batch index
     * 
     * @return int Current batch index
     */
    int getCurrentBatchIndex() const;

private:
    std::shared_ptr<DefectDataset> dataset_;
    int batchSize_;
    bool shuffle_;
    int currentIndex_;
    size_t datasetSize_;
    int totalBatches_;
    std::vector<size_t> indices_;
    std::mt19937 rng_;  ///< Random number generator for shuffling
    
    /**
     * @brief Get batch indices
     * 
     * @param start Start index
     * @return std::vector<size_t> Batch indices
     */
    std::vector<size_t> getBatchIndices(size_t start) const;
};

#endif // DATA_LOADER_H
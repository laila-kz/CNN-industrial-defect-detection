//Visualization interface
//responsible for images , overlaying predictions and metrics 

#ifndef VISUALIZER_H
#define VISUALIZER_H

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <chrono>
#include <ctime>
#include <opencv2/opencv.hpp>

namespace cv{
    class Mat;
}

//structure display config
struct DisplayConfig{
    //window setting 
    std::string windowName = "Defect Detection System";
    int windowWidth = 1024;
    int windowHeight = 768;
    bool fullscreen = false;

    //txt display setting
    struct textConfig
    {
        std::string fontFace = "HersheySimplex";
        double fontScale = 1.0;
        int thickness = 2;
        int lineSpacing = 30;
        
        // Colors (BGR format)
        std::vector<int> textColor = {255, 255, 255};      // White
        std::vector<int> backgroundColor = {0, 0, 0};      // Black
        int backgroundOpacity = 80;                        // 0-255
        
        // Position
        int marginX = 20;
        int marginY = 20;
        
        // Text alignment
        bool alignRight = false;

    }textConfig;

    //overlayng setting 
    struct OverlayConfig{
        bool showBoundingBox = true;
        bool showConfidence = true;
        bool showLabel = true;
        bool showTimestamp = true;
        bool showFPS = true;
        bool showStatistics = true;

        //box setting 
        int boxThickness =3;
        int boxPadding = 5;

        //colors
        std::vector<int> okColor = {0, 255, 0};           // Green
        std::vector<int> defectColor = {0, 0, 255};       // Red
        std::vector<int> uncertainColor = {0, 255, 255};  // Yellow
        std::vector<int> errorColor = {255, 255, 0};      // Cyan

        //confidence bar 
        bool showConfidenceBar = true;
        int confidenceBarHeight = 20;
        int confidenceBarWidth = 200;

    }overlayConfig;

    //perfermance setting 
    struct PerformanceConfig {
        bool calculateFPS = true;
        int fpsUpdateInterval = 30;  // Update every N frames
        double fpsSmoothing = 0.9;   // Exponential moving average factor
        
        // Memory usage display
        bool showMemoryUsage = false;
    } performanceConfig;

    //layout setting 
    struct LayoutConfig{
        enum class LayoutMode{
            SINGLE_VIEW,        // Single image
            SPLIT_VIEW,         // Original vs processed
            QUAD_VIEW,          // 2x2 grid
            CUSTOM
        }layoutMode = LayoutMode::SINGLE_VIEW;
        // Grid settings for multi-view
        int gridRows = 2;
        int gridCols = 2;
        int gridSpacing = 10;
        
        // Zoom settings
        bool enableZoom = false;
        double zoomFactor = 2.0;
        std::vector<int> zoomRegion = {0, 0, 200, 200};  // x, y, width, height
    }layoutConfig;

    //recording setting 
    struct RecordingConfig{
        bool recordOutput = false;
        std::string outputPath = "./output/recording.avi";
        int outputFPS = 30;
        int outputCodec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    }recordingConfig;

    DisplayConfig() = default ;
    void validate() const;

    
};

//Data structure containing all information needed for visualization
struct VisualData{
    cv::Mat originalImage;
    cv::Mat processedImage;

    struct DetectionResult{
        std::string label;               // "OK", "DEFECT", "REVIEW"
        float confidence;                // 0.0 to 1.0
        std::vector<int> color;          // BGR color
        std::vector<int> boundingBox;    // [x, y, width, height]
        std::string explanation;         // Detailed explanation
        
        // Multiple defects support
        std::vector<std::string> defectTypes;
        std::vector<float> defectConfidences;

        DetectionResult() 
            : label("UNKNOWN"), 
              confidence(0.0f), 
              color({255, 255, 255}),
              boundingBox({0, 0, 0, 0}) {}

    }detectionResult;

    //sys infos 
    struct Systeminfo{
        double fps = 0.0;
        double processingTime = 0.0;  // ms
        int frameNumber = 0;
        std::string timestamp;
        std::map<std::string, std::string> metadata;
        
        // Model information
        std::string modelName;
        float modelVersion = 1.0f;
        
        // Performance metrics
        double cpuUsage = 0.0;
        double memoryUsage = 0.0;
    } systemInfo;

    // Statistics
    struct Statistics {
        int totalFrames = 0;
        int okCount = 0;
        int defectCount = 0;
        int reviewCount = 0;
        int errorCount = 0;
        
        float okPercentage = 0.0f;
        float defectPercentage = 0.0f;
        float reviewPercentage = 0.0f;
        
        // Time statistics
        std::chrono::steady_clock::time_point sessionStart;
        double sessionDuration = 0.0;  // seconds
    } statistics;

    VisualData(){
        systemInfo.timestamp = getCurrentTimestamp();
        statistics.sessionStart = std::chrono::steady_clock::now();
    }
private:
    std::string getCurrentTimestamp() const {
        auto now = std::chrono::system_clock::now();
        auto now_c = std::chrono::system_clock::to_time_t(now);
        char buffer[26] = {0};
        if (ctime_s(buffer, sizeof(buffer), &now_c) != 0) {
            return "";
        }
        std::string ts(buffer);
        if (!ts.empty() && ts.back() == '\n') {
            ts.pop_back();  // Remove newline
        }
        return ts;
    }

};


//class visualizer : Professional visualization interface for defect detection system
 /* 
 * 1. Real-time display of detection results
 * 2. Color-coded overlays for different decisions
 * 3. Performance metrics display (FPS, processing time)
 * 4. Statistical information
 * 5. Recording capabilities
 * 6. Multi-view layouts */

class Visualizer{
    public:
        Visualizer();
        explicit Visualizer(const DisplayConfig& config);
        virtual ~Visualizer();

        //configration methods
        void setConfig(const DisplayConfig& config);
        const DisplayConfig& getConfig() const;
        bool initialize(); //Initialize visualization window true if successful false if failed

        void shutdown();

        //main visualization methods
        cv::Mat visualize(const VisualData& visualData); //visualizer a single frame with detection results

        int display(const cv::Mat& frame, int waitKey = 1); //Display the visualized frame in a window

        int updateAndDisplay(const VisualData& visualData, int waitKey = 1);

        //overlay methods 
        
        //add detection overlay to an image
        cv::Mat addDetectionOverlay(const cv::Mat& image, const VisualData& visualData);

        //add txt overlay to an img
        cv::Mat addTextOverlay(const cv::Mat& image, 
                          const std::vector<std::string>& texts,
                          const cv::Point& position = cv::Point(20, 20),
                          const cv::Scalar& color = cv::Scalar(255, 255, 255));
        
        //add bounding box overlay 
        cv::Mat addBoundingBox(const cv::Mat& image,
                          const std::vector<int>& box,
                          const std::string& label,
                          float confidence,
                          const cv::Scalar& color);

        //add confidence bar overlay 
        cv::Mat addConfidenceBar(const cv::Mat& image,
                            float confidence,
                            const cv::Point& position,
                            const cv::Size& size);
        //add stats panel overlay 
        cv::Mat addStatisticsPanel(const cv::Mat& image,
                              const VisualData::Statistics& stats,
                              const cv::Point& position);

        
        //performance monitoring 

        double updateFPS(); //compute and update fps

        double getFPS(); //get cuurent fps

        void resetPerformanceCounters();

        //recording methods

        bool startRecording(const std::string& outputPath, int fps = 30); //start recording output to video file 

        void stopRecording();

        bool isRecording() const;

        bool saveFrame(const cv::Mat& image, const std::string& filePath); //Save current frame to image file

        //utility methods

        //Create a multi-view layout from multiple images
        cv::Mat createMultiView(const std::vector<cv::Mat>& images,
                           const DisplayConfig::LayoutConfig& layout);

        //Apply zoom to a region of interest
        cv::Mat applyZoom(const cv::Mat& image, const std::vector<int>& region, double zoomFactor);

        //print current config  
        void printConfig() const;

        //get key pressd by user 
        int getLastKey() const;

        bool isWindowOpen() const; //check if visualiser window is open

    private:
        DisplayConfig config_;                  ///< Visualization configuration
        cv::VideoWriter videoWriter_;          ///< For recording output
        std::string windowName_;               ///< Window name

        // Performance tracking
    struct PerformanceTracker {
        std::chrono::steady_clock::time_point lastTime;
        std::chrono::steady_clock::time_point startTime;
        double fps = 0.0;
        double fpsEMA = 0.0;  // Exponential Moving Average
        int frameCount = 0;
        int fpsUpdateCounter = 0;
    } perfTracker_;
    
    // Window state
    bool windowInitialized_ = false;
    bool isRecording_ = false;
    int lastKey_ = -1;
    
    // Statistics
    VisualData::Statistics sessionStats_;

    //private helper methods 

    //Draw text with background for better visibility
    void drawTextWithBackground(cv::Mat& image,
                               const std::string& text,
                               const cv::Point& position,
                               int fontFace,
                               double fontScale,
                               const cv::Scalar& textColor,
                               int thickness,
                               const cv::Scalar& backgroundColor,
                               int backgroundOpacity);
    
    cv::Mat createLabelImage(const std::string& label,
                            double fontScale,
                            int thickness,
                            const cv::Scalar& textColor,
                            const cv::Scalar& bgColor);

    void updateStatistics(const VisualData& visualData);
    cv::Scalar getColorForLabel(const std::string& label) const;
    void validateImage(const cv::Mat& image) const;
    std::string getCurrentTimestamp() const;
    std::string formatConfidence(float confidence) const;
    std::string formatDuration(double seconds) const;

};

#endif 
#include "../include/Visualizer.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <ctime>


using namespace std ;
using namespace cv;


//display config implementation 

void DisplayConfig::validate() const{
    //validate window dim
    if(windowWidth <=0 || windowHeight <= 0){
        throw invalid_argument("Window dimensions must be positive");
    }
    //txt validate config
    if (textConfig.fontScale <= 0) {
        throw invalid_argument("Font scale must be positive");
    }
    if (textConfig.thickness <= 0) {
        throw invalid_argument("Text thickness must be positive");
    }
    
    if (textConfig.backgroundOpacity < 0 || textConfig.backgroundOpacity > 255) {
        throw invalid_argument("Background opacity must be between 0 and 255");
    }

    //validate colors 
    auto validateColor =[](const vector<int>& color, const string& name) {
        if (color.size() != 3) {
            throw invalid_argument(name + " must have 3 values (BGR)");
        }
        for (int c : color) {
            if (c < 0 || c > 255) {
                throw invalid_argument(name + " values must be between 0 and 255");
            }
        }
    };
    validateColor(textConfig.textColor, "Text color");
    validateColor(textConfig.backgroundColor, "Background color");
    validateColor(overlayConfig.okColor, "OK color");
    validateColor(overlayConfig.defectColor, "Defect color");
    validateColor(overlayConfig.uncertainColor, "Uncertain color");
    validateColor(overlayConfig.errorColor, "Error color");

    // Validate performance settings
    if (performanceConfig.fpsUpdateInterval <= 0) {
        throw invalid_argument("FPS update interval must be positive");
    }
    
    if (performanceConfig.fpsSmoothing < 0 || performanceConfig.fpsSmoothing > 1) {
        throw invalid_argument("FPS smoothing must be between 0 and 1");
    }
    
    // Validate layout settings
    if (layoutConfig.gridRows <= 0 || layoutConfig.gridCols <= 0) {
        throw invalid_argument("Grid dimensions must be positive");
    }
    
    if (layoutConfig.zoomFactor <= 0) {
        throw invalid_argument("Zoom factor must be positive");
    }
}

//visualizer implementation 
Visualizer::Visualizer()
: config_(),
      windowName_(config_.windowName),
      windowInitialized_(false),
      isRecording_(false),
      lastKey_(-1) {
    
    // Initialize performance tracker
    perfTracker_.startTime = chrono::steady_clock::now();
    perfTracker_.lastTime = perfTracker_.startTime;
    perfTracker_.fps = 0.0;
    perfTracker_.fpsEMA = 0.0;
    perfTracker_.frameCount = 0;
    perfTracker_.fpsUpdateCounter = 0;
    
    cout << "[Visualizer] Initialized with default configuration" << endl;
}

Visualizer::Visualizer(const DisplayConfig& config)
    : config_(config),
      windowName_(config.windowName),
      windowInitialized_(false),
      isRecording_(false),
      lastKey_(-1) {
    config_.validate();
        // Initialize performance tracker
    perfTracker_.startTime = chrono::steady_clock::now();
    perfTracker_.lastTime = perfTracker_.startTime;
    perfTracker_.fps = 0.0;
    perfTracker_.fpsEMA = 0.0;
    perfTracker_.frameCount = 0;
    perfTracker_.fpsUpdateCounter = 0;
    
    cout << "[Visualizer] Initialized with custom configuration" << endl;
    printConfig();

      }

Visualizer::~Visualizer(){
    shutdown();
    cout << "[Visaulizer] Destroyed" << endl;
}

//configuration methods
void Visualizer::setConfig(const DisplayConfig& config){
    config_ = config;
    config_.validate();
    windowName_ = config_.windowName;
    cout << "[Visualizer] Configuration updated" << endl;
    printConfig();
}

const DisplayConfig& Visualizer::getConfig() const{
    return config_;
}

bool Visualizer::initialize(){
    if(windowInitialized_){
        cout << "[Visualizer] Window already initialized" << endl;
        return true;
    }
    int flags = WINDOW_AUTOSIZE;
    if(config_.fullscreen){
        flags |= WINDOW_FULLSCREEN;
    }
    namedWindow(windowName_, flags);
    windowInitialized_ = true;
    cout << "[Visualizer] Window initialized: " << windowName_ << endl;
    return true;
}

void Visualizer::shutdown(){
    if(windowInitialized_){
        destroyWindow(windowName_);
        windowInitialized_ = false;
        cout << "[Visualizer] Window shutdown: " << windowName_ << endl;
    }
    if(isRecording_){
        stopRecording();
    }
}


//main visualization methods
Mat Visualizer::visualize(const VisualData& visualData) {
    // Validate input
    validateImage(visualData.originalImage);
    
    // Start with original image
    Mat displayImage = visualData.originalImage.clone();
    
    // Add detection overlay if requested
    if (config_.overlayConfig.showBoundingBox || 
        config_.overlayConfig.showLabel ||
        config_.overlayConfig.showConfidence) {
        
        displayImage = addDetectionOverlay(displayImage, visualData);
    }
    
    // Prepare text overlay
    vector<string> overlayTexts;
    
    // Add label and confidence
    if (config_.overlayConfig.showLabel) {
        string labelText = "Status: " + visualData.detectionResult.label;
        overlayTexts.push_back(labelText);
    }
    
    if (config_.overlayConfig.showConfidence) {
        string confidenceText = "Confidence: " + formatConfidence(visualData.detectionResult.confidence);
        overlayTexts.push_back(confidenceText);
    }
    
    // Add timestamp if requested
    if (config_.overlayConfig.showTimestamp) {
        overlayTexts.push_back("Time: " + visualData.systemInfo.timestamp);
    }
    
    // Add FPS if requested
    if (config_.overlayConfig.showFPS && config_.performanceConfig.calculateFPS) {
        string fpsText = "FPS: " + to_string(static_cast<int>(perfTracker_.fps));
        overlayTexts.push_back(fpsText);
    }
    
    // Add processing time
    if (visualData.systemInfo.processingTime > 0) {
        string procTime = "Processing: " + to_string(static_cast<int>(visualData.systemInfo.processingTime)) + " ms";
        overlayTexts.push_back(procTime);
    }
    
    // Add frame number
    string frameText = "Frame: " + to_string(visualData.systemInfo.frameNumber);
    overlayTexts.push_back(frameText);
    
    // Add explanation if available
    if (!visualData.detectionResult.explanation.empty()) {
        // Split long explanations
        string explanation = visualData.detectionResult.explanation;
        if (explanation.length() > 50) {
            // Simple splitting for display
            overlayTexts.push_back("Details: " + explanation.substr(0, 50) + "...");
        } else {
            overlayTexts.push_back("Details: " + explanation);
        }
    }
    
    // Add text overlay
    if (!overlayTexts.empty()) {
        Point textPosition(config_.textConfig.marginX, config_.textConfig.marginY);
        
        // Adjust position for right alignment
        if (config_.textConfig.alignRight) {
            // Calculate maximum text width
            int maxWidth = 0;
            for (const auto& text : overlayTexts) {
                Size textSize = getTextSize(text, FONT_HERSHEY_SIMPLEX, 
                                          config_.textConfig.fontScale, 
                                          config_.textConfig.thickness, nullptr);
                maxWidth = max(maxWidth, textSize.width);
            }
            textPosition.x = displayImage.cols - maxWidth - config_.textConfig.marginX;
        }
        
        Scalar textColor(config_.textConfig.textColor[0], 
                        config_.textConfig.textColor[1], 
                        config_.textConfig.textColor[2]);
        
        Scalar bgColor(config_.textConfig.backgroundColor[0],
                      config_.textConfig.backgroundColor[1],
                      config_.textConfig.backgroundColor[2]);
        
        // Draw text with background
        for (size_t i = 0; i < overlayTexts.size(); ++i) {
            int yOffset = static_cast<int>(i) * config_.textConfig.lineSpacing;
            Point linePosition = textPosition + Point(0, yOffset);
            
            drawTextWithBackground(displayImage,
                                  overlayTexts[i],
                                  linePosition,
                                  FONT_HERSHEY_SIMPLEX,
                                  config_.textConfig.fontScale,
                                  textColor,
                                  config_.textConfig.thickness,
                                  bgColor,
                                  config_.textConfig.backgroundOpacity);
        }
    }
    
    // Add statistics panel if requested
    if (config_.overlayConfig.showStatistics) {
        Point statsPosition(displayImage.cols - 300, config_.textConfig.marginY);
        displayImage = addStatisticsPanel(displayImage, visualData.statistics, statsPosition);
    }
    
    // Update session statistics
    updateStatistics(visualData);
    
    return displayImage;
}

int Visualizer::display(const Mat& frame, int waitKey) {
    if (!windowInitialized_) {
        cerr << "[Visualizer] Window not initialized. Call initialize() first." << endl;
        return -1;
    }
    
    // Record frame if recording is active
    if (isRecording_ && videoWriter_.isOpened()) {
        videoWriter_.write(frame);
    }
    
    // Display the frame
    imshow(windowName_, frame);
    
    // Wait for key press
    lastKey_ = cv::waitKey(waitKey);
    
    // Handle special keys
    switch (lastKey_) {
        case 27:  // ESC key
            cout << "[Visualizer] ESC pressed - shutting down" << endl;
            shutdown();
            break;
        case 's':  // Save frame
        case 'S':
            saveFrame(frame, "./output/capture_" + getCurrentTimestamp() + ".jpg");
            cout << "[Visualizer] Frame saved" << endl;
            break;
        case 'r':  // Toggle recording
        case 'R':
            if (isRecording_) {
                stopRecording();
            } else {
                startRecording(config_.recordingConfig.outputPath, config_.recordingConfig.outputFPS);
            }
            break;
    }
    
    return lastKey_;
}

int Visualizer::updateAndDisplay(const VisualData& visualData, int waitKey) {
    // Update FPS
    if (config_.performanceConfig.calculateFPS) {
        updateFPS();
    }
    
    // Create visualization
    Mat displayFrame = visualize(visualData);
    
    // Display it
    return display(displayFrame, waitKey);
}

//overlay methods
Mat Visualizer::addDetectionOverlay(const Mat& image, const VisualData& visualData) {
    Mat overlayImage = image.clone();
    
    const auto& detection = visualData.detectionResult;
    
    // Draw bounding box if enabled
    if (config_.overlayConfig.showBoundingBox) {
        overlayImage = addBoundingBox(overlayImage,
                                     detection.boundingBox,
                                     detection.label,
                                     detection.confidence,
                                     Scalar(detection.color[0], detection.color[1], detection.color[2]));
    }
    
    // Draw confidence bar if enabled
    if (config_.overlayConfig.showConfidenceBar) {
        Point barPosition(20, image.rows - 40);
        Size barSize(config_.overlayConfig.confidenceBarWidth, config_.overlayConfig.confidenceBarHeight);
        
        overlayImage = addConfidenceBar(overlayImage,
                                       detection.confidence,
                                       barPosition,
                                       barSize);
    }
    
    return overlayImage;
}

Mat Visualizer::addTextOverlay(const Mat& image,
                             const vector<string>& texts,
                             const Point& position,
                             const Scalar& color) {
    Mat overlayImage = image.clone();
    
    Point textPosition = position;
    
    for (size_t i = 0; i < texts.size(); ++i) {
        int yOffset = static_cast<int>(i) * config_.textConfig.lineSpacing;
        Point linePosition = textPosition + Point(0, yOffset);
        
        putText(overlayImage,
                texts[i],
                linePosition,
                FONT_HERSHEY_SIMPLEX,
                config_.textConfig.fontScale,
                color,
                config_.textConfig.thickness);
    }
    
    return overlayImage;
}

Mat Visualizer::addBoundingBox(const Mat& image,
                             const vector<int>& box,
                             const string& label,
                             float confidence,
                             const Scalar& color) {
    Mat overlayImage = image.clone();
    
    // Extract box coordinates
    int x = box[0];
    int y = box[1];
    int width = box[2];
    int height = box[3];
    
    // Draw rectangle
    rectangle(overlayImage,
              Point(x, y),
              Point(x + width, y + height),
              color,
              config_.overlayConfig.boxThickness);
    
    // Prepare label text
    string labelText = label + " (" + formatConfidence(confidence) + ")";
    
    // Calculate text size
    int baseLine = 0;
    Size textSize = getTextSize(labelText, FONT_HERSHEY_SIMPLEX,
                               config_.textConfig.fontScale,
                               config_.textConfig.thickness,
                               &baseLine);
    
    // Draw filled rectangle for text background
    rectangle(overlayImage,
              Point(x, y - textSize.height - baseLine),
              Point(x + textSize.width, y),
              color,
              FILLED);
    
    // Put label text
    putText(overlayImage,
            labelText,
            Point(x, y - baseLine),
            FONT_HERSHEY_SIMPLEX,
            config_.textConfig.fontScale,
            Scalar(255, 255, 255),  // White text
            config_.textConfig.thickness);
    
    return overlayImage;
}

Mat Visualizer::addConfidenceBar(const Mat& image,
                               float confidence,
                               const Point& position,
                               const Size& size) {
    Mat overlayImage = image.clone();
    
    // Draw background bar
    rectangle(overlayImage,
              position,
              Point(position.x + size.width, position.y + size.height),
              Scalar(50, 50, 50),  // Dark gray background
              FILLED);
    
    // Draw confidence level
    int filledWidth = static_cast<int>(size.width * confidence);
    rectangle(overlayImage,
              position,
              Point(position.x + filledWidth, position.y + size.height),
              Scalar(0, 255, 0),  // Green bar
              FILLED);
    
    // Draw border
    rectangle(overlayImage,
              position,
              Point(position.x + size.width, position.y + size.height),
              Scalar(255, 255, 255),  // White border
              2);
    
    return overlayImage;
}
Mat Visualizer::addStatisticsPanel(const Mat& image,
                                 const VisualData::Statistics& stats,
                                 const Point& position) {
    Mat overlayImage = image.clone();
    
    vector<string> statsTexts;
    statsTexts.push_back("Total Frames: " + to_string(stats.totalFrames));
    statsTexts.push_back("Defects: " + to_string(stats.defectCount));
    statsTexts.push_back("OK: " + to_string(stats.okCount));
    statsTexts.push_back("Uncertain: " + to_string(stats.reviewCount));
    statsTexts.push_back("Errors: " + to_string(stats.errorCount));
    
    // Draw panel background
    int panelWidth = 250;
    int panelHeight = static_cast<int>(statsTexts.size()) * config_.textConfig.lineSpacing + 10;
    
    rectangle(overlayImage,
              position,
              Point(position.x + panelWidth, position.y + panelHeight),
              Scalar(0, 0, 0),  // Black background
              FILLED);
    
    // Draw each statistic line
    for (size_t i = 0; i < statsTexts.size(); ++i) {
        int yOffset = static_cast<int>(i + 1) * config_.textConfig.lineSpacing;
        Point linePosition = position + Point(5, yOffset);
        
        putText(overlayImage,
                statsTexts[i],
                linePosition,
                FONT_HERSHEY_SIMPLEX,
                config_.textConfig.fontScale,
                Scalar(255, 255, 255),  // White text
                config_.textConfig.thickness);
    }
    
    return overlayImage;
}


//performance monitoring

double Visualizer::updateFPS() {
    auto currentTime = chrono::steady_clock::now();
    perfTracker_.frameCount++;
    perfTracker_.fpsUpdateCounter++;
    
    // Update FPS every fpsUpdateInterval frames
    if (perfTracker_.fpsUpdateCounter >= config_.performanceConfig.fpsUpdateInterval) {
        chrono::duration<double> elapsed = currentTime - perfTracker_.lastTime;
        double currentFPS = perfTracker_.fpsUpdateCounter / elapsed.count();
        
        // Exponential moving average for smoothing
        perfTracker_.fpsEMA = (config_.performanceConfig.fpsSmoothing * perfTracker_.fpsEMA) +
                             ((1.0 - config_.performanceConfig.fpsSmoothing) * currentFPS);
        
        perfTracker_.fps = perfTracker_.fpsEMA;
        
        // Reset counters
        perfTracker_.lastTime = currentTime;
        perfTracker_.fpsUpdateCounter = 0;
    }
    
    return perfTracker_.fps;
}

double Visualizer::getFPS() {
    return perfTracker_.fps;
}

void Visualizer::resetPerformanceCounters() {
    perfTracker_.startTime = chrono::steady_clock::now();
    perfTracker_.lastTime = perfTracker_.startTime;
    perfTracker_.fps = 0.0;
    perfTracker_.fpsEMA = 0.0;
    perfTracker_.frameCount = 0;
    perfTracker_.fpsUpdateCounter = 0;
    cout << "[Visualizer] Performance counters reset" << endl;
}

//recording methods 
bool Visualizer::startRecording(const string& outputPath, int fps) {
    if (isRecording_) {
        cout << "[Visualizer] Recording already in progress" << endl;
        return false;
    }
    
    videoWriter_.open(outputPath,
                      config_.recordingConfig.outputCodec,
                      fps,
                      Size(config_.windowWidth, config_.windowHeight));
    
    if (!videoWriter_.isOpened()) {
        cerr << "[Visualizer] Failed to start recording to " << outputPath << endl;
        return false;
    }
    
    isRecording_ = true;
    cout << "[Visualizer] Recording started: " << outputPath << endl;
    return true;
}

void Visualizer::stopRecording() {
    if (!isRecording_) {
        cout << "[Visualizer] No active recording to stop" << endl;
        return;
    }
    
    videoWriter_.release();
    isRecording_ = false;
    cout << "[Visualizer] Recording stopped" << endl;
}
bool Visualizer::isRecording() const {
    return isRecording_;
}
bool Visualizer::saveFrame(const Mat& image, const std::string& filePath) {
    try {
        imwrite(filePath, image);
        cout << "[Visualizer] Frame saved to " << filePath << endl;
        return true;
    } catch (const cv::Exception& e) {
        cerr << "[Visualizer] Error saving frame to " << filePath << ": " << e.what() << endl;
        return false;
    }
}

//utility methods
Mat Visualizer::createMultiView(const vector<Mat>& images,
                               const DisplayConfig::LayoutConfig& layout) {
    if (images.empty()) {
        return Mat();
    }
    
    // Determine grid dimensions
    int rows = layout.gridRows;
    int cols = layout.gridCols;
    int cellCount = rows * cols;
    int imageCount = static_cast<int>(images.size());
    int displayCount = min(imageCount, cellCount);
    
    // Resize all images to same size
    Size cellSize(400, 300);  // Default cell size
    
    // Find maximum dimensions among images
    for (const auto& img : images) {
        if (!img.empty()) {
            cellSize.width = max(cellSize.width, img.cols);
            cellSize.height = max(cellSize.height, img.rows);
        }
    }
    
    // Create grid canvas
    Mat grid(cellSize.height * rows + layout.gridSpacing * (rows - 1),
            cellSize.width * cols + layout.gridSpacing * (cols - 1),
            CV_8UC3,
            Scalar(50, 50, 50));  // Dark gray background
    
    // Place images in grid
    for (int i = 0; i < displayCount; ++i) {
        int row = i / cols;
        int col = i % cols;
        
        Point cellOrigin(col * (cellSize.width + layout.gridSpacing),
                        row * (cellSize.height + layout.gridSpacing));
        
        // Resize image to fit cell
        Mat resized;
        if (!images[i].empty()) {
            resize(images[i], resized, cellSize);
            
            // Copy to grid
            resized.copyTo(grid(Rect(cellOrigin, cellSize)));
            
            // Add cell border
            rectangle(grid,
                     Rect(cellOrigin, cellSize),
                     Scalar(200, 200, 200),
                     1);
        }
    }
    
    return grid;
}
Mat Visualizer::applyZoom(const Mat& image, const vector<int>& region, double zoomFactor) {
    if (region.size() != 4) {
        cerr << "[Visualizer] Invalid region for zoom" << endl;
        return image;
    }
    
    int x = region[0];
    int y = region[1];
    int width = region[2];
    int height = region[3];
    
    // Validate region
    x = max(0, min(x, image.cols - 1));
    y = max(0, min(y, image.rows - 1));
    width = max(1, min(width, image.cols - x));
    height = max(1, min(height, image.rows - y));
    
    Rect roi(x, y, width, height);
    Mat cropped = image(roi);
    
    // Resize cropped region
    Mat zoomed;
    resize(cropped, zoomed, Size(), zoomFactor, zoomFactor, INTER_LINEAR);
    
    return zoomed;
}

void Visualizer::printConfig() const {
    cout << "=== Visualizer Configuration ===" << endl;
    cout << "Window Name: " << config_.windowName << endl;
    cout << "Window Size: " << config_.windowWidth << "x" << config_.windowHeight << endl;
    cout << "Fullscreen: " << (config_.fullscreen ? "Yes" : "No") << endl;
    cout << "--- Text Configuration ---" << endl;
    cout << "Font Scale: " << config_.textConfig.fontScale << endl;
    cout << "Text Color (BGR): (" 
         << config_.textConfig.textColor[0] << ", "
         << config_.textConfig.textColor[1] << ", "
         << config_.textConfig.textColor[2] << ")" << endl;
    cout << "Background Color (BGR): (" 
         << config_.textConfig.backgroundColor[0] << ", "
         << config_.textConfig.backgroundColor[1] << ", "
         << config_.textConfig.backgroundColor[2] << ")" << endl;
    cout << "--- Overlay Configuration ---" << endl;
    cout << "Show Bounding Box: " << (config_.overlayConfig.showBoundingBox ? "Yes" : "No") << endl;
    cout << "Box Thickness: " << config_.overlayConfig.boxThickness << endl;
    cout << "--- Performance Configuration ---" << endl;
    cout << "Calculate FPS: " << (config_.performanceConfig.calculateFPS ? "Yes" : "No") << endl;
    cout << "FPS Update Interval: " << config_.performanceConfig.fpsUpdateInterval << endl;
    cout << "FPS Smoothing: " << config_.performanceConfig.fpsSmoothing << endl;
    cout << "===============================" << endl;
}

int Visualizer::getLastKey() const {
    return lastKey_;
}

bool Visualizer::isWindowOpen() const {
    return windowInitialized_;
}

//private helper methods
void Visualizer::drawTextWithBackground(Mat& image,
                                     const string& text,
                                     const Point& position,
                                     int fontFace,
                                     double fontScale,
                                     const Scalar& textColor,
                                     int thickness,
                                     const Scalar& bgColor,
                                     int bgOpacity) {
    // Calculate text size
    int baseLine = 0;
    Size textSize = getTextSize(text, fontFace, fontScale, thickness, &baseLine);
    
    // Define background rectangle
    Rect bgRect(position.x - 5,
                position.y - textSize.height - 5,
                textSize.width + 10,
                textSize.height + baseLine + 10);
    
    // Create overlay for background with opacity
    Mat overlay;
    image(bgRect).copyTo(overlay);
    rectangle(overlay, Rect(0, 0, bgRect.width, bgRect.height), bgColor, FILLED);
    addWeighted(overlay, bgOpacity / 255.0, image(bgRect), 1 - (bgOpacity / 255.0), 0, image(bgRect));
    
    // Put the text over the background
    putText(image,
            text,
            Point(position.x, position.y),
            fontFace,
            fontScale,
            textColor,
            thickness);
}

Mat Visualizer::createLabelImage(const string& label,
                                 double fontScale,
                                 int thickness,
                                 const Scalar& textColor,
                                 const Scalar& bgColor) {
    // Calculate text size
    int baseLine = 0;
    Size textSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseLine);
    
    // Create image with background
    Mat labelImage(textSize.height + baseLine + 10,
                   textSize.width + 10,
                   CV_8UC3,
                   bgColor);
    
    // Put text on the label image
    putText(labelImage,
            label,
            Point(5, textSize.height + 5),
            FONT_HERSHEY_SIMPLEX,
            fontScale,
            textColor,
            thickness);
    
    return labelImage;
}

void Visualizer::updateStatistics(const VisualData& visualData) {
    sessionStats_.totalFrames++;
    
    const string& label = visualData.detectionResult.label;
    if (label == "Defect") {
        sessionStats_.defectCount++;
    } else if (label == "OK") {
        sessionStats_.okCount++;
    } else if (label == "Uncertain") {
        sessionStats_.reviewCount++;
    } else if (label == "Error") {
        sessionStats_.errorCount++;
    }
}

Scalar Visualizer::getColorForLabel(const string& label) const {
    if (label == "OK") {
        return Scalar(config_.overlayConfig.okColor[0],
                      config_.overlayConfig.okColor[1],
                      config_.overlayConfig.okColor[2]);
    } else if (label == "Defect") {
        return Scalar(config_.overlayConfig.defectColor[0],
                      config_.overlayConfig.defectColor[1],
                      config_.overlayConfig.defectColor[2]);
    } else if (label == "Uncertain") {
        return Scalar(config_.overlayConfig.uncertainColor[0],
                      config_.overlayConfig.uncertainColor[1],
                      config_.overlayConfig.uncertainColor[2]);
    } else {
        return Scalar(config_.overlayConfig.errorColor[0],
                      config_.overlayConfig.errorColor[1],
                      config_.overlayConfig.errorColor[2]);
    }
}

void Visualizer::validateImage(const Mat& image) const {
    if (image.empty()) {
        throw invalid_argument("Image is empty");
    }
    
    if (image.channels() != 3) {
        throw invalid_argument("Image must have 3 channels (BGR)");
    }
}

string Visualizer::getCurrentTimestamp() const {
    time_t now = time(nullptr);
    tm localTime = {};
    if (localtime_s(&localTime, &now) != 0) {
        return "";
    }

    char buffer[80];
    if (strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", &localTime) == 0) {
        return "";
    }

    return string(buffer);
}

string Visualizer::formatConfidence(float confidence) const {
    stringstream ss;
    ss << fixed << setprecision(1) << (confidence * 100) << "%";
    return ss.str();
}

string Visualizer::formatDuration(double seconds) const {
    int hours = static_cast<int>(seconds) / 3600;
    int minutes = (static_cast<int>(seconds) % 3600) / 60;
    int secs = static_cast<int>(seconds) % 60;
    
    stringstream ss;
    if (hours > 0) {
        ss << hours << "h ";
    }
    if (minutes > 0 || hours > 0) {
        ss << minutes << "m ";
    }
    ss << secs << "s";
    
    return ss.str();
}

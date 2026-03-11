//cordinates all the modules 
#include <iostream>
#include <string>
#include <memory>
#include <chrono>
#include <csignal>

#include "../include/ImageLoader.h"      // First: loads images
#include "../include/Preprocessor.h"     // Second: processes images  
#include "../include/CNNModel.h"         // Third: model inference
#include "../include/DecisionEngine.h"   // Fourth: makes decisions
#include "../include/Visualizer.h"       // Fifth: displays results
#include "../include/Evaluator.h"        // Sixth: evaluates performance

#include <opencv2/opencv.hpp>

// Forward declarations
void printBanner();
void printHelp();
bool parseArguments(int argc, char* argv[], std::string& configPath);
void signalHandler(int signal);
void cleanup();
void runDemoMode(ImageLoader& loader, Preprocessor& preprocessor);
void runImageMode(ImageLoader& loader, Preprocessor& preprocessor);
void runFolderMode(ImageLoader& loader, Preprocessor& preprocessor);
void runWebcamMode(ImageLoader& loader, Preprocessor& preprocessor);

// Global variable for signal handling
volatile sig_atomic_t g_signalReceived = 0;

//MAIN FUNCTION
int main(int argc, char* argv[]){
    printBanner();
    std::string configPath = "config/config.yaml";
    if(!parseArguments(argc, argv, configPath)){
        
        return 1;
    }

    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);
    
    std::cout << "\n[MAIN] Initializing Industrial Defect Detection System...\n";
    std::cout << "       Config file: " << configPath << "\n";
    
    // 2. LOAD CONFIGURATION (hardcoded for now)
    std::cout << "[MAIN] Loading configuration...\n";
    int imageWidth = 224;
    int imageHeight = 224;
    float defectThreshold = 0.75f;
    std::string runtimeMode = "demo";  // Change to "image" later
    
    std::cout << "       Image size: " << imageWidth << "x" << imageHeight << "\n";
    std::cout << "       Defect threshold: " << defectThreshold << "\n";
    std::cout << "       Runtime mode: " << runtimeMode << "\n";

    // 3. INITIALIZE MODULES
    std::cout << "\n[MAIN] Initializing system modules...\n";
    
    try {
        // Initialize ImageLoader
        std::cout << "[MAIN]   • Initializing ImageLoader... ";
        std::unique_ptr<ImageLoader> imageLoader = std::make_unique<ImageLoader>();
        std::cout << "OK\n";
        
        // Initialize Preprocessor
        std::cout << "[MAIN]   • Initializing Preprocessor... ";
        PreprocessorConfig preprocessConfig;
        preprocessConfig.targetWidth = imageWidth;
        preprocessConfig.targetHeight = imageHeight;
        preprocessConfig.mean = {0.485, 0.456, 0.406};
        preprocessConfig.stdDev = {0.229, 0.224, 0.225};
        
        std::unique_ptr<Preprocessor> preprocessor = std::make_unique<Preprocessor>(preprocessConfig);
        std::cout << "OK\n";
        
        // Model placeholder
        std::cout << "[MAIN]   • Initializing CNN Model... ";
        std::cout << "SKIPPED (will add later)\n";
        
        // Visualizer placeholder
        std::cout << "[MAIN]   • Initializing Visualizer... ";
        std::cout << "SKIPPED (will add later)\n";
        
        // 4. START INFERENCE LOOP
        std::cout << "\n[MAIN] Starting " << runtimeMode << " mode...\n";
        
        if (runtimeMode == "demo") {
            runDemoMode(*imageLoader, *preprocessor);
        } 
        else if (runtimeMode == "image") {
            runImageMode(*imageLoader, *preprocessor);
        }
        else if (runtimeMode == "folder") {
            runFolderMode(*imageLoader, *preprocessor);
        }
        else if (runtimeMode == "webcam") {
            runWebcamMode(*imageLoader, *preprocessor);
        }
        else {
            std::cout << "[ERROR] Unknown mode: " << runtimeMode << "\n";
        }
        
    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] Exception: " << e.what() << "\n";
        cleanup();
        return 1;
    }
    
    // 5. CLEANUP
    cleanup();
    std::cout << "\n[MAIN] System shutdown complete.\n";
    return 0;
}

//helper functions implementation

void printBanner(){
    std::cout << "\n";
    std::cout << "======================================================\n";
    std::cout << "    INDUSTRIAL DEFECT DETECTION SYSTEM\n";
    std::cout << "    CNN-Based Quality Control\n";
    std::cout << "======================================================\n";
    std::cout << "\n";

}

void printHelp(){
    std::cout << "Usage: defect_detection [options]\n";
    std::cout << "Options:\n";
    std::cout << "  -h, --help            Show this help message\n";
    std::cout << "  -c, --config <path>   Path to configuration file (default: config/config.yaml)\n";
    std::cout << "  -m, --mode <mode>     Runtime mode: demo, image, folder, webcam (default: demo)\n";
    std::cout << "\n";
}

bool parseArguments(int argc, char* argv[], std::string& configPath){
    for(int i=1; i< argc; i++){
        std::string arg = argv[i];
        if(arg == "-h" || arg == "--help"){
            printHelp();
            return false;
        } else if(arg == "-c" || arg == "--config"){
            if(i +1 < argc){
                configPath = argv[++i];
            } else{
                std::cerr << "[ERROR] Missing argument for " << arg << "\n";
                return false;
            }
        } else{
            std::cerr << "[ERROR] Unknown argument: " << arg << "\n";
            return false;
        }
    }
    return true;
}

void signalHandler(int signal){
    g_signalReceived = signal;
    std::cout << "\n[MAIN] Signal " << signal << " received. Initiating shutdown...\n";
}

void cleanup(){
    std::cout << "[MAIN] Cleaning up resources...\n";
    cv::destroyAllWindows();
}


// runtime mode functions 
void runDemoMode(ImageLoader& loader, Preprocessor& preprocessor) {
    std::cout << "\n[DEMO] Running demonstration mode\n";
    std::cout << "       Testing basic functionality...\n\n";
    
    try {
        // Create a test image
        std::cout << "[DEMO] 1. Creating test image...\n";
        cv::Mat testImage(300, 400, CV_8UC3, cv::Scalar(50, 100, 150));
        
        // Add some text
        cv::putText(testImage, "DEFECT SIMULATION", cv::Point(50, 150),
                   cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
        cv::putText(testImage, "Steel Surface", cv::Point(100, 200),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(200, 200, 200), 1);
        
        std::cout << "       Test image created: " 
                  << testImage.cols << "x" << testImage.rows << "\n";
        
        // Preprocess the image
        std::cout << "\n[DEMO] 2. Preprocessing image...\n";
        cv::Mat processedImage = preprocessor.preprocess(testImage);
        std::cout << "       Processed to: " 
                  << processedImage.cols << "x" << processedImage.rows << "\n";
        
        // Display results
        std::cout << "\n[DEMO] 3. Displaying results...\n";
        std::cout << "       Press any key to continue...\n";
        
        // Resize for display
        cv::Mat displayOriginal;
        cv::resize(testImage, displayOriginal, cv::Size(400, 300));
        
        cv::Mat displayProcessed;
        cv::resize(processedImage, displayProcessed, cv::Size(400, 300));
        
        // Convert to BGR for display
        cv::Mat displayProcessedBGR;
        cv::cvtColor(displayProcessed, displayProcessedBGR, cv::COLOR_RGB2BGR);
        
        // Create combined display
        cv::Mat combined(300, 800, CV_8UC3, cv::Scalar(40, 40, 40));
        displayOriginal.copyTo(combined(cv::Rect(0, 0, 400, 300)));
        displayProcessedBGR.copyTo(combined(cv::Rect(400, 0, 400, 300)));
        
        // Add labels
        cv::putText(combined, "Original", cv::Point(150, 30),
                   cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
        cv::putText(combined, "Processed (224x224)", cv::Point(450, 30),
                   cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
        
        cv::imshow("Demo Mode - Image Preprocessing", combined);
        cv::waitKey(0);
        
        std::cout << "\n[DEMO] ✅ Demonstration completed!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "[DEMO ERROR] " << e.what() << "\n";
    }
}

void runImageMode(ImageLoader& loader, Preprocessor& preprocessor) {
    std::cout << "\n[IMAGE MODE] Processing single image\n";
    std::cout << "             (Not implemented yet)\n";
    std::cout << "             Will process: ./data/organized/test/DEFECT/0002cc93b.jpg\n";
}

void runFolderMode(ImageLoader& loader, Preprocessor& preprocessor) {
    std::cout << "\n[FOLDER MODE] Not implemented yet\n";
}

void runWebcamMode(ImageLoader& loader, Preprocessor& preprocessor) {
    std::cout << "\n[WEBCAM MODE] Not implemented yet\n";
}

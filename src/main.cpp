/**
 * @file main.cpp
 * @brief Inference entry-point for the Industrial Defect Detection System.
 *
 * This executable coordinates all pipeline stages:
 *   1. ImageLoader  — acquire images from file, folder, or webcam
 *   2. Preprocessor — resize, color-convert, normalize
 *   3. CNNModel     — TorchScript forward pass
 *   4. DecisionEngine — apply business-logic threshold
 *   5. Visualizer   — render results to screen / file
 *   6. Evaluator    — collect and report metrics
 *
 * Usage:
 *   CNNIndustrialDefectsDetection [options]
 *
 * Options:
 *   -c, --config <path>   Path to config YAML (default: config/config.yaml)
 *   -m, --mode   <mode>   Runtime mode: demo | image | folder | webcam (default: demo)
 *   -i, --input  <path>   Input image or folder path (overrides config)
 *   -h, --help            Print this help and exit
 */

#include <iostream>
#include <string>
#include <memory>
#include <chrono>
#include <csignal>
#include <filesystem>

#include <opencv2/opencv.hpp>

#include "../include/ImageLoader.h"
#include "../include/Preprocessor.h"
#include "../include/CNNModel.h"
#include "../include/DecisionEngine.h"
#include "../include/Visualizer.h"
#include "../include/Evaluator.h"

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Global graceful-shutdown flag (used by signal handler)
// ---------------------------------------------------------------------------
volatile sig_atomic_t g_stop = 0;

// ---------------------------------------------------------------------------
// Forward declarations
// ---------------------------------------------------------------------------
void printBanner();
void printHelp();
bool parseArguments(int argc, char* argv[],
                    std::string& configPath,
                    std::string& runtimeMode,
                    std::string& inputOverride);
void signalHandler(int sig);
void cleanup();

void runDemoMode   (Preprocessor& preprocessor);
void runImageMode  (const std::string& imagePath,
                    ImageLoader& loader,
                    Preprocessor& preprocessor,
                    CNNModel& model,
                    DecisionEngine& engine,
                    Visualizer& visualizer);
void runFolderMode (const std::string& folderPath,
                    ImageLoader& loader,
                    Preprocessor& preprocessor,
                    CNNModel& model,
                    DecisionEngine& engine,
                    Visualizer& visualizer,
                    Evaluator& evaluator);

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    printBanner();

    // 1. Parse arguments
    std::string configPath    = "config/config.yaml";
    std::string runtimeMode   = "demo";
    std::string inputOverride = "";

    if (!parseArguments(argc, argv, configPath, runtimeMode, inputOverride)) {
        return 1;
    }

    // 2. Register signal handlers for graceful Ctrl-C / SIGTERM shutdown
    std::signal(SIGINT,  signalHandler);
    std::signal(SIGTERM, signalHandler);

    // 3. Validate configuration file
    if (!fs::exists(configPath)) {
        std::cerr << "[ERROR] Configuration file not found: " << configPath << "\n";
        std::cerr << "        Expected at: " << fs::absolute(configPath) << "\n";
        return 1;
    }
    std::cout << "[MAIN] Config : " << configPath << "\n";
    std::cout << "[MAIN] Mode   : " << runtimeMode << "\n";

    // 4. Hard-coded defaults (YAML parser not implemented yet — see NOTES.md)
    const int   imageWidth       = 224;
    const int   imageHeight      = 224;
    const float defectThreshold  = 0.75f;
    const std::string modelPath  = "models/defect_model.pt";

    // 5. Validate model file for non-demo modes
    if (runtimeMode != "demo" && !fs::exists(modelPath)) {
        std::cerr << "[ERROR] Model file not found: " << modelPath << "\n";
        std::cerr << "        Run CNNIndustrialDefectsTraining first, or place a\n";
        std::cerr << "        pre-trained model at " << fs::absolute(modelPath) << "\n";
        return 1;
    }

    std::cout << "[MAIN] Image size       : " << imageWidth << "x" << imageHeight << "\n";
    std::cout << "[MAIN] Defect threshold : " << defectThreshold << "\n";

    // 6. Initialise all pipeline modules
    std::cout << "\n[MAIN] Initializing pipeline modules...\n";
    try {
        // ImageLoader
        std::cout << "[MAIN]   • ImageLoader     ... ";
        auto imageLoader = std::make_unique<ImageLoader>();
        std::cout << "OK\n";

        // Preprocessor
        std::cout << "[MAIN]   • Preprocessor    ... ";
        PreprocessorConfig ppCfg;
        ppCfg.targetWidth   = imageWidth;
        ppCfg.targetHeight  = imageHeight;
        ppCfg.mean          = {0.485, 0.456, 0.406};
        ppCfg.stdDev        = {0.229, 0.224, 0.225};
        ppCfg.convertToRGB  = true;
        auto preprocessor = std::make_unique<Preprocessor>(ppCfg);
        std::cout << "OK\n";

        // CNNModel (skip for demo mode — no model file required)
        std::cout << "[MAIN]   • CNNModel        ... ";
        std::unique_ptr<CNNModel> cnnModel;
        if (runtimeMode != "demo") {
            ModelConfig modelCfg;
            modelCfg.modelPath           = modelPath;
            modelCfg.inputWidth          = imageWidth;
            modelCfg.inputHeight         = imageHeight;
            modelCfg.numClasses          = 2;
            modelCfg.classNames          = {"OK", "DEFECT"};
            modelCfg.confidenceThreshold = defectThreshold;
            cnnModel = std::make_unique<CNNModel>(modelCfg);
            if (!cnnModel->LoadModel(modelPath)) {
                std::cerr << "FAILED\n";
                std::cerr << "[ERROR] Could not load model from: " << modelPath << "\n";
                return 1;
            }
            std::cout << "OK\n";
        } else {
            std::cout << "SKIPPED (demo mode)\n";
        }

        // DecisionEngine
        std::cout << "[MAIN]   • DecisionEngine  ... ";
        DecisionConfig decCfg;
        decCfg.defectThreshold       = defectThreshold;
        decCfg.useUncertaintyThreshold = true;
        auto decisionEngine = std::make_unique<DecisionEngine>(decCfg);
        std::cout << "OK\n";

        // Visualizer
        std::cout << "[MAIN]   • Visualizer      ... ";
        DisplayConfig visCfg;
        visCfg.windowName = "Industrial Defect Detection System";
        auto visualizer = std::make_unique<Visualizer>(visCfg);
        visualizer->initialize();
        std::cout << "OK\n";

        // Evaluator
        std::cout << "[MAIN]   • Evaluator       ... ";
        Evaluator evaluator(defectThreshold);
        std::cout << "OK\n";

        // 7. Dispatch to selected runtime mode
        std::cout << "\n[MAIN] Starting runtime mode: " << runtimeMode << "\n";
        std::cout << "       Press Ctrl-C to exit cleanly.\n\n";

        if (runtimeMode == "demo") {
            runDemoMode(*preprocessor);

        } else if (runtimeMode == "image") {
            const std::string src = inputOverride.empty() ? "data/organized/test" : inputOverride;
            runImageMode(src, *imageLoader, *preprocessor, *cnnModel, *decisionEngine, *visualizer);

        } else if (runtimeMode == "folder") {
            const std::string src = inputOverride.empty() ? "data/organized/test" : inputOverride;
            runFolderMode(src, *imageLoader, *preprocessor, *cnnModel,
                          *decisionEngine, *visualizer, evaluator);
            // Print final metrics
            auto metrics = evaluator.computeMetrics();
            metrics.print();

        } else {
            std::cerr << "[ERROR] Unknown mode: " << runtimeMode
                      << "  (valid options: demo | image | folder | webcam)\n";
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "\n[FATAL] Unhandled exception: " << e.what() << "\n";
        cleanup();
        return 1;
    }

    cleanup();
    std::cout << "\n[MAIN] System shutdown complete.\n";
    return 0;
}

// ---------------------------------------------------------------------------
// Helper implementations
// ---------------------------------------------------------------------------

void printBanner() {
    std::cout << R"(
======================================================
    INDUSTRIAL DEFECT DETECTION SYSTEM
    CNN-Based Quality Control — C++ / LibTorch
======================================================
)" << "\n";
}

void printHelp() {
    std::cout <<
        "Usage: CNNIndustrialDefectsDetection [options]\n"
        "\n"
        "Options:\n"
        "  -h, --help                 Show this message and exit\n"
        "  -c, --config  <path>       Path to config.yaml (default: config/config.yaml)\n"
        "  -m, --mode    <mode>       Runtime mode: demo | image | folder | webcam\n"
        "  -i, --input   <path>       Input image or folder (overrides config value)\n"
        "\n"
        "Examples:\n"
        "  # Run the built-in preprocessing demo\n"
        "  ./CNNIndustrialDefectsDetection --mode demo\n"
        "\n"
        "  # Run inference on a single image\n"
        "  ./CNNIndustrialDefectsDetection --mode image --input data/organized/test/DEFECT/0002cc93b.jpg\n"
        "\n"
        "  # Evaluate all images in the test folder\n"
        "  ./CNNIndustrialDefectsDetection --mode folder --input data/organized/test\n"
        "\n";
}

bool parseArguments(int argc, char* argv[],
                    std::string& configPath,
                    std::string& runtimeMode,
                    std::string& inputOverride) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            printHelp();
            return false;
        } else if ((arg == "-c" || arg == "--config") && i + 1 < argc) {
            configPath = argv[++i];
        } else if ((arg == "-m" || arg == "--mode") && i + 1 < argc) {
            runtimeMode = argv[++i];
        } else if ((arg == "-i" || arg == "--input") && i + 1 < argc) {
            inputOverride = argv[++i];
        } else {
            std::cerr << "[ERROR] Unknown or incomplete argument: " << arg << "\n";
            printHelp();
            return false;
        }
    }
    return true;
}

void signalHandler(int sig) {
    g_stop = sig;
    std::cout << "\n[MAIN] Signal " << sig << " received — shutting down...\n";
}

void cleanup() {
    std::cout << "[MAIN] Releasing OpenCV windows...\n";
    cv::destroyAllWindows();
}

// ---------------------------------------------------------------------------
// Runtime mode implementations
// ---------------------------------------------------------------------------

void runDemoMode(Preprocessor& preprocessor) {
    std::cout << "[DEMO] Testing basic preprocessing pipeline...\n\n";
    try {
        // Synthesize a test image
        cv::Mat testImage(300, 400, CV_8UC3, cv::Scalar(50, 100, 150));
        cv::putText(testImage, "DEFECT SIMULATION",
                    cv::Point(40, 150), cv::FONT_HERSHEY_SIMPLEX,
                    1.0, cv::Scalar(0, 0, 255), 2);
        cv::putText(testImage, "Steel Surface",
                    cv::Point(100, 210), cv::FONT_HERSHEY_SIMPLEX,
                    0.7, cv::Scalar(200, 200, 200), 1);
        std::cout << "[DEMO] Synthetic image created: "
                  << testImage.cols << "x" << testImage.rows << "\n";

        // Preprocess
        cv::Mat processed = preprocessor.preprocess(testImage);
        std::cout << "[DEMO] Preprocessed to: "
                  << processed.cols << "x" << processed.rows << "\n";

        // Side-by-side display
        cv::Mat displayOrig, displayProc, displayProcBGR, combined;
        cv::resize(testImage,  displayOrig, cv::Size(400, 300));
        cv::resize(processed,  displayProc, cv::Size(400, 300));
        cv::cvtColor(displayProc, displayProcBGR, cv::COLOR_RGB2BGR);

        combined = cv::Mat(300, 800, CV_8UC3, cv::Scalar(40, 40, 40));
        displayOrig.copyTo(combined(cv::Rect(0,   0, 400, 300)));
        displayProcBGR.copyTo(combined(cv::Rect(400, 0, 400, 300)));

        cv::putText(combined, "Original",
                    cv::Point(140, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0,
                    cv::Scalar(255, 255, 255), 2);
        cv::putText(combined, "Preprocessed (224x224)",
                    cv::Point(420, 30), cv::FONT_HERSHEY_SIMPLEX, 0.9,
                    cv::Scalar(255, 255, 255), 2);

        cv::imshow("Demo — Preprocessing Pipeline", combined);
        std::cout << "[DEMO] Press any key to exit...\n";
        cv::waitKey(0);

        std::cout << "[DEMO] OK — Demo completed.\n";
    } catch (const std::exception& e) {
        std::cerr << "[DEMO ERROR] " << e.what() << "\n";
    }
}

void runImageMode(const std::string& imagePath,
                  ImageLoader& loader,
                  Preprocessor& preprocessor,
                  CNNModel& model,
                  DecisionEngine& engine,
                  Visualizer& visualizer) {
    std::cout << "[IMAGE] Processing: " << imagePath << "\n";
    try {
        cv::Mat image  = loader.loadImage(imagePath);
        cv::Mat prep   = preprocessor.preprocess(image);
        auto output    = model.predict(prep);
        float defectP  = output.probabilities.size() > 1 ? output.probabilities[1] : output.confidence;
        auto decision  = engine.makeDecision(defectP);

        std::cout << "[IMAGE] Result: " << decision.label
                  << "  (defect probability: " << defectP * 100.0f << "%)\n";

        VisualData vd;
        vd.originalImage = image;
        vd.detectionResult.label      = decision.label;
        vd.detectionResult.confidence = decision.confidence;
        vd.detectionResult.color      = decision.getDisplayColor();

        auto frame = visualizer.visualize(vd);
        cv::imshow("Result", frame);
        cv::waitKey(0);
    } catch (const std::exception& e) {
        std::cerr << "[IMAGE ERROR] " << e.what() << "\n";
    }
}

void runFolderMode(const std::string& folderPath,
                   ImageLoader& loader,
                   Preprocessor& preprocessor,
                   CNNModel& model,
                   DecisionEngine& engine,
                   Visualizer& visualizer,
                   Evaluator& evaluator) {
    std::cout << "[FOLDER] Processing folder: " << folderPath << "\n";

    if (!fs::exists(folderPath) || !fs::is_directory(folderPath)) {
        std::cerr << "[FOLDER ERROR] Directory not found: " << folderPath << "\n";
        return;
    }

    int processed = 0;
    for (const auto& entry : fs::recursive_directory_iterator(folderPath)) {
        if (g_stop) break;
        if (!entry.is_regular_file()) continue;

        std::string ext = entry.path().extension().string();
        for (auto& c : ext) c = static_cast<char>(std::tolower(c));
        if (ext != ".jpg" && ext != ".jpeg" && ext != ".png") continue;

        try {
            cv::Mat image = loader.loadImage(entry.path().string());
            cv::Mat prep  = preprocessor.preprocess(image);
            auto output   = model.predict(prep);
            float defectP = output.probabilities.size() > 1
                            ? output.probabilities[1]
                            : output.confidence;
            auto decision = engine.makeDecision(defectP);

            // Determine ground-truth from parent folder name (OK or DEFECT)
            std::string parentFolder = entry.path().parent_path().filename().string();
            int trueLabel = (parentFolder == "DEFECT") ? 1 : 0;
            evaluator.addResult(trueLabel, output.predictedClass, output.confidence,
                                entry.path().string());

            ++processed;
            if (processed % 50 == 0) {
                std::cout << "[FOLDER] Processed " << processed << " images...\n";
            }
        } catch (const std::exception& e) {
            std::cerr << "[FOLDER WARNING] " << entry.path().string() << ": " << e.what() << "\n";
        }
    }
    std::cout << "[FOLDER] Done. Total images processed: " << processed << "\n";
}

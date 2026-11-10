#include <iostream>
#include <fstream>
#include <chrono>
#include <opencv2/opencv.hpp>

#include "nmdlp.h"
#include "nmdlp_image_converter.h"
#include "detection_results.hpp"

NMDLP_RESULT Call(NMDLP_RESULT result, const std::string &function_name)
{
    switch (result) {
    case NMDLP_RESULT_OK:
        return NMDLP_RESULT_OK;
    case NMDLP_RESULT_INVALID_FUNC_PARAMETER:
        throw std::runtime_error(function_name + ": INVALID_FUNC_PARAMETER");
    case NMDLP_RESULT_NO_LOAD_LIBRARY:
        throw std::runtime_error(function_name + ": NO_LOAD_LIBRARY");
    case NMDLP_RESULT_NO_BOARD:
        throw std::runtime_error(function_name + ": NO_BOARD");
    case NMDLP_RESULT_BOARD_RESET_ERROR:
        throw std::runtime_error(function_name + ": BOARD_RESET_ERROR");
    case NMDLP_RESULT_INIT_CODE_LOADING_ERROR:
        throw std::runtime_error(function_name + ": INIT_CODE_LOADING_ERROR");
    case NMDLP_RESULT_CORE_HANDLE_RETRIEVAL_ERROR:
        throw std::runtime_error(function_name + ": CORE_HANDLE_RETRIEVAL_ERROR");
    case NMDLP_RESULT_FILE_LOADING_ERROR:
        throw std::runtime_error(function_name + ": FILE_LOADING_ERROR");
    case NMDLP_RESULT_MEMORY_WRITE_ERROR:
        throw std::runtime_error(function_name + ": MEMORY_WRITE_ERROR");
    case NMDLP_RESULT_MEMORY_READ_ERROR:
        throw std::runtime_error(function_name + ": MEMORY_READ_ERROR");
    case NMDLP_RESULT_MEMORY_ALLOCATION_ERROR:
        throw std::runtime_error(function_name + ": MEMORY_ALLOCATION_ERROR");
    case NMDLP_RESULT_MODEL_LOADING_ERROR:
        throw std::runtime_error(function_name + ": MODEL_LOADING_ERROR");
    case NMDLP_RESULT_INVALID_MODEL:
        throw std::runtime_error(function_name + ": INVALID_MODEL");
    case NMDLP_RESULT_BOARD_SYNC_ERROR:
        throw std::runtime_error(function_name + ": BOARD_SYNC_ERROR");
    case NMDLP_RESULT_BOARD_MEMORY_ALLOCATION_ERROR:
        throw std::runtime_error(function_name + ": BOARD_MEMORY_ALLOCATION_ERROR");
    case NMDLP_RESULT_NN_CREATION_ERROR:
        throw std::runtime_error(function_name + ": NN_CREATION_ERROR");
    case NMDLP_RESULT_NN_LOADING_ERROR:
        throw std::runtime_error(function_name + ": NN_LOADING_ERROR");
    case NMDLP_RESULT_NN_INFO_RETRIEVAL_ERROR:
        throw std::runtime_error(function_name + ": NN_INFO_RETRIEVAL_ERROR");
    case NMDLP_RESULT_MODEL_IS_TOO_BIG:
        throw std::runtime_error(function_name + ": MODEL_IS_TOO_BIG");
    case NMDLP_RESULT_NOT_INITIALIZED:
        throw std::runtime_error(function_name + ": NOT_INITIALIZED");
    case NMDLP_RESULT_INCOMPLETE:
        throw std::runtime_error(function_name + ": INCOMPLETE");
    case NMDLP_RESULT_UNKNOWN_ERROR:
        throw std::runtime_error(function_name + ": UNKNOWN_ERROR");
    default:
        throw std::runtime_error(function_name + ": UNKNOWN ERROR");
    };
}

template <typename T>
std::vector<T> ReadFile(const std::string &filename)
{
    std::ifstream ifs(filename, std::ios::binary | std::ios::ate);
    if (!ifs.is_open()) {
        throw std::runtime_error("Unable to open input file: " + filename);
    }
    auto fsize = static_cast<std::size_t>(ifs.tellg());
    ifs.seekg(0);
    std::vector<T> data(fsize / sizeof(T));
    ifs.read(reinterpret_cast<char *>(data.data()), data.size() * sizeof(T));
    return data;
}

void ShowNMDLVersion()
{
    std::uint32_t major = 0;
    std::uint32_t minor = 0;
    std::uint32_t patch = 0;
    Call(NMDLP_GetLibVersion(&major, &minor, &patch), "GetLibVersion");
    std::cout << "Lib version: " << major << "." << minor << "." << patch << std::endl;
}

void CheckBoard(std::uint32_t required_board_type)
{
    std::uint32_t boards = 0;
    std::uint32_t board_number = -1;
    Call(NMDLP_GetBoardCount(required_board_type, &boards), "GetBoardCount");
    std::cout << "Detected boards: " << boards << std::endl;
    if (!boards) {
        throw std::runtime_error("Board not found");
    }
}

NMDLP_ModelInfo GetModelInformation(NMDLP_HANDLE nmdl, std::uint32_t unit_num)
{
    NMDLP_ModelInfo model_info;
    Call(NMDLP_GetModelInfo(nmdl, unit_num, &model_info), "GetModelInfo");

    std::cout << "Model info on unit: " << unit_num << std::endl;
    std::cout << "  Input tensors: " << model_info.input_tensor_num << std::endl;
    for (std::size_t i = 0; i < model_info.input_tensor_num; ++i) {
        std::cout << "  Input tensor " << i << ": " << model_info.input_tensors[i].width << ", " << model_info.input_tensors[i].height << ", " << model_info.input_tensors[i].depth << std::endl;
    }
    std::cout << "  Output tensors: " << model_info.output_tensor_num << std::endl;
    for (std::size_t i = 0; i < model_info.output_tensor_num; ++i) {
        std::cout << "  Output tensor " << i << ": " << model_info.output_tensors[i].width << ", " << model_info.output_tensors[i].height << ", " << model_info.output_tensors[i].depth << std::endl;
    }
    return model_info;
}

void WaitForOutput(NMDLP_HANDLE nmdl, std::uint32_t unit_num, float *outputs[], double &fps)
{
    std::uint32_t status = NMDLP_PROCESS_FRAME_STATUS_INCOMPLETE;
    while (status == NMDLP_PROCESS_FRAME_STATUS_INCOMPLETE) {
        NMDLP_GetStatus(nmdl, unit_num, &status);
    }
    Call(NMDLP_GetOutput(nmdl, unit_num, outputs, &fps), "GetOutput");
}

cv::Mat BuildLetterBox(const cv::Mat& img, int target_width, int target_height, cv::Rect &frameRect)
{
    // Scale with saving aspect ration
    float scale = std::min((float)target_width / img.cols, (float)target_height / img.rows);
    int new_w = img.cols * scale;
    int new_h = img.rows * scale;
    // Resize frame
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
    // Create a square image with padding
    // left = (640 - 640) / 2, top = (640 - 400) / 2 = 120
    cv::Mat output = cv::Mat::ones(target_height, target_width, CV_8UC3) * 114.0f / 255.0f;
    int top = (target_height - new_h) / 2;
    int left = (target_width - new_w) / 2;

    frameRect = cv::Rect(left, top, new_w, new_h);
    resized.copyTo(output(cv::Rect(left, top, new_w, new_h)));
    return output;
}

int main(int argc, char* argv[])
{
    std::string modelPath = "models/model_v8n/model_mu.onnx";
    std::string labelsPath = "models/model.names";
    std::string inputVideoPath = "data/video.mp4";
    std::string outputVideoPath = "video_out.mp4";
    if (argc > 1) modelPath = argv[1];
    if (argc > 2) labelsPath = argv[2];
    if (argc > 3) inputVideoPath = argv[3];

    auto start_time = std::chrono::high_resolution_clock::now();

    // Load and initialize model on board 0, cluster 0, multi cluster mode
    const NMDLP_BOARD_TYPE BOARD_TYPE = NMDLP_BOARD_TYPE_NMQUAD;
    NMDLP_HANDLE nmdlp = 0;
    Call(NMDLP_Create(&nmdlp), "Create");
    auto model = ReadFile<float>(modelPath);
    std::array<const float *, NMDLP_MAX_UNITS> models = {model.data()};

    Call(NMDLP_Initialize(nmdlp, BOARD_TYPE, 0, 0, models.data()), "Initialize");

    // Get model info and create output vectors
    auto model_info = GetModelInformation(nmdlp, 0);
    std::vector<std::vector<float>> output_tensors(model_info.output_tensor_num);
    std::vector<float *> outputs(model_info.output_tensor_num);
    for (std::size_t i = 0; i < model_info.output_tensor_num; ++i) {
        output_tensors[i].resize(
            static_cast<std::size_t>(model_info.output_tensors[i].width) * model_info.output_tensors[i].height * model_info.output_tensors[i].depth);
        outputs[i] = output_tensors[i].data();
    }

    // Open input video file
    cv::VideoCapture cap(inputVideoPath);
    if (!cap.isOpened()) {
        std::cerr << "Could not open video: " << inputVideoPath << std::endl;
        return EXIT_FAILURE;
    }
    int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    int totalFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    std::cout << "Input video file: " << frameWidth << "x" << frameHeight << " @ " << fps
              << " FPS, " << totalFrames << " frames" << std::endl;
    // Prepare video writer for output video
    cv::VideoWriter writer(outputVideoPath, cv::VideoWriter::fourcc('m','p','4','v'),
                           fps, cv::Size(frameWidth, frameHeight));
    if (!writer.isOpened()) {
        std::cerr << "Could not open video writer" << std::endl;
        return -1;
    }

    // Precompute params for Yolo model
    int modelFrameWidth = 640;
    int modelFrameHeight = 640;
    // Allocate memory for Yolo's input frame
    int reqsize = NMDLP_IMAGE_CONVERTER_RequiredSize(modelFrameWidth, modelFrameHeight, NMDLP_IMAGE_CONVERTER_COLOR_FORMAT_RGB);
    std::vector<float> nmcFrame(reqsize);
    std::cout << "NMC frame size (float): " << reqsize << std::endl;

    const float NM_FRAME_RGB_DIVIDER[3] = {255.0, 255.0, 255.0};
    const float NM_FRAME_RGB_ADDER[3] = {0.0, 0.0, 0.0};

    // Yolo model info for postprocessing
    S_NetworkInfo network_info;
    network_info.number_of_classes = 80;
    network_info.input_image_height = modelFrameHeight;
    network_info.input_image_width = modelFrameWidth;
    for (std::size_t i = 0; i < model_info.output_tensor_num; ++i) {
        network_info.output_width[i] = model_info.output_tensors[i].width;
        network_info.output_height[i] = model_info.output_tensors[i].height;
        network_info.output_channels[i] = model_info.output_tensors[i].depth;
    }
    network_info.confidence_threshold = 0.25;
    network_info.iou_threshold = 0.7;

    cv::Mat frame;
    int frameCount = 0;
    auto startTime = std::chrono::high_resolution_clock::now();
    std::chrono::milliseconds totalFileIODuration(0);

    while (true) {
        // For each frame in input video
        auto fileioStart = std::chrono::high_resolution_clock::now();
        cap >> frame;
        auto fileioStop = std::chrono::high_resolution_clock::now();
        auto fileioDuration = std::chrono::duration_cast<std::chrono::milliseconds>(fileioStop - fileioStart);
        totalFileIODuration += fileioDuration;

        if (frame.empty()) break;
        frameCount++;

        // Run detection
        auto convertStart = std::chrono::high_resolution_clock::now();
        // Scale frame from input video resolution to model frame size (i.e. 640x640)
        cv::Rect resizedFrame;
        auto scaledFrame = BuildLetterBox(frame, modelFrameWidth, modelFrameHeight, resizedFrame);

        // Convert scaled frame to NMMDLP BMP format
        std::vector<unsigned char> bmpFrame;
        cv::imencode(".bmp", scaledFrame, bmpFrame);
        const char *bmpfp = reinterpret_cast<const char *>(bmpFrame.data());
        //cv::imwrite("bmp_nmc" + std::to_string(frameCount) + ".bmp", frameScaled);
        NMDLP_IMAGE_CONVERTER_Convert(bmpfp, nmcFrame.data(),
            static_cast<std::uint32_t>(bmpFrame.size()), modelFrameWidth, modelFrameHeight,
            NMDLP_IMAGE_CONVERTER_COLOR_FORMAT_RGB, NM_FRAME_RGB_DIVIDER, NM_FRAME_RGB_ADDER);
        auto convertEnd = std::chrono::high_resolution_clock::now();
        auto convertDuration = std::chrono::duration_cast<std::chrono::milliseconds>(convertEnd - convertStart);

        // Run inference
        auto detectStart = std::chrono::high_resolution_clock::now();
        std::array<const float *, 1> inputs = {nmcFrame.data()};
        double nmdlp_fps;
        Call(NMDLP_Process(nmdlp, 0, inputs.data()), "Process");
        // Wait for output tensors
        WaitForOutput(nmdlp, 0, outputs.data(), nmdlp_fps);
        auto detectEnd = std::chrono::high_resolution_clock::now();
        auto detectDuration = std::chrono::duration_cast<std::chrono::milliseconds>(detectEnd - detectStart);

        // Draw detections
        auto postprocessStart = std::chrono::high_resolution_clock::now();
        auto output = GetBoxes(output_tensors, network_info);
        //cv::imwrite("bmp_640" + std::to_string(frameCount) + ".bmp", frame_scaled);

        // Rescale from model frame resolution (640x640) ot original input video frame size (1780x1080)
        float scale_x = (float)modelFrameWidth / frameWidth;
        float scale_y = (float)modelFrameHeight / frameHeight;
        float scale = std::min(scale_x, scale_y);
        for (int i = 0; i < output.size(); i++) {
            output[i].left_coordinate = (output[i].left_coordinate - resizedFrame.x) / scale;
            output[i].top_coordinate = static_cast<int>((output[i].top_coordinate - resizedFrame.y) / scale);
            output[i].width = static_cast<int>(output[i].width / scale);
            output[i].height = static_cast<int>(output[i].height / scale);
        }

        // Draw boxes on frame
        for (int i = 0; i < output.size(); i++) {
            cv::Rect box(output[i].left_coordinate, output[i].top_coordinate,
                         output[i].width, output[i].height);
            cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 2);
        }

        // Calculate FPS
        auto currentTime = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(currentTime - startTime);
        double fps = elapsed.count() > 0 ? frameCount / static_cast<double>(elapsed.count()) : 0;
        std::string info = "FPS: " + std::to_string(static_cast<int>(fps)) + " NMDLP FPS: " + std::to_string(static_cast<int>(nmdlp_fps)) +
                            " Time (ms): " + std::to_string(detectDuration.count());
        std::cout << info << "\n";
        cv::putText(frame, info, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

        fileioStart = std::chrono::high_resolution_clock::now();
        writer.write(frame);
        fileioStop = std::chrono::high_resolution_clock::now();
        fileioDuration = std::chrono::duration_cast<std::chrono::milliseconds>(fileioStop - fileioStart);
        totalFileIODuration += fileioDuration;

        auto postprocessEnd = std::chrono::high_resolution_clock::now();
        auto postprocessDuration = std::chrono::duration_cast<std::chrono::milliseconds>(postprocessEnd - postprocessStart);
        #if 0
        std::cout << "Time (ms): convert " << convertDuration.count() << ", inference " << detectDuration.count()
                  << ", postprocess " << postprocessDuration.count() << std::endl;
        #endif
    }
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    double avgFps = duration.count() > 0 ? (frameCount * 1000.0) / duration.count() : 0.0;

    cap.release();
    writer.release();
    cv::destroyAllWindows();

    auto end_time = std::chrono::high_resolution_clock::now();
    auto d = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Total time (ms): " << d.count()
              << " file IO (ms)" << totalFileIODuration.count() << std::endl;
    return 0;
}

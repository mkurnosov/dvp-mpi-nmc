#include <fstream>
#include <iostream>
#include <chrono>
#include <cstdio>

#include <mpi.h>
#include <opencv2/opencv.hpp>

#include "nmdlp.h"
#include "nmdlp_image_converter.h"
#include "yolo_v8n_detection_results.hpp"

NMDLP_RESULT Call(NMDLP_RESULT result, const std::string& function_name);
void ShowNMDLVersion();
void CheckBoards(std::uint32_t required_board_type);
NMDLP_ModelInfo GetModelInformation(NMDLP_HANDLE nmdlp_handle, std::uint32_t unit);
void WaitForOutput(NMDLP_HANDLE nmdlp_handle, std::uint32_t unit, float* outputs[], double& fps);
cv::Mat BuildLetterBox(const cv::Mat& img, int target_width, int target_height, cv::Rect& frame_rect);

template <typename T>
std::vector<T> load_model(const std::string& filename);

void producer(const std::string& model_file, const std::string& input_file, const std::string& output_file)
{
    double tvid = 0, tcomm = 0, tcopy = 0;
    double ttotal = -MPI_Wtime();
    int commrank, commsize;
    MPI_Comm_rank(MPI_COMM_WORLD, &commrank);
    MPI_Comm_size(MPI_COMM_WORLD, &commsize);
    int nworkers = commsize - 1;
    std::printf("# producer: start %4d processes: producer + %d workers\n", commsize, nworkers);

    // Open input video file
    tvid = -MPI_Wtime();
    cv::VideoCapture cap(input_file);
    if (!cap.isOpened()) {
        std::cerr << "Error: could not open video: " << input_file << std::endl;
        return;
    }
    tvid += MPI_Wtime();

    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    int nframes = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

    cv::Mat first_frame;
    tvid -= MPI_Wtime();
    cap >> first_frame;
    tvid += MPI_Wtime();
    if (first_frame.empty()) {
        std::cerr << "Error: could not read first frame" << std::endl;
        return;
    }
    // Reset position on first frame
    tvid -= MPI_Wtime();
    cap.set(cv::CAP_PROP_POS_FRAMES, 0);
    tvid += MPI_Wtime();

    // Frame size in bytes
    size_t frame_size = first_frame.total() * first_frame.elemSize();
    //std::vector<uchar> buffer(nframes * frame_size);
    std::printf("# producer: input video: %dx%d @ %.1f FPS, %d frames, %lu frame size (bytes)\n", frame_width, frame_height,  fps, nframes, frame_size);

    int frame_params[3] = {nframes, frame_width, frame_height};
    tcomm -= MPI_Wtime();
    MPI_Bcast(frame_params, 3, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&frame_size, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    tcomm += MPI_Wtime();

    std::vector<uchar> frames_buf((nframes / nworkers + 1) * frame_size);

    for (int dest_rank = 1; dest_rank < commsize; dest_rank++) {
        // Read frames
        int dest_rank_frames = nframes / nworkers;
        if (dest_rank <= (nframes % nworkers))
            dest_rank_frames++;

        // Read frames for each rank dest_rank
        double tread = MPI_Wtime();
        for (int i = 0; i < dest_rank_frames; i++) {
            cv::Mat frame;
            tvid -= MPI_Wtime();
            cap >> frame;
            tvid += MPI_Wtime();
            if (frame.empty()) {
                std::cerr << "Warning: frame " << i << " is empty" << std::endl;
                break;
            }
            tcopy -= MPI_Wtime();
            uchar* frame_offset = frames_buf.data() + i * frame_size;
            if (frame.isContinuous()) {
                std::memcpy(frame_offset, frame.data, frame_size);
            } else {
                cv::Mat contig_frame;
                frame.copyTo(contig_frame);
                memcpy(frame_offset, contig_frame.data, frame_size);
            }
            tcopy += MPI_Wtime();
        }
        tread = MPI_Wtime() - tread;
        // Send frames to dest rank
        tcomm -= MPI_Wtime();
        size_t frames_buf_size = dest_rank_frames * frame_size;
        double t = MPI_Wtime();
        MPI_Send(frames_buf.data(), frames_buf_size, MPI_BYTE, dest_rank, 0, MPI_COMM_WORLD);
        t = MPI_Wtime() - t;
        tcomm += MPI_Wtime();
        std::printf("#producer: dest %d, read %.6f, send %.6f\n", dest_rank, tread, t);
    }

    // Prepare video writer
    tvid -= MPI_Wtime();
    cv::VideoWriter writer(output_file, cv::VideoWriter::fourcc('m','p','4','v'), fps, cv::Size(frame_width, frame_height));
    if (!writer.isOpened()) {
        std::cerr << "Error: could not open video writer" << std::endl;
        return;
    }
    // Reset position on first frame
    cap.set(cv::CAP_PROP_POS_FRAMES, 0);
    tvid += MPI_Wtime();

    for (int dest_rank = 1; dest_rank < commsize; dest_rank++) {
        int dest_rank_frames = nframes / nworkers;
        if (dest_rank <= (nframes % nworkers))
            dest_rank_frames++;

        std::vector<int> frame_num_boxes(dest_rank_frames, 0);
        tcomm -= MPI_Wtime();
        MPI_Recv(frame_num_boxes.data(), dest_rank_frames, MPI_INT, dest_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        tcomm += MPI_Wtime();
        int nboxes = std::reduce(frame_num_boxes.begin(), frame_num_boxes.end());
        int total_boxes_params = nboxes * 6; // left, top, width, height, confidence, class
        std::vector<int> boxes(total_boxes_params, 0);
        tcomm -= MPI_Wtime();
        MPI_Recv(boxes.data(), total_boxes_params, MPI_INT, dest_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        tcomm += MPI_Wtime();

        // Draw boxes on each frame
        int pos = 0;
        for (int i = 0; i < dest_rank_frames; i++) {
            cv::Mat frame;
            tvid -= MPI_Wtime();
            cap >> frame;
            tvid += MPI_Wtime();
            for (int box = 0; box < frame_num_boxes[i]; box++) {
                int left = boxes[pos + 0];
                int top = boxes[pos + 1];
                int width = boxes[pos + 2];
                int height = boxes[pos + 3];
                int confidence = boxes[pos + 4];
                int class_index = boxes[pos + 5];
                pos += 6;
                cv::Rect r(left, top, width, height);
                cv::rectangle(frame, r, cv::Scalar(0, 255, 0), 2);
            }
            tvid -= MPI_Wtime();
            writer.write(frame);
            tvid += MPI_Wtime();
        }
    }
    ttotal += MPI_Wtime();
    tvid -= MPI_Wtime();
    writer.release();
    tvid += MPI_Wtime();
    std::printf("# producer: total time (sec) %.6f, video %.6f, tcomm %.6f, tcopy %.6f\n", ttotal, tvid, tcomm, tcopy);
}

void worker(const std::string& model_file)
{
    double tcomm = 0, tconvert = 0, tprocess = 0, tboxes = 0, tnmdlp = 0;
    double ttotal = -MPI_Wtime();
    ttotal += MPI_Wtime();
    int rank, commsize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &commsize);
    int nworkers = commsize - 1;

    int frame_params[3];
    tcomm -= MPI_Wtime();
    MPI_Bcast(frame_params, 3, MPI_INT, 0, MPI_COMM_WORLD);
    tcomm += MPI_Wtime();
    int nframes = frame_params[0];
    int frame_width = frame_params[1];
    int frame_height = frame_params[2];
    size_t frame_size;
    tcomm -= MPI_Wtime();
    MPI_Bcast(&frame_size, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    tcomm += MPI_Wtime();
    int rank_frames = nframes / nworkers;
    if (rank <= (nframes % nworkers))
        rank_frames++;

    size_t frames_buf_size = rank_frames * frame_size;
    std::vector<uchar> frames_buf(frames_buf_size);
    double trecv = MPI_Wtime();
    tcomm -= MPI_Wtime();
    MPI_Recv(frames_buf.data(), frames_buf_size, MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    tcomm += MPI_Wtime();
    trecv = MPI_Wtime() - trecv;

    // Initialize NMDLP and load model
    tnmdlp -= MPI_Wtime();
    const NMDLP_BOARD_TYPE board_type = NMDLP_BOARD_TYPE_NMQUAD;
    NMDLP_HANDLE nmdlp_handle = 0;
    Call(NMDLP_Create(&nmdlp_handle), "Create");
    auto model = load_model<float>(model_file);
    std::array<const float *, NMDLP_MAX_UNITS> models = {model.data()};

    //CheckBoards(board_type);
    // Map rank to SoC: 8 SoC per server
    int board_number = (rank - 1) % 8; // rank 1 -> SoC 0, rank 2 -> 1, ..., 8 -> 7, 9 -> 0, 10 -> 1, ... 16 -> 7
    int proc_number = 0;
    Call(NMDLP_Initialize(nmdlp_handle, board_type, board_number, proc_number, models.data()), "Initialize");
    // Get model info and create output vectors
    auto model_info = GetModelInformation(nmdlp_handle, 0);
    std::vector<std::vector<float>> output_tensors(model_info.output_tensor_num);
    std::vector<float *> outputs(model_info.output_tensor_num);
    for (std::size_t i = 0; i < model_info.output_tensor_num; ++i) {
        output_tensors[i].resize(
            static_cast<std::size_t>(model_info.output_tensors[i].width) * model_info.output_tensors[i].height * model_info.output_tensors[i].depth);
        outputs[i] = output_tensors[i].data();
    }
    tnmdlp += MPI_Wtime();

    // Model runtime parameters
    int model_frame_width = 640;
    int model_frame_height = 640;
    int req_size = NMDLP_IMAGE_CONVERTER_RequiredSize(model_frame_width, model_frame_height, NMDLP_IMAGE_CONVERTER_COLOR_FORMAT_RGB);
    std::vector<float> nmc_frame(req_size);
    const float nmdlp_frame_rgb_divider[3] = {255.0, 255.0, 255.0};
    const float nmdlp_frame_rgb_adder[3] = {0.0, 0.0, 0.0};

    // Yolo params for postprocessing
    S_NetworkInfo network_info;
    network_info.number_of_classes = 80;
    network_info.input_image_height = model_frame_height;
    network_info.input_image_width = model_frame_width;
    for (std::size_t i = 0; i < model_info.output_tensor_num; ++i) {
        network_info.output_width[i] = model_info.output_tensors[i].width;
        network_info.output_height[i] = model_info.output_tensors[i].height;
        network_info.output_channels[i] = model_info.output_tensors[i].depth;
    }
    network_info.confidence_threshold = 0.25;
    network_info.iou_threshold = 0.7;

    std::printf("# worker %4d: NM SoC %d, frames %6d, frame size %lu, received %.2f (MiB)\n",
                rank, board_number, rank_frames, frame_size, frames_buf_size / 1024.0 / 1024.0);
    // Detected number of boxes on each frame
    std::vector<int> frame_num_boxes(rank_frames, 0);
    std::vector<int> boxes;

    for (int i = 0; i < rank_frames; i++) {
        tconvert -= MPI_Wtime();
        // Build frame from raw bytes
        uchar* frame_offset = frames_buf.data() + i * frame_size;
        cv::Mat frame(frame_height, frame_width, CV_8UC3, frame_offset);

        // Build letterbox -- scale frame from input video resolution to model frame size (i.e. 640x640)
        cv::Rect resized_frame;
        auto scaled_frame = BuildLetterBox(frame, model_frame_width, model_frame_height, resized_frame);

        // Convert scaled frame to NMDLP BMP format
        std::vector<unsigned char> bmp_frame;
        cv::imencode(".bmp", scaled_frame, bmp_frame);
        const char *bmpfp = reinterpret_cast<const char *>(bmp_frame.data());
        NMDLP_IMAGE_CONVERTER_Convert(bmpfp, nmc_frame.data(),
            static_cast<std::uint32_t>(bmp_frame.size()), model_frame_width, model_frame_height,
            NMDLP_IMAGE_CONVERTER_COLOR_FORMAT_RGB, nmdlp_frame_rgb_divider, nmdlp_frame_rgb_adder);
        tconvert += MPI_Wtime();

        tprocess -= MPI_Wtime();
        // Run inference: multicluster mode
        std::array<const float *, 1> inputs = {nmc_frame.data()};
        double nmdlp_fps;
        Call(NMDLP_Process(nmdlp_handle, 0, inputs.data()), "Process");
        WaitForOutput(nmdlp_handle, 0, outputs.data(), nmdlp_fps);
        tprocess += MPI_Wtime();

        tboxes -= MPI_Wtime();
        // Draw detections
        auto output = GetBoxes(output_tensors, network_info);
        // Rescale from model frame resolution (640x640) ot original input video frame size (1780x1080)
        float scale_x = (float)model_frame_width / frame_width;
        float scale_y = (float)model_frame_height / frame_height;
        float scale = std::min(scale_x, scale_y);
        for (unsigned int i = 0; i < output.size(); i++) {
            output[i].left_coordinate = (output[i].left_coordinate - resized_frame.x) / scale;
            output[i].top_coordinate = static_cast<int>((output[i].top_coordinate - resized_frame.y) / scale);
            output[i].width = static_cast<int>(output[i].width / scale);
            output[i].height = static_cast<int>(output[i].height / scale);

            // Add box to list for all rank frames
            boxes.push_back(output[i].left_coordinate);
            boxes.push_back(output[i].top_coordinate);
            boxes.push_back(output[i].width);
            boxes.push_back(output[i].height);
            boxes.push_back(static_cast<int>(output[i].confidence * 100.0));
            boxes.push_back(output[i].class_index);
        }
        frame_num_boxes[i] = static_cast<int>(output.size());
        tboxes += MPI_Wtime();
    }

    // Send number of boxes per frame
    tcomm -= MPI_Wtime();
    MPI_Send(frame_num_boxes.data(), rank_frames, MPI_INT, 0, 0, MPI_COMM_WORLD);
    //int nboxes = std::reduce(frame_num_boxes.begin(), frame_num_boxes.end());
    //int total_boxes_params = nboxes * 6; // left, top, width, height, confidence, class
    MPI_Send(boxes.data(), boxes.size(), MPI_INT, 0, 0, MPI_COMM_WORLD);
    tcomm += MPI_Wtime();

    NMDLP_Release(nmdlp_handle);
    NMDLP_Destroy(nmdlp_handle);

    ttotal += MPI_Wtime();
    std::printf("# worker %4d: total time (sec) %.6f, nmdlp init %.6f, convert %.6f, process %.6f, get boxes %.6f, comm %.6f, frames recv %.6f\n",
                rank, ttotal, tnmdlp, tconvert, tprocess, tboxes, tcomm, trecv);
}

int main(int argc, char **argv)
{
    std::string model_file = "model/model_v8n/model_mu.onnx";
    std::string input_file = "data/video.mp4";
    std::string output_file = "video_out.mp4";
    if (argc > 1) model_file = argv[1];
    if (argc > 2) input_file = argv[2];
    if (argc > 3) output_file = argv[3];

    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        producer(model_file, input_file, output_file);
    } else {
        worker(model_file);
    }
    MPI_Finalize();
    return 0;
}

NMDLP_RESULT Call(NMDLP_RESULT result, const std::string& function_name)
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
std::vector<T> load_model(const std::string& filename)
{
    std::ifstream ifs(filename, std::ios::binary | std::ios::ate);
    if (!ifs.is_open()) {
        throw std::runtime_error("Error: could not open file: " + filename);
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
    std::cout << "# NMDLP version " << major << "." << minor << "." << patch << std::endl;
}

void CheckBoards(std::uint32_t required_board_type)
{
    std::uint32_t boards = 0;
    Call(NMDLP_GetBoardCount(required_board_type, &boards), "GetBoardCount");
    std::cout << "# Detected NM boards: " << boards << std::endl;
    if (!boards) {
        throw std::runtime_error("Board not found");
    }
}

NMDLP_ModelInfo GetModelInformation(NMDLP_HANDLE nmdlp_handle, std::uint32_t unit)
{
    NMDLP_ModelInfo model_info;
    Call(NMDLP_GetModelInfo(nmdlp_handle, unit, &model_info), "GetModelInfo");
    #if 0
    std::cout << "# Model info on unit: " << unit << std::endl;
    std::cout << "#   input tensors: " << model_info.input_tensor_num << std::endl;
    for (std::size_t i = 0; i < model_info.input_tensor_num; ++i) {
        std::cout << "#   input tensor " << i << ": " << model_info.input_tensors[i].width
                  << ", " << model_info.input_tensors[i].height << ", " << model_info.input_tensors[i].depth << std::endl;
    }
    std::cout << "#  output tensors: " << model_info.output_tensor_num << std::endl;
    for (std::size_t i = 0; i < model_info.output_tensor_num; ++i) {
        std::cout << "#  output tensor " << i << ": " << model_info.output_tensors[i].width
                  << ", " << model_info.output_tensors[i].height << ", " << model_info.output_tensors[i].depth << std::endl;
    }
    #endif
    return model_info;
}

void WaitForOutput(NMDLP_HANDLE nmdlp_handle, std::uint32_t unit, float* outputs[], double& fps)
{
    std::uint32_t status = NMDLP_PROCESS_FRAME_STATUS_INCOMPLETE;
    while (status == NMDLP_PROCESS_FRAME_STATUS_INCOMPLETE) {
        NMDLP_GetStatus(nmdlp_handle, unit, &status);
    }
    Call(NMDLP_GetOutput(nmdlp_handle, unit, outputs, &fps), "GetOutput");
}

cv::Mat BuildLetterBox(const cv::Mat& img, int target_width, int target_height, cv::Rect& frame_rect)
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

    frame_rect = cv::Rect(left, top, new_w, new_h);
    resized.copyTo(output(cv::Rect(left, top, new_w, new_h)));
    return output;
}

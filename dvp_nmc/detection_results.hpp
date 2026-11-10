// NMDL Plus example
#include <algorithm>
#include <cmath>
#include <iterator>
#include <numeric>
#include <vector>
#include <cstdint>

struct S_NetworkInfo {
    static const std::size_t N_OUTPUT_TENSORS = 3;
    std::size_t number_of_classes;
    std::size_t input_image_width;
    std::size_t input_image_height;
    std::size_t output_width[N_OUTPUT_TENSORS];
    std::size_t output_height[N_OUTPUT_TENSORS];
    std::size_t output_channels[N_OUTPUT_TENSORS];
    double confidence_threshold;
    double iou_threshold;
};

struct Box {
    std::int32_t left_coordinate = 0;
    std::int32_t top_coordinate = 0;
    std::int32_t width = 0;
    std::int32_t height = 0;
    double confidence = 0.0;
    std::int32_t class_index = 0;
    static double overlap(double x1, double w1, double x2, double w2) {
        auto l1 = x1 - w1 / 2;
        auto l2 = x2 - w2 / 2;
        auto left = l1 > l2 ? l1 : l2;
        auto r1 = x1 + w1 / 2;
        auto r2 = x2 + w2 / 2;
        auto right = r1 < r2 ? r1 : r2;
        return right - left;
    }

    static double box_intersection(const Box &a, const Box &b) {
        auto w = overlap(a.left_coordinate, a.width, b.left_coordinate, b.width);
        auto h = overlap(a.top_coordinate, a.height, b.top_coordinate, b.height);
        if (w <= 0 || h <= 0) {
            return 0.0;
        }
        auto area = w * h;
        return area;
    }

    static double box_union(const Box &a, const Box &b) {
        auto i = box_intersection(a, b);
        auto u = static_cast<double>(a.width) * a.height + static_cast<double>(b.width) * b.height - i;
        return u;
    }

    static double box_iou(const Box &a, const Box &b) {
        return box_intersection(a, b) / box_union(a, b);
    }
};

std::vector<Box> GetBoxes(const std::vector<std::vector<float>> &scores, const S_NetworkInfo &network_info)
{
    std::vector<Box> dst_tmp;
    const std::size_t NUMBER_OF_CLASSES = network_info.number_of_classes;
    for (std::size_t t = 0; t < scores.size(); ++t) {
        const std::size_t OUTPUT_HEIGHT = network_info.output_height[t];
        const std::size_t OUTPUT_WIDTH = network_info.output_width[t];
        const std::size_t OUTPUT_CHANNELS = network_info.output_channels[t];
        const std::size_t BOUNDING_BOXES = OUTPUT_CHANNELS / (4 + NUMBER_OF_CLASSES);
        const std::size_t BOX_SIZE = NUMBER_OF_CLASSES + 4;
        // Set tensor
        const float *cur_tensor;
        cur_tensor = scores[t].data();

        for (std::size_t y = 0; y < OUTPUT_HEIGHT; ++y) {
            for (std::size_t x = 0; x < OUTPUT_WIDTH; ++x) {
                for (std::size_t b = 0; b < BOUNDING_BOXES; ++b) {
                    auto cur_scores = cur_tensor + y * OUTPUT_WIDTH * OUTPUT_CHANNELS + x * OUTPUT_CHANNELS + b * BOX_SIZE;
                    // Coordinates
                    double tx = *cur_scores++;
                    double ty = *cur_scores++;
                    double tw = *cur_scores++;
                    double th = *cur_scores++;
                    // Classes
                    std::vector<double> classes(cur_scores, cur_scores + NUMBER_OF_CLASSES);
                    auto max_probability = std::max_element(classes.begin(), classes.end());
                    auto confidence = *max_probability;
                    if (confidence <= network_info.confidence_threshold) {
                        continue;
                    }

                    std::int32_t left, top, right, bottom;
                    left = std::max(0, static_cast<std::int32_t>((tx - 0.5 * tw) * 1));
                    top = std::max(0, static_cast<std::int32_t>((ty - 0.5 * th) * 1));
                    right = std::min(static_cast<std::int32_t>(network_info.input_image_width) - 1, static_cast<std::int32_t>((tx + 0.5 * tw) * 1));
                    bottom = std::min(static_cast<std::int32_t>(network_info.input_image_height) - 1, static_cast<std::int32_t>((ty + 0.5 * th) * 1));

                    Box yolo_box;
                    yolo_box.left_coordinate = left;
                    yolo_box.top_coordinate = top;
                    yolo_box.width = right - left;
                    yolo_box.height = bottom - top;
                    yolo_box.confidence = confidence;
                    yolo_box.class_index = std::distance(classes.begin(), max_probability);
                    dst_tmp.push_back(yolo_box);
                }
            }
        }
    }
    std::stable_sort(dst_tmp.begin(), dst_tmp.end(), [](auto &lhs, auto &rhs)
                     { return lhs.confidence > rhs.confidence; });

    std::vector<Box> dst;
    for (std::size_t i = 0; i < dst_tmp.size(); ++i) {
        if (dst_tmp[i].class_index == -1) {
            continue;
        }
        dst.push_back(dst_tmp[i]);

        // Zero other
        for (auto j = i + 1; j < dst_tmp.size(); ++j) {
            if (Box::box_iou(dst_tmp[i], dst_tmp[j]) > network_info.iou_threshold) {
                dst_tmp[j].class_index = -1; // for Wega
            }
        }
    }
    return dst;
}

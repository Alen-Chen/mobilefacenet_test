#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <cmath>

// Preprocess the image for MobileFaceNet
cv::Mat preprocessImage(const cv::Mat& face, cv::Size targetSize) {
    cv::Mat resized, normalized;
    cv::resize(face, resized, targetSize);
    resized.convertTo(normalized, CV_32F, 1.0 / 255); // Scale to [0, 1]
    normalized = (normalized - 0.5) / 0.5;            // Normalize to [-1, 1]
    cv::dnn::blobFromImage(normalized, normalized, 1.0, targetSize, cv::Scalar(), true, false);
    return normalized;
}

// Calculate cosine similarity
float cosineSimilarity(const cv::Mat& vec1, const cv::Mat& vec2) {
    double dot = vec1.dot(vec2);
    double norm1 = cv::norm(vec1);
    double norm2 = cv::norm(vec2);
    return dot / (norm1 * norm2);
}

int main() {
    // Load MobileFaceNet model
    std::string modelPath = "mobilefacenet.onnx";  // Path to ONNX model
    cv::dnn::Net faceNet = cv::dnn::readNetFromONNX(modelPath);

    // Load Haar Cascade for face detection
    std::string cascadePath = cv::samples::findFile("haarcascade_frontalface_default.xml");
    cv::CascadeClassifier faceCascade;
    if (!faceCascade.load(cascadePath)) {
        std::cerr << "Error loading Haar cascade\n";
        return -1;
    }

    // Initialize camera
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video stream\n";
        return -1;
    }

    cv::Mat frame, gray, referenceEmbedding;
    bool referenceSet = false;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        std::vector<cv::Rect> faces;
        faceCascade.detectMultiScale(gray, faces, 1.1, 4);

        for (const auto& face : faces) {
            cv::Mat faceROI = frame(face).clone();

            // Preprocess and forward pass
            cv::Mat inputBlob = preprocessImage(faceROI, cv::Size(112, 112));
            faceNet.setInput(inputBlob);
            cv::Mat embedding = faceNet.forward();

            if (!referenceSet) {
                referenceEmbedding = embedding.clone();
                referenceSet = true;
                cv::putText(frame, "Reference Set", cv::Point(face.x, face.y - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
            }
            else {
                float similarity = cosineSimilarity(referenceEmbedding, embedding);
                //for (int k = 0; k < referenceEmbedding.cols; k++)
                //{
                //    std::cout << "[" << k << "] = (" << referenceEmbedding.ptr<float>(0)[k] << ", " << embedding.ptr<float>(0)[k] << ")" << std::endl;
                //}
                std::string label = (similarity > 0.5) ? "Matched" : "Not Matched";
                cv::putText(frame, label + " (" + std::to_string(similarity) + ")",
                    cv::Point(face.x, face.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 255, 0), 1);
            }

            cv::rectangle(frame, face, cv::Scalar(255, 0, 0), 2);
        }

        cv::imshow("Face Recognition", frame);
        if (cv::waitKey(1) == 27) break; // Exit on ESC
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}

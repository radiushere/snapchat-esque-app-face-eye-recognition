#include <opencv2/opencv.hpp>
#include <iostream>
#include <ctime> // For generating unique filenames

using namespace cv;
using namespace std;

int main() {
    CascadeClassifier faceCascade, eyeCascade;
    string faceCascadePath = "cascades/haarcascade_frontalface_default.xml";
    string eyeCascadePath = "cascades/haarcascade_eye.xml";

    if (!faceCascade.load(faceCascadePath)) {
        cout << "Error loading face cascade!" << endl;
        return -1;
    }

    if (!eyeCascade.load(eyeCascadePath)) {
        cout << "Error loading eye cascade!" << endl;
        return -1;
    }

    // Load sunglasses images
    Mat sunglasses1 = imread("Images/sunglasses.png", IMREAD_UNCHANGED);
    Mat sunglasses2 = imread("Images/memesunglasses.png", IMREAD_UNCHANGED);
    Mat sunglasses3 = imread("Images/blacksunglasses.png", IMREAD_UNCHANGED);

    if (sunglasses1.empty() || sunglasses2.empty() || sunglasses3.empty()) {
        cout << "Error loading sunglasses images!" << endl;
        return -1;
    }

    Mat currentSunglasses = sunglasses1; // Default sunglasses

    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cout << "Error opening camera!" << endl;
        return -1;
    }

    Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        vector<Rect> faces;
        faceCascade.detectMultiScale(gray, faces, 1.1, 5);

        for (const Rect& face : faces) {
            double scale = (double)face.width / currentSunglasses.cols;
            Mat resizedSunglasses;
            resize(currentSunglasses, resizedSunglasses, Size(), scale, scale);

            int y = face.y + face.height / 4;

            for (int i = 0; i < resizedSunglasses.rows; i++) {
                for (int j = 0; j < resizedSunglasses.cols; j++) {
                    Vec4b pixel = resizedSunglasses.at<Vec4b>(i, j);
                    if (pixel[3] > 0) {
                        frame.at<Vec3b>(y + i, face.x + j) = Vec3b(pixel[0], pixel[1], pixel[2]);
                    }
                }
            }
        }

        imshow("Snapchat Sunglasses Filter", frame);

        // Key press events
        char key = waitKey(1);
        if (key == '1') {
            currentSunglasses = sunglasses1;
            cout << "Switched to Sunglasses 1" << endl;
        }
        else if (key == '2') {
            currentSunglasses = sunglasses2;
            cout << "Switched to Sunglasses 2" << endl;
        }
        else if (key == '3') {
            currentSunglasses = sunglasses3;
            cout << "Switched to Sunglasses 3" << endl;
        }
        else if (key == 's') { // Snapshot feature
            // Generate a unique filename using the current timestamp
            string filename = "snapshot_" + to_string(time(0)) + ".png";
            // Save the current frame as an image file
            imwrite(filename, frame);
            cout << "Snapshot saved as " << filename << endl;
        }
        else if (key == 27) { // ESC key
            break;
        }
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
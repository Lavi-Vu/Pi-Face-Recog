#include <iostream>
#include <chrono>
#include <string>
#include <opencv2/opencv.hpp>
#include "TMtCNN.h"
#include "TArcface.h"
#include "TRetina.h"
#include "TWarp.h"
#include "TLive.h"
#include "TBlur.h"
#include "curl/curl.h"
#include <nlohmann/json.hpp>

//----------------------------------------------------------------------------------------
// Adjustable Parameters
//----------------------------------------------------------------------------------------
const int MaxItemsDatabase = 3000;
const int MinHeightFace = 90;
const float MinFaceThreshold = 0.40;
const float FaceLiving = 0.93;
const double MaxBlur = -25.0; // more positive = sharper image
const double MaxAngle = 10.0;
//----------------------------------------------------------------------------------------
// Some globals
//----------------------------------------------------------------------------------------
const int RetinaWidth = 1080;
const int RetinaHeight = 720;
float ScaleX, ScaleY;
std::vector<cv::String> NameFaces;
//----------------------------------------------------------------------------------------
using namespace std;
using namespace cv;
//----------------------------------------------------------------------------------------
//  Computing the cosine distance between input feature and ground truth feature
//----------------------------------------------------------------------------------------
inline float CosineDistance(const cv::Mat v1, const cv::Mat v2)
{
    double dot = v1.dot(v2);
    double denom_v1 = norm(v1);
    double denom_v2 = norm(v2);
    return dot / (denom_v1 * denom_v2);
}
//----------------------------------------------------------------------------------------
// painting
//----------------------------------------------------------------------------------------
void DrawObjects(cv::Mat &frame, vector<FaceObject> &Faces)
{
    for (size_t i = 0; i < Faces.size(); i++)
    {
        FaceObject &obj = Faces[i];

        //----- rectangle around the face -------
        obj.rect.x *= ScaleX;
        obj.rect.y *= ScaleY;
        obj.rect.width *= ScaleX;
        obj.rect.height *= ScaleY;
        cv::rectangle(frame, obj.rect, cv::Scalar(0, 255, 0));
        //---------------------------------------

        //----- labels ----------------------------

        cv::String Str;
        cv::Scalar color;
        int baseLine = 0;

        switch (obj.Color)
        {
        case 0:
            color = cv::Scalar(255, 255, 255);
            break; // default white -> face ok
        case 1:
            color = cv::Scalar(80, 255, 255);
            break; // yellow ->stranger
        case 2:
            color = cv::Scalar(255, 237, 178);
            break; // blue -> too tiny
        case 3:
            color = cv::Scalar(127, 127, 255);
            break; // red -> fake
        default:
            color = cv::Scalar(255, 255, 255);
        }

        switch (obj.NameIndex)
        {
        case -1:
            Str = "Stranger";
            break;
        case -2:
            Str = "too tiny";
            break;
        case -3:
            Str = "Fake !";
            break;
        default:
            Str = NameFaces[obj.NameIndex];
        }

        cv::Size label_size = cv::getTextSize(Str, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseLine);
        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > frame.cols)
            x = frame.cols - label_size.width;

        cv::rectangle(frame, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)), color, -1);
        cv::putText(frame, Str, cv::Point(x, y + label_size.height + 2), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0));
    }
}

// Load Face Content
void LoadFaceCnt(size_t &FaceCnt, vector<cv::String> &NameFaces, vector<cv::Mat> &fc1, TArcFace &ArcFace, TRetina &Rtn, TWarp &Warp)
{
    std::vector<FaceObject> Faces;
    
    int n;
    cv::Mat faces;
    string pattern_jpg = "../img/*.jpg";
    cv::glob(pattern_jpg, NameFaces);
    FaceCnt = NameFaces.size();
    if (FaceCnt == 0)
    {
        cout << "No image files[jpg] in database" << endl;
    }
    else
    {
        cout << "Found " << FaceCnt << " pictures in database." << endl;
        for (size_t i = 0; i < FaceCnt; i++)
        {
            // convert to landmark vector and store into fc
            faces = cv::imread(NameFaces[i]);
            if (faces.cols > 112 && faces.rows > 112)
            {
                ScaleX = ((float)faces.cols) / RetinaWidth;
                ScaleY = ((float)faces.rows) / RetinaHeight;
                // copy/resize image to result_cnn as input tensor
                cv::resize(faces, faces, Size(RetinaWidth, RetinaHeight), INTER_LINEAR);
                Rtn.detect_retinaface(faces, Faces);
                if (Faces.size() == 1)
                {
                    cv::Mat aligned = Warp.Process(faces, Faces[0]);
                    Faces[0].Angle  = Warp.Angle;
                    fc1.push_back(ArcFace.GetFeature(aligned));
                }
            }
            else
            {
                fc1.push_back(ArcFace.GetFeature(faces));
            }
            // get a proper name
            string Str = NameFaces[i];
            n = Str.rfind('/');
            Str = Str.erase(0, n + 1);
            n = Str.find('#');
            if (n > 0)
                Str = Str.erase(n, Str.length() - 1); // remove # some numbers.jpg
            else
                Str = Str.erase(Str.length() - 4, Str.length() - 1); // remove .jpg
            NameFaces[i] = Str;
            if (FaceCnt > 1)
                printf("\rloading: %.2lf%% ", (i * 100.0) / (FaceCnt - 1));
        }
    }
    cout << "" << endl;
    cout << "Loaded " << FaceCnt << " faces in total" << endl;
}

void curlPost(std::string dataToPost, CURL *curl, std::string url)
{
    nlohmann::json jsonData;
    jsonData["personName"] = dataToPost;
    std::string jsonString = jsonData.dump();

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());

    curl_easy_setopt(curl, CURLOPT_POST, 1L);

    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, jsonString.c_str()); // Set POST data directly

    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK)
        std::cerr << "cURL request failed: " << curl_easy_strerror(res) << std::endl;

    else
        std::cout << "Data sent successfully." << std::endl;
    // std::cout << jsonString.c_str() << std::endl;
}
// Callback Function
static size_t WriteCallback(void *contents, size_t size, size_t nmemb, void *userp)
{
    ((std::string *)userp)->append((char *)contents, size * nmemb);
    return size * nmemb;
}

//----------------------------------------------------------------------------------------
// main
//----------------------------------------------------------------------------------------
int main()
{
    std::string url = "localhost:3000/api/";
    CURL *curl = curl_easy_init();

    float f;
    float FPS[16];
    int n, Fcnt = 0;
    size_t i;
    cv::Mat frame;
    cv::Mat result_cnn;
    cv::Mat faces;
    std::vector<FaceObject> Faces;
    std::vector<cv::Mat> fc1;
    string pattern_jpg = "../img/*.jpg";
    cv::String NewItemName;
    size_t FaceCnt;
    std::string personNamePrev;
    // the networks
    TLive Live;
    TWarp Warp;
    TMtCNN MtCNN;
    TArcFace ArcFace;
    TRetina Rtn(RetinaWidth, RetinaHeight, false); // no Vulkan support on a RPi
    TBlur Blur;
    // some timing
    chrono::steady_clock::time_point Tbegin, Tend;

    Live.LoadModel();

    for (i = 0; i < 16; i++)
        FPS[i] = 0.0;

    LoadFaceCnt(FaceCnt, NameFaces, fc1, ArcFace, Rtn, Warp);

    cv::VideoCapture cap("/home/lavi/Desktop/Face-Recognition-Raspberry-Pi-64-bits/Norton_A.mp4");
    // cv::VideoCapture cap(0);
    // std::string cameraURL = "http://192.169.2.214:4747/video";

    // Open the IP camera stream
    // cv::VideoCapture cap(cameraURL);

    if (!cap.isOpened())
    {
        cerr << "ERROR: Unable to open the camera" << endl;
        return 0;
    }
    cout << "Start grabbing, press ESC on TLive window to terminate" << endl;
    while (1)
    {
        cap >> frame;
        if (frame.empty())
        {
            cerr << "End of movie" << endl;
            break;
        }
        ScaleX = ((float)frame.cols) / RetinaWidth;
        ScaleY = ((float)frame.rows) / RetinaHeight;

        // copy/resize image to result_cnn as input tensor
        cv::resize(frame, result_cnn, Size(RetinaWidth, RetinaHeight), INTER_LINEAR);

        Tbegin = chrono::steady_clock::now();

        Rtn.detect_retinaface(result_cnn, Faces);
        // MtCNN.detect(result_cnn, Faces);
        for (i = 0; i < Faces.size(); i++)
        {
            Faces[i].NameIndex = -2; //-2 -> too tiny (may be negative to signal the drawing)
            Faces[i].Color = 2;
            Faces[i].NameProb = 0.0;
            Faces[i].LiveProb = 0.0;
        }
        if (Faces.size() == 1)
        {
            for (i = 0; i < Faces.size(); i++)
            {
                if (Faces[i].FaceProb > MinFaceThreshold)
                {
                    // get centre aligned image
                    cv::Mat aligned = Warp.Process(result_cnn, Faces[i]);
                    Faces[i].Angle = Warp.Angle;
                    // features of camera image
                    cv::Mat fc2 = ArcFace.GetFeature(aligned);
                    // reset indicators
                    Faces[i].NameIndex = -1; // a stranger
                    Faces[i].Color = 1;

                    // the similarity score
                    if (FaceCnt > 0)
                    {
                        vector<double> score_;
                        for (size_t c = 0; c < FaceCnt; c++)
                            score_.push_back(CosineDistance(fc1[c], fc2));
                        int Pmax = max_element(score_.begin(), score_.end()) - score_.begin();
                        Faces[i].NameIndex = Pmax;
                        Faces[i].NameProb = score_[Pmax];
                        score_.clear();
                        if (Faces[i].NameProb >= MinFaceThreshold)
                        {
                            if (Faces[i].rect.height < MinHeightFace)
                            {
                                /// Found in database but too small
                                // std::cout << NameFaces[Faces[i].NameIndex] << std::endl;
                                Faces[i].Color = 2;
                            }
                            else
                            {
                                Faces[i].Color = 0; // Found in DB
                                std::string personName = NameFaces[Faces[i].NameIndex];
                                if (personName != personNamePrev)
                                    curlPost(personName, curl, url);
                                personNamePrev = personName;
                                // test fake face
                                // float x1 = Faces[i].rect.x;
                                // float y1 = Faces[i].rect.y;
                                // float x2 = Faces[i].rect.width + x1;
                                // float y2 = Faces[i].rect.height + y1;
                                // struct LiveFaceBox LiveBox = {x1, y1, x2, y2};

                                // Faces[i].LiveProb = Live.Detect(result_cnn, LiveBox);
                                // if (Faces[i].LiveProb <= FaceLiving)
                                // {
                                //     Faces[i].Color = 3; // fake
                                //     Faces[i].NameIndex = -3;
                                // }
                            }
                        }
                        else
                        {
                            Faces[i].NameIndex = -1; // a stranger
                            Faces[i].Color = 1;
                        }
                    }
                }
            }
        }

        Tend = chrono::steady_clock::now();

        DrawObjects(frame, Faces);

        // calculate frame rate
        f = chrono::duration_cast<chrono::milliseconds>(Tend - Tbegin).count();
        if (f > 0.0)
            FPS[((Fcnt++) & 0x0F)] = 1000.0 / f;
        for (f = 0.0, i = 0; i < 16; i++)
        {
            f += FPS[i];
        }
        cv::putText(frame, cv::format("FPS %0.2f", f / 16), cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(180, 180, 0));

        // show output
        cv::imshow("RPi 64 OS - 1,95 GHz - 2 Mb RAM", frame);
        char esc = cv::waitKey(5);
        if (esc == 27)
            break;
    }
    cv::destroyAllWindows();

    return 0;
}

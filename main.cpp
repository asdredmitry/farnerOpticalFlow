#include <iostream>
#include <cmath>
#include <ctype.h>

#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include <opencv2/highgui.hpp>

#include <unistd.h>

using namespace cv;
using namespace std;

int main(int argv, char ** argc)
{
    namedWindow("OpticalFlowDense");
    if(argv == 1)
    {
        std :: cout << "Enter name of file" << std :: endl;
        return 0;
    }
    else if(argv != 2)
    {
        std :: cout  << "Unknown input" << std :: endl;
        return 0;
    }
    VideoCapture cap(argc[1]);
    if(!cap.isOpened())
    {
        std :: cout << "Cannot open fideo flow" << std :: endl;
        return 0;
    }
    Mat image, grayCur, grayPrev;
    Mat flowUmat;
    while(1)
    {
        cap >> image;
        if(image.empty())
            break;
        cvtColor(image, grayCur, CV_RGB2GRAY);
        if(grayPrev.empty())
            grayPrev = grayCur.clone();
        calcOpticalFlowFarneback(grayPrev, grayCur, flowUmat, 0.4, 1, 12, 2, 8, 1.2, 0);
        //imshow("OpticalFlowDense",flowUmat);
        std :: cout << flowUmat << std :: endl;
        waitKey(20);

    }

    return 0;
}

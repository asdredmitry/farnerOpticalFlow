#include <iostream>
#include <cmath>
#include <ctype.h>

#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include <opencv2/highgui.hpp>

#include <unistd.h>
#include <string>

using namespace cv;
using namespace std;

void drawArrow(Mat & mat, Point2f  pt1, Point2f  pt2, Scalar sc)
{
    if(norm(pt2 - pt1) < 0.2)
        return ;
    Point2f v = (pt2 - pt1)/norm(pt1 - pt2);
    Point2f end = pt1 + 15*v;
    line(mat, pt1, end, sc, 1.5);
    line(mat, end, Point2f(end.x - 2*(v.x*cos(M_PI/4) + v.y*sin(M_PI/4)), end.y - 2*(-v.x*sin(M_PI/4) + v.y*cos(M_PI/4))), sc, 1.5);
    line(mat, end, Point2f(end.x - 2*(v.x*cos(M_PI/4) - v.y*sin(M_PI/4)), end.y - 2*(v.x*sin(M_PI/4) + v.y*cos(M_PI/4))), sc, 1.5);
}
void drawGrid(Mat & mat, int countX, int countY, Mat & flow)
{
    for(int i = 1; i < countY; i++)
    {
        for(int j = 1; j < countX; j++)
        {
            int x = j*mat.cols/countX;
            int y = i*mat.rows/countY;
            circle(mat, Point2f(x, y), 2, Scalar(100, 100, 100));
            line(mat, Point2f(x, y), Point2f(x, y) + flow.at<Point2f>(y, x),Scalar(0, 0 , 255));
        }
    }
}
int main(int argv, char ** argc)
{
    namedWindow("OpticalFlowDense", WINDOW_NORMAL);
    namedWindow("Image", WINDOW_NORMAL);
    namedWindow("ImageGrid",WINDOW_NORMAL);
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
    int counter = 0;
    string str(".jpg");
    string strTmp("/home/dmitry/flow");
    string strTmp2("/home/dmitry/image");
    while(1)
    {
        cap >> image;
        if(image.empty())
            break;
        cvtColor(image, grayCur, CV_RGB2GRAY);
        if(grayPrev.empty())
            grayPrev = grayCur.clone();
        calcOpticalFlowFarneback(grayPrev, grayCur, flowUmat, 0.4, 1, 12, 2, 8, 1.2, 0);
        Mat xy[2];
        split(flowUmat, xy);

        Mat magnitude, angle;
        cartToPolar(xy[0], xy[1], magnitude, angle, true);

        double mag_max;
        cv::minMaxLoc(magnitude, 0, &mag_max);
        magnitude.convertTo(magnitude, -1, 10.0 / mag_max);

        Mat _hsv[3], hsv;
        _hsv[0] = angle;
        _hsv[1] = cv::Mat::ones(angle.size(), CV_32F);
        _hsv[2] = magnitude;
        cv::merge(_hsv, 3, hsv);

        Mat bgr;
        cvtColor(hsv, bgr, CV_HSV2BGR);
        imshow("OpticalFlowDense", bgr);
        imshow("Image", image);
        drawGrid(image, 40, 50, flowUmat);
        imshow("ImageGrid", image);

        char p = waitKey(20);
        if(p == 'p')
            imwrite("/home/dmitry/ImageGrid.jpg", image);
        grayPrev = grayCur.clone();
    }

    return 0;
}

/*#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"

#include <iostream>

using namespace cv;
using namespace std;

static void help()
{
    cout <<
            "\nThis program demonstrates dense optical flow algorithm by Gunnar Farneback\n"
            "Mainly the function: calcOpticalFlowFarneback()\n"
            "Call:\n"
            "./fback\n"
            "This reads from video camera 0\n" << endl;
}
static void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step,
                    double, const Scalar& color)
{
    for(int y = 0; y < cflowmap.rows; y += step)
        for(int x = 0; x < cflowmap.cols; x += step)
        {
            const Point2f& fxy = flow.at<Point2f>(y, x);
            line(cflowmap, Point(x,y), Point(cvRound(x+fxy.x), cvRound(y+fxy.y)),
                 color);
            circle(cflowmap, Point(x,y), 2, color, -1);
        }
}

int main(int argc, char** argv)
{
    VideoCapture cap(argv[1]);
    help();
    if( !cap.isOpened() )
        return -1;

    Mat flow, cflow, frame;
    UMat gray, prevgray, uflow;
    namedWindow("flow", 1);

    for(;;)
    {
        cap >> frame;
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        if( !prevgray.empty() )
        {
            calcOpticalFlowFarneback(prevgray, gray, uflow, 0.5, 3, 15, 3, 5, 1.2, 0);
            cvtColor(prevgray, cflow, COLOR_GRAY2BGR);
            uflow.copyTo(flow);
            drawOptFlowMap(flow, cflow, 16, 1.5, Scalar(0, 255, 0));
            imshow("flow", cflow);
        }
        if(waitKey(30)>=0)
            break;
        std::swap(prevgray, gray);
    }
    return 0;
}
*/

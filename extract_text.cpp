#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include "tesseract/baseapi.h"

//compile with g++ -std=c++11 extract_text.cpp -ggdb `pkg-config --cflags opencv` `pkg-config --cflags tesseract` -o a `pkg-config --libs opencv` `pkg-config --libs tesseract`

using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
    const char* lang = "eng";

    Mat image = imread( argv[1] );
    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    Mat fin;
    threshold(gray, fin, 160, 255, THRESH_BINARY_INV);
    Mat kernel =  getStructuringElement(MORPH_CROSS, Size(2, 4));
    Mat dilated;
    dilate(fin, dilated, kernel, Point(-1,-1), 6);
    //imwrite("dilated.png",dilated); uncomment to see dilated image
    vector<Vec4i> hierarchy;
    vector<vector<Point> >contours;
    findContours(dilated, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);

    vector<Rect> rects;
    Rect big_rect = Rect(image.cols/2,image.rows/2,1,1);
    for (const auto& c : contours)
    {
        //                x        y
        //one columns is 850px x 5400px
        Rect r = boundingRect(c);
        float percentage_height = (float)r.height / (float)image.rows;
        float percentage_width = (float)r.width / (float)image.cols;
        
        //Should be robust to different dimensions of test image / test set
        if ( percentage_height > 0.92 || percentage_width > 0.20)
            continue;
        if (percentage_height < 0.05 || percentage_width < 0.04)
            continue;

        big_rect = big_rect | r; // finds bounding box of all Rects
        rects.push_back( r ); // stores Rects 
    }

    for ( size_t i = 0; i < rects.size(); i++ )
    {
        // sets y and height of all rects 
        rects[i].y = big_rect.y;
        rects[i].height = big_rect.height;
    }
    
    sort(rects.begin(), rects.end(), [] (Rect a, Rect b) { return a.x < b.x; });
    for ( size_t i = 1; i < rects.size(); i++ )
    {
        //removes redundant boundingRects
        //minimum area Rect that contains both [i-1] and [i]
        Rect big_rect = rects[i-1] | rects[i];

        //If [i-1] and [i] bound same column
        if( big_rect.width < rects[i-1].width + rects[i].width )
        {
            //has 0 width
            rects[i-1] = Rect();
            rects[i] = big_rect;
        }
    }

    vector<Rect> rect;
    for ( size_t i = 1; i < rects.size(); i++ )
    {
        //filters for 0 width
        if( rects[i].width > 0 )
        {
            //draw Rects
            rectangle(image, rects[i], Scalar(255,0,255), 2);
            rect.push_back( rects[i] );
        }
    }
    
    //imwrite("result.png", image); uncomment to see image with Rects drawn around columns

    sort(rects.begin(), rects.end(), [] (Rect a, Rect b) { return a.x < b.x; }); //sort by x
    
    //Optimize images (seven columns) before passing to Tesseract
    Mat col0 = image(rect[0]);
    cvtColor(col0, col0, COLOR_BGR2GRAY);
    threshold(col0, col0, 152, 255, THRESH_BINARY);

    Mat col1 = image(rect[1]);
    cvtColor(col1, col1, COLOR_BGR2GRAY);
    threshold(col1, col1, 152, 255, THRESH_BINARY);

    Mat col2 = image(rect[2]);
    cvtColor(col2, col2, COLOR_BGR2GRAY);
    threshold(col2, col2, 152, 255, THRESH_BINARY);

    Mat col3 = image(rect[3]);
    cvtColor(col3, col3, COLOR_BGR2GRAY);
    threshold(col3, col3, 152, 255, THRESH_BINARY);

    Mat col4 = image(rect[4]);
    cvtColor(col4, col4, COLOR_BGR2GRAY);
    threshold(col4, col4, 152, 255, THRESH_BINARY);

    Mat col5 = image(rect[5]);
    cvtColor(col5, col5, COLOR_BGR2GRAY);
    threshold(col5, col5, 152, 255, THRESH_BINARY);

    Mat col6 = image(rect[6]);
    cvtColor(col6, col6, COLOR_BGR2GRAY);
    threshold(col6, col6, 152, 255, THRESH_BINARY);


    //instantiate Tesseract and pass it one image at a time
    tesseract::TessBaseAPI tess;
    tess.Init(NULL, lang, tesseract::OEM_DEFAULT);
    tess.SetPageSegMode(tesseract::PSM_SINGLE_BLOCK);
    tess.SetImage((unsigned char*)col0.data, col0.cols, col0.rows, 1, col0.cols);

    char* out = tess.GetUTF8Text();
    cout << out << std::endl;
    cout << "COLUMN BREAK" << endl;

    tess.SetImage((unsigned char*)col1.data, col1.cols, col1.rows, 1, col1.cols);

    out = tess.GetUTF8Text();
    cout << out << std::endl;
    cout << "COLUMN BREAK" << endl;

    tess.SetImage((unsigned char*)col2.data, col2.cols, col2.rows, 1, col2.cols);

    out = tess.GetUTF8Text();
    cout << out << std::endl;
    cout << "COLUMN BREAK" << endl;

    tess.SetImage((unsigned char*)col3.data, col3.cols, col3.rows, 1, col3.cols);

    out = tess.GetUTF8Text();
    cout << out << std::endl;
    cout << "COLUMN BREAK" << endl;

    tess.SetImage((unsigned char*)col4.data, col4.cols, col4.rows, 1, col4.cols);

    out = tess.GetUTF8Text();
    cout << out << std::endl;
    cout << "COLUMN BREAK" << endl;

    tess.SetImage((unsigned char*)col5.data, col5.cols, col5.rows, 1, col5.cols);

    out = tess.GetUTF8Text();
    cout << out << std::endl;
    cout << "COLUMN BREAK" << endl;

    tess.SetImage((unsigned char*)col6.data, col6.cols, col6.rows, 1, col6.cols);

    out = tess.GetUTF8Text();
    cout << out << std::endl;
    cout << "COLUMN BREAK" << endl;
    
    waitKey(0);
    return 0;
}

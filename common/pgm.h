#ifndef PGM_H
#define PGM_H

#include <opencv2/opencv.hpp>

class PGMImage {
public:
    PGMImage(char *);
    PGMImage(int x, int y, int col);
    ~PGMImage();
    bool write(char *);
    bool to_jpg(char *);

    int x_dim;
    int y_dim;
    int num_colors;
    unsigned char *pixels;
private:
    cv::Mat make_image();
};

#endif

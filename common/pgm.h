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
    bool to_jpg_with_line(char *, int *accumulator, int threshold, int total_degree_bins, int degree_increment, int total_radial_bins);

    int x_dim;
    int y_dim;
    int num_colors;
    unsigned char *pixels;
private:
    cv::Mat make_image();
    int get_y(float radius, float degree, int x);
};

#endif

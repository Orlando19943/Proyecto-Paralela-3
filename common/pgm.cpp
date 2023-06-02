/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : Used in different projects to handle PGM I/O
 To build use  : 
 ============================================================================
 */
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <opencv2/opencv.hpp>
#include <math.h>
#include "pgm.h"

using namespace std;
using namespace cv;

//-------------------------------------------------------------------
PGMImage::PGMImage(char *fname)
{
   x_dim=y_dim=num_colors=0;
   pixels=NULL;
   
   FILE *ifile;
   ifile = fopen(fname, "rb");
   if(!ifile) return;

   char *buff = NULL;
   size_t temp;

   fscanf(ifile, "%*s %i %i %i", &x_dim, &y_dim, &num_colors);

   getline((char **)&buff, &temp, ifile); // eliminate CR-LF
   
   assert(x_dim >1 && y_dim >1 && num_colors >1);
   pixels = new unsigned char[x_dim * y_dim];
   fread((void *) pixels, 1, x_dim*y_dim, ifile);   
   
   fclose(ifile);
}
//-------------------------------------------------------------------
PGMImage::PGMImage(int x=100, int y=100, int col=16)
{
   num_colors = (col>1) ? col : 16;
   x_dim = (x>1) ? x : 100;
   y_dim = (y>1) ? y : 100;
   pixels = new unsigned char[x_dim * y_dim];
   memset(pixels, 0, x_dim * y_dim);
   assert(pixels);
}
//-------------------------------------------------------------------
PGMImage::~PGMImage()
{
  if(pixels != NULL)
     delete [] pixels;
  pixels = NULL;
}
//-------------------------------------------------------------------
bool PGMImage::write(char *fname)
{
   int i,j;
   FILE *ofile;
   ofile = fopen(fname, "w+t");
   if(!ofile) return 0;

   fprintf(ofile,"P5\n%i %i\n%i\n",x_dim, y_dim, num_colors);
   fwrite(pixels, 1, x_dim*y_dim, ofile);
   fclose(ofile);
   return 1;
}
//-------------------------------------------------------------------
Mat PGMImage::make_image() {
    unsigned char pixel_value;
    Mat image = Mat::zeros(Size(x_dim,y_dim), CV_8UC3);

    for(int x = 0; x < x_dim; x++) {
        for(int y = 0; y < y_dim; y++) {
            pixel_value = pixels[x_dim * y + x];
            if (pixel_value != 0) {
                image.at<Vec3b>(Point(x, y)) = Vec3b(pixel_value, pixel_value, pixel_value);
            }
        }
    }
    return image;
}

bool PGMImage::to_jpg(char *location) {
    Mat image = make_image();
    return imwrite(location, image);
}
//-------------------------------------------------------------------
int PGMImage::get_y(float radius, float degree, int x) {
    return y_dim / 2 - ((radius - (x - x_dim / 2) * cos(degree)) / sin(degree));
}

bool PGMImage::to_jpg_with_line(char *location, int *accumulator, int threshold, int total_degree_bins, int degree_increment, int total_radial_bins) {
    Mat image = make_image();

    float max_radius = sqrt(1.0 * x_dim * x_dim + 1.0 * y_dim * y_dim) / 2;
    float radial_bin_width = 2 * max_radius / total_radial_bins;

    const float degree_bin_width = degree_increment * M_PI / 180;

    for (int degree_bin = 0; degree_bin < total_degree_bins; degree_bin++){
        for (int radial_bin = 0; radial_bin < total_radial_bins; radial_bin++) {
            if (accumulator[radial_bin * total_degree_bins + degree_bin] > threshold) {
                float radius = radial_bin * radial_bin_width - max_radius;
                float degree = degree_bin * degree_bin_width;

                int initial_y = get_y(radius, degree, 0);
                int final_y = get_y(radius, degree, x_dim - 1);
                printf("initial %d, final %d \n", initial_y, final_y);

                line(image, Point(0, initial_y), Point(x_dim - 1, final_y), (0, 0, 255), 1);
            }
        }
    }
    return imwrite(location, image);
}

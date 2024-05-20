#define _USE_MATH_DEFINES

//includes
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <random>
#include <chrono>
using Clock = std::chrono::high_resolution_clock;

#include <complex>
#include <math.h>
#include <omp.h>


//transform types
const bool FORWARD = 1;
const bool INVERSE = 0;

const bool SPECTRUM_IMG = 1;
const bool NORMAL_IMG = 0;


//display type
const bool BRIGHTEN = 1;
const bool NO_BRIGHTEN = 0;


//phillips spectrum constants
const double L = 1000;
const double V = 40;
const double g = 9.80665;
const double L_ = V*V/g;

const double W1 = 1;
const double W2 = 1;
const double w = sqrt(W1*W1 + W2*W2);

const double A = 4;
const double l = 0.5;


//texture constants
#define M 256
constexpr int cx_log2(int n) { 
    if (n <= 2) {
        return 1;
    }
    else {
        return cx_log2(n/2)+1;
    }
}
#define bitsConst cx_log2(M)


//namespaces
using namespace cv;
using namespace std;




//Fourier Transform Stuff
static double pingpong0[M][M][4];
static double pingpong1[M][M][4];
static double butterflyForward[M][bitsConst][4];
static double butterflyInverse[M][bitsConst][4];

void Normalization(double data[M][M][4], bool forward, bool spectrum) {
    if (spectrum) {
        if (forward) {
            #pragma omp parallel for collapse(2)
            for (int y = 0; y < M; y++) {
                for (int x = 0; x < M; x++) {
                    data[y][x][0] = 0;
                    data[y][x][3] = 1;
                }
            }
        }
        else {
            #pragma omp parallel for collapse(2)
            for (int y = 0; y < M; y++) {
                for (int x = 0; x < M; x++) {
                    double perm = 1.0;
                    int idx = int(fmod(int(y + x), 2));
                    perm = idx ? 1.0 : -1.0;

                    double h = (data[y][x][2]/(M*M))*perm;

                    data[y][x][2] = h;
                    data[y][x][1] = 0;
                    data[y][x][0] = 0;
                    data[y][x][3] = 1;
                }
            }
        }
    }

    else {
        if (forward) {
            #pragma omp parallel for collapse(2)
            for (int y = 0; y < M; y++) {
                for (int x = 0; x < M; x++) {
                    data[y][x][2] = data[y][x][2]/(M*M);
                    data[y][x][1] = data[y][x][1]/(M*M);
                    data[y][x][0] = 0;
                    data[y][x][3] = 1;
                }
            }
        }
        else {
            #pragma omp parallel for collapse(2)
            for (int y = 0; y < M; y++) {
                for (int x = 0; x < M; x++) {
                    data[y][x][0] = 0;
                    data[y][x][3] = 1;
                }
            }
        }
    }
}



void ft2D_(double signal[M][M][4], double X[M][M][4], bool forward, bool spectrum) {
    double factor = (forward ? -1.0 : 1.0);
    #pragma omp parallel for collapse(2)
    for (int v = 0; v < M; v++) {
        for (int u = 0; u < M; u++) {

            double sum_real = 0;
            double sum_imag = 0;

            #pragma omp parallel for collapse(2) reduction(+:sum_real) reduction(+:sum_imag)
            for (int y = 0; y < M; y++) {
                for (int x = 0; x < M; x++) {
                    double exp = (2 * M_PI) * (x*u + y*v) / M;
                    double real = cos(exp);
                    double imag = factor * sin(exp);

                    complex<double> val = complex<double>(signal[y][x][2], signal[y][x][1]) * complex<double>(real, imag);
                    sum_real += val.real();
                    sum_imag += val.imag();
                }
            }

            //different channels
            X[v][u][2] = sum_real;
            X[v][u][1] = sum_imag;
        }
    }

    Normalization(X, forward, spectrum);
}



void ft1D(double x[M][M][4], double X[M][M][4], bool forward, bool horizontal, int row_col) {
    double factor = forward ? -1 : 1;

    if (horizontal) {
        #pragma omp parallel for
        for (int k = 0; k < M; k++) {

            double sum_real = 0;
            double sum_imag = 0;

            //#pragma omp parallel for reduction(+:sum_real) reduction(+:sum_imag)   WORKS, BUT WOULD RATHER COMMENT OUT IN-CASE
            for (int n = 0; n < M; n++) {
                double real =          cos(((2 * M_PI) / M) * k * n);
                double imag = factor * sin(((2 * M_PI) / M) * k * n);
                complex<double> val = complex<double>(x[row_col][n][2], x[row_col][n][1]) * complex<double>(real, imag);
                sum_real += val.real();
                sum_imag += val.imag();
            }

            X[row_col][k][2] = sum_real;
            X[row_col][k][1] = sum_imag;
        }
    }
    else {
        #pragma omp parallel for
        for (int k = 0; k < M; k++) {

            double sum_real = 0;
            double sum_imag = 0;

            //#pragma omp parallel for reduction(+:sum_real) reduction(+:sum_imag)   BROKEN
            for (int n = 0; n < M; n++) {
                double real =          cos(((2 * M_PI) / M) * k * n);
                double imag = factor * sin(((2 * M_PI) / M) * k * n);
                complex<double> val = complex<double>(x[n][row_col][2], x[n][row_col][1]) * complex<double>(real, imag);
                sum_real += val.real();
                sum_imag += val.imag();
            }

            X[k][row_col][2] = sum_real;
            X[k][row_col][1] = sum_imag;
        }
    }
}

void ft2D(double signal[M][M][4], double X[M][M][4], bool forward, bool spectrum) {
    static double X_mid[M][M][4];

    //Horizontal DFTS
    #pragma omp parallel for
    for (int y = 0; y < M; y++) {
        ft1D(signal, X_mid, forward, true, y);
    }

    //Vertical DFTS
    #pragma omp parallel for
    for (int x = 0; x < M; x++) {
        ft1D(X_mid, X, forward, false, x);
    }


    //Normalization
    Normalization(X, forward, spectrum);
}



unsigned int bitsReversed(unsigned int num, int bits) {
    unsigned int rev = 0;
    for (int i = 0; i < bits; i++) {
        unsigned int bit_n = (num >> i) & (unsigned int)1;
        rev = rev | (bit_n << (bits-1 - i));
    }
    return rev;
}

void changeOrder(double input[M][M][4], double output[M][M][4], bool horizontal) {
    if (horizontal) {
        #pragma omp parallel for collapse(2)
        for (int row_col = 0; row_col < M; row_col++) {
            for (int i = 0; i < M; i++) {
                unsigned int j = bitsReversed(i, bitsConst);
                
                memcpy(&output[row_col][i], &input[row_col][j], 4*sizeof(double));
            }
        }
    }
    else {
        #pragma omp parallel for collapse(2)
        for (int row_col = 0; row_col < M; row_col++) {
            for (int i = 0; i < M; i++) {
                unsigned int j = bitsReversed(i, bitsConst);

                memcpy(&output[i][row_col], &input[j][row_col], 4*sizeof(double));
            }
        }
    }
}



void HorizontalButterfly_(int n, int row, double pingpong0[M][M][4], double pingpong1[M][M][4], bool forward) {
    #pragma omp parallel for
    for (int v = 0; v < M; v+=n){

        double exp = 2 * M_PI / n * (forward ? -1 : 1);
        complex<double> w(1), wn(cos(exp), sin(exp));

        #pragma omp parallel for
        for (int i = 0; i < n/2; i++) {
            int evenIdx = v+  2*i;
            int oddIdx =  v+  2*i + 1;
            complex<double> even(pingpong0[row][evenIdx][2], pingpong0[row][evenIdx][1]);
            complex<double> odd(pingpong0[row][oddIdx][2], pingpong0[row][oddIdx][1]);

            complex<double> butterfly_1 = even + w * odd;
            complex<double> butterfly_2 = even - w * odd;

                
            pingpong1[row][v+  i][2] = butterfly_1.real();       pingpong1[row][v+  i][1] = butterfly_1.imag();
            pingpong1[row][v+  i + n/2][2] = butterfly_2.real(); pingpong1[row][v+  i + n/2][1] = butterfly_2.imag();
            w *= wn;
        }
    }

    //even-odd merge step
    if (n < M){
        #pragma omp parallel for collapse(2)
        for (int v = 0; v < M; v+=2*n){
            for(int i = 0; i < 2*n; i+=2){
                memcpy(&pingpong0[row][v + i], &pingpong1[row][v + i/2], 4*sizeof(double));
                memcpy(&pingpong0[row][v + i+1], &pingpong1[row][v + i/2 + n], 4*sizeof(double));
            }
        }
    }
}

void VerticalButterfly_(int n, int col, double pingpong0[M][M][4], double pingpong1[M][M][4], bool forward) {
    #pragma omp parallel for
    for (int v = 0; v < M; v+=n){

        double exp = 2 * M_PI / n * (forward ? -1 : 1);
        complex<double> w(1), wn(cos(exp), sin(exp));

        #pragma omp parallel for
        for (int i = 0; i < n/2; i++) {
            int evenIdx = v+  2*i;
            int oddIdx =  v+  2*i + 1;
            complex<double> even(pingpong0[evenIdx][col][2], pingpong0[evenIdx][col][1]);
            complex<double> odd(pingpong0[oddIdx][col][2], pingpong0[oddIdx][col][1]);

            complex<double> butterfly_1 = even + w * odd;
            complex<double> butterfly_2 = even - w * odd;

                
            pingpong1[v+  i][col][2] = butterfly_1.real();       pingpong1[v+  i][col][1] = butterfly_1.imag();
            pingpong1[v+  i + n/2][col][2] = butterfly_2.real(); pingpong1[v+  i + n/2][col][1] = butterfly_2.imag();
            w *= wn;
        }
    }

    //even-odd merge step
    if (n < M){
        #pragma omp parallel for collapse(2)
        for (int v = 0; v < M; v+=2*n){
            for(int i = 0; i < 2*n; i+=2){
                memcpy(&pingpong0[v + i][col], &pingpong1[v + i/2][col], 4*sizeof(double));
                memcpy(&pingpong0[v + i+1][col], &pingpong1[v + i/2 + n][col], 4*sizeof(double));
            }
        }
    }
}

//My 2D FFT implementation
void fft2D_(double signal[M][M][4], double X[M][M][4], bool forward, bool spectrum) {
    //horizontal butterflies
    changeOrder(signal, pingpong0, true);
    for (int n = 2; n <= M; n*=2) {
        #pragma omp parallel for
        for (int y = 0; y < M; y++) {
            HorizontalButterfly_(n, y, pingpong0, pingpong1, forward);
        }
    }

    //vertical butterflies
    changeOrder(pingpong1, pingpong0, false);
    for (int n = 2; n <= M; n *= 2) {
        #pragma omp parallel for
        for (int x = 0; x < M; x++) {
            VerticalButterfly_(n, x, pingpong0, X, forward);
        }
    }



    //Normalization
    Normalization(X, forward, spectrum);
}



static void makeButterflyTexture(double butterfly[M][bitsConst][4], bool forward) {
    #pragma omp parallel for collapse(2)
    for (int m = 0; m < M; m++) {
        for (int stage = 0; stage < bitsConst; stage++) {
            double k = fmod(double(m) * (double(M)/pow(2, stage + 1)), M);
            double exp = 2.0 * M_PI * k / double(M) * (forward ? -1 : 1);
            complex<double> twiddle(cos(exp), sin(exp));

            int butterflyspan = int(pow(2, stage));

            int butterflywing;
            if(fmod(m, pow(2, stage + 1)) < pow(2, stage))
                butterflywing = 1;
            else butterflywing = 0;


            if (stage == 0) {
                if (butterflywing == 1){
                             //M-1 - y
                    butterfly[m][stage][2] = twiddle.real();//r
                    butterfly[m][stage][1] = twiddle.imag();//g
                    butterfly[m][stage][0] = bitsReversed(m, bitsConst);//b
                    butterfly[m][stage][3] = bitsReversed(m + 1, bitsConst);//a
                }

                else{
                    butterfly[m][stage][2] = twiddle.real();//r
                    butterfly[m][stage][1] = twiddle.imag();//g
                    butterfly[m][stage][0] = bitsReversed(m - 1, bitsConst);//b
                    butterfly[m][stage][3] = bitsReversed(m, bitsConst);//a
                }
            }
            else {
                if (butterflywing == 1){
                    butterfly[m][stage][2] = twiddle.real();//r
                    butterfly[m][stage][1] = twiddle.imag();//g
                    butterfly[m][stage][0] = m;//b
                    butterfly[m][stage][3] = m + butterflyspan;//a
                }

                else{
                    butterfly[m][stage][2] = twiddle.real();//r
                    butterfly[m][stage][1] = twiddle.imag();//g
                    butterfly[m][stage][0] = m - butterflyspan;//b
                    butterfly[m][stage][3] = m;//a
                }
            }
        }
    }
}


void HorizontalButterfly(int stage, int y, int x, double pingpongIn[M][M][4], double pingpongOut[M][M][4], double butterfly[M][bitsConst][4]) {
    double data[4];
    memcpy(&data, &butterfly[x][stage], 4*sizeof(double));

    double p_[2];              
    int pIdx = (int)data[0]; //blue
    p_[0] = pingpongIn[y][pIdx][2]; //red
    p_[1] = pingpongIn[y][pIdx][1]; //green

    double q_[2];              
    int qIdx = (int)data[3]; //alpha
    q_[0] = pingpongIn[y][qIdx][2]; //red 
    q_[1] = pingpongIn[y][qIdx][1]; //green


    complex<double> p(p_[0], p_[1]);
    complex<double> q(q_[0], q_[1]);
    complex<double> w(data[2], data[1]);

    complex<double> H = w*q + p;

    pingpongOut[y][x][2] = H.real();
    pingpongOut[y][x][1] = H.imag();
}

void VerticalButterfly(int stage, int y, int x, double pingpongIn[M][M][4], double pingpongOut[M][M][4], double butterfly[M][bitsConst][4]) {
    double data[4];
    memcpy(&data, &butterfly[y][stage], 4*sizeof(double));

    double p_[2];              //blue  
    p_[0] = pingpongIn[(int)data[0]][x][2]; //red
    p_[1] = pingpongIn[(int)data[0]][x][1]; //green

    double q_[2];              //alpha
    q_[0] = pingpongIn[(int)data[3]][x][2]; //red 
    q_[1] = pingpongIn[(int)data[3]][x][1]; //green


    complex<double> p(p_[0], p_[1]);
    complex<double> q(q_[0], q_[1]);
    complex<double> w(data[2], data[1]);

    complex<double> H = w*q + p;

    pingpongOut[y][x][2] = H.real();
    pingpongOut[y][x][1] = H.imag();
}

//N.B. signal will be changed, make sure not to use it again after running the FFT
void fft2DWrapper(double signal[M][M][4], double X[M][M][4], bool forward, bool spectrum, double butterfly[M][bitsConst][4]) {
    bool pingpong = 0;
    //memcpy(pingpong0, signal, M*M*4 * sizeof(double));
    
    //horizontal butterflies
    for (int stage = 0; stage < bitsConst; stage++){
        if (pingpong == 0) {
            #pragma omp parallel for collapse(2)
            for (int y = 0; y < M; y++){
                for (int x = 0; x < M; x++){
                    HorizontalButterfly(stage, y, x, signal, pingpong1, butterfly);
                }
            }
        }
        else if (pingpong == 1) {
            #pragma omp parallel for collapse(2)
            for (int y = 0; y < M; y++){
                for (int x = 0; x < M; x++){
                    HorizontalButterfly(stage, y, x, pingpong1, signal, butterfly);
                }
            }
        }

        pingpong = !pingpong;
    }

    //vertical butterflies
    for (int stage = 0; stage < bitsConst; stage++){
        if (pingpong == 0) {
            #pragma omp parallel for collapse(2)
            for (int y = 0; y < M; y++){
                for (int x = 0; x < M; x++){
                    VerticalButterfly(stage, y, x, signal, pingpong1, butterfly);
                }
            }
        }
        else if (pingpong == 1) {
            #pragma omp parallel for collapse(2)
            for (int y = 0; y < M; y++){
                for (int x = 0; x < M; x++){
                    VerticalButterfly(stage, y, x, pingpong1, signal, butterfly);
                }
            }
        }

        pingpong = !pingpong;
    }


    //copyyy
    if (pingpong == 0) {
        memcpy(X, signal, M*M*4 * sizeof(double));
    }
    else {
        memcpy(X, pingpong1, M*M*4 * sizeof(double));
    }


    //Normalization
    Normalization(X, forward, spectrum);
}

//Standard 2D FFT implementation
void fft2D(double signal[M][M][4], double X[M][M][4], bool forward, bool spectrum) {
    if (forward) {
        fft2DWrapper(signal, X, forward, spectrum, butterflyForward);
    }
    else {
        fft2DWrapper(signal, X, forward, spectrum, butterflyInverse);
    }
}




//Make Oceanographic Spectra 
void CreateNoiseTexture(double noise[M][M][4]) {
    double lower_bound = -1;
    double upper_bound = 1;
    uniform_real_distribution<double> unif(lower_bound, upper_bound);
    default_random_engine re;


    for (int y = 0; y < M; y++){
        for (int x = 0; x < M; x++) {
            noise[y][x][2] = unif(re);
            noise[y][x][1] = unif(re);
            noise[y][x][0] = unif(re);
            noise[y][x][3] = unif(re);
        }
    }
}

double Phillips(double K1, double K2, double k) {
    double K1_N = K1/k;
    double K2_N = K2/k;
    double W1_N = W1/w;
    double W2_N = W2/w;

    return A * (exp(-pow(k*L_, -2)) * pow(k, -4)) * pow(K1_N*W1_N + K2_N*W2_N, 2);
}

const double bound = 4000;
void Create_h0(double noise[M][M][4], double h0k[M][M][4], double h0minusk[M][M][4]) {
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < M; y++){
        for (int x = 0; x < M; x++) {

            double K1 = 2 * M_PI * (x - M/2) / L;
            double K2 = 2 * M_PI * (y - M/2) / L;
            double k = sqrt(K1*K1 + K2*K2);
            
            double suppressionFactor = exp(-k*k*l*l);
            
            double h0k_val = max(-bound, min(bound, sqrt((suppressionFactor * Phillips(K1, K2, k)) * 0.5)));
            h0k[y][x][2] = noise[y][x][2] * h0k_val;
            h0k[y][x][1] = noise[y][x][1] * h0k_val;
            //h0k[y][x][0] = 0;
            //h0k[y][x][3] = 1;

            double h0minusk_val = max(-bound, min(bound, sqrt((suppressionFactor * Phillips(-K1, -K2, k)) * 0.5)));
            h0minusk[y][x][2] = noise[y][x][0] * h0minusk_val;
            h0minusk[y][x][1] = noise[y][x][3] * h0minusk_val;
            //h0minusk[y][x][0] = 0;
            //h0minusk[y][x][3] = 1;
        }
    }
}

void Create_hk(double h0k[M][M][4], double h0minusk[M][M][4], double hk[M][M][4], double t) {
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < M; y++){
        for (int x = 0; x < M; x++) {

            double K1 = 2 * M_PI * (x - M/2) / L;
            double K2 = 2 * M_PI * (y - M/2) / L;
            double k = sqrt(K1*K1 + K2*K2);

            double omega = sqrt(g * k);
            double expo = omega * t;
            complex<double> comp_1(cos(expo), sin(expo));
            complex<double> comp_2(comp_1.real(), -comp_1.imag());

            complex<double> h0k_val(h0k[y][x][2], h0k[y][x][1]);
            complex<double> h0minusk_val(h0minusk[y][x][2], h0minusk[y][x][1]);

            complex<double> hk_val = h0k_val * comp_1  +  h0minusk_val * comp_2;
            hk[y][x][2] = hk_val.real();
            hk[y][x][1] = hk_val.imag();
            hk[y][x][0] = 0;
            hk[y][x][3] = 1;
        }
    }
}



//Display Stuff
void fftshift(double X[M][M][4], double X_shift[M][M][4]) {
    for (int v = 0; v < M/2; v++) {
        memcpy(&X_shift[v + M/2][M/2], &X[v], M*4 * sizeof(double));
        memcpy(&X_shift[v], &X[v + M/2][M/2], M*4 * sizeof(double));
    }

    for (int v = 0; v < M/2; v++) {
        memcpy(&X_shift[v + M/2][-M/2], &X[v], M*4 * sizeof(double));
        memcpy(&X_shift[v], &X[v + M/2][-M/2], M*4 * sizeof(double));
    }
}


void imshow1C(string name, double image[M][M], bool brighten) {
    Mat cv_img(M, M, CV_64FC1);
    memcpy(cv_img.data, image, M*M * sizeof(double));
    if (brighten)
        cv_img *= 255;

    imshow(name, cv_img);
}

void imshow4C(string name, double image[M][M][4], bool brighten) {
    Mat cv_img(M, M, CV_64FC4);
    memcpy(cv_img.data, image, M*M*4 * sizeof(double));
    if (brighten)
        cv_img *= 255;
    
    imshow(name, cv_img);
}


void im1CTo4C(double image_1C[M][M], double image_4C[M][M][4]) {
    for (int y = 0; y < M; y++) {
        for (int x = 0; x < M; x++) {
            image_4C[y][x][2] = image_1C[y][x];
            image_4C[y][x][1] = image_1C[y][x];
            image_4C[y][x][0] = image_1C[y][x];
            image_4C[y][x][3] = 1;
        }
    }
}

void im4CTo1C(double image_4C[M][M][4], double image_1C[M][M]) {
    for (int y = 0; y < M; y++) {
        for (int x = 0; x < M; x++) {
            image_1C[y][x] = image_4C[y][x][2];
        }
    }
}


void imshow1CTo4C(string name, double image_1C[M][M], bool brighten) {
    static double image_4C[M][M][4];
    im1CTo4C(image_1C, image_4C);

    imshow4C(name, image_4C, brighten);
}

void imshow4CTo1C(string name, double image_4C[M][M][4], bool brighten){
    static double image_1C[M][M];
    im4CTo1C(image_4C, image_1C);

    imshow1C(name, image_1C, brighten);
}


void imshow1CTo4C_PreAlloc(string name, double image_1C[M][M], double image_4C[M][M][4], bool brighten) {
    im1CTo4C(image_1C, image_4C);
    imshow4C(name, image_4C, brighten);
}

void imshow4CTo1C_PreAlloc(string name, double image_4C[M][M][4], double image_1C[M][M], bool brighten){
    im4CTo1C(image_4C, image_1C);
    imshow1C(name, image_1C, brighten);
}


Mat loadImage(string image_path) {
    Mat signal_image = imread(image_path, IMREAD_GRAYSCALE);
    signal_image.convertTo(signal_image, CV_64FC1);
    signal_image /= 255;
    //cout << signal_image.at<Vec3d>(43, 52);
    resize(signal_image, signal_image, { M, M }, 0, 0, cv::INTER_NEAREST);

    return signal_image;
}




int main(){
    //pre-compute butterfly textures
    makeButterflyTexture(butterflyForward, true);
    makeButterflyTexture(butterflyInverse, false);
    //IMSHOW butterfly texture
    /*Mat cv_img(M, bitsConst, CV_64FC4);
    memcpy(cv_img.data, butterflyInverse, M*bitsConst*4 * sizeof(double));
    resize(cv_img, cv_img, { M, M }, 0, 0, cv::INTER_NEAREST);
    imshow("butterflyInverse", cv_img);*/



    //Standard Image
    /*Mat signal_img = loadImage("./images/lena_color_256.tif");
    //imshow("Image", signal_img);
    
    //1Channel -> 4Channel
    static double signal1C[M][M];
    memcpy(signal1C, signal_img.data, M*M * sizeof(double));
    static double signal[M][M][4];
    //im1CTo4C(signal1C, signal);
    //imshow4C("signal", signal, NO_BRIGHTEN);
    
    
    //FT
    static double X[M][M][4];
    fft2D_(signal, X, FORWARD, NORMAL_IMG);
    //imshow4C("X", X, BRIGHTEN);
    
    //Shift
    static double X_shift[M][M][4];
    fftshift(X, X_shift);
    imshow4C("X Shift", X_shift, BRIGHTEN);
    
    
    //IFT
    //static double x[M][M][4];
    //fft2D_(X, x, INVERSE, NORMAL_IMG);
    //imshow4CTo1C("x", x, NO_BRIGHTEN);*/
    



    //Spectrum Image
    //noise image
    static double noise[M][M][4];
    CreateNoiseTexture(noise);
    //imshow4C("noise", noise, NO_BRIGHTEN);
    
    //spectrum texture init
    static double h0k[M][M][4];
    static double h0minusk[M][M][4];
    Create_h0(noise, h0k, h0minusk);
    //imshow4C("h0k", h0k, NO_BRIGHTEN);
    //imshow4C("h0minusk", h0minusk, NO_BRIGHTEN);
    
    
    //spectrum texture(t)
    static double hk[M][M][4];
    static double heightmap4C[M][M][4];
    static double heightmap1C[M][M];

    chrono::steady_clock::time_point start = Clock::now();
    chrono::steady_clock::time_point current;
    double t = 0;
    while (1) {
        //hk
        Create_hk(h0k, h0minusk, hk, t);
        //imshow4C("hk", hk, NO_BRIGHTEN);
    
        //height texture (IFT)
        fft2D(hk, heightmap4C, INVERSE, SPECTRUM_IMG);
        //imshow4C("Height Map", heightmap4C, NO_BRIGHTEN);
        imshow4CTo1C_PreAlloc("Height Map", heightmap4C, heightmap1C, NO_BRIGHTEN);


        //framerate
        int key = waitKey(16.6);
        if (key == 'q') {
            break;
        }

        //advance time
        current = Clock::now();                                                    //to seconds   //simulation speed
        t = (chrono::duration_cast<chrono::nanoseconds>(current - start).count() * 0.000000001) * 10;
        cout << t << endl;
    }
    
    
    //FT(IFT(hx)) = hx
    //static double hx_shifted[M][M][4];
    //fft2D(heightmap, hx_shifted, FORWARD, SPECTRUM_IMG);
    //static double hx_[M][M][4];
    //fftshift(hx_shifted, hx_);
    //imshow4C("hx_", hx_, NO_BRIGHTEN);


    //display video
    //waitKey(0);

    return 0;
}

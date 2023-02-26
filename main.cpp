#include <iostream>
#include "bitmap_image.hpp"
#include "NeuralNet.h"
#include <string>
#include <fstream>

using namespace std;

ostream & operator << (ostream & out, const vector<double> & v ) {
    for (const auto & x : v)
        out<<x<<' ';
    return out;
}

void set_target(vector <vector <double> > &targets) {
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            if (j == i) {
                continue;
            }

            targets[i][j] = 0;
        }
    }

    return;
}

void set_data(vector <vector <double> > &data, vector <vector <bitmap_image> > data_img) {
    for (int n = 0; n <= 10; n++) {
        for (int m = 0; m <= 9; m++) {
            for (int j = 0; j < 32; j++) {
                for (int k = 0; k < 32; k++) {
                    rgb_t colour;
                    data_img[n][m].get_pixel(k, j, colour);

                    double cur = colour.red;
                    if (cur == 255) {
                        data[m + n * 10][k + j * 32] = 0;
                    } else if (cur < 255) {
                        data[m + n * 10][k + j * 32] = 1;
                    }
                    
                }
            }
        }
    }

    return;
}

vector <vector <vector <double> > > upload_weights() {
    vector <vector <vector <double> > > ans;
    vector <vector <double> > lay0 (1025, vector <double>(512));
    vector <vector <double> > lay1 (513, vector <double>(10));

    ifstream input("weights_machine.txt"); 
    if (input.is_open()) {
        double cur = 0.0;
        for (int i = 0; i < 1025; i++) {
             for (int j = 0; j < 512; j++) {
                input >> cur;
                lay0[i][j] = cur;
            }
        }

        for (int i = 0; i < 513; i++) {
            for (int j = 0; j < 10; j++) {
                input >> cur;
                lay1[i][j] = cur;
            }
        }
    }

    input.close();

    ans.push_back(lay0);
    ans.push_back(lay1);
    return ans;
}

int main() {
    vector <vector <vector <double> > > weights = upload_weights();

    myNeuro N(weights, {1024, 512, 10});

    vector <vector <bitmap_image> > data_img (11, vector <bitmap_image> (10));

    for (int i = 0; i <= 10; i++) {
        for (int j = 0; j <= 9; j++) {
            string path = "data/" + to_string(i) + "/" + to_string(j) + ".bmp";
            
            data_img[i][j] = bitmap_image(path);
        }
    }

    vector <vector <double> > targets (10, vector <double> (10, 1));
    set_target(targets);

    vector <vector <double> > data (110, (vector <double> (1024)));
    set_data(data, data_img);

    for (int i = 0; i < 10000; i++) {
        for (int n = 0; n < 10; n++) {
            for (int m = 0; m < 10; m++) {
                N.train(data[n * 10 + m], targets[m]);
            }
        }
    }
    cout<<"----------------------------------------------------------------------------------------------------"<<endl; 

    for (int i = 0; i <= 9; i++) {
        vector <double> prediction = N.query(data[100 + i]); 
        double max = 0;
        int id = 0;

        for (int j = 0; j < prediction.size(); j++) {
            if (prediction[j] > max) {
                max = prediction[j];
                id = j; 
            }
        }

        cout << "Prediction: " << id << endl;
        cout << N.query(data[100 + i]) << endl;
        cout << endl;
    }

    N.print_weights();
}
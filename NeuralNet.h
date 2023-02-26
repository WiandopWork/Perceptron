#pragma once
#include <stddef.h>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <iostream>

using nn_matrix = std::vector<std::vector<double>>;
#define randWeight(x) (((double)rand() / (double)RAND_MAX) - 0.5) * pow(x, -0.5)
#define learnRate 0.1


class NeuralNetLay {
    size_t in_num;
    size_t out_num;
    nn_matrix w;
    std::vector<double> output;
    std::vector<double> error;

   public:
    NeuralNetLay(size_t inputs, size_t outputs);
    NeuralNetLay(size_t inputs, size_t outputs, nn_matrix weights);
    size_t get_in_num() const;
    size_t get_out_num() const;
    nn_matrix get_weights() const;
    std::vector<double> get_output() const;
    std::vector<double> get_error() const;
    void calc_output(const std::vector<double>& input);
    void calc_out_error(const std::vector<double>& targets);
    void calc_hidden_error(const std::vector<double>& targets,
                           const nn_matrix& outWeights);
    void fix_weight(const std::vector<double>& val);
};

class myNeuro {
    std::vector<NeuralNetLay> list;
    int inputNeurons;
    int outputNeurons;
    int nlCount;
    std::vector<double> input;
    std::vector<double> target;
    void forward_propagate();
    void back_propagate();
    void update_weights();

   public:
    myNeuro(const std::vector<int> & structure = {100,10,2});
    myNeuro(const std::vector<std::vector <std::vector <double> > > & cur_weights, const std::vector<int> & structure = {100,10,2});
    void train(const std::vector<double>& in, const std::vector<double>& targ);
    void print_weights();
    std::vector<double> query(const std::vector<double>& in);
};
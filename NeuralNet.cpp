#include "NeuralNet.h"
#include <fstream>

float sigmoida(double val) { return (1.0 / (1.0 + exp(-val))); }
float sigmoidasDerivate(double val) { return (val * (1.0 - val)); }

NeuralNetLay::NeuralNetLay(size_t inputs, size_t outputs)
    : output(outputs), w(inputs + 1, std::vector<double>(outputs))
{
    in_num = inputs;
    out_num = outputs;
    for (int inp = 0; inp < in_num + 1; inp++)
    {
        for (int outp = 0; outp < out_num; outp++)
        {
            w[inp][outp] = randWeight(out_num);
        }
    }
}

NeuralNetLay::NeuralNetLay(size_t inputs, size_t outputs, nn_matrix weights)
    : output(outputs), w(inputs + 1, std::vector<double>(outputs))
{
    in_num = inputs;
    out_num = outputs;
    for (int inp = 0; inp < in_num + 1; inp++)
    {
        for (int outp = 0; outp < out_num; outp++)
        {
            w[inp][outp] = weights[inp][outp];
        }
    }
}

void NeuralNetLay::fix_weight(const std::vector<double> &val)
{
    for (int ou = 0; ou < out_num; ou++)
    {
        for (int hid = 0; hid < in_num; hid++)
        {
            w[hid][ou] += (learnRate * error[ou] * val[hid]);
        }
        w[in_num][ou] += (learnRate * error[ou]);
    }
}

void NeuralNetLay::calc_output(const std::vector<double> &input)
{
    for (int hid = 0; hid < out_num; hid++)
    {
        double tmpS = 0.0;
        for (int inp = 0; inp < in_num; inp++)
        {
            tmpS += input[inp] * w[inp][hid];
        }
        tmpS += w[in_num][hid];
        output[hid] = sigmoida(tmpS);
    }
}

void NeuralNetLay::calc_out_error(const std::vector<double> &targets)
{
    error.resize(out_num);
    for (int ou = 0; ou < out_num; ou++)
    {
        error[ou] = (targets[ou] - output[ou]) * sigmoidasDerivate(output[ou]);
    }
}

void NeuralNetLay::calc_hidden_error(const std::vector<double> &targets,
                                     const nn_matrix &outWeights)
{
    int inS = outWeights.size();
    int outS = outWeights[0].size();
    error.resize(inS);
    for (int hid = 0; hid < inS; hid++)
    {
        error[hid] = 0.0;
        for (int ou = 0; ou < outS; ou++)
        {
            error[hid] += targets[ou] * outWeights[hid][ou];
        }
        error[hid] *= sigmoidasDerivate(output[hid]);
    }
}

size_t NeuralNetLay::get_in_num() const
{
    return in_num;
}
size_t NeuralNetLay::get_out_num() const
{
    return out_num;
}
nn_matrix NeuralNetLay::get_weights() const
{
    return w;
}
std::vector<double> NeuralNetLay::get_output() const
{
    return output;
}
std::vector<double> NeuralNetLay::get_error() const
{
    return error;
}


void myNeuro::train(const std::vector<double>& in, const std::vector<double>& targ){
    input = in;
    target = targ;
    forward_propagate();
    back_propagate();
    update_weights();
}

std::vector<double> myNeuro::query(const std::vector<double>& in){
    input = in;
    forward_propagate();
    return list.back().get_output();
}

void myNeuro::forward_propagate(){
    list[0].calc_output(input);
    for (int i = 1; i < nlCount; ++i)
        list[i].calc_output(list[i-1].get_output());
}

void myNeuro::back_propagate(){
    list[nlCount-1].calc_out_error(target);
    for (int i = nlCount - 2; i>=0; --i) {
        list[i].calc_hidden_error(list[i+1].get_error(), list[i+1].get_weights());
    }
}

void myNeuro::update_weights(){
    for (int i =nlCount-1; i>0; --i)
        list[i].fix_weight(list[i-1].get_output());
    list[0].fix_weight(input);
}

myNeuro::myNeuro(const std::vector<int> & structure ){
    nlCount = structure.size()-1;
    inputNeurons = structure.front();
    outputNeurons = structure.back();
    for (int i =0; i < nlCount ;++i)
        list.emplace_back(structure[i],structure[i+1]);
}

void myNeuro::print_weights() {
    std::ofstream fout;
    fout.open("weights_people1.txt");

    for (int n = 0; n < list.size(); n++) {
        std::vector <std::vector <double> > cur = list[n].get_weights();

        fout << "LAY: " << n;
        fout << std::endl << std::endl;
        for (int i = 0; i < cur.size(); i++) {
            fout << i << ": ";
            for (int j = 0; j < cur[i].size(); j++) {
                fout << cur[i][j] << " ";
            }

            fout << std::endl;
        }
        fout << std::endl << std::endl;
    }
    
    fout.close();

    fout.open("weights_machine1.txt");

    for (int n = 0; n < list.size(); n++) {
        std::vector <std::vector <double> > cur = list[n].get_weights();
        fout << std::endl;
        for (int i = 0; i < cur.size(); i++) {
            for (int j = 0; j < cur[i].size(); j++) {
                fout << cur[i][j] << " ";
            }

            fout << std::endl;
        }
        fout << std::endl;
    }
    
    fout.close();
}

myNeuro::myNeuro(const std::vector<std::vector <std::vector <double> > > & cur_weights, const std::vector<int> & structure) {
    nlCount = structure.size()-1;
    inputNeurons = structure.front();
    outputNeurons = structure.back();

    for (int i =0; i < nlCount ;++i)
        list.emplace_back(structure[i], structure[i+1], cur_weights[i]);
}
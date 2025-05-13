#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <stdexcept>
#include <chrono>

using namespace std;

namespace activation{
    inline double relu(double x) { return max(0.0, x); }
    inline double reluDerivative(double x) { return (x > 0) ? 1.0 : 0.0; }
    inline double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
    inline double sigmoidDerivative(double x){
        double s = sigmoid(x);
        return s * (1.0 - s);
    }
}

class Matrix{
private:
    vector<vector<double>> data;
    size_t rows, cols;
public:
    Matrix(size_t r, size_t c):rows(r), cols(c){
        data.resize(r, vector<double>(c, 0.0));
    }

    double &operator()(size_t i, size_t j){ return data[i][j]; }
    const double &operator()(size_t i, size_t j) const { return data[i][j]; }

    size_t getRows() const { return rows; }
    size_t getCols() const { return cols; }
};

class NeuralNetwork{
private:
    vector<int> layerSizes;
    Matrix weights1, weights2, weights3;
    vector<double> bias1, bias2, bias3;
    mt19937 gen;

    void initializeWeights(){
        normal_distribution<> dist(0.0, 1.0);
        double scale1 = sqrt(2.0 / layerSizes[0]);
        double scale2 = sqrt(2.0 / layerSizes[1]);
        double scale3 = sqrt(2.0 / layerSizes[2]);

        for(size_t i = 0; i < weights1.getRows(); i++){
            for(size_t j = 0; j < weights1.getCols(); j++){
                weights1(i, j) = dist(gen) * scale1;
            }
        }

        for(size_t i = 0; i < weights2.getRows(); i++){
            for(size_t j = 0; j < weights2.getCols(); j++){
                weights2(i, j) = dist(gen) * scale2;
            }
        }

        for(size_t i = 0; i < weights3.getRows(); i++){
            for(size_t j = 0; j < weights3.getCols(); j++){
                weights3(i, j) = dist(gen) * scale3;
            }
        }

        fill(bias1.begin(), bias1.end(), 0.0);
        fill(bias2.begin(), bias2.end(), 0.0);
        fill(bias3.begin(), bias3.end(), 0.0);
    }
public:
    NeuralNetwork(int inputSize, int hidden1Size, int hidden2Size, int outputSize)
    : layerSizes{inputSize, hidden1Size, hidden2Size, outputSize},
        weights1(inputSize, hidden1Size),
        weights2(hidden1Size, hidden2Size),
        weights3(hidden2Size, outputSize),
        bias1(hidden1Size), bias2(hidden2Size), bias3(outputSize),
        gen(random_device{}())
    {
        cout << "NeuralNetwork constructor called" << endl;
        if(inputSize <= 0 || hidden1Size <= 0 || hidden2Size <= 0 || outputSize <= 0){
            throw invalid_argument("Layer sizes must be positive !");
        }
        initializeWeights();
    }

    vector<double> forward(const vector<double> &input){
        if(input.size() != static_cast<size_t>(layerSizes[0])){
            throw invalid_argument("Input size does not match the input layer size !");
        }

        vector<double> hidden1(layerSizes[1]);
        vector<double> hidden2(layerSizes[2]);
        vector<double> output(layerSizes[3]);

        for(int i = 0; i < layerSizes[1]; i++){
            hidden1[i] = bias1[i];
            for(int j = 0; j < layerSizes[0]; j++){
                hidden1[i] += input[j] * weights1(j, i);
            }
            hidden1[i] = activation::relu(hidden1[i]);
        }

        for(int i = 0; i < layerSizes[2]; i++){
            hidden2[i] = bias2[i];
            for(int j = 0; j < layerSizes[1]; j++){
                hidden2[i] += hidden1[j] * weights2(j, i);
            }
            hidden2[i] = activation::relu(hidden2[i]);
        }

        for(int i = 0; i < layerSizes[3]; i++){
            output[i] = bias3[i];
            for(int j = 0; j < layerSizes[2]; j++){
                output[i] += hidden2[j] * weights3(j, i);
            }
            output[i] = activation::sigmoid(output[i]);
        }

        return output;
    }

    void train(const vector<vector<double>> &inputs, const vector<vector<double>> &targets, int epochs, double learningRate){
        if(inputs.size() != targets.size()){
            throw runtime_error("Input and output sizes do not match !");
        }

        for(int epoch = 0; epoch < epochs; epoch++){
            double totalError = 0.0;

            for(size_t k = 0; k < inputs.size(); k++){

                vector<double> hidden1(layerSizes[1]);
                vector<double> hidden1Pre(layerSizes[1]);
                for(int i = 0; i < layerSizes[1]; i++){
                    double sum = bias1[i];
                    for(int j = 0; j < layerSizes[0]; j++){
                        sum += inputs[k][j] * weights1(j, i);
                    }
                    hidden1Pre[i] = sum;
                    hidden1[i] = activation::relu(hidden1Pre[i]);
                }

                vector<double> hidden2(layerSizes[2]);
                vector<double> hidden2Pre(layerSizes[2]);
                for(int i = 0; i < layerSizes[2]; i++){
                    double sum = bias2[i];
                    for(int j = 0; j < layerSizes[1]; j++){
                        sum += hidden1[j] * weights2(j, i);
                    }
                    hidden2Pre[i] = sum;
                    hidden2[i] = activation::relu(hidden2Pre[i]);
                }

                vector<double> output(layerSizes[3]);
                vector<double> outputPre(layerSizes[3]);
                for(int i = 0; i < layerSizes[3]; i++){
                    double sum = bias3[i];
                    for(int j = 0; j < layerSizes[2]; j++){
                        sum += hidden2[j] * weights3(j, i);
                    }
                    outputPre[i] = sum;
                    output[i] = activation::sigmoid(outputPre[i]);
                }

                for(int i = 0; i < layerSizes[3]; i++){
                    double error = targets[k][i] - output[i];
                    totalError += error * error;
                }

                vector<double> outputGradient(layerSizes[3]);
                for(int i = 0; i < layerSizes[3]; i++){
                    outputGradient[i] = (output[i] - targets[k][i]) * activation::sigmoidDerivative(outputPre[i]);
                }

                vector<double> hidden2Gradient(layerSizes[2]);
                for(int i = 0; i < layerSizes[2]; i++){
                    hidden2Gradient[i] = 0.0;
                    for(int j = 0; j < layerSizes[3]; j++){
                        hidden2Gradient[i] += outputGradient[j] * weights3(i, j);
                    }
                    hidden2Gradient[i] *= activation::reluDerivative(hidden2Pre[i]);
                }

                vector<double> hidden1Gradient(layerSizes[1]);
                for(int i = 0; i < layerSizes[1]; i++){
                    hidden1Gradient[i] = 0.0;
                    for(int j = 0; j < layerSizes[2]; j++){
                        hidden1Gradient[i] += hidden2Gradient[j] * weights2(i, j);
                    }
                    hidden1Gradient[i] *= activation::reluDerivative(hidden1Pre[i]);
                }

                for(int i = 0; i < layerSizes[3]; i++){
                    for(int j = 0; j < layerSizes[2]; j++){
                        weights3(j, i) -= learningRate * outputGradient[i] * hidden2[j];
                    }
                    bias3[i] -= learningRate * outputGradient[i];
                }
                for(int i = 0; i < layerSizes[2]; i++){
                    for(int j = 0; j < layerSizes[1]; j++){
                        weights2(j, i) -= learningRate * hidden2Gradient[i] * hidden1[j];
                    }
                    bias2[i] -= learningRate * hidden2Gradient[i];
                }
                for(int i = 0; i < layerSizes[1]; i++){
                    for(int j = 0; j < layerSizes[0]; j++){
                        weights1(j, i) -= learningRate * hidden1Gradient[i] * inputs[k][j];
                    }
                    bias1[i] -= learningRate * hidden1Gradient[i];
                }

            }
            if(epoch % 100 == 0){
                cout << "Epoch " << epoch << " MSE : " << totalError / inputs.size() << "\n";
            }
        }
    }
};

int main(){

    try{

        NeuralNetwork nn(2, 8, 4, 1);
        mt19937 gen(random_device{}());
        uniform_real_distribution<> dist(-2.0, 2.0);
        const int numSamples = 1000;
        vector<vector<double>> inputs(numSamples);
        vector<vector<double>> targets(numSamples);

        for(int i = 0; i < numSamples;i++){
            double x = dist(gen);
            double y = dist(gen);
            inputs[i] = {x, y};
            double distance = sqrt(x * x + y * y);
            targets[i] = {distance < 1.0 ? 1.0 : 0.0};
        }

        auto start = chrono::high_resolution_clock::now();
        nn.train(inputs, targets, 100, 0.01);
        auto end = chrono::high_resolution_clock::now();
        cout << "Training Time : "
             << chrono::duration_cast<chrono::milliseconds>(end - start).count()
             << " ms\n";
        vector<vector<double>> testPoints = {
            {0.0, 0.0},
            {1.0, 1.0},
            {0.5, 0.5},
            {2.0, 0.0}};

        cout << "\n Test Results (1 = inside , 0 = outside) : \n";
        for(const auto &point : testPoints)
        {
            auto output = nn.forward(point);
            double actual = sqrt(point[0] * point[0] + point[1] * point[1]) < 1.0 ? 1.0 : 0.0;
            cout << " point ( " << point[0] << "," << point[1] << " ) :"
                 << output[0] << " ( actual : " << actual
                 << " , error : " << abs(output[0] - actual) << " ) \n";
        }
    }catch(const exception &e){
        cerr << " ERROR : " << e.what() << endl;
        return 1;
    }

    return 0;
}
/**************************************************************************
 *
 *   SOME STARTER CODE FOR WORKING WITH NMIST, © Ellan Esenaliev 2018
 *
 **************************************************************************/
#include <iostream>
#include <math.h>

#include "randlib.h"
#include "mnist/mnist.h"

using namespace std;

#define numOfInputNodes 785
#define numOfOutputNodes 10

void randomizeWeightMatrixForOutPut(float weights[numOfOutputNodes][numOfInputNodes]) {
    for(int i = 0; i < numOfOutputNodes; i++) {
        for(int j = 0; j < numOfInputNodes; j++) {
            weights[i][j] = rand_weight();
        }
    }
}

void initTarget(float target[], int numberOnPicture) {

    for(int i = 0; i < numOfOutputNodes; i++) {
        if (i == numberOnPicture) {
            target[i] = 1;
        } else {
            target[i] = 0;
        }
    }
    
}

void get_output(float output[], int input[], float weights[numOfOutputNodes][numOfInputNodes]) {
    
    for(int i = 0; i < numOfOutputNodes; i++) {
        float resultOfMultiplication = 0;
        for(int j = 0; j < numOfInputNodes; j++) {
            resultOfMultiplication += input[j] * weights[i][j];;
        }
        output[i] = resultOfMultiplication;
    }
}

void squash_output(float output[]) {
    
    for(int i = 0; i < numOfOutputNodes; i++) {
        output[i] = 1.0 / (1.0 + pow(M_E, -1 * output[i]));
        // printf("squashed output[%d] = %f\n", i, output[i]);
    }
}

void get_error_for_output(float errors[], float target[], float output[]) {
    for(int i = 0; i < numOfOutputNodes; i++) {
        errors[i] = target[i] - output[i];
    }
}

float getAverageError(float error[]) {
    float errorsSum = 0;
    for (int i = 0; i < numOfOutputNodes; i++) {
        errorsSum += fabs(error[i]);
    }
    
    return (errorsSum / numOfOutputNodes);
}

void update_weights_output(float learningRate, int input[], float errors[], float weights[numOfOutputNodes][numOfInputNodes]) {
    float deltaWeights[numOfOutputNodes][numOfInputNodes];
    for(int i = 0; i < numOfOutputNodes; i++) {
        for(int j = 0; j < numOfInputNodes; j++) {
            deltaWeights[i][j] = learningRate  * input[j] * errors[i];
            weights[i][j] += deltaWeights[i][j];
        }
    }
}


int main(int argc, char const *argv[]) {
    // --- an example for working with random numbers
    seed_randoms();
    
    float sampNoise = 0;
    
    // --- a simple example of how to set params from the command line
    if(argc == 2){ // if an argument is provided, it is SampleNoise
        sampNoise = atof(argv[1]);
        if (sampNoise < 0 || sampNoise > .5){
            printf("Error: sample noise should be between 0.0 and 0.5\n");
            return 0;
        }
    }
    
    mnist_data *zData;      // each image is 28x28 pixels
    unsigned int sizeData;  // depends on loadType
    int loadType = 1; // loadType may be: 0, 1, or 2
    if (mnistLoad(&zData, &sizeData, loadType)){
        printf("something went wrong loading data set\n");
        return -1;
    }
    
    mnist_data *zTestingData;      // each image is 28x28 pixels
    unsigned int sizeTestingData;  // depends on loadType
    int loadTypeTesing = 2; // loadType may be: 0, 1, or 2
    if (mnistLoad(&zTestingData, &sizeTestingData, loadTypeTesing)){
        printf("something went wrong loading data set\n");
        return -1;
    }
    
    int inputNodes[numOfInputNodes];
    float outputNodes[numOfOutputNodes];
    
    float errorsOutput[numOfOutputNodes];
    
    float target[numOfOutputNodes];
    
    float weightsOutput[numOfOutputNodes][numOfInputNodes];
    
    for (float learningRate = 0.1; learningRate >= 0.00001; learningRate /= 10) {
        
        randomizeWeightMatrixForOutPut(weightsOutput);
        
        for(int epoch = 0; epoch < 20; epoch++) {
            
            float resultError = 0;
            
            for(int picIndex = 0; picIndex < sizeTestingData; picIndex++) {
                get_input(inputNodes, zTestingData, picIndex, sampNoise);
                
                initTarget(target, zTestingData[picIndex].label);
                
                get_output(outputNodes, inputNodes, weightsOutput);
                squash_output(outputNodes);
                
                get_error_for_output(errorsOutput, target, outputNodes);
                resultError += getAverageError(errorsOutput);
                
            }
            
            cout << (resultError / sizeTestingData) << ", ";
            
            for(int picIndex = 0; picIndex < sizeData; picIndex++) {
                
                get_input(inputNodes, zData, picIndex, sampNoise);
                
                initTarget(target, zData[picIndex].label);
                
                get_output(outputNodes, inputNodes, weightsOutput);
                squash_output(outputNodes);
                
                get_error_for_output(errorsOutput, target, outputNodes);
                update_weights_output(learningRate, inputNodes, errorsOutput, weightsOutput);
            }
            
        }
        cout << endl;
        
    }
    
    
    
    
    return 0;
}

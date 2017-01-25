/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package diuf.diva.dia.ms.ml.layer;

import java.io.DataInputStream;
import java.io.IOException;

/**
 *
 * @author Mathias Seuret
 */
public class SigmoidLayer extends NeuralLayer {
    /**
     * Creates a neural layer.
     * @param inputArr a float array
     * @param outputSize the number of neurons
     */
    public SigmoidLayer(float[] inputArr, int outputSize) {
        this(inputArr, inputArr.length, outputSize);
    }

    /**
     * Creates a neural layer.
     * @param inputArr   a float array
     * @param inputSize  size of the input array
     * @param outputSize the number of neurons
     */
    public SigmoidLayer(float[] inputArr, int inputSize, int outputSize) {
        this(inputArr, inputSize, outputSize, null, null);
    }

    /**
     * Constructs a neural layer with all the parameters needed.
     * @param inputArray input array
     * @param inputSize size of the input array
     * @param outputSize size of the output array
     * @param weight of the layer - if null an array is created
     * @param bias of the layer - if null an array is created
     */
    public SigmoidLayer(float[] inputArray, int inputSize, int outputSize, float[][] weight, float[] bias) {
        super(inputArray, inputSize, outputSize, weight, bias);
    }

    /**
     * Creates a layer from a stream.
     * @param is input stream
     * @throws IOException if the stream cannot be read
     */
    SigmoidLayer(DataInputStream is) throws IOException {
        super(is);
    }

    /**
     * Creates a layer from a file.
     * @param fname name of a file storing a layer
     * @throws IOException if the file cannot be read
     */
    SigmoidLayer(String fname) throws IOException {
        super(fname);
    }
    
    /**
     * Computes the output of the layer.
     */
    @Override
    public void compute() {
        for (int o = 0; o < outputSize; o++) {
            wSum[o] = bias[o];
            for (int i = 0; i < inputSize; i++) {
                wSum[o] += weight[i][o] * input[i];
            }
            output[o] = 1.0f / (1.0f + (float)Math.exp(-wSum[o]));
        }
    }
    
    /**
     * Applies the backpropagation if needed.
     * @return the mean average error of the top layer
     */
    @Override
    public float backPropagate() {
        float errSum = 0.0f;
        // It does not look nice, but it decreases MUCH the number
        // of conditions executed - I don't think the Java compiler
        // would do this
        if (prevErr==null) {
            for (int o = 0; o < outputSize; o++) {
                errSum += Math.abs(err[o]);
                float fact = output[o] * (1.0f - output[o]) * err[o];
                for (int i = 0; i < inputSize; i++) {
                    gradient[i][o] += fact * input[i];
                }
                biasGradient[o] += fact;
            }
        } else {
            for (int o = 0; o < outputSize; o++) {
                errSum += Math.abs(err[o]);
                float fact = output[o] * (1.0f - output[o]) * err[o];
                for (int i = 0; i < inputSize; i++) {
                    gradient[i][o] += fact * input[i];
                    prevErr[i] += fact * weight[i][o];
                }
                biasGradient[o] += fact;
            }
        }
        
        return errSum / outputSize;
    }
}

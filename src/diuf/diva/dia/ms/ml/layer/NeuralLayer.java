/*****************************************************
  N-light-N
  
  A Highly-Adaptable Java Library for Document Analysis with
  Convolutional Auto-Encoders and Related Architectures.
  
  -------------------
  Author:
  2016 by Mathias Seuret <mathias.seuret@unifr.ch>
      and Michele Alberti <michele.alberti@unifr.ch>
  -------------------

  This software is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License as published by the Free Software Foundation version 3.

  This software is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this software; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 ******************************************************************************/

package diuf.diva.dia.ms.ml.layer;

import java.io.DataInputStream;
import java.io.IOException;

/**
 * This is a simple neural layer class using backpropagation and a soft-sign
 * activation function.
 * @author Mathias Seuret, Michele Alberti
 */
public class NeuralLayer extends AbstractLayer {

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Constructor
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * Creates a neural layer.
     * @param inputArr a float array
     * @param outputSize the number of neurons
     */
    public NeuralLayer(float[] inputArr, int outputSize) {
        this(inputArr, inputArr.length, outputSize);
    }

    /**
     * Creates a neural layer.
     * @param inputArr   a float array
     * @param inputSize  size of the input array
     * @param outputSize the number of neurons
     */
    public NeuralLayer(float[] inputArr, int inputSize, int outputSize) {
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
    public NeuralLayer(float[] inputArray, int inputSize, int outputSize, float[][] weight, float[] bias) {
        super(inputArray, inputSize, outputSize, weight, bias);
    }

    /**
     * Creates a layer from a stream.
     * @param is input stream
     * @throws IOException if the stream cannot be read
     */
    NeuralLayer(DataInputStream is) throws IOException {
        super(is);
    }

    /**
     * Creates a layer from a file.
     * @param fname name of a file storing a layer
     * @throws IOException if the file cannot be read
     */
    NeuralLayer(String fname) throws IOException {
        super(fname);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Computing
    ///////////////////////////////////////////////////////////////////////////////////////////////
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
            output[o] = wSum[o] / (1 + Math.abs(wSum[o]));
        }
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Learning
    ///////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * Applies the gradient descent.
     */
    @Override
    public void learn() {
        for (int o = 0; o < outputSize; o++) {
            for (int i = 0; i < inputSize; i++) {
                weight[i][o] = (1.0f-decay)*weight[i][o] - learningSpeed * gradient[i][o];
                gradient[i][o] = 0.0f;
            }
            bias[o] = (1.0f-decay)*bias[o] - learningSpeed * biasGradient[o];
            biasGradient[o] = 0.0f;
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
                float bot = 1 + Math.abs(wSum[o]);
                float fact = 1 / (bot * bot) * err[o];
                for (int i = 0; i < inputSize; i++) {
                    gradient[i][o] += fact * input[i];
                }
                biasGradient[o] += fact;
            }
        } else {
            for (int o = 0; o < outputSize; o++) {
                errSum += Math.abs(err[o]);
                float bot = 1 + Math.abs(wSum[o]);
                float fact = 1 / (bot * bot) * err[o];
                for (int i = 0; i < inputSize; i++) {
                    gradient[i][o] += fact * input[i];
                    prevErr[i] += fact * weight[i][o];
                }
                biasGradient[o] += fact;
            }
        }
        
        return errSum / outputSize;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Utility
    ///////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * Typical clone methods
     * @return a deep copy of the object
     */
    @Override
    public Layer clone() {
        return new NeuralLayer(
                input,
                inputSize,
                outputSize,
                weight,
                bias
        );

    }
}

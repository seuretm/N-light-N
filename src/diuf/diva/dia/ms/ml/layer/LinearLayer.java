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
 * This is a simple linear associator layer class using backpropagation and a no
 * activation function.
 *
 * @author Michele Alberti
 */
public class LinearLayer extends AbstractLayer {

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Constructor
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * Creates a neural layer.
     *
     * @param inputArr   a float array
     * @param outputSize the number of neurons
     */
    public LinearLayer(float[] inputArr, int outputSize) {
        this(inputArr, inputArr.length, outputSize);
    }

    /**
     * Creates a neural layer.
     *
     * @param inputArr   a float array
     * @param inputSize  size of the input array
     * @param outputSize the number of neurons
     */
    public LinearLayer(float[] inputArr, int inputSize, int outputSize) {
        this(inputArr, inputSize, outputSize, null, null);
    }

    /**
     * Constructs a neural layer with all the parameters needed.
     *
     * @param inputArray input array
     * @param inputSize  size of the input array
     * @param outputSize size of the output array
     * @param weight     of the layer - if null an array is created
     * @param bias       of the layer - if null an array is created
     */
    public LinearLayer(float[] inputArray, int inputSize, int outputSize, float[][] weight, float[] bias) {
        super(inputArray, inputSize, outputSize, weight, bias);
    }

    /**
     * Creates a layer from a stream.
     *
     * @param is input stream
     * @throws IOException if the stream cannot be read
     */
    LinearLayer(DataInputStream is) throws IOException {
        super(is);
    }

    /**
     * Creates a layer from a file.
     *
     * @param fname name of a file storing a layer
     * @throws IOException if the file cannot be read
     */
    LinearLayer(String fname) throws IOException {
        super(fname);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Computing
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * Computes the output of the layer.
     */
    public void compute() {
        for (int o = 0; o < outputSize; o++) {
            wSum[o] = bias[o];
            for (int i = 0; i < inputSize; i++) {
                wSum[o] += weight[i][o] * input[i];
            }
            output[o] = wSum[o];
        }
        /*
        // Here rescale output ?
        float sum = 0;
        for (int o = 0; o < outputSize; o++) {
            sum += output[o];
        }

        for (int o = 0; o < outputSize; o++) {
            output[o] /= sum;
        }
        */
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Learning
    ///////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * Applies the gradient descent.
     */
    public void learn() {
        for (int o = 0; o < outputSize; o++) {
            for (int i = 0; i < inputSize; i++) {
                weight[i][o] = (1.0f - decay) * weight[i][o] - learningSpeed * gradient[i][o];
                gradient[i][o] = 0.0f;
                // Emergency normalization
                if (weight[i][o] > 10) {
                    System.out.println("!WARNING! Weights are too big! Normalizing!");
                    float norm = 0;
                    for (int x = 0; x < weight.length; x++) {
                        for (int y = 0; y < weight[x].length; y++) {
                            norm += Math.sqrt(Math.pow(weight[i][o], 2));
                        }
                    }
                    for (int x = 0; x < weight.length; x++) {
                        for (int y = 0; y < weight[x].length; y++) {
                            weight[i][o] /= norm;
                        }
                    }
                }
            }
            bias[o] = (1.0f - decay) * bias[o] - learningSpeed * biasGradient[o];
            biasGradient[o] = 0.0f;
        }
    }

    /**
     * Applies the backpropagation if needed.
     */
    public float backPropagate() {
        float errSum = 0.0f;
        if (prevErr == null) {
            for (int o = 0; o < outputSize; o++) {
                errSum += Math.abs(err[o]);
                for (int i = 0; i < inputSize; i++) {
                    gradient[i][o] += err[o] * input[i];
                }
                biasGradient[o] += err[o];
            }
        } else {
            for (int o = 0; o < outputSize; o++) {
                errSum += Math.abs(err[o]);
                for (int i = 0; i < inputSize; i++) {
                    gradient[i][o] += err[o] * input[i];
                    prevErr[i] += err[o] * weight[i][o];
                }
                biasGradient[o] += err[o];
            }
        }

        return errSum / outputSize;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Utility
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * Typical clone methods
     *
     * @return a deep copy of the object
     */
    @Override
    public Layer clone() {
        return new LinearLayer(
                input,
                inputSize,
                outputSize,
                weight,
                bias
        );

    }

}

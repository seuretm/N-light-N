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
 * This is a simple linear associator trained with the Oja's algorithm
 * Oja's rule WIKI "https://en.wikipedia.org/wiki/Oja%27s_rule"
 *
 * @author Michele Alberti
 */
public class OjasLayer extends AbstractLayer {

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Constructor
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * Creates a neural layer.
     *
     * @param inputArr   a float array
     * @param outputSize the number of neurons
     */
    public OjasLayer(float[] inputArr, int outputSize) {
        this(inputArr, inputArr.length, outputSize);
    }

    /**
     * Creates a neural layer.
     *
     * @param inputArr   a float array
     * @param inputSize  size of the input array
     * @param outputSize the number of neurons
     */
    public OjasLayer(float[] inputArr, int inputSize, int outputSize) {
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
    public OjasLayer(float[] inputArray, int inputSize, int outputSize, float[][] weight, float[] bias) {
        super(inputArray, inputSize, outputSize, weight, bias);
        learningSpeed = 1e-10f;
    }

    /**
     * Creates a layer from a stream.
     *
     * @param is input stream
     * @throws IOException if the stream cannot be read
     */
    OjasLayer(DataInputStream is) throws IOException {
        super(is);
        learningSpeed = 1e-10f;
    }

    /**
     * Creates a layer from a file.
     *
     * @param fname name of a file storing a layer
     * @throws IOException if the file cannot be read
     */
    OjasLayer(String fname) throws IOException {
        super(fname);
        learningSpeed = 1e-10f;
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
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Learning
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * Applies the Oja's algorithm
     */
    public void learn() {

        for (int o = 0; o < outputSize; o++) {
            // Computing phi
            double phi = 0;
            for (int i = 0; i < inputSize; i++) {
                phi += weight[i][o] * input[i];
            }

            for (int i = 0; i < inputSize; i++) {
                // Updating weight
                weight[i][o] += learningSpeed * phi * (input[i] - (phi * weight[i][o]));
                if (Float.isNaN(weight[i][o])) {
                    throw new RuntimeException("NaN detected. Something went wrong.");
                }

                // Subtracting mean
                input[i] -= phi * weight[i][o];
            }

            // Updating learning speed
            //learningSpeed *= 0.9999;
        }
    }

    /**
     * Applies the backpropagation if needed.
     */
    public float backPropagate() {

        float errSum = 0.0f;
        for (int o = 0; o < outputSize; o++) {
            errSum += Math.abs(err[o]);
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
        return new OjasLayer(
                input,
                inputSize,
                outputSize,
                weight,
                bias
        );

    }

}

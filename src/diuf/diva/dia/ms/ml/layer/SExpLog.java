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
 * This layer is experimental and has NOT been tested. It corresponds to a
 * Simplified version of the ExpLog. Its activation function is
 * y = log(exp(sum e^x_i))
 * @author Mathias Seuret, Michele Alberti
 */
public class SExpLog extends AbstractLayer {
    //TODO: implement this in a better way
    public static float INERTIA = 0.99f;
    
    float[][] exp;

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Constructor
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * Creates a neural layer.
     * @param inputArr a float array
     * @param outputSize the number of neurons
     */
    public SExpLog(float[] inputArr, int outputSize) {
        this(inputArr, inputArr.length, outputSize);
    }

    /**
     * Creates a neural layer.
     * @param inputArr   a float array
     * @param inputSize  size of the input array
     * @param outputSize the number of neurons
     */
    public SExpLog(float[] inputArr, int inputSize, int outputSize) {
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
    public SExpLog(float[] inputArray, int inputSize, int outputSize, float[][] weight, float[] bias) {
        super(inputArray, inputSize, outputSize, weight, bias);
        if (outputSize!=inputSize) {
            throw new Error("SExpLog cannot modify the dimensionality - had "+inputSize+", requests "+outputSize+" outputs.");
        }
        // By def, the first unit has 1-initialized weights
        for (int in=0; in<inputSize; in++) {
            weight[in][0] = 1.0f;
        }
        exp = new float[inputSize][outputSize];
    }

    /**
     * Creates a layer from a stream.
     * @param is input stream
     * @throws IOException if the stream cannot be read
     */
    SExpLog(DataInputStream is) throws IOException {
        super(is);
    }

    /**
     * Creates a layer from a file.
     * @param fname name of a file storing a layer
     * @throws IOException if the file cannot be read
     */
    SExpLog(String fname) throws IOException {
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
            wSum[o] = 0;
            for (int i = 0; i < inputSize; i++) {
                exp[i][o] = (float)Math.exp(input[i]);
                wSum[o] += exp[i][o];
            }
            output[o] = (float)Math.log(wSum[o]);
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
        // Nothing to do
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
        if (prevErr!=null) {
            for (int o = 0; o < outputSize; o++) {
                errSum += Math.abs(err[o]);
                for (int i = 0; i < inputSize; i++) {
                    prevErr[i] += 1 / wSum[o] * exp[i][o] * err[o];;
                }
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
        SExpLog el = new SExpLog(
                input,
                inputSize,
                outputSize,
                weight,
                bias
        );
        for (int i=0; i<inputSize; i++) {
            for (int o=0; o<outputSize; o++) {
                el.weight[i][o] = weight[i][o];
            }
        }
        return el;
    }
}

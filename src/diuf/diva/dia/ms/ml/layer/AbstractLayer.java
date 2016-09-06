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

import java.io.*;

/**
 * This is a simple class which serves as "starting point" when creating a new kind of layer.
 * Without this class all layers would repeat a lot of code making them bigger and more
 * complicated than necessary. This way, all different kinds of layers can focus on logic
 * rather than utility methods as setInput() or so. However, this is not mandatory, one can
 * also implements hiw own layer from the scratch. Each layer mus anyway have an array of floats
 * as input and another array of floats as output.
 *
 * @author Michele Alberti
 */
public abstract class AbstractLayer implements Layer, Serializable {
    /**
     * Number of inputs.
     */
    protected int inputSize;
    /**
     * Inputs of the layer.
     */
    protected float[] input;
    /**
     * Number of outputs.
     */
    protected int outputSize;
    /**
     * Outputs of the layer.
     */
    protected float[] output;
    /**
     * Stores the bias of the output.
     */
    protected float[] bias;
    /**
     * Gradient for the bias
     */
    protected float[] biasGradient;
    /**
     * Gradient, which is used for storing some inertia.
     */
    protected float[][] gradient;
    /**
     * Weights of the layer.
     */
    protected float[][] weight;
    /**
     * Learning speed of the network. A value of 0.0001 seems
     * to work well in most cases, if the inputs have values
     * between -1 and +1.
     */
    protected float learningSpeed = 1e-3f;
    /**
     * Weighted sum (= outputs before calling the activation function).
     */
    protected float[] wSum;
    /**
     * Error of the layer, for each neuron.
     */
    protected float[] err;
    /**
     * Stores a reference to the error of the previous layer.
     */
    protected float[] prevErr;
    /**
     * Weight decay factor.
     */
    protected float decay = 0.0f;

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Constructor
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * Constructs a layer with all the parameters needed.
     *
     * @param inputArray input array
     * @param inputSize  size of the input array
     * @param outputSize size of the output array
     * @param weight     of the layer - if null an array is created
     * @param bias       of the layer - if null an array is created
     */
    public AbstractLayer(float[] inputArray,
                         int inputSize,
                         int outputSize,
                         float[][] weight,
                         float[] bias) {
        // Store input
        assert (inputSize > 0);

        this.inputSize = inputSize;
        this.input = inputArray;

        // Store output
        assert (outputSize > 0);

        this.outputSize = outputSize;
        this.output = new float[outputSize];

        // Init the error array
        err = new float[outputSize];

        // Store or init weights
        this.gradient = new float[inputSize][outputSize];
        if (weight != null) {
            this.weight = copy(weight);
            if (weight.length != inputSize || weight[0].length != outputSize) {
                throw new IllegalArgumentException(
                        "bad input weight size: " + weight.length + "x" + weight[0].length + ", expected " + inputSize + "x" + outputSize
                );
            }
        } else {
            this.weight = new float[inputSize][outputSize];
            for (int i = 0; i < inputSize; i++) {
                for (int o = 0; o < outputSize; o++) {
                    this.weight[i][o] = (float) ((1 - 2 * Math.random()) / Math.sqrt(inputSize));
                }
            }
        }

        // Store or init bias
        if (bias != null) {
            // Verify sizes
            if (bias.length != outputSize) {
                throw new IllegalArgumentException("Illegal bias size: " + bias.length + ", expected " + outputSize);
            }
            // Store new values
            this.bias = bias.clone();
        } else {
            this.bias = new float[outputSize];
        }

        this.biasGradient = new float[outputSize];

        // Init weighs sums array
        wSum = new float[outputSize];
    }

    /**
     * Creates a layer from a stream.
     *
     * @param is input stream
     * @throws IOException if the stream cannot be read
     */
    AbstractLayer(DataInputStream is) throws IOException {
        load(is);
    }

    /**
     * Creates a layer from a file.
     *
     * @param fname name of a file storing a layer
     * @throws IOException if the file cannot be read
     */
    AbstractLayer(String fname) throws IOException {
        load(fname);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Input related
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * @return the dimension of the input
     */
    public int getInputSize() {
        return inputSize;
    }

    /**
     * @return the input array, not a copy
     */
    @Override
    public float[] getInputArray() {
        return input;
    }

    /**
     * Changes the input array.
     *
     * @param in new array
     */
    public void setInputArray(float[] in) {
        input = in;
    }

    /**
     * Deletes an input and shifts all weights accordingly.
     *
     * @param num input number
     */
    public void deleteInput(int num) {
        float[][] nWeight = new float[inputSize - 1][outputSize];

        for (int i = 0; i < inputSize - 1; i++) {
            int ii = (i >= num) ? i + 1 : i;
            System.arraycopy(weight[ii], 0, nWeight[i], 0, outputSize - 1);
        }
        weight = nWeight;

        inputSize--;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Computing
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * Computes the output of the layer.
     */
    public abstract void compute();

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Learning
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * Applies the gradient descent.
     */
    public abstract void learn();

    /**
     * Applies the backpropagation if needed.
     */
    public abstract float backPropagate();

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Output related
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * @return the output array, not a copy
     */
    @Override
    public float[] getOutputArray() {
        return output;
    }

    /**
     * Changes the output array
     *
     * @param out new output array
     */
    public void setOutputArray(float[] out) {
        output = out;
    }

    /**
     * @return the number of outputs (neurons)
     */
    public int getOutputSize() {
        return outputSize;
    }

    /**
     * Deletes an output, and shifts all weights accordingly.
     *
     * @param num output number
     */
    public void deleteOutput(int num) {
        float[][] nWeight = new float[inputSize][outputSize - 1];
        float[] nBias = new float[outputSize - 1];

        for (int o = 0; o < outputSize - 1; o++) {
            int oo = (o >= num) ? o + 1 : o;
            for (int i = 0; i < inputSize; i++) {
                nWeight[i][o] = weight[i][oo];
            }
            nBias[o] = bias[oo];
        }

        bias = nBias;
        weight = nWeight;

        outputSize--;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Error related
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * @return the error array
     */
    public float[] getError() {
        return err;
    }

    /**
     * Tells the neural layer which array should be
     * used for storing errors.
     *
     * @param err an array
     */
    public void setError(float[] err) {
        this.err = err;
    }

    /**
     * Increases the error of an output.
     *
     * @param o output number
     * @param e error to add
     */
    public void addError(int o, float e) {
        err[o] += e;
    }

    /**
     * Clears the error
     */
    public void clearError() {
        for (int i = 0; i < err.length; i++) {
            err[i] = 0;
        }
    }

    /**
     * @return the previous error, NOT a copy
     */
    @Override
    public float[] getPreviousError() {
        return prevErr;
    }

    /**
     * Tells the layer to which array the error should be backpropagated.
     *
     * @param e typically the error of a previous layer
     */
    public void setPreviousError(float[] e) {
        prevErr = e;
    }

    /**
     * Clears previous error
     */
    public void clearPreviousError() {
        if (prevErr == null) {
            return;
        }
        for (int i = 0; i < prevErr.length; i++) {
            prevErr[i] = 0;
        }
    }

    /**
     * Sets the expected value of an output
     *
     * @param o output number
     * @param v expected value
     */
    public void setExpected(int o, float v) {
        float e = output[o] - v;
        addError(o, e);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Getters&Setters
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * @return the weight array
     */
    public float[][] getWeights() {
        return weight;
    }

    /**
     * Sets the provided matrix as weights matrix.
     *
     * @param w the new weight matrix. The size must correspond to: inputSize*outputSize
     */
    public void setWeights(float[][] w) {
        if (w != null) {
            // Verify sizes
            if (w.length != inputSize || w[0].length != outputSize) {
                throw new IllegalArgumentException("bad input weight size: " + w.length + "x" + w[0].length + ", expected " + inputSize + "x" + outputSize);
            }
            // Store new values
            weight = copy(w);
        } else {
            throw new IllegalArgumentException("the weights provided are null!");
        }

    }

    /**
     * @return the bias array
     */
    public float[] getBias() {
        return bias;
    }

    /**
     * Set the bias array
     */
    public void setBias(float[] b) {
        if (b != null) {
            // Verify sizes
            if (b.length != outputSize) {
                throw new IllegalArgumentException("bad input bias size: " + b.length + ", expected " + outputSize);
            }
            // Store new values
            bias = b.clone();
        } else {
            throw new IllegalArgumentException("the bias provided is null!");
        }
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Utility
    ///////////////////////////////////////////////////////////////////////////////////////////////

    protected void printBias() {
        System.out.print("Bias=[");
        for (int i = 0; i < bias.length; i++) {
            System.out.printf("%.2f", bias[i]);
            if (i == (bias.length - 1)) continue;
            System.out.print(", ");
        }
        System.out.println("]");
    }

    /**
     * It is fundamental that the layer can be correctly cloned, otherwise
     * AEs cannot properly clone themselves.
     */
    @Override
    public abstract Layer clone();

    /**
     * 2D array copy
     *
     * @param m an array
     * @return a copy of m
     */
    protected float[][] copy(float[][] m) {
        float[][] n = new float[m.length][];
        for (int i = 0; i < m.length; i++) {
            n[i] = new float[m[i].length];
            System.arraycopy(m[i], 0, n[i], 0, m[i].length);
        }
        return n;
    }

    /**
     * Saves the layer to a file.
     *
     * @param fname file name
     * @throws IOException if the file cannot be written to
     */
    public void save(String fname) throws IOException {
        try (DataOutputStream os = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(fname)))) {
            save(os);
        }
    }

    /**
     * Saves the layer to a stream.
     *
     * @param os output stream
     * @throws IOException if the layer cannot be written
     */
    public void save(DataOutputStream os) throws IOException {
        os.writeInt(inputSize);
        os.writeInt(outputSize);
        for (int i = 0; i < inputSize; i++) {
            for (int o = 0; o < outputSize; o++) {
                os.writeFloat(weight[i][o]);
            }
        }
        for (int o = 0; o < outputSize; o++) {
            os.writeFloat(bias[o]);
        }
        os.flush();
    }

    /**
     * Loads the layer from a file.
     *
     * @param fname file name
     * @throws IOException if the file cannot be read
     */
    public void load(String fname) throws IOException {
        try (DataInputStream is = new DataInputStream(new FileInputStream(fname))) {
            load(is);
        }
    }

    /**
     * Loads the layer from a stream
     *
     * @param is input stream
     * @throws IOException if the stream cannot be read
     */
    public void load(DataInputStream is) throws IOException {
        inputSize = is.readInt();
        outputSize = is.readInt();
        weight = new float[inputSize][outputSize];
        for (int i = 0; i < inputSize; i++) {
            for (int o = 0; o < outputSize; o++) {
                this.weight[i][o] = is.readFloat();
            }
        }
        for (int o = 0; o < this.outputSize; o++) {
            bias[o] = is.readFloat();
        }

        // Init the error array
        err = new float[this.outputSize];

        // Init weighs sums array
        wSum = new float[this.outputSize];
    }

}

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

package diuf.diva.dia.ms.ml.ae;

import diuf.diva.dia.ms.ml.layer.Layer;
import diuf.diva.dia.ms.util.DataBlock;

import java.io.Serializable;
import java.util.Arrays;

/**
 * This abstract defines the basic interface of an Auto Encoder. Any class which extends and properly implements the
 * abstract methods can be used as a Unit in a SCAE.
 * @author Michele Alberti, Mathias Seuret
 */
public abstract class AutoEncoder implements Serializable {
    /**
     * Reference to the input - can be modified.
     */
    protected DataBlock input;
    /**
     * X coordinate of the input area.
     */
    protected int inputX;
    /**
     * Y coordinate of the input area.
     */
    protected int inputY;
    /**
     * Width of the input area.
     */
    protected final int inputWidth;
    /**
     * Height of the input area.
     */
    protected final int inputHeight;
    /**
     * Depth of the input area.
     */
    protected final int inputDepth;
    /**
     * Length of the input if represented in a 1D array
     */
    protected int inputLength;
    /**
     * Input represented in a 1D array. Beware this copy needs to
     * be manually updated as it is not a reference to real input
     */
    protected float[] inputArray;
    /**
     * Reference to the output.
     */
    protected DataBlock output;
    /**
     * X coordinate of the output.
     */
    protected int outputX;
    /**
     * Y coordinate of the output.
     */
    protected int outputY;
    /**
     * Depth of the output, corresponds to the number of outputs.
     */
    protected int outputDepth;
    /**
     * Learning speed of the unit.
     */
    protected float learningSpeed = 1e-3f;
    /**
     * Encoding layer
     */
    protected Layer encoder;
    /**
     * Decoding layer
     */
    protected Layer decoder;
    /**
     * Reference to the output error.
     */
    protected DataBlock error;
    /**
     * Reference to the previous error.
     */
    protected DataBlock prevErr;
    /**
     * Stores the decoded data.
     */
    protected float[] decoded;

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Constructor
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * Constructs an auto-encoder.
     *
     * PLEASE MAKE SURE THAT YOU INIT THE AUTO ENCODER PROPERLY AFTER!!
     *
     * To do so, don't forget to:
     * setEncoder();
     * setDecoder();
     * setInput();
     *
     * @param inputWidth input width
     * @param inputHeight input height
     * @param inputDepth input depth
     * @param outputDepth  output depth
     */
    protected AutoEncoder(
            int inputWidth,
            int inputHeight,
            int inputDepth,
            int outputDepth
    ) {

        // Saving input param
        assert (inputWidth > 0);
        assert (inputHeight > 0);
        assert (inputDepth > 0);

        this.input = null;
        this.inputX = 0;
        this.inputY = 0;
        this.inputWidth = inputWidth;
        this.inputHeight = inputHeight;
        this.inputDepth = inputDepth;

        // Computing input length
        this.inputLength = inputWidth * inputHeight * inputDepth;

        // Init input array
        inputArray = new float[inputLength];

        // Saving output param
        assert (outputDepth > 0);

        this.output = new DataBlock(1, 1, outputDepth);
        this.outputX = 0;
        this.outputY = 0;
        this.outputDepth = outputDepth;

        // Saving enc/dec layers
        this.encoder = null;
        this.decoder = null;

        // Saving previousError and error
        this.prevErr = null;
        this.error = new DataBlock(1, 1, outputDepth);

        // Init the decoded array
        decoded = new float[inputLength];
    }

    /**
     * PROTECTED
     * <p>
     * Constructs an auto-encoder. If all fields are provided this AE is ready to use.
     * In case is not possible to have all this parameters at the creation moment do please
     * use the constructor above but don't forget to eventually init the AE properly.
     *
     * @param input       DataBlock which represent the input
     * @param inputX      x coordinate on which the AE is reading input patch
     * @param inputY      y coordinate on which the AE is reading input patch
     * @param inputWidth  width of the input patch for the AE
     * @param inputHeight height of the input patch for the AE
     * @param inputDepth  depth of the input patch for the AE
     * @param output      DataBlock which represent the output
     * @param outputX     x coordinate on which the AE is storing the output
     * @param outputY     y coordinate on which the AE is storing the output
     * @param outputDepth depth of the output
     * @param encoder     Layer which represent the encoder of the AE
     * @param decoder     Layer which represent the decoder of the AE
     * @param prevErr     DataBlock which represent the previous error for the AE
     * @param error       DataBlock which represent the error of the AE
     */
    protected AutoEncoder(
            DataBlock input,
            int inputX,
            int inputY,
            int inputWidth,
            int inputHeight,
            int inputDepth,
            DataBlock output,
            int outputX,
            int outputY,
            int outputDepth,
            Layer encoder,
            Layer decoder,
            DataBlock prevErr,
            DataBlock error
    ) {

        // Saving input param
        assert (input != null);
        assert (inputX >= 0);
        assert (inputY >= 0);
        assert (inputWidth > 0);
        assert (inputHeight > 0);
        assert (inputDepth > 0);

        this.input = input;
        this.inputX = inputX;
        this.inputY = inputY;
        this.inputWidth = inputWidth;
        this.inputHeight = inputHeight;
        this.inputDepth = inputDepth;

        // Computing input length
        this.inputLength = inputWidth * inputHeight * inputDepth;

        // Init input array
        inputArray = new float[inputLength];

        // Computing the input as 1D array form
        this.input.patchToArray(inputArray, this.inputX, this.inputY, this.inputWidth, this.inputHeight);

        assert (inputArray.length == inputLength);

        // Saving output param
        assert (output != null);
        assert (outputX >= 0);
        assert (outputY >= 0);
        assert (outputDepth > 0);

        this.output = output;
        this.outputX = outputX;
        this.outputY = outputY;
        this.outputDepth = outputDepth;

        // Saving enc/dec layers
        assert (encoder != null);
        assert (decoder != null);
        assert (outputDepth == encoder.getOutputSize());

        this.encoder = encoder;
        this.decoder = decoder;

        // Saving previousError and error
        assert (error != null);

        /* prevError can be null as the first layer (most bottom) does not need to have one. */
        this.prevErr = prevErr;
        this.error = error;

        // Init the decoded array
        decoded = new float[inputLength];

        // Setting input and output of encoder
        this.encoder.setInputArray(this.inputArray);
        this.encoder.setOutputArray(this.output.getValues(this.outputX, this.outputY));

        // Setting input and output of decoder
        this.decoder.setInputArray(encoder.getOutputArray());
        this.decoder.setOutputArray(decoded);

        // Setting previous error and error of encoder
        this.encoder.setPreviousError((prevErr != null) ? prevErr.patchToArray(this.inputX, this.inputY, this.inputWidth, this.inputHeight) : null);
        this.encoder.setError(error.getValues(this.outputX, this.outputY));

        // Setting previous error and error of decoder
        this.decoder.setPreviousError(encoder.getError());

    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Computing
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * Encodes the input and stores the result in the output.
     */
    public void encode(){
        /* Copying the input patch as array for the encoder.
         * This is necessary because despite the input (dataBlock) reference
         * might be up to date, its content may have changed. This happens for instance
         * when the AE is convoluted, as previous layer changes content of
         * the input for the current layer. For this reason we just paste again
         * the input patch on array.
         */
        input.patchToArray(inputArray, inputX, inputY, inputWidth, inputHeight);
        encoder.compute();
    }

    /**
     * Decodes the output and stores it in a temporary array.
     */
    public void decode(){
        decoder.compute();
    }

    /**
     * Trains the auto-encoder.
     * @return an estimation of the reconstruction error
     */
    public float train() {

        // Compute output
        encoder.compute();
        decoder.compute();

        // Set expected for all input
        for (int i = 0; i < inputLength; i++) {
            decoder.setExpected(i, inputArray[i]);
        }

        // Backpropagate
        assert (decoder.getPreviousError() != null);
        float err = decoder.backPropagate();
        encoder.backPropagate();
        
        encoder.clearError();
        decoder.clearError();

        // Learn
        encoder.learn();
        decoder.learn();
        
        return err;
    }

    /**
     * Applies a gradient descent.
     *
     * @return the error
     */
    public void learn() {
        encoder.learn();
    }

    /**
     * Backpropagate the error, if needed.
     */
    public float backPropagate() {
        // Backpropagate
        float err = encoder.backPropagate();

        // Set the previous error from layer to datablock!
        if (prevErr!=null) {
            prevErr.weightedPatchPaste(encoder.getPreviousError(), inputX, inputY, inputWidth, inputHeight);
        }
        return err;
    }

    /**
     * This method MUST be called when the training is done.
     * Clearly those AE which need this must override this method
     */
    public void trainingDone() {
        // Nothing to do.
    }

    /**
     * @param outputNumber output to activate or not
     * @param state true if activated
     */
    public void activateOutput(int outputNumber, boolean state) {
        output.setValue(outputNumber, outputX, outputY, (state) ? 1 : 0);
    }

    /**
     * Deletes the given features.
     * @param number of the features
     */
    public void deleteFeatures(int... number) {
        int[] num = number.clone();
        Arrays.sort(num);
        for (int i = num.length - 1; i >= 0; i--) {
            encoder.deleteOutput(num[i]);
            decoder.deleteInput(num[i]);
        }

        outputDepth -= number.length;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Input related
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * @return the input datablock
     */
    public DataBlock getInput() {
        return input;
    }

    /**
     * Sets the input.
     *
     * @param db input DataBlock
     * @param x  position x
     * @param y  position y
     */
    public void setInput(DataBlock db, int x, int y) {
        assert (db != null);
        assert (x >= 0);
        assert (y >= 0);

        assert (x + inputWidth <= db.getWidth());
        assert (y + inputHeight <= db.getHeight());
        assert (inputDepth == db.getDepth());

        // Set the input parameters
        input = db;
        inputX = x;
        inputY = y;

        // Computing the new input as 1D array form
        input.patchToArray(inputArray, inputX, inputY, inputWidth, inputHeight);

        // Set input for encoder
        encoder.setInputArray(inputArray);
    }

    /**
     * @return the input width
     */
    public int getInputWidth() {
        return inputWidth;
    }

    /**
     * @return the inputHeight
     */
    public int getInputHeight() {
        return inputHeight;
    }

    /**
     * @return the input depth
     */
    public int getInputDepth() {
        return inputDepth;
    }

    /**
     * @return the number of values in the input
     */
    public int getInputSize() {
        return inputWidth * inputHeight * inputDepth;
    }

    /**
     * @return the input array as 1D array form, NOT a copy
     */
    public float[] getInputArray(){
        return inputArray;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Output related
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * @return the output datablock
     */
    public DataBlock getOutput() {
        return output;
    }

    /**
     * Sets the output
     *
     * @param db output DataBlock
     * @param x  position x
     * @param y  position y
     */
    public void setOutput(DataBlock db, int x, int y) {
        assert (db != null);
        assert (encoder != null); // Forgot to init AutoEncoder properly?
        assert (decoder != null); // Forgot to init AutoEncoder properly?
        assert (db.getDepth() == outputDepth);
        assert (x < db.getWidth());
        assert (y < db.getHeight());

        // Assign the new output
        output = db;
        outputX = x;
        outputY = y;

        // Set output for encoder
        encoder.setOutputArray(output.getValues(x, y));

        // Set input for decoder (which is the same as the output of the encoder!)
        decoder.setInputArray(output.getValues(x, y));
    }

    /**
     * @return the number of values in the output
     */
    public int getOutputDepth() {
        return outputDepth;
    }

    /**
     * @return the output array, not a copy
     */
    public float[] getOutputArray() {
        return output.getValues(outputX,outputY);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Error related
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * Sets the previous error to use.
     *
     * @param db previous error data block
     */
    public void setPrevError(final DataBlock db) {
        assert (db != null);
        assert (input.getWidth() == db.getWidth());
        assert (input.getHeight() == db.getHeight());
        assert (inputDepth == db.getDepth());

        prevErr = db;

        // Set the previous error for the encoder
        encoder.setPreviousError(prevErr.patchToArray(this.inputX, this.inputY, this.inputWidth, this.inputHeight));
    }

    /**
     * Sets the error to use.
     *
     * @param db error data block
     */
    public void setError(DataBlock db) {
        assert (db != null);
        assert (output.getWidth() == db.getWidth());
        assert (output.getHeight() == db.getHeight());
        assert (outputDepth == db.getDepth());

        error = db;

        // Set error for the encoder
        encoder.setError(error.getValues(outputX, outputY));

        // Setting previous error of decoder
        this.decoder.setPreviousError(encoder.getError());
    }

    /**
     * Clear the error of the encoder and decoder
     */
    public void clearError() {
        encoder.clearError();
        encoder.clearPreviousError();
        decoder.clearError();
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Other getters&setters
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * @return the encoder
     */
    public Layer getEncoder() {
        return encoder;
    }

    /**
     * Set the encoder
     *
     * @param encoder that will be used
     */
    public void setEncoder(Layer encoder) {
        this.encoder = encoder;

        // Setting input and output of encoder
        encoder.setInputArray(inputArray);
        encoder.setOutputArray((output != null) ? output.getValues(outputX, outputY) : null);

        // Setting previous error and error of encoder
        encoder.setPreviousError((prevErr != null) ? prevErr.patchToArray(this.inputX, this.inputY, this.inputWidth, this.inputHeight) : null);
        encoder.setError(error.getValues(outputX, outputY));
    }

    /**
     * @return the decoder
     */
    public Layer getDecoder() {
        return decoder;
    }

    /**
     * Set the decoder
     *
     * @param decoder that will be used
     */
    public void setDecoder(Layer decoder) {
        this.decoder = decoder;

        // Setting input and output of decoder
        decoder.setInputArray(encoder.getOutputArray());
        decoder.setOutputArray(decoded);

        // Setting previous error and error of decoder
        decoder.setPreviousError(encoder.getError());
    }

    /**
     * @return the decoded array, not a copy
     */
    public float[] getDecoded() {
        return decoded;
    }

    /**
     * Set the decoded array, not a copy
     */
    public void setDecoded(float[] d) {
        decoded = d;

        // Set the output for the decoder layer
        decoder.setOutputArray(decoded);
    }

    /**
     * @return the current learning speed
     */
    public float getLearningSpeed() {
        return learningSpeed;
    }

    /**
     * Sets the learning speed of all layers. Default value: 1e-3f.
     *
     * @param s new learning speed
     */
    public void setLearningSpeed(float s) {
        learningSpeed = s;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Properties
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * Child of this class should override this method if necessary
     *
     * @return true if the autoencoder has binary outputs
     */
    public boolean hasBinaryOutput() {
        return false;
    }

    /**
     * Child of this class should override this method if necessary
     *
     * @return true if the autoencoder cannot have anything else than binary inputs
     */
    public boolean needsBinaryInput() {
        return false;
    }

    /**
     * Child of this class should override this method if necessary
     *
     * @return true if the autoencoder has denoising abilities
     */
    public boolean isDenoising() {
        return false;
    }

    /**
     * Child of this class should override this method if necessary
     *
     * @return a character indicating what kind of autoencoder this is
     */
    public abstract char getTypeChar();

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Utility
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * Of utmost importance for the correct behaviour of the ConvolutionLayer, the method
     * clone should return a proper copy of the AE. Take special care of deep copy the layers!
     * Note however than the dataBlocks should not be cloned!
     * In fact, we want to copy the AE and not his environment. It is duty of who uses the copy to
     * change the input, output and errors dataBlock meaningfully!
     */
    public abstract AutoEncoder clone();

    /**
     * Parses the short class name
     *
     * @param c the class to be examinated (e.g "diuf.diva.dia.ms.ml.layer.NeuralLayer")
     * @return a string which contains only the class name (e.g NeuralLayer)
     */
    protected String parseClassName(Class c) {
        String[] cn = c.getName().split("\\.");
        return cn[cn.length - 1];
    }

    /**
     * 2D array copy and conversion to float[][]
     *
     * @param b an array
     * @return a copy of b, converted to float[][]
     */
    protected float[][] copy(double[][] b) {
        // Get dimensions
        int n = b.length;
        int m = b[0].length;

        // Deep copy the matrix and cast it to float
        float[][] a = new float[n][m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                a[i][j] = (float) b[i][j];
            }
        }

        return a;
    }

    /**
     * Pastes the decoded data onto the given data block.
     * @param data target
     * @param x position
     * @param y  position
     */
    public void pasteDecoded(DataBlock data, int x, int y) {
        int p = 0;
        for (int i = 0; i < getInputWidth(); i++) {
            for (int j = 0; j < getInputHeight(); j++) {
                data.weightedPaste(decoded, p, x + i, y + j);
                p += input.getDepth();
            }
        }
    }

    /**
     * @return a string describing the structure of the classifier
     */
    @Override
    public String toString() {
        return getTypeChar()+":"+getInputWidth()+"x"+getInputHeight()+"x"+getOutputDepth();
    }

}

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

import diuf.diva.dia.ms.ml.Trainable;
import diuf.diva.dia.ms.ml.layer.Layer;
import diuf.diva.dia.ms.util.DataBlock;

import java.io.Serializable;
import java.util.Arrays;

/**
 * This abstract defines the basic interface of an Auto Encoder. Any class which extends and properly implements the
 * abstract methods can be used as a Unit in a SCAE.
 * @author Michele Alberti, Mathias Seuret
 */
public abstract class AutoEncoder implements Serializable, Trainable {

    private static final long serialVersionUID = -3741751341348339527l;
    /**
     * Width of the input area.
     */
    public final int inputWidth;
    /**
     * Height of the input area.
     */
    public final int inputHeight;
    /**
     * Depth of the input area.
     */
    public final int inputDepth;
    /**
     * Reference to the input - can be modified.
     */
    public DataBlock input;
    /**
     * X coordinate of the input area.
     */
    public int inputX;
    /**
     * Y coordinate of the input area.
     */
    public int inputY;
    /**
     * Length of the input if represented in a 1D array
     */
    public int inputLength;
    /**
     * Input represented in a 1D array. Beware this copy needs to
     * be manually updated as it is not a reference to real input
     */
    public float[] inputArray;
    /**
     * Reference to the output.
     */
    public DataBlock output;
    /**
     * X coordinate of the output.
     */
    public int outputX;
    /**
     * Y coordinate of the output.
     */
    public int outputY;
    /**
     * Depth of the output, corresponds to the number of outputs.
     */
    public int outputDepth;
    /**
     * Encoding layer
     */
    public Layer encoder;
    /**
     * Decoding layer
     */
    public Layer decoder;
    /**
     * Reference to the output error.
     */
    public DataBlock error;
    /**
     * Reference to the previous error.
     */
    public DataBlock prevErr;
    /**
     * Stores the decoded data.
     */
    public float[] decoded;
    /**
     * Keeps track whether trainingDone() has been already called or not
     */
    public boolean trainingDone = false;

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Constructor
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * Constructs an auto-encoder without encoder nor decoder. You <i>must</i>
     * add them manually afterward, otherwise the instance won't be usable.
     * To do so, you will have to use the following methods:
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

        this.input = new DataBlock(inputWidth, inputHeight, inputDepth);
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
     * Constructs an auto-encoder. If all fields are provided this AE is ready
     * to use. In case is not possible to have all this parameters at the
     * creation moment do please use the constructor above but don't forget to
     * eventually init the AE properly.
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
        inputPatchToArray(inputArray);

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
        if (prevErr==null) {
            this.encoder.setPreviousError(null);
        } else {
            this.encoder.setPreviousError(
                    prevErr.patchToArray(this.inputX, this.inputY, this.inputWidth, this.inputHeight)
            );
        }
        
        this.encoder.setError(error.getValues(this.outputX, this.outputY));

        // Setting previous error and error of decoder
        this.decoder.setPreviousError(encoder.getError());

    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Computing
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * 2D array copy and conversion to float[][]
     *
     * @param b an array
     * @return a copy of b, converted to float[][]
     */
    protected static float[][] copy(double[][] b) {
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
        inputPatchToArray(inputArray);
        encoder.compute();
    }

    /**
     * Decodes the output and stores it in a temporary array.
     */
    public void decode(){
        decoder.compute();
    }

    /**
     * Trains the auto-encoder. The semantical meaning of the return value
     * might depend on the kind of encoder and decoder used.
     * @return an estimation of the reconstruction error
     */
    public float train() {

        // Compute output
        encode();
        decode();

        // Set expected for all output
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
     * Applies a gradient descent when doing supervised training.
     */
    public void learn() {
        encoder.learn();
    }

    /**
     * Backpropagate the error, if needed.
     * @return the mean absolute error of the top layer
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
        // Place holder for sub-classes to override
    }

    /**
     * This method will be replaced by startTraining() / stopTraining().
     * These methods should be used, as this one will be removed in the future.
     * @return true if the training has been indicated as done
     */
    @Deprecated
    public boolean isTrainingDone() {
        return trainingDone;
    }

    /**
     * Can be used to manually activate an output (value 1) or deactivate it
     * (value 0). This is used when displaying features.
     * @param outputNumber output to activate or not
     * @param activated true if activated
     */
    public void activateOutput(int outputNumber, boolean activated) {
        output.setValue(outputNumber, outputX, outputY, (activated) ? 1 : 0);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Input related
    ///////////////////////////////////////////////////////////////////////////////////////////////

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

    /**
     * @return the input datablock
     */
    public DataBlock getInput() {
        return input;
    }

    /**
     * Sets the input, based on the coordinates of the top-left corner of
     * the input patch.
     *
     * @param db input DataBlock
     * @param x  position x
     * @param y  position y
     */
    public void setInput(DataBlock db, int x, int y) {
        if (db==null) {
            return;
        }

        // assert (x >= 0);
        // assert (y >= 0);

        //       assert (x + inputWidth <= db.getWidth());
//        assert (y + inputHeight <= db.getHeight());
        assert (inputDepth == db.getDepth());

        // Set the input parameters
        input = db;
        inputX = x;
        inputY = y;

        // Computing the new input as 1D array form
        inputPatchToArray(inputArray);

        // Set input for encoder
        // TODO why is next line commented?!?! -> update apparently the reference is updated with inputPatchToArray().
        //encoder.setInputArray(inputArray);
    }

    /**
     * Copies the input patch to an array. Can be overwritten.
     * @param array the destination array
     */
    protected void inputPatchToArray(float[] array) {
        input.patchToArray(array, inputX, inputY, inputWidth, inputHeight);
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

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Output related
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * Do not use, this will be removed in the future.
     * @return the input array as 1D array form, NOT a copy
     */
    @Deprecated
    public float[] getInputArray(){
        return inputArray;
    }

    /**
     * @return the output datablock
     */
    public DataBlock getOutput() {
        return output;
    }

    /**
     * Sets the output location. Coordinates are based on the top-left corner
     * of the input patch.
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

        // If needed also set its error
        if (error!=null && output.getWidth() == error.getWidth() && output.getHeight() == error.getHeight()) {
            encoder.setError(error.getValues(x, y));
        }

        // Set input for decoder (which is the same as the output of the encoder!)
        decoder.setInputArray(output.getValues(x, y));
    }

    /**
     * @return the number of values in the output
     */
    public int getOutputDepth() {
        return outputDepth;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Error related
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * Returns the output array itself, not a copy. For performance reasons,
     * the same array is always used. If you need to store several outputs,
     * then you will have to clone it with myAE.getOutputArray().clone();
     *
     * @return the output array
     */
    public float[] getOutputArray() {
        return output.getValues(outputX,outputY);
    }

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

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Other getters&setters
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * Clear the error of the encoder and decoder
     */
    public void clearError() {
        encoder.clearError();
        encoder.clearPreviousError();
        decoder.clearError();
    }

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
        if (prevErr==null) {
            encoder.setPreviousError(null);
        } else {
            encoder.setPreviousError(
                    prevErr.patchToArray(this.inputX, this.inputY, this.inputWidth, this.inputHeight)
            );
        }
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
     * @param d the array to which decoded data will be written
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
        return encoder.getLearningSpeed();
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Properties
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * Sets the learning speed of all layers. Default value: 1e-3f.
     *
     * @param s new learning speed
     */
    public void setLearningSpeed(float s) {
        if (encoder!=null) {
            encoder.setLearningSpeed(s);
        }
        if (decoder!=null) {
            decoder.setLearningSpeed(s);
        }
    }

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
     * @return true if the autoencoder need a supervised training type
     */
    public boolean isSupervised() {
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

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Utility
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * Child of this class should override this method if necessary
     *
     * @return a character indicating what kind of autoencoder this is
     */
    public abstract String getTypeName();

    /**
     * This very important method is used when convolving the autoencoder in
     * ConvolutionLayers. An exact copy of the AE must be returned. Take special
     * care of deep copy the layers! Note however than the dataBlocks should not
     * be cloned, as we want to copy the AE and not his environment.
     * It is duty of who uses the copy to change the input, output and errors
     * dataBlock meaningfully.
     * @return a copy of the autoencoder
     */
    @Override
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
     * Normalise matrix
     *
     * @param m matrix to normalise
     * @return the matrix itself
     */
    public float[][] normalise(float[][] m) {
        double norm = 0;
        for (int x = 0; x < m.length; x++) {
            for (int y = 0; y < m[x].length; y++) {
                norm += Math.pow(m[x][y], 2);
            }
        }
        norm = Math.sqrt(norm);
        for (int x = 0; x < m.length; x++) {
            for (int y = 0; y < m[x].length; y++) {
                m[x][y] /= norm;
            }
        }
        return m;
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
        // Computing the new input as 1D array form
        inputPatchToArray(inputArray);
    }

    /**
     * @return a string describing the structure of the classifier
     */
    @Override
    public String toString() {
        return getTypeName()+":"+getInputWidth()+"x"+getInputHeight()+"x"+getOutputDepth();
    }

    public abstract void clearGradient();
}

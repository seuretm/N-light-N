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

package diuf.diva.dia.ms.ml.ae.ffcnn;

import diuf.diva.dia.ms.ml.ae.AutoEncoder;
import diuf.diva.dia.ms.ml.ae.StandardAutoEncoder;
import diuf.diva.dia.ms.ml.ae.scae.Convolution;
import diuf.diva.dia.ms.util.DataBlock;

import java.io.Serializable;

/**
 * Convolution layer used by FFCNN.
 * @author Mathias Seuret, Michele Alberti
 */
public class SingleUnitConvolution extends ConvolutionalLayer implements Serializable {
    /**
     * Number of units on X axis.
     */
    int outWidth;
    /**
     * Number of units on Y axis
     */
    int outHeight;
    /**
     * Number of outputs of the unit.
     */
    int outDepth;
    /**
     * Width of the input area.
     */
    int inputWidth;
    /**
     * Height of the input area.
     */
    int inputHeight;
    /**
     * Depth of the input area.
     */
    int inputDepth;
    /**
     * By how much the units are offset.
     */
    int offsetX;
    /**
     * By how much the units are offset.
     */
    int offsetY;
    /**
     * Position X of the input area.
     */
    int inputX;
    /**
     * Position Y of the input area.
     */
    int inputY;
    /**
     * Units.
     */
    AutoEncoder unit;
    /**
     * Input data block.
     */
    DataBlock input;
    /**
     * Output data block.
     */
    DataBlock output;
    /**
     * Data block storing the error.
     */
    DataBlock error;
    /**
     * Accumulator of the previous layer.
     */
    DataBlock prevError = null;

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Constructor
    ///////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * Creates a convolution layer out of an autoencoder convolution (basically is where the SCAE
     * becomes part of the FFCNN).
     * @param convolution autoencoder convolution
     */
    public SingleUnitConvolution(Convolution convolution) {
        
        // Copying the input parameters of the layer
        inputWidth = convolution.getInputPatchWidth();
        inputHeight = convolution.getInputPatchHeight();
        inputDepth = convolution.getInputPatchDepth();

        offsetX = convolution.getInputOffsetX();
        offsetY = convolution.getInputOffsetY();

        // Copying output parameters of the layer
        outWidth = convolution.getOutputWidth();
        outHeight = convolution.getOutputHeight();
        outDepth = convolution.getOutputDepth();

        // Init two new dataBlocks for output and error
        output = new DataBlock(outWidth, outHeight, outDepth);
        error  = new DataBlock(outWidth, outHeight, outDepth);

        /* This setup of the output is necessary because the loaded SCAE
         * might have some 'memory' of last command which would made the
         * clone() of layers crash. Specifically commands like 'recode' and
         * 'show features' leave the SCAE with output set on the last convoluted
         * unit. This is in general not a problem, but when accessing coordinate
         * of output (outputX, outputY) while cloning, raises an out of bounds
         * exception on the creation of the cloning of the AE.
         */
        convolution.getBase().setOutput(output, 0, 0);
        unit = convolution.getBase().clone();
        unit.setError(error);
    }

    /**
     * Creates a convolution layer, on top of a previous one, and with a given number
     * of outputs. Take note that it is not a real convolution, as we do not do
     * offsets.
     * @param prev previous layer
     * @param layerClassName kind of layer which has to be used for encoding and decoding
     * @param nbNeurons number of neurons
     */
    public SingleUnitConvolution(ConvolutionalLayer prev, String layerClassName, int nbNeurons) {
        assert (nbNeurons>0);
        assert (prev!=null);

        outWidth = 1;
        outHeight = 1;
        outDepth = nbNeurons;

        inputWidth = prev.getOutput().getWidth();
        inputHeight = prev.getOutput().getHeight();
        inputDepth = prev.getOutput().getDepth();

        offsetX = 1; // not used in this case but cannot be null
        offsetY = 1;

        // Init two new dataBlocks for output and error
        output = new DataBlock(outWidth, outHeight, outDepth);
        error  = new DataBlock(outWidth, outHeight, outDepth);

        unit = new StandardAutoEncoder(inputWidth, inputHeight, inputDepth, outDepth, layerClassName);
        unit.setOutput(output, 0, 0);
        unit.setError(error);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Computing
    ///////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * Computes the output.
     */
    public void compute() {
        for (int x=0; x<outWidth; x++) {
            for (int y=0; y<outHeight; y++) {
                unit.setInput(input, inputX + x * offsetX, inputY + y * offsetY);
                unit.setOutput(output, x, y);
                unit.encode();
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Learning
    ///////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * Sets the expected value, assuming the layer is not convolved.
     * @param z output number
     * @param ex expected value
     */
    public void setExpected(int z, float ex) {
        setExpected(0, 0, z, ex);
    }

    /**
     * Sets which value was expected at a given output.
     * @param x output position x
     * @param y output position y
     * @param z output position z
     * @param ex expected value
     */
    public void setExpected(int x, int y, int z, float ex) {
        float e = output.getValue(z, x, y) - ex;
        addError(x, y, z, e);
    }

    /**
     * Learn the units
     */
    public void learn() {
        for (int x=0; x<outWidth; x++) {
            for (int y=0; y<outHeight; y++) {
                unit.setInput(input, inputX + x * offsetX, inputY + y * offsetY);
                unit.setOutput(output, x, y);
                unit.setError(error);
                unit.learn();
            }
        }
    }

    /**
     * Backpropagate the error, if needed.
     * @return the mean absolute error of the outputs
     */
    public float backPropagate() {
        // Backpropagate on all the units of this layer
        float errSum = 0.0f;
        for (int x = 0; x < outWidth; x++) {
            for (int y = 0; y < outHeight; y++) {
                unit.setInput(input, inputX + x * offsetX, inputY + y * offsetY);
                unit.setOutput(output, x, y);
                unit.setError(error);
                errSum += unit.backPropagate();
            }
        }

        // Return the cumulated error
        return errSum / (outWidth * outHeight);
    }
    
    /**
     * @param x should be 0
     * @param y should be 0
     * @return the autoencoder
     */
    public AutoEncoder getAutoEncoder(int x, int y) {
        assert (x==0);
        assert (y==0);
        
        return unit;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Input related
    ///////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * @return the perception area width
     */
    public int getInputWidth() {
        return inputWidth;
    }

    /**
     * @return the perception area height
     */
    public int getInputHeight() {
        return inputHeight;
    }

    /**
     * @return the perception area depth
     */
    public int getInputDepth() {
        return inputDepth;
    }

    /**
     * @return the input data block
     */
    DataBlock getInput() {
        return input;
    }

    /**
     * Select the input data and the position.
     * @param db input data block
     * @param posX position x of the input
     * @param posY position y of the input
     */
    public void setInput(DataBlock db, int posX, int posY) {
        assert (db.getDepth() == inputDepth);
        assert (posX + inputWidth <= db.getWidth());
        assert (posY + inputHeight <= db.getHeight());

        input = db;
        inputX = posX;
        inputY = posY;
        
        for (int x = 0; x < outWidth; x++) {
            for (int y = 0; y < outHeight; y++) {
                unit.setInput(input, inputX + x * offsetX, inputY + y * offsetY);
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Output related
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * @return the output of the layer
     */
    public DataBlock getOutput() {
        return output;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Error related
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * @return the error data block
     */
    public DataBlock getError() {
        return error;
    }
    /**
     * Changes the error data block
     * @param db new data block
     */
    public void setError(DataBlock db) {
        error = db;
    }

    /**
     * Clears the error in all units
     */
    public void clearError() {
        for (int x = 0; x < outWidth; x++) {
            for (int y = 0; y < outHeight; y++) {
                unit.setInput(input, inputX + x * offsetX, inputY + y * offsetY);
                unit.setOutput(output, x, y);
                unit.setError(error);
                unit.clearError();
            }
        }
    }

    /**
     * Selects the data block to which the error has to be backpropagated.
     * @param db data block
     */
    public void setPrevError(DataBlock db) {
        assert (db != null);
        assert (db.getDepth() == inputDepth);
        assert (inputWidth == db.getWidth());
        assert (inputHeight == db.getHeight());

        prevError = db;
        unit.setPrevError(db);
    }

    /**
     * Adds a value to the error of an output.
     * @param x output position x
     * @param y output position y
     * @param z output position z
     * @param e error to add
     */
    public void addError(int x, int y, int z, float e) {
        error.setValue(z, x, y, error.getValue(z, x, y) + e / (outWidth*outHeight));
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Getters & Setters
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * @return the current learning speed
     */
    public float getLearningSpeed() {
        return unit.getLearningSpeed();
    }

    /**
     * Sets the learning speed. Default value: 1e-3f.
     *
     * @param s new learning speed
     */
    public void setLearningSpeed(float s) {
        unit.setLearningSpeed(s);
    }
    
    @Override
    public DataBlock getPrevError() {
        return prevError;
    }
}

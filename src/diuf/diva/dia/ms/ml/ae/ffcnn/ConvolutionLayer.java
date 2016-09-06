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
public class ConvolutionLayer implements Serializable {
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
    AutoEncoder[][] unit;
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
    DataBlock prevAccumulator = null;

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Constructor
    ///////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * Creates a convolution layer out of an autoencoder convolution (basically is where the SCAE
     * becomes part of the FFCNN).
     * @param convolution autoencoder convolution
     */
    public ConvolutionLayer(Convolution convolution) {
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

        // Get the autoencoder out of the convolution
        AutoEncoder ae = convolution.getBase();

        /* This setup of the output is necessary because the loaded SCAE
         * might have some 'memory' of last command which would made the
         * clone() of layers crash. Specifically commands like 'recode' and
         * 'show features' leave the SCAE with output set on the last convoluted
         * unit. This is in general not a problem, but when accessing coordinate
         * of output (outputX, outputY) while cloning, raises an out of bounds
         * exception on the creation of the cloning of the AE.
         */
        ae.setOutput(output, 0, 0);

        /* Replicate the autoencoder for all the units in the convoluted layer such
         * that the weight of each AE can now be trained separately.
         * Input and previousError are set in the FFCNN!
         */
        unit = new AutoEncoder[outWidth][outHeight];
        for (int x=0; x<outWidth; x++) {
            for (int y=0; y<outHeight; y++) {
                unit[x][y] = ae.clone();
                unit[x][y].setOutput(output, x, y);
                unit[x][y].setError(error);
            }
        }
    }

    /**
     * Creates a convolution layer, on top of a previous one, and with a given number
     * of outputs. Take note that it is not a real convolution, as we do not do
     * offsets.
     * @param prev previous layer
     * @param nbNeurons number of neurons
     */
    public ConvolutionLayer(ConvolutionLayer prev, String layerClassName, int nbNeurons) {
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

        unit = new AutoEncoder[1][1];

        unit[0][0] = new StandardAutoEncoder(inputWidth, inputHeight, inputDepth, outDepth, layerClassName);
        unit[0][0].setOutput(output, 0, 0);
        unit[0][0].setError(error);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Computing
    ///////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * Computes the output.
     */
    void compute() {
        for (int x=0; x<outWidth; x++) {
            for (int y=0; y<outHeight; y++) {
                unit[x][y].encode();
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
                unit[x][y].learn();
            }
        }
    }

    /**
     * Backpropagate the error, if needed.
     */
    public float backPropagate() {
        // Backpropagate on all the units of this layer
        float errSum = 0.0f;
        for (int x = 0; x < outWidth; x++) {
            for (int y = 0; y < outHeight; y++) {
                errSum += unit[x][y].backPropagate();
            }
        }

        // Reset the error datablock. It clears the units error as well as it is a reference
        //error.clear();

        // Return the cumulated error
        return errSum / (outWidth * outHeight);
    }
    
    public AutoEncoder getAutoEncoder(int x, int y) {
        return unit[x][y];
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
                unit[x][y].setInput(db, posX + x * offsetX, posY + y * offsetY);
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Output related
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * @return the output of the layer
     */
    DataBlock getOutput() {
        return output;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Error related
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * @return the error data block
     */
    DataBlock getError() {
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
    void clearError() {
        for (int x = 0; x < outWidth; x++) {
            for (int y = 0; y < outHeight; y++) {
                unit[x][y].clearError();
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

        prevAccumulator = db;
        for (int x=0; x<outWidth; x++) {
            for (int y=0; y<outHeight; y++) {
                unit[x][y].setPrevError(db);
            }
        }
    }

    /**
     * Adds something to the error of an output.
     * @param x output position x
     * @param y output position y
     * @param z output position z
     * @param e error to add
     */
    public void addError(int x, int y, int z, float e) {
        error.setValue(z, x, y, error.getValue(z, x, y) + e);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Getters & Setters
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * @return the current learning speed
     */
    public float getLearningSpeed() {
        float sum = 0.0f;
        for (int x=0; x<outWidth; x++) {
            for (int y=0; y<outHeight; y++) {
                unit[x][y].getLearningSpeed();
            }
        }
        return sum / (outWidth*outHeight);
    }

    /**
     * Sets the learning speed. Default value: 1e-3f.
     *
     * @param s new learning speed
     */
    public void setLearningSpeed(float s) {
        for (int x=0; x<outWidth; x++) {
            for (int y=0; y<outHeight; y++) {
                unit[x][y].setLearningSpeed(s);
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Utility
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * Executes on the whole layer: evaluateInputImportance()
     */
    void evaluateInputImportance() {
//        for (int x=0; x<outWidth; x++) {
//            for (int y=0; y<outHeight; y++) {
//                unit[x][y].evaluateInputImportance();
//            }
//        }
    }
}

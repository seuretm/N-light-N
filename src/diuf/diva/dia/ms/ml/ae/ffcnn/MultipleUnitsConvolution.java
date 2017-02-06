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
import diuf.diva.dia.ms.ml.ae.scae.Convolution;
import diuf.diva.dia.ms.util.DataBlock;
import java.io.Serializable;

/**
 * This subclass of ConvolutionLayer implements the same behavior as its
 * parent class, but allows the units to learn different weights at the
 * different positions within the convolution. Note that this was the
 * behavior of N-light-N before the creation of this class. In order to
 * use it, call the deconvolve() method from a FFCNN instance.
 * @author Mathias Seuret
 */
public class MultipleUnitsConvolution implements ConvolutionalLayer, Serializable {
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
    
    AutoEncoder[][] unit;
    
    /**
     * Turns a convolution into a deconvolved layer.
     * @param prev to transform
     */
    public MultipleUnitsConvolution(SingleUnitConvolution prev) {
        output = prev.getOutput().clone();
        
        inputWidth = prev.inputWidth;
        inputHeight = prev.inputHeight;
        inputDepth = prev.inputDepth;
        outWidth = prev.outWidth;
        outHeight = prev.outHeight;
        offsetX = prev.offsetX;
        offsetY = prev.offsetY;
        
        error = new DataBlock(outWidth, outHeight, output.getDepth());
        
        unit = new AutoEncoder[outWidth][outHeight];
        prev.unit.setError(error);
        for (int x=0; x<outWidth; x++) {
            for (int y=0; y<outHeight; y++) {
                unit[x][y] = prev.unit.clone();
                unit[x][y].setError(error);
                unit[x][y].setInput(input, x, y);
                unit[x][y].setOutput(output, x, y);
            }
        }
    }
    
    /**
     * Creates a deconvolved layer and "connects" it to a previous convolution
     * or deconvolved layer.
     * @param prev previous layer
     * @param layerClassName class name of the encoder
     * @param nbNeurons number of neurons in the encoder
     */
    public MultipleUnitsConvolution(SingleUnitConvolution prev, String layerClassName, int nbNeurons) {
        outWidth = prev.outWidth;
        outHeight = prev.outHeight;
        outDepth = prev.outDepth;
        error = new DataBlock(outWidth, outHeight, outDepth);
        for (int x=0; x<outWidth; x++) {
            for (int y=0; y<outHeight; y++) {
                unit[x][y] = prev.unit.clone();
                unit[x][y].setError(error);
            }
        }
    }
    
    /**
     * Creates a deconvolved layer out of a convolved layer.
     * @param model convolved layer to use
     */
    /*public MultipleUnitsConvolution(SingleUnitLayer model) {
        this(
                new Convolution(
                        model.unit,
                        model.outWidth,
                        model.outHeight,
                        model.offsetX,
                        model.offsetY
                )
        );
    }*/
    
    /**
     * Computes the output.
     */
    @Override
    public void compute() {
        for (int x=0; x<outWidth; x++) {
            for (int y=0; y<outHeight; y++) {
                unit[x][y].encode();
            }
        }
    }
    
    /**
     * Sets the expected value, assuming the layer is not convolved.
     * @param z output number
     * @param ex expected value
     */
    @Override
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
    @Override
    public void setExpected(int x, int y, int z, float ex) {
        float e = output.getValue(z, x, y) - ex;
        addError(x, y, z, e);
    }
    
    /**
     * Adds a value to the error of an output.
     * @param x output position x
     * @param y output position y
     * @param z output position z
     * @param e error to add
     */
    public void addError(int x, int y, int z, float e) {
        error.setValue(z, x, y, error.getValue(z, x, y) + e);
    }

    
    /**
     * Learn the units
     */
    @Override
    public void learn() {
        for (int x=0; x<outWidth; x++) {
            for (int y=0; y<outHeight; y++) {
                unit[x][y].learn();
            }
        }
    }
    
    /**
     * Backpropagate the error, if needed.
     * @return the mean absolute error of the outputs
     */
    @Override
    public float backPropagate() {
        // Backpropagate on all the units of this layer
        float errSum = 0.0f;
        for (int x = 0; x < outWidth; x++) {
            for (int y = 0; y < outHeight; y++) {
                errSum += unit[x][y].backPropagate();
            }
        }

        // Return the cumulated error
        return errSum / (outWidth * outHeight);
    }
    
    /**
     * @param x coordinate in the convolution
     * @param y coordinate in the convolution
     * @return the autoencoder at the specified coordinates
     */
    @Override
    public AutoEncoder getAutoEncoder(int x, int y) {
        return unit[x][y];
    }
    
    /**
     * Select the input data and the position.
     * @param db input data block
     * @param posX position x of the input
     * @param posY position y of the input
     */
    @Override
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
    
    /**
     * Clears the error in all units
     */
    @Override
    public void clearError() {
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
    @Override
    public void setPrevError(DataBlock db) {
        assert (db != null);
        assert (db.getDepth() == inputDepth);
        assert (inputWidth == db.getWidth());
        assert (inputHeight == db.getHeight());

        prevError = db;
        for (int x=0; x<outWidth; x++) {
            for (int y=0; y<outHeight; y++) {
                unit[x][y].setPrevError(db);
            }
        }
    }
    
    /**
     * @return the current learning speed
     */
    @Override
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
    @Override
    public void setLearningSpeed(float s) {
        for (int x=0; x<outWidth; x++) {
            for (int y=0; y<outHeight; y++) {
                unit[x][y].setLearningSpeed(s);
            }
        }
    }

    @Override
    public DataBlock getOutput() {
        return output;
    }
    
    /**
     * @return the error data block
     */
    public DataBlock getError() {
        return error;
    }

    @Override
    public int getInputWidth() {
        return inputWidth;
    }

    @Override
    public int getInputHeight() {
        return inputHeight;
    }

    @Override
    public DataBlock getPrevError() {
        return prevError;
    }

    @Override
    public int getXoffset() {
        return offsetX;
    }

    @Override
    public int getYoffset() {
        return offsetY;
    }
}

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

package diuf.diva.dia.ms.ml.ae.scae;

import diuf.diva.dia.ms.ml.Trainable;
import diuf.diva.dia.ms.ml.ae.AutoEncoder;
import diuf.diva.dia.ms.ml.ae.SupervisedAutoEncoder;
import diuf.diva.dia.ms.util.DataBlock;

import java.io.Serializable;

/**
 * This corresponds to a convolution of an autoencoder on an input data block.
 * @author Mathias Seuret, Michele Alberti
 */
public class Convolution  implements Serializable, Trainable {

    private static final long serialVersionUID = 5381615971840980344l;

    /**
     * The autoencoder.
     */
    public AutoEncoder base;
    /**
     * The input.
     */
    public DataBlock input;
    /**
     * The output. Its dimensions must match the dimensions of the convolution.
     */
    public DataBlock output;
    /**
     * The number of X positions of the autoencoder.
     */
    public int outWidth;
    /**
     * The number of Y positions of the autoencoder.
     */
    public int outHeight;
    /**
     * By how much the autoencoder is offset.
     */
    public int inputOffsetX;
    /**
     * By how much the autoencoder is offset.
     */
    public int inputOffsetY;
    /**
     * X position of the convolution.
     */
    public int inputX;
    /**
     * Y position of the convolution.
     */
    public int inputY;
    
    /**
     * Set to true during training phases.
     */
    protected boolean isTraining = false;

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Constructor
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * Prepares a convolution with the given autoencoder and offset, but
     * does not convolve it yet.
     * @param base the encoder
     * @param inOffX the X offset between two autoencoders
     * @param inOffY the Y offset
     */
    public Convolution(AutoEncoder base, int inOffX, int inOffY) {
        this(base, 1, 1, inOffX, inOffY);
    }

    /**
     * Creates a convolution of the given autoencoder.
     * @param base the encoder
     * @param outW the convolution's width (in autoencoders)
     * @param outH the convolution's height (in autoencoders)
     * @param inOffX the X offset between two autoencoders
     * @param inOffY the Y offset between two autoencoders
     */
    public Convolution(AutoEncoder base, int outW, int outH, int inOffX, int inOffY) {
        assert (base!=null);
        assert (outW>=1);
        assert (outH>=1);
        assert (inOffX>=1);
        assert (inOffY>=1);

        this.base         = base;
        this.outWidth     = outW;
        this.outHeight    = outH;
        this.inputOffsetX = inOffX;
        this.inputOffsetY = inOffY;

        // Output of the convolution is number of AE(x axis) x AE(y axis) x AE.depth()
        output = new DataBlock(outWidth, outHeight, this.base.getOutputDepth());
        this.base.setOutput(output, 0, 0);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Computing
    ///////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * Encodes the input area and outputs the result to the output block.
     * Maybe this part could be multi-threaded :-)
     */
    public void encode() {
        for (int ox = 0; ox < outWidth; ox++) {
            int ix = inputX + ox * getInputOffsetX();
            for (int oy = 0; oy < outHeight; oy++) {
                int iy = inputY + oy * getInputOffsetY();
                base.setInput(input, ix, iy);
                base.setOutput(output, ox, oy);
                base.encode();
            }
        }
        // Reset the output to initial position. This is necessary for saving/loading AE correctly
        base.setOutput(output, 0, 0);
    }

    /**
     * Decodes the encoded data and pastes it onto the inputs, you can specify
     * if the inputs have to be cleared before.
     * @param clearInputs must the input be cleared ?
     */
    public void rebuildInput(boolean clearInputs) {
        if (clearInputs) {
            input.clear();
        }

        for (int ox=0; ox<outWidth; ox++) {
            int ix = inputX + ox*getInputOffsetX();
            for (int oy=0; oy<outHeight; oy++) {
                int iy = inputY + oy*getInputOffsetY();

                base.setOutput(output, ox, oy);
                base.decode();
                base.pasteDecoded(input, ix, iy);
            }
        }
        if (clearInputs) {
            input.normalizeWeights();
        }
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Learning
    ///////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * Trains the autoencoder.
     *
     * @return the training error
     */
    public float train() {
        float err = 0.0f;
        for (int ox = 0; ox < outWidth; ox++) {
            int ix = inputX + ox * getInputOffsetX();
            for (int oy = 0; oy < outHeight; oy++) {
                int iy = inputY + oy * getInputOffsetY();
                base.setInput(input, ix, iy);
                base.setOutput(output, ox, oy);
                err += base.train();
            }
        }
        return (err / outWidth) / outHeight;
    }

    /**
     * Trains the autoencoder in a supervised fashion
     * @param label expected label number
     * @return the training error
     */
    public float train(int label) {
        float err = 0.0f;
        for (int ox = 0; ox < outWidth; ox++) {
            int ix = inputX + ox * getInputOffsetX();
            for (int oy = 0; oy < outHeight; oy++) {
                int iy = inputY + oy * getInputOffsetY();
                base.setInput(input, ix, iy);
                base.setOutput(output, ox, oy);
                err += ((SupervisedAutoEncoder) base).train(label);
            }
        }
        return (err / outWidth) / outHeight;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Getters & Setters
    ///////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * @return the input data block
     */
    public DataBlock getInput() {
        return input;
    }

    /**
     * Sets the input block of the convolution.
     * @param db data block
     */
    public void setInput(DataBlock db) {
        assert (db != null);

        this.input = db;
        inputX = 0;
        inputY = 0;
        base.setInput(db, inputX, inputY);
    }

    /**
     * Sets the input block and position.
     * @param db DataBlock
     * @param x new position x on the input
     * @param y new position y on the input
     */
    public void setInput(DataBlock db, int x, int y) {
        assert (db != null);
        assert (x >= 0);
        assert (y >= 0);
        assert (x + getInputPatchWidth() <= db.getWidth());
        assert (y + getInputPatchHeight() <= db.getHeight());
        assert (getInputPatchDepth() == db.getDepth());

        this.input = db;
        inputX = x;
        inputY = y;
        base.setInput(db, inputX, inputY);
    }

    /**
     * @return the inputOffsetX
     */
    public int getInputOffsetX() {
        return inputOffsetX;
    }

    /**
     * @return the inputOffsetY
     */
    public int getInputOffsetY() {
        return inputOffsetY;
    }

    /**
     * @return the width of the input patch.
     */
    public int getInputPatchWidth() {
        return (outWidth-1)*getInputOffsetX() + base.getInputWidth();
    }

    /**
     * @return the height of the input patch.
     */
    public int getInputPatchHeight() {
        return (outHeight-1)*getInputOffsetY() + base.getInputHeight();
    }

    /**
     * @return the depth of the input patch.
     */
    public int getInputPatchDepth() {
        return base.getInputDepth();
    }

    /**
     * Thew whole input patch is reconstructed and concatenated. The concatenation way
     * is z,y,x. This means that in an array you have [p_0,0,red][p_0,0,green][p_0,0,blue][p_,0,1,red][p_0,1,green][p_0,1,blue],...
     * @return the WHOLE input patch, considering the convolution of the base
     */
    public float[] getInputPatch() {
        return input.patchToArray(inputX,inputY,getInputPatchWidth(),getInputPatchHeight());
    }

    /**
     * @return the width of the output.
     */
    public int getOutputWidth() {
        return outWidth;
    }

    /**
     * @return the height of the output.
     */
    public int getOutputHeight() {
        return outHeight;
    }

    /**
     * @return the depth of the output.
     */
    public int getOutputDepth() {
        return output.getDepth();
    }

    /**
     * @return the output data block
     */
    public DataBlock getOutput() {
        return output;
    }

    /**
     * @return the autoencoder
     */
    public AutoEncoder getBase() {
        return base;
    }

    /**
     * Changes the autoencoder - use it with care.
     *
     * @param ae new autoencoder
     */
    public void setAE(AutoEncoder ae) {
        base = ae;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Utility
    ///////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * Changes the size of the convolution. The output DataBlock will be recreated,
     * therefore all objects making use of it will need to update their reference.
     * @param mW new width
     * @param mH new height
     */
    public void resize(int mW, int mH) {
        this.outWidth = mW;
        this.outHeight = mH;

        output = new DataBlock(outWidth, outHeight, base.getOutputDepth());
        base.setOutput(output, 0, 0);
        base.setError(new DataBlock(outWidth, outHeight, base.getOutputDepth()));
    }

    /**
     * Copies the features at the center of the convolution on the feature vector,
     * at the given position.
     *
     * @param featureVector feature vector
     * @param pos           position of the first feature
     */
    void fillFeatureVector(float[] featureVector, int pos) {
        int cx = outWidth / 2;
        int cy = outHeight / 2;
        for (int n = 0; n < output.getDepth(); n++) {
            featureVector[pos + n] = output.getValue(n, cx, cy);
        }
    }


    void deleteFeatures(int[] number) {
        if (outWidth != 1 || outHeight != 1) {
            throw new IllegalStateException("Cannot delete features of convolved convolutions");
        }
        base.deleteFeatures(number);
        output = new DataBlock(output.getWidth(), output.getHeight(), output.getDepth() - number.length);
        base.setOutput(output, 0, 0);
    }

    /**
     * @return a string indicating the structure of the convolution
     */
    @Override
    public String toString() {
        return base.toString()+"+"+inputOffsetX+"+"+inputOffsetY;
    }

    @Override
    public void startTraining() {
        base.startTraining();
        isTraining = true;
    }

    @Override
    public void stopTraining() {
        base.stopTraining();
        isTraining = false;
    }

    @Override
    public boolean isTraining() {
        return isTraining;
    }


}

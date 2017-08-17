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

import diuf.diva.dia.ms.util.DataBlock;

/**
 * The max pooler is often used in convolutional neural networks. It computes,
 * for a patch, the maximum value of each channel.
 * @author Mathias Seuret
 */
public class MaxPooler extends AutoEncoder {

    /**
     * Set to true when the network is being trained.
     */
    protected boolean isTraining = false;
    
    /**
     * Constructs a max pooler. The number of outputs corresponds to
     * the input patch depth.
     * @param inputWidth input patch width
     * @param inputHeight input patch height
     * @param inputDepth input patch depth
     */
    public MaxPooler(int inputWidth, int inputHeight, int inputDepth) {
        super(inputWidth, inputHeight, inputDepth, inputDepth);
    }

    /**
     * Computes the max in a patch.
     */
    @Override
    public void encode() {
        for (int z=0; z<inputDepth; z++) {
            float max = Float.NEGATIVE_INFINITY;
            for (int ox=0; ox<inputWidth; ox++) {
                for (int oy=0; oy<inputHeight; oy++) {
                    max = Math.max(max, input.getValue(z, inputX+ox, inputY+oy));
                }
            }
            output.getValues(outputX, outputY)[z] = max;
        }
    }

    @Override
    public void decode() {
        int n = 0;
        for (int x=0; x<inputWidth; x++) {
            for (int y=0; y<inputHeight; y++) {
                for (int z=0; z<inputDepth; z++) {
                    decoded[n++] = output.getValues(outputX, outputY)[z];
                }
            }
        }
    }

    @Override
    public float train() {
        throw new Error("Max poolers cannot be trained");
    }

    @Override
    public void learn() {
        // Nothing to do
    }

    /**
     * Backpropagate the error, if needed.
     * @return the mean absolute error of the top layer
     */
    @Override
    public float backPropagate() {
        float sum = 0;
        float[] e = this.error.getValues(outputX, outputY);
        for (float f : e) {
            sum += Math.abs(f);
        }

        if (prevErr==null) {
            return sum / outputDepth;
        }

        for (int z=0; z<inputDepth; z++) {
            int maxPx = 0;
            int maxPy = 0;
            for (int ox=0; ox<inputWidth; ox++) {
                for (int oy=0; oy<inputHeight; oy++) {
                    if (input.getValue(z, ox+inputX, oy+inputY)>input.getValue(z, maxPx+inputX, maxPy+inputY)) {
                        maxPx = ox;
                        maxPy = oy;
                    }
                }
            }
            prevErr.addValue(
                    z,
                    maxPx+inputX,
                    maxPy+inputY,
                    e[z]
            );
        }
        
        return sum / outputDepth;
    }

    /**
     * Sets the input, based on the coordinates of the top-left corner of
     * the input patch.
     *
     * @param db input DataBlock
     * @param x  position x
     * @param y  position y
     */
    @Override
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
        inputPatchToArray(inputArray);
    }

    /**
     * Sets the output location. Coordinates are based on the top-left corner
     * of the input patch.
     *
     * @param db output DataBlock
     * @param x  position x
     * @param y  position y
     */
    @Override
    public void setOutput(DataBlock db, int x, int y) {
        assert (db != null);
        assert (db.getDepth() == outputDepth);
        assert (x < db.getWidth());
        assert (y < db.getHeight());

        // Assign the new output
        output = db;
        outputX = x;
        outputY = y;
    }

    /**
     * Sets the previous error to use.
     *
     * @param db previous error data block
     */
    @Override
    public void setPrevError(final DataBlock db) {
        assert (db != null);
        assert (input.getWidth() == db.getWidth());
        assert (input.getHeight() == db.getHeight());
        assert (inputDepth == db.getDepth());

        prevErr = db;
    }

    /**
     * Sets the error to use.
     *
     * @param db error data block
     */
    @Override
    public void setError(DataBlock db) {
        assert (db != null);
        assert (output.getWidth() == db.getWidth());
        assert (output.getHeight() == db.getHeight());
        assert (outputDepth == db.getDepth());

        error = db;
    }

    /**
     * Clear the error of the encoder and decoder
     */
    @Override
    public void clearError() {
        error.clear();
        prevErr.clear();
    }

    @Override
    public String getTypeName() {
        return "[MAX]";
    }

    @Override
    public AutoEncoder clone() {
        return new MaxPooler(inputWidth, inputHeight, inputDepth);
    }

    @Override
    public void startTraining() {
        isTraining = true;
    }

    @Override
    public void stopTraining() {
        isTraining = false;
    }

    @Override
    public boolean isTraining() {
        return isTraining;
    }

    @Override
    public void clearGradient() {
        // Nothing to do
    }

}

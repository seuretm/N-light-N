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


public interface Layer {

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Computing
    ///////////////////////////////////////////////////////////////////////////////////////////////
    void compute();

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Learning
    ///////////////////////////////////////////////////////////////////////////////////////////////

    void setExpected(int pos, float expectedValue);

    /**
     * Backpropagate the unit's error.
     * @return the unit's error
     */
    float backPropagate();

    /**
     * Applies the gradient descent.
     */
    void learn();

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Input related
    ///////////////////////////////////////////////////////////////////////////////////////////////
    int getInputSize();

    float[] getInputArray();

    void setInputArray(float[] inputArray);

    void deleteInput(int num);

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Output related
    ///////////////////////////////////////////////////////////////////////////////////////////////
    int getOutputSize();

    float[] getOutputArray();

    void setOutputArray(float[] inputArray);

    void deleteOutput(int num);

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Error related
    ///////////////////////////////////////////////////////////////////////////////////////////////
    float[] getPreviousError();

    void setPreviousError(float[] prevError);

    void clearPreviousError();

    float[] getError();

    void setError(float[] error);

    void clearError();

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Getters&Setters
    ///////////////////////////////////////////////////////////////////////////////////////////////
    float[][] getWeights();

    void setWeights(float[][] w);

    float[] getBias();

    void setBias(float[] b);

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Utility
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * It is fundamental that the layer can be correctly cloned, otherwise
     * AEs cannot properly clone themselves.
     * @return a full copy of the Layer
     */
    Layer clone();
}

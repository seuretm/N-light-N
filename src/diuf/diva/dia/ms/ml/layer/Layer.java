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
    /**
     * Computes the output of the layer and stores it in the right array.
     */
    void compute();

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Learning
    ///////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * Sets the expected value for a given output.
     * @param outputNum output number
     * @param expectedValue value that should ideally have been outputted
     */
    void setExpected(int outputNum, float expectedValue);

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
    /**
     * @return the number of values composing the input
     */
    int getInputSize();

    /**
     * @return the array that has to be used as input
     */
    float[] getInputArray();

    /**
     * Sets which array has to be used as input.
     * @param inputArray the new array
     */
    void setInputArray(float[] inputArray);

    /**
     * Sets the learning speed if this feature is supported by the layer.
     * @param s new learning speed
     */
    void setLearningSpeed(float s);
    
    /**
     * Experimental, not supported by all layers. Deletes an input from the
     * layer - useful if an output of the previous layer has been removed.
     * @param num input number
     */
    void deleteInput(int num);

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Output related
    ///////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * @return the number of values composing the output
     */
    int getOutputSize();

    /**
     * @return the array used as output
     */
    float[] getOutputArray();

    /**
     * Indicates which array has to be used as output.
     * @param outputArray the new array
     */
    void setOutputArray(float[] outputArray);

    /**
     * Experimental, not supported by all layers. Deletes an output.
     * @param num number of the output
     */
    void deleteOutput(int num);

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Error related
    ///////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * @return the error array that will be sent by the AE to the previous layer
     */
    float[] getPreviousError();

    /**
     * Indicates to which array errors have to be transmitted when backpropagating.
     * @param prevError the new array
     */
    void setPreviousError(float[] prevError);

    /**
     * Erases the content of the array storing errors for the previous layer.
     */
    void clearPreviousError();

    /**
     * @return the error array of this layer
     */
    float[] getError();

    /**
     * Replaces the error array of this layer.
     * @param error new array
     */
    void setError(float[] error);

    /**
     * Resets the error of this layer.
     */
    void clearError();

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Getters&Setters
    ///////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * If the layer is based on weights/bias, this method should
     * return the weights. Otherwise it should throw an error.
     * @return the weight array
     */
    float[][] getWeights();

    /**
     * If the layer is based on weights/bias, this method should set
     * the weights, otherwise it should throw an error.
     * @param w new weight array
     */
    void setWeights(float[][] w);

    /**
     * If the layer is based on weights/bias, this method should
     * return the bias. Otherwise it should throw an error.
     * @return the bias array
     */
    float[] getBias();

    /**
     * If the layer is based on weights/bias, this method should set
     * the bias, otherwise it should throw an error.
     * @param b new bias array
     */
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

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

import diuf.diva.dia.ms.util.Random;

/**
 * This layer is based on a correlation computation, instead of the usual dot
 * product followed by an activation function.
 * @author Mathias Seuret
 */
public class Correlator implements Layer {
    /**
     * Set to true during training phases.
     */
    protected boolean isTraining = false;
    /**
     * Learned data representation.
     */
    float[][] y;
    /**
     * Stores gradients.
     */
    float[][] gY;
    /**
     * Mean value of the data representation.
     */
    float[] meanY;
    /**
     * Output of the layer.
     */
    float[] output;
    /**
     * Error of the layer.
     */
    float[] error;
    /**
     * Input of the layer.
     */
    float[] input;
    /**
     * Ref. to the error of the previous layer - might be null.
     */
    float[] prevError;
    /**
     * Number of inputs.
     */
    int inputSize;
    /**
     * Number of outputs.
     */
    int outputSize;
    /**
     * Learning speed. As this is a new kind of neuron, we have
     * no knowledge yet about good values. The same default value
     * as for the NeuralLayer has been used.
     */
    float learningSpeed = 1e-2f;
    
    
    /**
     * Creates a correlator.
     *
     * @param inputArr   a float array
     * @param outputSize the number of elements to learn
     */
    public Correlator(float[] inputArr, int outputSize) {
        this(inputArr, inputArr.length, outputSize);
    }

    /**
     * Creates a correlator.
     *
     * @param inputArr   a float array
     * @param inputSize  size of the input array
     * @param outputSize the number of elements to learn
     */
    public Correlator(float[] inputArr, int inputSize, int outputSize) {
        this(inputArr, inputSize, outputSize, null, null);
    }

    /**
     * Constructs a correlator with all the parameters needed.
     *
     * @param inputArray input array
     * @param inputSize  size of the input array
     * @param outputSize size of the output array
     * @param y elements to learn - initialized randomly if empty
     * @param bias needed for interface reasons but should be null as no bias are used by the correlator.
     */
    public Correlator(float[] inputArray, int inputSize, int outputSize, float[][] y, float[] bias) {
        assert (bias==null);
        
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        
        output   = new float[outputSize];
        error    = new float[outputSize];
        
        meanY = new float[outputSize];
        this.y = new float[outputSize][inputSize];
        gY = new float[outputSize][inputSize];
        for (int o=0; o<outputSize; o++) {
            for (int i=0; i<inputSize; i++) {
                this.y[o][i] = (y==null) ? Random.nextFloat() : y[o][i];
            }
            meanY[o] = mean(this.y[o]);
        }
        
    }

    /**
     * Computes the mean value of an array.
     *
     * @param v array
     * @return mean value of v
     */
    private static float mean(float[] v) {
        float sum = 0;
        for (float f : v) {
            sum += f;
        }
        return sum / v.length;
    }

    /**
     * Computes the output of the correlator.
     */
    @Override
    public void compute() {
        float meanX = mean(input);
        for (int o=0; o<outputSize; o++) {
            float sigmaXY = 0;
            float sigmaX2 = 0;
            float sigmaY2 = 0;
            float meanY = mean(y[o]);
            for (int i=0; i<inputSize; i++) {
                sigmaXY += (input[i]-meanX) * (y[o][i]-meanY);
                sigmaX2 += (input[i]-meanX) * (input[i]-meanX);
                sigmaY2 += (y[o][i]-meanY) * (y[o][i]-meanY);
            }
            float bot = (float)Math.sqrt(sigmaX2 * sigmaY2);
            output[o] = (bot==0.0f) ? 0 : sigmaXY / bot;
        }
    }
    
    /**
     * Sets the expected value.
     * @param o output number
     * @param v expected value
     */
    @Override
    public void setExpected(int o, float v) {
        float e = output[o] - v;
        addError(o, e);
    }
    
    /**
     * Increases the error of an output.
     *
     * @param o output number
     * @param e error to add
     */
    public void addError(int o, float e) {
        error[o] += e;
    }

    /**
     * Computes the gradient and backpropagates it through the network.
     * @return the mean error of the outputs
     */
    @Override 
    public float backPropagate() {
        float errSum = 0;
        float meanX = mean(input);
        
        float sigmaX = 0;
        for (int i=0; i<inputSize; i++) {
            sigmaX += (input[i]-meanX) * (input[i]-meanX);
        }
        if (sigmaX==0.0f) {
            for (int o=0; o<outputSize; o++) {
                errSum += Math.abs(error[o]);
            }
            return errSum; // by def nothing can be done
        }
        float rtSigmaX = (float)Math.sqrt(sigmaX);
        
        for (int o=0; o<outputSize; o++) {
            errSum += Math.abs(error[o]);
            
            float sigmaXY = 0;
            float sigmaY = 0;
            for (int i=0; i<inputSize; i++) {
                sigmaXY += (input[i]-meanX) * (y[o][i]-meanY[o]);
                sigmaY += (y[o][i]-meanY[o]) * (y[o][i]-meanY[o]);
            }
            
            if (sigmaY==0.0f) {
                continue; // by def
            }
            
            float rtSigmaXSigmaY = (float)Math.sqrt(sigmaX*sigmaY);
            float rtSigmaY  = (float)Math.sqrt(sigmaY);
            
            for (int i=0; i<inputSize; i++) {
                gY[o][i] += error[o] * (rtSigmaXSigmaY * (input[i]-meanX) - sigmaXY * rtSigmaX / rtSigmaY * (y[o][i] - meanY[o])) / (sigmaX * sigmaY);
                if (prevError==null) {
                    continue;
                }
                prevError[i] += error[o] * y[o][i] * (rtSigmaXSigmaY * (y[o][i]-meanY[o]) - sigmaXY * rtSigmaY / rtSigmaX * (input[i]-meanX)) / (sigmaX * sigmaY);
            }
        }
        return errSum;
    }

    /**
     * Applies the gradient descents.
     */
    @Override
    public void learn() {
        float max = 0;
        for (int o=0; o<outputSize; o++) {
            for (int i=0; i<inputSize; i++) {
                y[o][i] -= learningSpeed * gY[o][i];
                gY[o][i] = 0;
                if (Math.abs(y[o][i])>max) {
                    max = Math.abs(y[o][i]);
                }
            }
            meanY[o] = mean(y[o]);
        }
        if (max==0.0f) {
            return;
        }
        for (int o=0; o<outputSize; o++) {
            for (int i=0; i<inputSize; i++) {
                y[o][i] /= max;
            }
        }
    }

    /**
     * @return the number of inputs
     */
    @Override
    public int getInputSize() {
        return inputSize;
    }

    /**
     * @return a reference to the input array
     */
    @Override
    public float[] getInputArray() {
        return input;
    }

    /**
     * Changes the input array.
     * @param inputArray new array
     */
    @Override
    public void setInputArray(float[] inputArray) {
        input = inputArray;
    }

    /**
     * @return the current learning speed
     */
    public float getLearningSpeed() {
        return learningSpeed;
    }

    /**
     * Updates the learning speed.
     * @param s new learning speed
     */
    @Override
    public void setLearningSpeed(float s) {
        learningSpeed = s;
    }

    /**
     * Not implemented yet
     * @param num number of the input to delete
     */
    @Override
    public void deleteInput(int num) {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    /**
     * @return the number of outputs
     */
    @Override
    public int getOutputSize() {
        return outputSize;
    }

    /**
     * @return the output array
     */
    @Override
    public float[] getOutputArray() {
        return output;
    }

    /**
     * Changes the output array.
     * @param outputArray new array
     */
    @Override
    public void setOutputArray(float[] outputArray) {
        output = outputArray;
    }

    /**
     * Not implemented yet.
     * @param num output number
     */
    @Override
    public void deleteOutput(int num) {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    /**
     * @return a reference to the error array of the previous layer - might be null
     */
    @Override
    public float[] getPreviousError() {
        return prevError;
    }

    /**
     * Indicates where's the error of the previous layer
     * @param prevError reference
     */
    @Override
    public void setPreviousError(float[] prevError) {
        this.prevError = prevError;
    }

    /**
     * Erases the previous error array.
     */
    @Override
    public void clearPreviousError() {
        if (prevError==null) {
            return;
        }
        for (int i=0; i<prevError.length; i++) {
            prevError[i] = 0;
        }
    }

    /**
     * @return the error array of this layer
     */
    @Override
    public float[] getError() {
        return error;
    }

    /**
     * Tells the neural layer which array should be
     * used for storing errors.
     *
     * @param err an array
     */
    @Override
    public void setError(float[] err) {
        error = err;
    }

    /**
     * Resets the error of this layer.
     */
    @Override
    public void clearError() {
        for (int i=0; i<error.length; i++) {
            error[i] = 0;
        }
    }

    /**
     * @return the learned data
     */
    @Override
    public float[][] getWeights() {
        return y;
    }

    /**
     * Sets the learned data.
     * @param w new values
     */
    @Override
    public void setWeights(float[][] w) {
        y = w.clone();
        for (int i=0; i<y.length; i++) {
            y[i] = w[i].clone();
        }
    }

    /**
     * @return nothing as the Correlator has no bias
     */
    @Override
    public float[] getBias() {
        return null;
    }

    /**
     * Does nothing as the Correlator has no bias.
     * @param b should be null
     */
    @Override
    public void setBias(float[] b) {
        // nothing to do
    }
    
    /**
     * @return a copy of the Correlator
     */
    @Override
    public Layer clone() {
        return new Correlator(input, inputSize, outputSize, y, null);
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
        for (int i=0; i<inputSize; i++) {
            for (int o=0; o<outputSize; o++) {
                gY[i][o] = 0;
            }
        }
    }
}

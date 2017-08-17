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
import java.io.Serializable;

/**
 * This layer is based on a correlation computation, instead of the usual dot
 * product followed by an activation function.
 * @author Mathias Seuret
 */
public class Correlator2 implements Layer, Serializable {
    /**
     * Learned patterns.
     */
    float[][] p;
    
    /**
     * Stores gradients for the patterns.
     */
    float[][] gP;
    
    /**
     * Learned weights.
     */
    float[][] w;
    
    /**
     * Stores gradients for the weights.
     */
    float[][] gW;
    
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
    float learningSpeed = 1e-3f;
    float weightingSpeed = 1e-3f;
    
    /**
     * True during training phases.
     */
    boolean isTraining = false;
    
    
    /**
     * Creates a correlator.
     *
     * @param inputArr   a float array
     * @param outputSize the number of elements to learn
     */
    public Correlator2(float[] inputArr, int outputSize) {
        this(inputArr, inputArr.length, outputSize);
    }

    /**
     * Creates a correlator.
     *
     * @param inputArr   a float array
     * @param inputSize  size of the input array
     * @param outputSize the number of elements to learn
     */
    public Correlator2(float[] inputArr, int inputSize, int outputSize) {
        this(inputArr, inputSize, outputSize, null, null);
    }

    /**
     * Constructs a correlator with all the parameters needed.
     *
     * @param inputArray input array
     * @param inputSize  size of the input array
     * @param outputSize size of the output array
     * @param p elements to learn - initialized randomly if empty
     * @param bias needed for interface reasons but should be null as no bias are used by the correlator.
     */
    public Correlator2(float[] inputArray, int inputSize, int outputSize, float[][] p, float[] bias) {
        assert (bias==null);
        
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        
        output   = new float[outputSize];
        error    = new float[outputSize];
        
        this.p = new float[outputSize][inputSize];
        gP = new float[outputSize][inputSize];
        for (int o=0; o<outputSize; o++) {
            for (int i=0; i<inputSize; i++) {
                this.p[o][i] = (p==null) ? Random.nextFloat() : p[o][i];
            }
        }
        
        w = new float[outputSize][inputSize];
        gW = new float[outputSize][inputSize];
        
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
            float sigXPW = 0;
            float sigXW2 = 0;
            float sigPW2 = 0;
            for (int i=0; i<inputSize; i++) {
                float ew = fw(w[o][i]);
                sigXPW += (input[i]-meanX) * p[o][i] * ew;
                sigXW2 += ((input[i]-meanX) * ew) * ((input[i]-meanX) * ew);
                sigPW2 += p[o][i] * p[o][i] * ew * ew;
            }
            float bot = (float)Math.sqrt(sigXW2 * sigPW2);
            output[o] = (bot==0.0f) ? 0 : sigXPW / bot;
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
        
        for (int o=0; o<outputSize; o++) {
            errSum += Math.abs(error[o]);
            float sigXPW = 0;
            float sigXW2 = 0;
            float sigPW2 = 0;
            for (int i=0; i<inputSize; i++) {
                float fwj = fw(w[o][i]);
                sigXPW += (input[i]-meanX) * p[o][i] * fwj;
                sigXW2 += ((input[i]-meanX) * fwj) * ((input[i]-meanX) * fwj);
                sigPW2 += p[o][i] * p[o][i] * fwj * fwj;
            }
            float fact = (sigXW2 * sigPW2);
            if (fact==0.0f) {
                continue;
            }
            float bot  = 1.0f / fact;
            float root = (float)Math.sqrt(fact);
            
            
            // Note: if another function than exp(w[o][j]), the first fwj in dwj must be replaced by
            // its derivative, as well as the fwj i nthe prevError computation
            for (int j=0; j<inputSize; j++) {
                float fwj = fw(w[o][j]);
                float pwj = fpw(w[o][j]);
                float dpj = bot * (root * (input[j]-meanX)*fwj - sigXPW / root * sigXW2 * p[o][j] * fwj * fwj);
                float dwj = bot * pwj * (root * (input[j]-meanX)*p[o][j] - sigXPW * fwj / root * (sigXW2*p[o][j]*p[o][j] + sigPW2*(input[j]-meanX)*(input[j]-meanX)));
                
                gP[o][j] += dpj * error[o];
                gW[o][j] += dwj * error[o];
                
                if (prevError!=null && !Float.isNaN(root)) {
                    float dxj = bot * (root * p[o][j] * fwj - sigXPW / root * sigPW2 * (input[j] - meanX) * fwj * fwj);
                    float b =  dxj * fwj * p[o][j] * error[o];
                    prevError[j] += b;
                }
            }
        }
        
        return errSum;
    }
    
    protected float fw(float w) {
        return 1.0f / (1.0f + (float)Math.exp(-w));
    }
    
    protected float fpw(float w) {
        float y = fw(w);
        return y * (1.0f - y);
    }

    /**
     * Applies the gradient descents.
     */
    @Override
    public void learn() {
        float max = 0;
        for (int o=0; o<outputSize; o++) {
            for (int i=0; i<inputSize; i++) {
                p[o][i] -= learningSpeed * gP[o][i];
                w[o][i] -= weightingSpeed * gW[o][i];
                gP[o][i] = 0;
                gW[o][i] = 0;
                max = Math.max(max, Math.abs(p[o][i]));
            }
        }
        if (max==0.0f) {
            return;
        }
        for (int o=0; o<outputSize; o++) {
            for (int i=0; i<inputSize; i++) {
                p[o][i] /= max;
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
     * Updates the learning speed.
     * @param s new learning speed
     */
    public void setWeightingSpeed(float s) {
        weightingSpeed = s;
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
        return p;
    }
    
    public float[][] getImportance() {
        return w;
    }

    /**
     * Sets the learned data.
     * @param w new values
     */
    @Override
    public void setWeights(float[][] w) {
        p = w.clone();
        for (int i=0; i<p.length; i++) {
            p[i] = w[i].clone();
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
        return new Correlator2(input, inputSize, outputSize, p, null);
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
                gP[i][o] = 0;
                gW[i][o] = 0;
            }
        }
    }
}

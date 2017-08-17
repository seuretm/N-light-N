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

import java.util.Random;

/**
 * This is a simple neural layer class using backpropagation and a soft-sign
 * activation function.
 * @author Mathias Seuret, Michele Alberti
 */
public class SoftSign extends AbstractLayer {
    
    /**
     * Stores which neurons are active for the dropout.
     */
    protected boolean[] active;
    
    /**
     * Value in [0, 1[, indicates which fraction of the neurons
     * are dropped out during training.
     */
    private float dropoutRate = 0.0f;
    
    /**
     * SetRandom number generator for selecting which
     * neurons are dropped out.
     */
    private Random rand = new Random();
    
    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Constructor
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * Creates a neural layer.
     * @param inputArr a float array
     * @param outputSize the number of neurons
     */
    public SoftSign(float[] inputArr, int outputSize) {
        this(inputArr, inputArr.length, outputSize);
    }

    /**
     * Creates a neural layer.
     * @param inputArr   a float array
     * @param inputSize  size of the input array
     * @param outputSize the number of neurons
     */
    public SoftSign(float[] inputArr, int inputSize, int outputSize) {
        this(inputArr, inputSize, outputSize, null, null);
    }

    /**
     * Constructs a neural layer with all the parameters needed.
     * @param inputArray input array
     * @param inputSize size of the input array
     * @param outputSize size of the output array
     * @param weight of the layer - if null an array is created
     * @param bias of the layer - if null an array is created
     */
    public SoftSign(float[] inputArray, int inputSize, int outputSize, float[][] weight, float[] bias) {
        super(inputArray, inputSize, outputSize, weight, bias);
        active = new boolean[outputSize];
        setDropoutRate(0.5f);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Computing
    ///////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * Computes the output of the layer.
     */
    @Override
    public void compute() {
        if (isTraining()) {
            computeDO();
            return;
        }
        for (int o = 0; o < outputSize; o++) {
            wSum[o] = bias[o];
            for (int i = 0; i < inputSize; i++) {
                wSum[o] += weight[i][o] * input[i];
            }
            wSum[o] *= (1.0f-dropoutRate);
            output[o] = wSum[o] / (1 + Math.abs(wSum[o]));
        }
    }
    
    /**
     * Computes the output of the layer taking into account
     * the dropout.
     */
    protected void computeDO() {
        for (int o = 0; o < outputSize; o++) {
            if (!active[o]) {
                output[o] = 0;
                wSum[o] = 0;
                continue;
            }
            wSum[o] = bias[o];
            for (int i = 0; i < inputSize; i++) {
                wSum[o] += weight[i][o] * input[i];
            }
            output[o] = wSum[o] / (1 + Math.abs(wSum[o]));
        }
    }
    
    protected void randomizeActivatedNeurons() {
        for (int o=0; o<outputSize; o++) {
            int p = rand.nextInt(outputSize);
            boolean b = active[o];
            active[o] = active[p];
            active[p] = b;
        }
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Learning
    ///////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * Applies the gradient descent.
     */
    @Override
    public void learn() {
        for (int o = 0; o < outputSize; o++) {
            for (int i = 0; i < inputSize; i++) {
                if (active[o]) {
                    weight[i][o] = (1.0f-decay)*weight[i][o] - learningSpeed * gradient[i][o];
                }
                gradient[i][o] = 0.0f;
            }
            if (active[o]) {
                bias[o] = (1.0f-decay)*bias[o] - learningSpeed * biasGradient[o];
            }
            biasGradient[o] = 0.0f;
        }
        randomizeActivatedNeurons();
    }

    /**
     * Applies the backpropagation if needed.
     * @return the mean average error of the top layer
     */
    @Override
    public float backPropagate() {
        assert (isTraining);
        
        float errSum = 0.0f;
        // It does not look nice, but it decreases MUCH the number
        // of conditions executed - I don't think the Java compiler
        // would do this
        if (prevErr==null) {
            for (int o = 0; o < outputSize; o++) {
                errSum += Math.abs(err[o]);
                if (!active[o]) {
                    continue;
                }
                float bot = 1 + Math.abs(wSum[o]);
                float fact = 1 / (bot * bot) * err[o];
                for (int i = 0; i < inputSize; i++) {
                    gradient[i][o] += fact * input[i];
                }
                biasGradient[o] += fact;
            }
        } else {
            for (int o = 0; o < outputSize; o++) {
                errSum += Math.abs(err[o]);
                if (!active[o]) {
                    continue;
                }
                float bot = 1 + Math.abs(wSum[o]);
                float fact = 1 / (bot * bot) * err[o];
                for (int i = 0; i < inputSize; i++) {
                    gradient[i][o] += fact * input[i];
                    prevErr[i] += fact * weight[i][o];
                }
                biasGradient[o] += fact;
            }
        }
        
        return errSum / outputSize;
    }
    
    /**
     * Indicates the desired dropout rate. As this might not be reached
     * because of the number of available neurons, the reached
     * rate is returned.
     * @param rate desired dropout rae
     * @return reached dropout rate
     */
    public float setDropoutRate(float rate) {
        assert (rate>=0);
        assert (rate<1);
        
        int nbKept = Math.max(1, Math.round(outputSize*(1-rate)));
        for (int i=0; i<outputSize; i++) {
            active[i] = (i<nbKept);
        }
        
        dropoutRate = 1.0f - nbKept / (float)outputSize;
        return dropoutRate;
    }
    
    /**
     * @return the dropout rate
     */
    public float getDropoutRate() {
        return dropoutRate;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Utility
    ///////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * Typical clone methods
     * @return a deep copy of the object
     */
    @Override
    public Layer clone() {
        SoftSign ss = new SoftSign(
                input,
                inputSize,
                outputSize,
                weight,
                bias
        );
        ss.setDropoutRate(dropoutRate);
        return ss;
    }
}

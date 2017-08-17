/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package diuf.diva.dia.ms.ml.layer;

/**
 *
 * @author Mathias Seuret
 */
public class SoftMax implements Layer {
    protected int inputSize;
    float[] input;
    float[] output;
    float[] error;
    float[] prevErr;
    float[] exp;
    float eSum;
    boolean isTraining = false;
    int expectedClass = -1;
    
    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Constructor
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * Creates a neural layer.
     * @param inputArr a float array
     * @param outputSize the number of neurons
     */
    public SoftMax(float[] inputArr, int outputSize) {
        this(inputArr, inputArr.length, outputSize);
    }

    /**
     * Creates a soft max.
     * @param inputArr   a float array
     * @param inputSize  size of the input array
     * @param outputSize the number of neurons
     */
    public SoftMax(float[] inputArr, int inputSize, int outputSize) {
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
    public SoftMax(float[] inputArray, int inputSize, int outputSize, float[][] weight, float[] bias) {
        if (inputSize!=outputSize) {
            throw new Error("SoftMax must have as many outputs as inputs");
        }
        if (weight!=null) {
            throw new Error("SoftMax require null weights");
        }
        if (weight!=null) {
            throw new Error("SoftMax require null bias");
        }
        this.inputSize = inputSize;
        exp    = new float[inputSize];
        input  = new float[inputSize];
        output = new float[inputSize];
        error  = new float[inputSize];
    }
    
    @Override
    public void compute() {
        eSum = 0;
        for (int i=0; i<inputSize; i++) {
            exp[i] = (float)Math.exp(input[i]);
            eSum += exp[i];
        }
        for (int i=0; i<inputSize; i++) {
            output[i] = exp[i] / eSum;
        }
    }

    @Override
    public void setExpected(int outputNum, float expectedValue) {
        error[outputNum] += output[outputNum] - expectedValue;
    }
    
    @Override
    public void setExpectedClass(int classNum) {
        expectedClass = classNum;
        float loss = (float)-Math.log(exp[expectedClass] / eSum);
        for (int i=0; i<inputSize; i++) {
            error[i] = loss;
        }
    }

    @Override
    public float backPropagate() {
        float sum = 0.0f;
        for (int i=0; i<inputSize; i++) {
            float derivative = exp[i] / eSum;
            if (i==expectedClass) {
                derivative -= 1.0f;
            }
            prevErr[i] += derivative * error[i];
            sum += error[i];
            error[i] = 0;
        }
        return sum / inputSize;
    }

    @Override
    public void learn() {
        // nothing to do
    }

    @Override
    public int getInputSize() {
        return inputSize;
    }

    @Override
    public float[] getInputArray() {
        return input;
    }

    @Override
    public void setInputArray(float[] inputArray) {
        this.input = inputArray;
    }

    @Override
    public float getLearningSpeed() {
        return 0;
    }

    @Override
    public void setLearningSpeed(float s) {
        // nothing to do
    }

    @Override
    public void deleteInput(int num) {
        // nothing to do
    }

    @Override
    public int getOutputSize() {
        return inputSize;
    }

    @Override
    public float[] getOutputArray() {
        return output;
    }

    @Override
    public void setOutputArray(float[] outputArray) {
        output = outputArray;
    }

    @Override
    public void deleteOutput(int num) {
        // nothing to do
    }

    @Override
    public float[] getPreviousError() {
        return prevErr;
    }

    @Override
    public void setPreviousError(float[] prevError) {
        this.prevErr = prevErr;
    }

    @Override
    public void clearPreviousError() {
        for (int i=0; i<prevErr.length; i++) {
            prevErr[i] = 0;
        }
    }

    @Override
    public float[] getError() {
        return error;
    }

    @Override
    public void setError(float[] error) {
        this.error = error;
    }

    @Override
    public void clearError() {
        for (int i=0; i<error.length; i++) {
            error[i] = 0;
        }
    }

    @Override
    public float[][] getWeights() {
        return null;
    }

    @Override
    public void setWeights(float[][] w) {
        throw new Error("Cannot set weights in SoftMax");
    }

    @Override
    public float[] getBias() {
        return null;
    }

    @Override
    public void setBias(float[] b) {
        throw new Error("Cannot set bias in SoftMax");
    }

    @Override
    public Layer clone() {
        return new SoftMax(input, inputSize);
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
        // nothing to do
    }
    
}

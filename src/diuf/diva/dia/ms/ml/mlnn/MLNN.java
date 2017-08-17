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

package diuf.diva.dia.ms.ml.mlnn;

import diuf.diva.dia.ms.ml.Trainable;
import diuf.diva.dia.ms.ml.layer.NeuralLayer;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;

/**
 * Multi-layered neural network.
 * @author Mathias Seuret, Alberti Michele
 */
public class MLNN implements Serializable, Trainable {

    /**
     * Layers of the network
     */
    private ArrayList<NeuralLayer> layers = new ArrayList<>();
    /**
     * Top layer (output)
     */
    private NeuralLayer top;
    /**
     * Bottom layer (input)
     */
    private NeuralLayer base;

    /**
     * Input array
     */
    private float[] input;
    /**
     * Number of outputs
     */
    private int nbOutputs;
    
    /**
     * Set to true during training phases.
     */
    protected boolean isTraining = false;

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Constructor
    ///////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * Creates a multi-layered neural network.
     * @param nbInputs number of inputs
     * @param nbOutputs number of outputs
     * @param nbNeurons number of neurons in each hidden
     */
    public MLNN(int nbInputs, int nbOutputs, int... nbNeurons) {
        if (nbNeurons==null || nbNeurons.length<1) {
            throw new Error(
                    "MLNN: requires at least one layer, you gave "
                            + nbNeurons.length
                            + " ("
                            + Arrays.toString(nbNeurons)
                            + ")"
            );
        }
        
        this.nbOutputs = nbOutputs;

        // Add the layers
        for (int i = 0; i < nbNeurons.length; i++) {
            // Set the input array, either by creating a new one or with output of prev layer.
            float[] in = (i == 0) ? new float[nbInputs] : layers.get(i - 1).getOutputArray();
            NeuralLayer layer = new NeuralLayer(in, in.length, nbNeurons[i]);
            layers.add(layer);
        }
        base = layers.get(0);
        
        // Top-layer
        top = new NeuralLayer(
                layers.get(layers.size() - 1).getOutputArray(),
                nbNeurons[nbNeurons.length-1],
                nbOutputs
        );
        layers.add(top);
        
        // Connecting error values
        for (int l=layers.size()-1; l>0; l--) {
            layers.get(l).setPreviousError(layers.get(l-1).getError());
        }

        input = base.getInputArray();
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Computing
    ///////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * Computes the neural network for the current input.
     */
    public void compute() {
        for (NeuralLayer layer : layers) {
            layer.compute();
        }
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Learning
    ///////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * Indicates which output was expected - call this between compute() and backPropagate().
     * @param outputNumber output number for which we want to indicate the expected value
     * @param value expected value
     */
    public void setExpected(int outputNumber, float value) {
        assert (value>=-1);
        assert (value<=1);
        
        top.setExpected(outputNumber, value);
    }

    /**
     * Backpropagates the error and train the layers. All of them.
     * @return the absolute error of the top layer
     */
    public float learn() {
        return learn(layers.size());
    }

    /**
     * Learn the specified amount of layers from the top
     *
     * @param nbLayers how many layers from the top ?
     * @return the absolute error of the top layer
     */
    public float learn(int nbLayers) {
        for (int i = layers.size() - 1; i >= 0 && i >= layers.size() - nbLayers; i--) {
            layers.get(i).learn();
        }
        // TODO temporary as maybe learn() will return void
        return -1;
    }


    /**
     * Backpropagate all layers*
     *
     * @return average of the absolute errors of each output of the top layer
     */
    public float backPropagate() {
        return backPropagate(layers.size());
    }

    /**
     * Backpropagate the specified amount of layers from the top*
     *
     * @param nbLayers how many layers from the top ?
     * @return average of the absolute errors of each output of the top layer
     */
    public float backPropagate(int nbLayers) {

        /* Backpropagate
         * The only reason the top layer is not in the for-loop is because we want to return
         * his error alone.
         */
        float err = layers.get(layers.size() - 1).backPropagate();
        for (int i = layers.size() - 2; i >= 0 && i >= layers.size() - nbLayers; i--) {
            layers.get(i).backPropagate();
        }

        // Clear error in whole network
        for (int i = layers.size() - 1; i >= 0 && i >= layers.size() - nbLayers; i--) {
            layers.get(i).clearError();
            layers.get(i).clearPreviousError();
        }

        return err;
    }
    
    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Output related
    ///////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * Sets the array to use as input
     * @param arr new input array
     */
    public void setInput(float[] arr) {
        base.setInputArray(arr);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Output related
    ///////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * @return the number of outputs
     */
    public int getOutputSize() {
        return nbOutputs;
    }
    
    /**
     * @return the output array - not a copy
     */
    public float[] getOutput() {
        return top.getOutputArray();
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Getters & Setters
    ///////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * @return the number of layers
     */
    public int getLayersCount() {
        return layers.size();
    }
    
    /**
     * @param n layer number
     * @return the layer number n
     */
    public NeuralLayer getLayer(int n) {
        return layers.get(n);
    }

    @Override
    public void startTraining() {
        for (NeuralLayer l : layers) {
            l.startTraining();
        }
        isTraining = true;
    }

    @Override
    public void stopTraining() {
        for (NeuralLayer l : layers) {
            l.stopTraining();
        }
        isTraining = false;
    }

    @Override
    public boolean isTraining() {
        return isTraining;
    }
}

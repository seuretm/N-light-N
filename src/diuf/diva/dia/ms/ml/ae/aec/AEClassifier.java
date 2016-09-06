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

package diuf.diva.dia.ms.ml.ae.aec;

import diuf.diva.dia.ms.ml.Classifier;
import diuf.diva.dia.ms.ml.mlnn.MLNN;
import diuf.diva.dia.ms.ml.ae.scae.SCAE;
import diuf.diva.dia.ms.util.DataBlock;

import java.io.*;

/**
 * This class represents a stacked convolution autoencoder coupled to
 * a classification feed forward neural network. The number of layers of the
 * feed forward neural network can be specified at the object creation.
 * @author Mathias Seuret,Michele Alberti
 */
public class AEClassifier implements Classifier, Serializable {
    /**
     * Reference to the autoencoder.
     */
    protected SCAE scae;
    
    /**
     * Number of classes.
     */
    protected int nbClasses;
    
    /**
     * The neural network on top of the classifier.
     */
    protected MLNN mlnn;

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Constructor
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * Constructor of the class
     * @param ae the autoencoder
     * @param nbClasses the number of classes
     * @param nbNeurons the number of neurons in the different classification layers
     */
    public AEClassifier(final SCAE ae, final int nbClasses, final int... nbNeurons) {
        this.scae = ae;
        this.nbClasses = nbClasses;

        mlnn = new MLNN(ae.getFeatureLength(), nbClasses, nbNeurons);
        mlnn.setInput(scae.getCentralMultilayerFeatures());
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Setting input
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * Selects the input to use.
     * @param db input data block
     * @param x position
     * @param y position
     */
    public void setInput(DataBlock db, int x, int y) {
        scae.setInput(db, x - scae.getInputPatchWidth() / 2, y - scae.getInputPatchHeight() / 2);
    }

    /**
     * Centers the classifier at the given coordinates.
     * @param db data block
     * @param cx center x
     * @param cy center y
     */
    public void centerInput(DataBlock db, int cx, int cy) {
        scae.setInput(db, cx - scae.getInputPatchWidth() / 2, cy - scae.getInputPatchHeight() / 2);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Computing
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * Computes the features and then the classes.
     */
    public void compute() {
        scae.forward();
        mlnn.compute();
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Getting the output/results
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * This methods returns the classification result of the last evaluated input. The classification
     * might be single or multiclass. In case of single class the result will be a normal integer, in
     * case of the multiclass use the result will be an integer whose bits will represent the classes
     * which the output have been assigned to.
     * e.g. Single: res = 5 means output got classified as belonging to class 5
     * e.g. Multi: res = 5 (0..0101) means output got "classified"as belonging to the class three and one).
     *
     * @param  multiClass defines whether or not the result will be multiclass
     * @return the index of the output with the highest value
     */
    public int getOutputClass(boolean multiClass) {
        int res = 0;
        if (multiClass) {
            for (int i = 0; i < mlnn.getOutputSize(); i++) {
                if (mlnn.getOutput()[i] > 0.5f) {
                    res |= (0x01 << i);
                }
            }
        } else {
            for (int i = 0; i < mlnn.getOutputSize(); i++) {
                if (mlnn.getOutput()[i] > mlnn.getOutput()[res]) {
                    res = i;
                }
            }
        }
        return res;
    }

    /**
     * Returns the size of the output. Used mainly for knowing how many maximal different classes are we
     * working with
     *
     * @return the size of the output
     */
    public int getOutputSize() {
        return mlnn.getOutputSize();
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Learning
    ///////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * Indicates what was expected for a given output.
     * @param expectedClass output number which should correspond to the class
     * @param expectedValue expected value for the expected class
     */
    public void setExpected(int expectedClass, float expectedValue) {
        mlnn.setExpected(expectedClass, expectedValue);
    }

    /**
     * Computed the training error and applies the gradient descent for the NeuralLayers.
     * @return the training error
     */
    public float learn() {
        return mlnn.learn(mlnn.getLayersCount());
    }

    /**
     * Computed the training error and applies the gradient descent for the specified
     * amount of NeuralLayers from the top.
     *
     * @return the training error
     */
    public float learn(int nbLayers) {
        return mlnn.learn(nbLayers);
    }

    /**
     * Backpropagate all layers
     * @return average of the absolute errors of each output of the top layer
     */
    public float backPropagate() {
        return mlnn.backPropagate(mlnn.getLayersCount());
    }

    /**
     * Backpropagate the specified amount of layers from the top
     *
     * @param nbLayers how many layers from the top ?
     * @return average of the absolute errors of each output of the top layer
     */
    public float backPropagate(int nbLayers) {
        return mlnn.backPropagate(nbLayers);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Getters
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * @return the input width of the autoencoder
     */
    public int getInputWidth() {
        return scae.getInputPatchWidth();
    }

    /**
     * @return the input height of the autoencoder
     */
    public int getInputHeight() {
        return scae.getInputPatchHeight();
    }

    /**
     * @return the autoencoder
     */
    public SCAE getSCAE() {
        return scae;
    }

    /**
     * @return the neural network used for classification
     */
    public MLNN getMLNN() {
        return mlnn;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Utility
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * Must return a string indicating the name of the classifier.
     * Useful to avoid using "instanceof"
     * @return the name of the classifier as string
     */
    public String name() {
        return "AEClassifier";
    }

    /**
     * Returns a string indicating the type of classifier.
     *
     * @return the type of the classifier as string
     */
    @Override
    public String type() {
        return "pixel";
    }

    /**
     * Useful to select how many layer we want to train from the top
     * @return the number of layers
     */
    public int getNumLayers() {
        return mlnn.getLayersCount();
    }

    /**
     * Saves the object.
     *
     * @param fName file name
     * @throws IOException if the file cannot be written to
     */
    public void save(final String fName) throws IOException {
        // Check whether the path is existing, if not create it
        File file = new File(fName);
        if (!file.isDirectory()) {
            file = file.getParentFile();
        }
        if (!file.exists()) {
            file.mkdirs();
        }

        DataBlock pIn = scae.getBase().getInput();
        scae.setInput(new DataBlock(scae.getInputPatchWidth(), scae.getInputPatchHeight(), scae.getInputPatchDepth()));
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(fName))) {
            oos.writeObject(this);
        }
        scae.getBase().setInput(pIn);
    }

    /**
     * Loads an AEClassifier.
     *
     * @param fileName file name
     * @return the new instance
     * @throws IOException            if the file cannot be loaded
     * @throws ClassNotFoundException if the file is not valid, or if the class has been modified
     */
    public AEClassifier load(String fileName) throws IOException, ClassNotFoundException {
        return (AEClassifier) Classifier.load(fileName);
    }

}

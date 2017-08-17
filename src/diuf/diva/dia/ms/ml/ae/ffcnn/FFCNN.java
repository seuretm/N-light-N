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

package diuf.diva.dia.ms.ml.ae.ffcnn;

import diuf.diva.dia.ms.ml.Classifier;
import diuf.diva.dia.ms.ml.Trainable;
import diuf.diva.dia.ms.ml.ae.scae.Convolution;
import diuf.diva.dia.ms.ml.ae.scae.SCAE;
import diuf.diva.dia.ms.util.DataBlock;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;

import java.util.ArrayList;

/**
 * This is a feed forward convolutional network built out of an SCAE.
 * @author Mathias Seuret,Michele Alberti
 */
public class FFCNN implements Classifier, Serializable, Cloneable, Trainable {

    /**
     * Width of the perception patch
     */
    protected final int inputWidth;

    /**
     * Height of the perception patch
     */
    protected final int inputHeight;

    /**
     * Depth of the perception patch
     */
    protected final int inputDepth;

    /**
     * List of layers.
     */
    protected ArrayList<ConvolutionalLayer> layers = new ArrayList<>();
    
    /**
     * Set to true during training phases.
     */
    protected boolean isTraining = false;

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Constructor
    ///////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * Creates an FFCNN out of a SCAE. No further layer will be added.
     * @param scae scae to use as model
     */
    public FFCNN(final SCAE scae) {

        // Get the input parameters from SCAE
        inputWidth = scae.getInputPatchWidth();
        inputHeight = scae.getInputPatchHeight();
        inputDepth = scae.getInputPatchDepth();

        // Add all basic layers from the scae
        for (Convolution c : scae.getLayers()) {
            layers.add(new SingleUnitConvolution(c));
        }

        // Setting input and previous error for all layers
        for (int i = 1; i < layers.size(); i++) {
            layers.get(i).setInput(layers.get(i - 1).getOutput(), 0, 0);
            layers.get(i).setPrevError(layers.get(i - 1).getError());
        }
        
        for (ConvolutionalLayer l : layers) {
            l.clearError();
        }
    }

    /**
     * Constructs an FFCNN out of an SCAE and a set of several neural layers.
     * @param base scae to use as model
     * @param layerClassName name of the class with which should the classifying layers built with
     * @param nbClasses specifies how many classes should be classified by the FFCNN
     */
    public FFCNN(final SCAE base, String layerClassName, int nbClasses) {
        this(base, layerClassName, nbClasses, new int[0]);
    }

    /**
     * Creates an FFCNN. With this constructor it is possible to specify, and then add, additional
     * layers on top of the converted SCAE. The number of classes directly specifies how many neurons should
     * have the top layer.
     * @param base scae to use as model
     * @param layerClassName name of the class with which should the classifying layers built with
     * @param nbClasses specifies how many classes should be classified by the FFCNN
     * @param additionalLayers number of neurons in the additional classification layers
     */
    public FFCNN(final SCAE base, String layerClassName, final int nbClasses, int[] additionalLayers) {
        inputWidth = base.getInputPatchWidth();
        inputHeight = base.getInputPatchHeight();
        inputDepth = base.getInputPatchDepth();

        // Add all basic layers from the base
        for (Convolution c : base.getLayers()) {
            layers.add(new SingleUnitConvolution(c));
        }

        // Add the additional layer required for classification
        for (int nbNeurons : additionalLayers) {
            ConvolutionalLayer top = layers.get(layers.size()-1);
            layers.add(new SingleUnitConvolution(top, layerClassName, nbNeurons));
        }

        // Add the top layer
        ConvolutionalLayer top = layers.get(layers.size() - 1);
        layers.add(new SingleUnitConvolution(top, layerClassName, nbClasses));

        // Adjust input/error datablocks references
        layers.get(0).setInput(new DataBlock(inputWidth, inputHeight, inputDepth), 0, 0);
        for (int i = 1; i < layers.size(); i++) {
            layers.get(i).setInput(layers.get(i - 1).getOutput(), 0, 0);
            layers.get(i).setPrevError(layers.get(i - 1).getError());
        }
        
        for (ConvolutionalLayer l : layers) {
            l.clearError();
        }
    }

    /**
     * Loads the network from a file
     *
     * @param fileName file name
     * @return a new instance
     * @throws IOException            if the file cannot be read for some reason
     * @throws ClassNotFoundException if the file contains an older version of the FFCNN
     */
    public static FFCNN load(String fileName) throws IOException, ClassNotFoundException {
        return (FFCNN) Classifier.load(fileName);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Setting input
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * Allows the units to learn different parameters for the
     * different positions in the convolutions. Calling this
     * method leads to having the legacy behavior of N-light-N.
     */
    public void deconvolve() {
        for (int i=0; i<layers.size(); i++) {
            layers.set(i, new MultipleUnitsConvolution((SingleUnitConvolution)layers.get(i)));
        }
        for (int i = 1; i < layers.size(); i++) {
            layers.get(i).setInput(layers.get(i - 1).getOutput(), 0, 0);
            layers.get(i).setPrevError(layers.get(i - 1).getError());
        }
    }

    /**
     * Selects the input to use.
     *
     * @param db input data block
     * @param x  position
     * @param y  position
     */
    @Override
    public void setInput(DataBlock db, int x, int y) {
        layers.get(0).setInput(db, x, y);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Computing
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * Select the input to use - specify the center, not the corner.
     * @param db input data block
     * @param cx center x
     * @param cy center y
     */
    @Override
    public void centerInput(DataBlock db, int cx, int cy) {
        //System.out.println("Setting center @ "+cx+","+cy);
        ConvolutionalLayer l = layers.get(0);
        l.setInput(
                db,
                cx - l.getInputWidth() / 2,
                cy - l.getInputHeight() / 2
        );
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Getting the output/results
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * Computes the output.
     */
    @Override
    public void compute() {
        for (ConvolutionalLayer layer : layers) {
            layer.compute();
        }
    }

    /**
     * @return the index of the output having the highest activation
     */
    public int getOutputClass() {
        DataBlock db = layers.get(layers.size() - 1).getOutput();
        int res = 0;
        float max = db.getValue(0, 0, 0);
        for (int i = 1; i < db.getDepth(); i++) {
            float v = db.getValue(i, 0, 0);
            if (v > max) {
                res = i;
                max = v;
            }
        }
        return res;
    }

    /**
     * This methods returns the classification result of the last evaluated input. The classification
     * might be single or multiclass. In case of single class the result will be a normal integer, in
     * case of the multiclass use the result will be an integer whose bits will represent the classes
     * which the output have been assigned to.
     * e.g. Single: res = 5 means output got classified as belonging to class 5
     * e.g. Multi: res = 5 (0..0101) means output got "classified"as belonging to the class three and one).
     *
     * @param multiClass defines whether or not the result will be multiclass
     * @return the index of the output with the highest value
     */
    @Override
    public int getOutputClass(boolean multiClass) {
        int res = 0;
        if (multiClass) {
            DataBlock db = layers.get(layers.size() - 1).getOutput();
            for (int i = 0; i < db.getDepth(); i++) {
                if (db.getValue(i, 0, 0) > 0.35f) { //TODO: replace multiClass by the use of a threshold
                    res |= (0x01 << i);
                }
            }
        } else {
            DataBlock db = layers.get(layers.size() - 1).getOutput();
            for (int i = 1; i < db.getDepth(); i++) {
                if (db.getValue(i, 0, 0) > db.getValue(res, 0, 0)) {
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
    @Override
    public int getOutputSize() {
        return layers.get(layers.size() - 1).getOutput().getDepth();
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Learning
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * @return the output data block
     */
    public DataBlock getOutput() {
        return layers.get(layers.size() - 1).getOutput();
    }
    
    /**
     * Indicates what was expected for a given output.
     *
     * @param expectedClass output number which should correspond to the class
     * @param expectedValue expected value for the expected class
     */
    @Override
    public void setExpected(int expectedClass, float expectedValue) {
        topLayer().setExpected(0, 0, expectedClass, expectedValue);
    }

    /**
     * Indicates which class is expected for the current sample.
     * @param classNum index of the class
     */
    @Override
    public void setExpectedClass(int classNum) {
        topLayer().setExpectedClass(0, 0, classNum);
    }

    /**
     * Applies the gradient descents on the different layers.
     */
    @Override
    public void learn() {
        learn(layers.size());
    }

    /**
     * Learn the specified amount of layers from the top
     * @param nbLayers how many layers from the top ?
     */
    @Override
    public void learn(int nbLayers) {
        for (int i = layers.size() - 1; i>=0 && i >= layers.size() - nbLayers; i--) {
            layers.get(i).learn();
        }
    }

    /**
     * Backpropagate all layers
     *
     * @return average of the absolute errors of each output of the top layer
     */
    @Override
    public float backPropagate() {
        float res = backPropagate(layers.size());
        return res;
    }

    /**
     * Backpropagate the specified amount of layers from the top
     *
     * @param nbLayers how many layers from the top ?
     * @return average of the absolute errors of each output of the top layer
     */
    @Override
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
        for (ConvolutionalLayer l : layers) {
            l.clearError();
        }
        return err;
    }
    
    /**
     * Removes all previously backpropagated gradients.
     */
    public void clearGradient() {
        for (ConvolutionalLayer l : layers) {
            l.clearGradient();
        }
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Getters&Setters
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * Adds an error to a given output.
     * @param z output number
     * @param e error to add
     */
    public void addError(int z, float e) {
        layers.get(layers.size()-1).addError(0, 0, z, e);
    }

    /**
     * @return the width of the perception patch
     */
    @Override
    public int getInputWidth() {
        return inputWidth;
    }

    /**
     * @return the height of the perception patch
     */
    @Override
    public int getInputHeight() {
        return inputHeight;
    }

    /**
     * @return the depth of the perception patch
     */
    public int getInputDepth() {
        return inputDepth;
    }

    /**
     * @param n layer number
     * @return the n-th layer of the FFCNN
     */
    public ConvolutionalLayer getLayer(int n) {
        return layers.get(n);
    }

    /**
     * @return the number of outputs of the FFCNN
     */
    public int getOutputDepth() {
        return layers.get(layers.size() - 1).getOutput().getDepth();
    }

    /**
     * @return the number of layers in the network
     */
    public int countLayers() {
        return layers.size();
    }

    /**
     * @return the error of the base layer
     */
    public DataBlock getAccumulator() {
        return layers.get(0).getPrevError();
    }

    /**
     * Sets the learning speed of all layers. Default value: 1e-3f.
     * @param speed new learning speed
     */
    public void setLearningSpeed(float speed) {
        for (int i = 0; i < countLayers(); i++) {
            setLearningSpeed(i, speed);
        }
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Utility
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * Sets the learning speed of a given layer. Default value: 1e-3f.
     * @param layerNumber layer number
     * @param speed new speed
     */
    public void setLearningSpeed(int layerNumber, float speed) {
        layers.get(layerNumber).setLearningSpeed(speed);
    }

    private ConvolutionalLayer topLayer() {
        return layers.get(layers.size() - 1);
    }

    /**
     * Must return a string indicating the name of the classifier.
     * Useful to avoid using "instanceof"
     *
     * @return the name of the classifier as string
     */
    public String name() {
        return "FFCNN";
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
     *
     * @return the number of layers
     */
    @Override
    public int getNumLayers() {
        return layers.size();
    }

    /**
     * Saves the network to a file.
     *
     * @param fileName file name
     * @throws IOException if the file cannot be written to
     */
    @Override
    public void save(String fileName) throws IOException {
        // Check whether the path is existing, if not create it
        File file = new File(fileName);
        if (!file.isDirectory()) {
            file = file.getParentFile();
        }
        if (file!=null && !file.exists()) {
            file.mkdirs();
        }

        ObjectOutputStream oop = new ObjectOutputStream(new FileOutputStream(fileName));
        // Dummy input
        setInput(new DataBlock(getInputWidth(), getInputHeight(), getInputDepth()), 0, 0);
        oop.writeObject(this);
        oop.close();
    }

    /**
     * Clones the FFCNN. Throws an error in case of failure.
     * @return a new FFCNN
     * @throws java.lang.CloneNotSupportedException in case of bad implementation of a layer
     */
    @Override
    public FFCNN clone() throws CloneNotSupportedException {
        super.clone();
        FFCNN res;
        try {
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            ObjectOutputStream oos = new ObjectOutputStream(baos);
            oos.writeObject(this);

            ByteArrayInputStream bais = new ByteArrayInputStream(baos.toByteArray());
            ObjectInputStream ois = new ObjectInputStream(bais);

            res = (FFCNN) ois.readObject();
        } catch (Exception e) {
            throw new Error("Could not clone the FFCNN");
        }
        return res;
    }

    @Override
    public void startTraining() {
        for (ConvolutionalLayer layer : layers) {
            layer.startTraining();
        }
        isTraining = true;
    }

    @Override
    public void stopTraining() {
        for (ConvolutionalLayer layer : layers) {
            layer.stopTraining();
        }
        isTraining = false;
    }

    @Override
    public boolean isTraining() {
        return isTraining;
    }


}
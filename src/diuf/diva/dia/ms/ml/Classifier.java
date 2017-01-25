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

package diuf.diva.dia.ms.ml;

import diuf.diva.dia.ms.util.DataBlock;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;

/**
 * This class defines the basic interface standard for a classifier in the framework.
 * Any class that implements this interface can be used as classifier and therefore be
 * the target of XML commands like:
 * -create classifier
 * -train classifier
 * -evaluate classifier
 *
 * @author Michele Alberti
 */
public interface Classifier {

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Setting input
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * Sets the input of the classifier at a given location of a DataBlock.
     * @param db DataBlock to use as input
     * @param x horizontal location of the classifier's top-left corner
     * @param y vertical location of the classifier's top-left corner
     */
    void setInput(DataBlock db, int x, int y);

    /**
     * Centers the input at the given position in a DataBlock.
     * @param db DataBlock
     * @param cx center x
     * @param cy center y
     */
    void centerInput(DataBlock db, int cx, int cy);

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Computing
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * Runs the classification task and makes sure that the next getOutputClass
     * call will return correct values.
     */
    void compute();

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Getting the output/results
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * This method returns either an int class number (corresponding to a
     * single-class classification result), or an int which bits indicate
     * whether the class has been selected or not (multi-class classification)
     * @param multiClass true in case of multi-class task
     * @return the classification result
     */
    int getOutputClass(boolean multiClass);         // Computes the classification with single or multiclass

    /**
     * @return the number of output values, i.e., of classes
     */
    int getOutputSize();

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Learning
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * This method is used during training, after compute(), and before
     * backpropagate(). Indicate for each output what is the expected
     * classification result.
     * @param outputNumber typically a class number
     * @param expectedValue the values which should ideally have been outputted
     */
    void setExpected(int expectedClass, float expectedValue);

    /**
     * Applies the gradients stocked after one or several backPropagate() calls.
     */
    void learn();

    /**
     * Applies the gradients computed with one or several backPropagate() calls,
     * but only for some of the top layers of the classifier.
     * @param n number of layers
     */
    void learn(int n);

    /**
     * Backpropagate the error of the top layer to the previous layers, compute
     * and accumulate error gradients.
     * @return the average absolute error of the top layer
     */
    float backPropagate();

    /**
     * Backpropagate the error of the top layer to the n-1 previous layers,
     * compute and accumulate error gradients.
     * @param n number of layers which should backpropagate
     * @return the average absolute error of the top layer
     */
    float backPropagate(int n);

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Utility
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * Must return a string indicating the name of the classifier (AEClassifier,FFCNN, ...).
     * Useful to avoid using "instanceof"
     */
    String name();

    /**
     * Must return a string indicating the type of the classifier (pixel, graph, XML ...).
     * Useful to for easily extend the framework to work with other data types
     * Possible values yet implemented:
     * <p>
     * -pixel
     */
    String type();

    /**
     * @return the number of layers in the classifier
     */
    int getNumLayers();

    /**
     * The input width corresponds to the width of the patch covered by the
     * classifier in the data.
     * @return the width of the input
     */
    int getInputWidth();

    /**
     * The input height corresponds to the height of the patch covered by the
     * classifier in the data.
     * @return the height of the input
     */
    int getInputHeight();

    /**
     * Saves the classifier to a file.
     * @param fName file name
     * @throws IOException if the file cannot be written to
     */
    void save(final String fName) throws IOException;

    /**
     * Loads a Classifier from a file.
     *
     * @param fName file name
     * @return the new instance
     * @throws IOException            if the file cannot be loaded
     * @throws ClassNotFoundException if the file is not valid, or if the class has been modified
     */
    static Classifier load(final String fName) throws IOException, ClassNotFoundException {
        Classifier classifier;
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(fName))) {
            classifier = (Classifier) ois.readObject();
        }
        return classifier;
    }


}

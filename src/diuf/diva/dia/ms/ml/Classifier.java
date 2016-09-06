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
    void setInput(DataBlock db, int x, int y);          // x and y are the coordinate of the top left corner of the patch

    /**
     * Centers the 
     * @param db
     * @param cx
     * @param cy 
     */
    void centerInput(DataBlock db, int cx, int cy);     // cx and cy are exactly the pixel coordinate

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Computing
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * This method will execute the classifier and make him produce an output
     */
    void compute();

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Getting the output/results
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * These methods provide tool to access the result of the computation.
     */
    int getOutputClass(boolean multiClass);         // Computes the classification with single or multiclass

    int getOutputSize();

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Learning
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * These methods provide the possibility to set the expected values (supervised training) and learn the classifier
     */
    void setExpected(int expectedClass, float expectedValue);

    float learn();

    float learn(int n);

    float backPropagate();

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
     * Useful to select how many layer we want to train from the top
     */
    int getNumLayers();

    /**
     * These are general purpose utility methods. This section only covers the basics stuff, each classifier
     * shall have an own set of utility methods which are specific to his nature.
     */
    int getInputWidth();

    int getInputHeight();

    void save(final String fName) throws IOException;

    /**
     * Loads a Classifier.
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

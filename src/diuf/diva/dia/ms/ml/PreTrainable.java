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

/**
 * This class defines the basic interface standard for a classifier whic
 * allows for pre-training in the framework. Currently (September 2016),
 * the only classifier which support such thing is FFCNN because it can
 * feature LDAAutoEncoder as layers. On the other hand, AEClassifier cannot
 * work with LDAAutoEncoder because it uses a MLNN which is fixed to be made
 * of NeuralLayer and does not allow the employment of AutoEncoders of any kind.
 *
 * @author Michele Alberti
 */
public interface PreTrainable {

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Setting input
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * This method makes sure that all the AEs get the training sample into their training dataset.
     * Before calling this is mandatory that the input is set correctly into the classifier.
     * @param correctClass class to be expected for the sample
     */
    void addTrainingSample(int correctClass);

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Learning
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * This method must be called at the end of training
     */
    void trainingDone();

    /**
     * @return true if the classifier has already been pre-trained, false otherwise
     */
    boolean isTrained();

}

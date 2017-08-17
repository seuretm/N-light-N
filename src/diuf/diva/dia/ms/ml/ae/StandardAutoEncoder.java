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

package diuf.diva.dia.ms.ml.ae;

import diuf.diva.dia.ms.ml.Trainable;
import diuf.diva.dia.ms.ml.layer.Layer;
/**
 * Autoencoder with standard book-like behaviour-
 * The type of the layer can be chosen through a parameter.
 *
 * @author Michele Alberti
 */
public class StandardAutoEncoder extends AutoEncoder implements Trainable {

    protected boolean isTraining = false;
    
    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Constructor
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * Constructor of the class
     *
     * @param inputWidth  input width
     * @param inputHeight input height
     * @param inputDepth  input depth
     * @param outputDepth output depth
     * @param layerClassName what kind of layer should be used for encoder and decoder
     */
    public StandardAutoEncoder(
            int inputWidth,
            int inputHeight,
            int inputDepth,
            int outputDepth,
            String layerClassName
    ) {
        this(inputWidth, inputHeight, inputDepth, outputDepth, null, null, null, null, layerClassName, layerClassName);
    }
    
    /**
     * Constructor of the class
     *
     * @param inputWidth  input width
     * @param inputHeight input height
     * @param inputDepth  input depth
     * @param outputDepth output depth
     * @param encoderClassName name of the encoder's class, without package name
     * @param decoderClassName name of the decoder's class, without package name
     */
    public StandardAutoEncoder(
            int inputWidth,
            int inputHeight,
            int inputDepth,
            int outputDepth,
            String encoderClassName,
            String decoderClassName
    ) {
        this(
                inputWidth,
                inputHeight,
                inputDepth,
                outputDepth,
                null,
                null,
                null,
                null,
                encoderClassName,
                decoderClassName
        );
    }

    /**
     * Constructor of the class.
     *
     * @param inputWidth    input width
     * @param inputHeight   input height
     * @param inputDepth    input depth
     * @param outputDepth   output depth
     * @param encoderWeight encoding weights
     * @param decoderWeight decoding weights
     * @param encoderBias   encoding bias
     * @param decoderBias   decoding bias
     * @param encClassName name of the encoder's class, without package name
     * @param decClassName name of the decoder's class, without package name
     */
    public StandardAutoEncoder(int inputWidth,
                               int inputHeight,
                               int inputDepth,
                               int outputDepth,
                               float[][] encoderWeight,
                               float[][] decoderWeight,
                               float[] encoderBias,
                               float[] decoderBias,
                               String encClassName,
                               String decClassName) {
        super(
                inputWidth,
                inputHeight,
                inputDepth,
                outputDepth
        );

        try {
            Class ec = Class.forName("diuf.diva.dia.ms.ml.layer." + encClassName);
            Class dc = Class.forName("diuf.diva.dia.ms.ml.layer." + decClassName);

            // Setting the encoder
            setEncoder(
                    (Layer) ec.getDeclaredConstructor(
                            float[].class,
                            int.class,
                            int.class,
                            float[][].class,
                            float[].class
                    ).newInstance(
                            null,
                            inputWidth * inputHeight * inputDepth,
                            outputDepth,
                            encoderWeight,
                            encoderBias
                    )
            );

            // Setting the decoder
            setDecoder(
                    (Layer) dc.getDeclaredConstructor(
                            float[].class,
                            int.class,
                            int.class,
                            float[][].class,
                            float[].class
                    ).newInstance(
                    null,
                    outputDepth,
                    inputWidth * inputHeight * inputDepth,
                    decoderWeight,
                            decoderBias
                    )
            );
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Utility
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * This creates a deep copy of the AE. Note however than the dataBlocks are not cloned on purpose.
     * In fact, we want to copy the AE and not his environment. It is duty of who uses the copy to
     * change the input, output and errors dataBlock meaningfully!
     *
     * @return a deep copy of the AE.
     */
    @Override
    public AutoEncoder clone() {

        // Create a StandardAutoEncoder the standard way
        StandardAutoEncoder standardAutoEncoder = new StandardAutoEncoder(
                inputWidth,
                inputHeight,
                inputDepth,
                outputDepth,
                parseClassName(encoder.getClass()),
                parseClassName(decoder.getClass())
        );

        // Set input
        standardAutoEncoder.setInput(input, inputX, inputY);

        // Set output
        standardAutoEncoder.setOutput(output, 0, 0);

        /* If error has not the same size of output, it means it was not used before.
         * Hence we do not set it. It is clear that one has to set the error manually after
         */
        if (output.getWidth() == error.getWidth() && output.getHeight() == error.getHeight() && output.getDepth()==error.getDepth()) {
            standardAutoEncoder.setError(error);
        }

        // Set previous error if not null
        if (prevErr != null) {
            standardAutoEncoder.setPrevError(prevErr);
        }

        // Set the encoder / decoder
        standardAutoEncoder.setEncoder(encoder.clone());
        standardAutoEncoder.setDecoder(decoder.clone());

        return standardAutoEncoder;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Properties
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * @return a character indicating what kind of autoencoder this is
     */
    @Override
    public String getTypeName() {
        return "[SAE]";
    }

    @Override
    public void startTraining() {
        encoder.startTraining();
        decoder.startTraining();
        isTraining = true;
    }

    @Override
    public void stopTraining() {
        encoder.stopTraining();
        decoder.stopTraining();
        isTraining = false;
    }

    @Override
    public boolean isTraining() {
        return isTraining;
    }

    @Override
    public void clearGradient() {
        encoder.clearGradient();
        decoder.clearGradient();
    }

}

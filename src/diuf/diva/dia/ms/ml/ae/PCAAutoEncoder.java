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

import Jama.Matrix;
import diuf.diva.dia.ms.ml.layer.Layer;
import diuf.diva.dia.ms.util.PCA;

import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.stream.IntStream;

/**
 * Autoencoder witch sets  initial weights with a PCA algorithm.
 * The type of the layer can be chosen through a parameter.
 *
 * @author Alberti Michele
 */

public class PCAAutoEncoder extends AutoEncoder {

    /**
     * Keeps track whether trainingDone() has been already called or not
     */
    protected boolean trainingDone = false;
    /**
     * List that stores all training data provided,
     * with which will be calculated the PCA transformation
     */
    private final List<double[]> trainingData = new ArrayList<>();

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
     */
    public PCAAutoEncoder(int inputWidth, int inputHeight, int inputDepth, int outputDepth, String layerClassName) {
        this(inputWidth, inputHeight, inputDepth, outputDepth, null, null, null, null, layerClassName);
        if (outputDepth > inputWidth * inputHeight * inputDepth) {
            throw new IllegalArgumentException(
                    "the projected subspace of PCA cannot have more dimensions than input has"
            );
        }
    }

    /**
     * Constructor of the class
     *
     * @param inputWidth    input width
     * @param inputHeight   input height
     * @param inputDepth    input depth
     * @param outputDepth   output depth
     * @param encoderWeight encoding weights
     * @param decoderWeight decoding weights
     * @param encoderBias   encoding bias
     * @param decoderBias   decoding bias
     * @param layerClassName specifies the type of the layer to be used. It should be the exact name of the class wanted
     */
    public PCAAutoEncoder(int inputWidth,
                          int inputHeight,
                          int inputDepth,
                          int outputDepth,
                          float[][] encoderWeight,
                          float[][] decoderWeight,
                          float[] encoderBias,
                          float[] decoderBias,
                          String layerClassName) {
        super(
                inputWidth,
                inputHeight,
                inputDepth,
                outputDepth
        );

        try {
            Class c = Class.forName("diuf.diva.dia.ms.ml.layer." + layerClassName);

            // Setting the encoder
            setEncoder((Layer) c.getDeclaredConstructor(float[].class, int.class, int.class, float[][].class, float[].class).newInstance(
                    null,
                    inputWidth * inputHeight * inputDepth,
                    outputDepth,
                    encoderWeight,
                    encoderBias));

            // Setting the decoder
            setDecoder((Layer) c.getDeclaredConstructor(float[].class, int.class, int.class, float[][].class, float[].class).newInstance(
                    null,
                    outputDepth,
                    inputWidth * inputHeight * inputDepth,
                    decoderWeight,
                    decoderBias));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Computing
    ///////////////////////////////////////////////////////////////////////////////////////////////
    @Override
    public void encode() {
        if (!trainingDone) {
            throw new IllegalStateException("cannot encode with PCA before training end");
        } else {
            super.encode();
        }
    }

    @Override
    public void decode() {
        if (!trainingDone) {
            throw new IllegalStateException("cannot encode with PCA before training end");
        } else {
            super.decode();
        }
    }

    @Override
    public float train() {
        if (!trainingDone) {
            // Store the input into the training data set
            double[] x = IntStream.range(0, inputArray.clone().length).mapToDouble(i -> inputArray.clone()[i]).toArray();
            trainingData.add(x);
            return 0;
        } else {
            return super.train();
        }
    }

    @Override
    public void trainingDone() {
        // Only if it was not done before
        if (!trainingDone) {
            SimpleDateFormat ft = new SimpleDateFormat("HH:mm:ss.SSS");
            System.out.println(ft.format(new Date()) + ": Converting training set");

            // Create the training dataset in double[][] form from the linked list
            int n = trainingData.size();
            int m = trainingData.get(0).length;
            double[][] tds = new double[n][m];
            for (int i = 0; i < n; i++) {
                System.arraycopy(trainingData.get(i), 0, tds[i], 0, m);
            }

            // Compute PCA
            System.out.print(ft.format(new Date()) + ": Computing PCA");

            Matrix mat = new Matrix(tds);
            for (int r = 0; r < mat.getRowDimension(); r++) {
                for (int c = 0; c < mat.getColumnDimension(); c++) {
                    if (mat.get(r, c) != mat.get(r, c)) {
                        throw new RuntimeException("NaN detected. Something went wrong.");
                    }
                }
            }

            PCA pca = new PCA(mat, outputDepth);

            // Get the transformation matrix W
            Matrix W = pca.getW();

            for (int r = 0; r < W.getRowDimension(); r++) {
                for (int c = 0; c < W.getColumnDimension(); c++) {
                    if (W.get(r, c) != W.get(r, c)) {
                        throw new RuntimeException("NaN detected. Something went wrong.");
                    }
                }
            }

            /* Verify size of W and modify encoder & decoder accordingly.
             * It might happen that some dimensions have been dropped even
             * tough user did not specified it */
            assert (encoder.getOutputSize() == decoder.getInputSize());
            if (W.getColumnDimension() < encoder.getOutputSize()) {
                System.out.print(" : !WARNING! Automatic dimension reduction from " + encoder.getOutputSize() + " to " + W.getColumnDimension());
            }
            while (W.getColumnDimension() < encoder.getOutputSize()) {
                encoder.deleteOutput(encoder.getOutputSize());
                decoder.deleteInput(decoder.getInputSize());
            }

            System.out.println("\n" + ft.format(new Date()) + ": PCA finished");

            // Set the new weights to the encoder and decoder
            encoder.setWeights(copy(W.getArray()));
            decoder.setWeights(copy(W.transpose().getArray()));

            // Set the flag to true
            trainingDone = true;

            // Free the memory of the training data
            trainingData.clear();
        }
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Utility
    ///////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * Sets the training done to a specified parameter
     * @param std the value to be used
     */
    protected void setTrainingDone(boolean std) {
        trainingDone = std;
    }

    /**
     * This creates a deep copy of the AE. Note however than the dataBlocks are not cloned on purpose.
     * In fact, we want to copy the AE and not his environment. It is duty of who uses the copy to
     * change the input, output and errors dataBlock meaningfully!
     *
     * @return a deep copy of the AE.
     */
    @Override
    public AutoEncoder clone() {

        // Create a PCAAutoEncoder the standard way
        PCAAutoEncoder pcaAutoEncoder = new PCAAutoEncoder(
                inputWidth,
                inputHeight,
                inputDepth,
                outputDepth,
                parseClassName(encoder.getClass())
        );

        // Set input
        pcaAutoEncoder.setInput(input, inputX, inputY);

        // Set output
        pcaAutoEncoder.setOutput(output, outputX, outputY);

        // Set previous error if not null
        if (prevErr != null) {
            pcaAutoEncoder.setPrevError(prevErr);
        }

        /* If error has not the same size of output, it means it was not used before.
         * Hence we do not set it. It is clear that one has to set the error manually after
         */
        if (output.getWidth() == error.getWidth() && output.getHeight() == error.getHeight()) {
            pcaAutoEncoder.setError(error);
        }

        // Set the encoder / decoder
        pcaAutoEncoder.setEncoder(encoder.clone());
        pcaAutoEncoder.setDecoder(decoder.clone());

        // Set training done
        pcaAutoEncoder.setTrainingDone(trainingDone);

        return pcaAutoEncoder;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Properties
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * @return a character indicating what kind of autoencoder this is
     */
    @Override
    public char getTypeChar() {
        return 'p';
    }

}

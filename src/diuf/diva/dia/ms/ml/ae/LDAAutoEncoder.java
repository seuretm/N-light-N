package diuf.diva.dia.ms.ml.ae;

import diuf.diva.dia.ms.ml.layer.Layer;
import diuf.diva.dia.ms.util.LDA;

import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.stream.IntStream;

/**
 * Autoencoder witch sets initial weights with a LDA algorithm.
 * The type of the layer can be chosen through a parameter.
 *
 * @author Alberti Michele
 */

public class LDAAutoEncoder extends AutoEncoder implements SupervisedAutoEncoder {
    /**
     * List that stores all training data provided,
     * with which will be calculated the LDA transformation
     */
    private final List<double[]> trainingData = new ArrayList<>();
    /**
     * List that stores all labels for the training data provided,
     * necessary to compute the LDA transformation
     */
    private final List<Integer> trainingDataLabels = new ArrayList<>();

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
     * @param layerClassName specifies the type of the layer to be used. It should be the exact name of the class wanted
     */
    public LDAAutoEncoder(int inputWidth, int inputHeight, int inputDepth, int outputDepth, String layerClassName) {
        this(inputWidth, inputHeight, inputDepth, outputDepth, null, null, null, null, layerClassName);
        if (outputDepth > inputWidth * inputHeight * inputDepth) {
            throw new IllegalArgumentException(
                    "the projected subspace of LDA cannot have more dimensions than input has!\n" +
                            "CHECK YOUR ARCHITECTURE: is every layer having the number of neurons you think they have?"
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
    public LDAAutoEncoder(int inputWidth,
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
            input.patchToArray(inputArray, inputX, inputY, inputWidth, inputHeight);
        } else {
            super.encode();
        }
    }

    @Override
    public void decode() {
        if (!trainingDone) {
            throw new IllegalStateException("cannot encode with LDA before training end");
        } else {
            super.decode();
        }
    }

    @Override
    public float train() {
        if (!trainingDone) {
            throw new IllegalStateException("LDA must be trained in a supervised fashion use train(int label) instead.");
        } else {
            return super.train();
        }
    }

    @Override
    public float train(int label) {
        if (!trainingDone) {
            // Store the input into the training data set
            double[] x = IntStream.range(0, inputArray.clone().length).mapToDouble(i -> inputArray.clone()[i]).toArray();
            trainingData.add(x);
            trainingDataLabels.add(label);
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
            int m = trainingData.iterator().next().length;
            double[][] data = new double[n][m];
            for (int i = 0; i < n; i++) {
                System.arraycopy(trainingData.get(i), 0, data[i], 0, m);
            }

            // Create the label of training set in double[] from the linked list
            int[] labels = new int[n];
            for (int i = 0; i < n; i++) {
                labels[i] = trainingDataLabels.get(i);
            }

            // Compute LDA
            System.out.print(ft.format(new Date()) + ": Computing LDA");

            LDA lda = new LDA(data, labels);

            // Output size must be at most as big as the number of dimensions in LDA (trivial)
            assert (encoder.getOutputSize() <= lda.getNumFeatures());

            // Get the transformation matrix L
            double[][] L = lda.getLinearDiscriminants(encoder.getOutputSize());

            System.out.println("\n" + ft.format(new Date()) + ": LDA finished");

            // Set encoder
            encoder.setWeights(copy(L));
            // Set decoder weights normalised, as from inverse of a matrix we get big numbers
            decoder.setWeights(normalise(copy(lda.getInverseLinearDiscriminants(decoder.getInputSize()))));

            // Set the flag to true
            trainingDone = true;

            // Free the memory of the training data
            trainingData.clear();
            trainingDataLabels.clear();
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

    // Return B = A^T
    private double[][] transpose(double[][] a) {
        int m = a.length;
        int n = a[0].length;
        double[][] b = new double[n][m];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                b[j][i] = a[i][j];
        return b;
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
        LDAAutoEncoder ldaAutoEncoder = new LDAAutoEncoder(
                inputWidth,
                inputHeight,
                inputDepth,
                outputDepth,
                parseClassName(encoder.getClass())
        );

        // Set input
        ldaAutoEncoder.setInput(input, inputX, inputY);

        // Set output
        ldaAutoEncoder.setOutput(output, outputX, outputY);

        // Set previous error if not null
        if (prevErr != null) {
            ldaAutoEncoder.setPrevError(prevErr);
        }

        /* If error has not the same size of output, it means it was not used before.
         * Hence we do not set it. It is clear that one has to set the error manually after
         */
        if (output.getWidth() == error.getWidth() && output.getHeight() == error.getHeight()) {
            ldaAutoEncoder.setError(error);
        }

        // Set the encoder / decoder
        ldaAutoEncoder.setEncoder(encoder.clone());
        ldaAutoEncoder.setDecoder(decoder.clone());

        // Set training done
        ldaAutoEncoder.setTrainingDone(trainingDone);

        return ldaAutoEncoder;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Properties
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * Child of this class should override this method if necessary
     *
     * @return true if the autoencoder need a supervised training type
     */
    public boolean isSupervised() {
        return true;
    }

    /**
     * @return a character indicating what kind of autoencoder this is
     */
    @Override
    public char getTypeChar() {
        return 'p';
    }

}

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
import diuf.diva.dia.ms.ml.Trainable;
import diuf.diva.dia.ms.ml.layer.Layer;
import diuf.diva.dia.ms.util.PCA;

import java.io.Serializable;
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

public class PCAAutoEncoder extends AutoEncoder implements Serializable, Trainable {

    /**
     * List that stores all training data provided,
     * with which will be calculated the PCA transformation
     */
    private final List<double[]> trainingData = new ArrayList<>();
    /**
     * Keeps track whether trainingDone() has been already called or not
     */
    protected boolean trainingDone = false;
    /**
     * Set to true during training phases.
     */
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
     * @param layerClassName indicates which kind of layer will be used in this AE
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
            setEncoder(
                    (Layer) c.getDeclaredConstructor(
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
                    (Layer) c.getDeclaredConstructor(
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
    // Computing
    ///////////////////////////////////////////////////////////////////////////////////////////////
    @Override
    public void encode() {
        if (!trainingDone) {
            inputPatchToArray(inputArray);
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
            double[] x = IntStream.range(
                    0,
                    inputArray.clone().length
            ).mapToDouble(i -> inputArray.clone()[i]).toArray();
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

            // Check for NaN in input data (better safe than sorry!)
            Matrix mat = new Matrix(tds);
            for (int r = 0; r < mat.getRowDimension(); r++) {
                for (int c = 0; c < mat.getColumnDimension(); c++) {
                    if (mat.get(r, c) != mat.get(r, c)) {
                        throw new RuntimeException("NaN detected in training set for PCA. Something went wrong.");
                    }
                }
            }

            // Compute PCA
            System.out.print(ft.format(new Date()) + ": Computing PCA");

            PCA pca = new PCA(mat, outputDepth);

            // Get the transformation matrix R
            final Matrix R = pca.getW();

            // Get means of the data
            Matrix means = doubleToMatrix(pca.getMeans());

            // Check for NaN in input data (better safe than sorry!)
            for (int r = 0; r < R.getRowDimension(); r++) {
                for (int c = 0; c < R.getColumnDimension(); c++) {
                    if (R.get(r, c) != R.get(r, c)) {
                        throw new RuntimeException("NaN detected in PCA matrix R. Something went wrong.");
                    }
                }
            }

            /* Verify size of R and modify encoder & decoder accordingly.
             * It might happen that some dimensions have been dropped even
             * tough user did not specified it */
            /* NOTE: Although a cool thing, it messes up with the means: they should be
               resized accordingly when this happens. As atm this is not done, this feature
               is not supported.
            assert (encoder.getOutputSize() == decoder.getInputSize());
            if (R.getColumnDimension() < encoder.getOutputSize()) {
                System.out.print(
                        " : !WARNING! Automatic dimension reduction from "
                                + encoder.getOutputSize()
                                + " to "
                                + R.getColumnDimension()
                );
            }
            while (R.getColumnDimension() < encoder.getOutputSize()) {
                encoder.deleteOutput(encoder.getOutputSize());
                decoder.deleteInput(decoder.getInputSize());
            }
            */

            System.out.println("\n" + ft.format(new Date()) + ": PCA finished");

            // Compute the encoder bias with b1 = -R*m
            float[] b1 = copy(means.times(-1).times(R).getArray())[0];

            // Set the new weights and bias to the encoder
            encoder.setWeights(copy(R.getArray()));
            encoder.setBias(b1);

            // Set the encoder and weights of the decoder
            if(!decoder.getClass().getName().contains("NeuralLayer")){
                // If it's a linear layer then the best decoder matrix is just the transpose of the PCA matrix
                decoder.setBias(copy(means.getArray())[0]);
                decoder.setWeights(copy(R.transpose().getArray()));
            }else{

                if(true) {
                /* For more info about the following procedure of creating the decoder see the paper:
                 * "PCA-Initialized Deep Neural Networks Applied To Document Image Analysis"
                 * Mathias Seuret, Michele Alberti, Rolf Ingold, and Marcus Liwicki
                 */

                    // This dimensions corresponds to
                    int dim = encoder.getOutputSize();

                    // Creating X from the training dataset in format: each column is a sample
                    /* Note that we activate it because we want that f^-1(X) is between ]-1,1[
                     * otherwise you end up having values which are impossible to achieve with
                     * a softSign function after the step of W2+ * Y+ = f^-1(X).
                     * Idd, if the values of f^-1(X) are close to -1 or 1, it produces
                     * enormous weights in W2+ (in the order of Infinity).
                     */
                    double[][] x = mat.transpose().getArray();
                    for (int i = 0; i < x.length; i++) {
                        for (int j = 0; j < x[0].length; j++) {
                            // Apply f (in this case softSign, maybe in the future we could ask encoder to provide the function)
                            x[i][j] /= (1 + Math.abs(x[i][j]));
                        }
                    }
                    assert (x.length == m);
                    assert (x[0].length == n);

                    // Creating X+
                    double[][] xp = new double[m + 1][n];
                    for (int i = 0; i < m; i++) {
                        // Just copy x
                        System.arraycopy(x[i], 0, xp[i], 0, n);
                    }
                    for (int j = 0; j < n; j++) {
                        // Add a row of 1' on the bottom of X+
                        xp[m][j] = 1;
                    }

                    // Creating W1 and W1+
                    double[][] w1 = R.transpose().getArray();
                    double[][] w1p = new double[dim][m + 1];
                    for (int i = 0; i < dim; i++) {
                        // Just copy W1
                        System.arraycopy(w1[i], 0, w1p[i], 0, m);
                        // Add a column made of the bias and the end
                        w1p[i][m] = b1[i];
                    }

                    // Creating Y = f(W1+ * X+)
                    double[][] y = new Matrix(w1p).times(new Matrix(xp)).getArray();
                    for (int i = 0; i < y.length; i++) {
                        for (int j = 0; j < y[0].length; j++) {
                            // Apply f (in this case softSign, maybe in the future we could ask encoder to provide the function)
                            y[i][j] /= (1 + Math.abs(y[i][j]));
                        }
                    }

                    // Creating Y+ by adding a row one 1' below Y
                    double[][] yp = new double[dim + 1][n];
                    for (int i = 0; i < dim; i++) {
                        // Just copy x
                        System.arraycopy(y[i], 0, yp[i], 0, n);
                    }
                    for (int j = 0; j < n; j++) {
                        // Add a row of 1' on the bottom of X+
                        yp[dim][j] = 1;
                    }

                    /* Next we want to find X by solving W2+ * Y+ = f^-1(X) */

                    // Deactivating X
                    for (int i = 0; i < x.length; i++) {
                        for (int j = 0; j < x[0].length; j++) {
                            // Apply inverse f (in this case softSign, maybe in the future we could ask encoder to provide the function)
                            x[i][j] = (x[i][j] > 0) ? -x[i][j] / (x[i][j] - 1) : x[i][j] / (x[i][j] + 1);

                        }
                    }

                    // Solve W2+
                    double[][] w2p = new Matrix(yp).solveTranspose(new Matrix(x)).transpose().getArray();

                    /*
                    // USe this to export matrices to MATLAB friendly format
                    Matrix YP = new Matrix(yp);
                    Matrix X = new Matrix(x);
                    Matrix W2P = YP.solveTranspose(X);

                    try {
                        BufferedWriter bw1 = new BufferedWriter(new FileWriter("./../../200-Matlab/yp.txt"));
                        BufferedWriter bw2 = new BufferedWriter(new FileWriter("./../../200-Matlab/x.txt"));
                        BufferedWriter bw3 = new BufferedWriter(new FileWriter("./../../200-Matlab/w2p.txt"));

                        for (int i = 0; i < yp.length; i++) {
                            for (int j = 0; j < yp[i].length; j++) {
                                bw1.write(yp[i][j] + ",");
                            }
                            bw1.newLine();
                        }
                        bw1.flush();

                        for (int i = 0; i < x.length; i++) {
                            for (int j = 0; j < x[i].length; j++) {
                                bw2.write(x[i][j] + ",");
                            }
                            bw2.newLine();
                        }
                        bw2.flush();

                        for (int i = 0; i < w2p.length; i++) {
                            for (int j = 0; j < w2p[i].length; j++) {
                                bw3.write(w2p[i][j] + ",");
                            }
                            bw3.newLine();
                        }
                        bw3.flush();

                    } catch (IOException e) {
                    }
                    */

                    // Get W2
                    double[][] w2 = new double[m][dim];
                    for (int i = 0; i < m; i++) {
                        for (int j = 0; j < dim; j++) {
                            w2[i][j] = w2p[i][j];
                        }
                    }

                    // Get b2
                    float[] b2 = new float[m];
                    for (int i = 0; i < m; i++) {
                        b2[i] = (float) w2p[i][dim];
                    }

                    //Set bias and weights
                    decoder.setBias(b2);
                    decoder.setWeights(copy(new Matrix(w2).transpose().getArray()));
                }else{

                    /**
                     * READ HERE:
                     *
                     * n = samples
                     * m = pca.inputSize()
                     * dim = pca.outputSize() (numbers of hidden neurons)
                     */

                    // This dimensions corresponds to
                    int dim = encoder.getOutputSize();

                    // The backward matrix of PCA is its transpose (inverse=transpose for this matrix)
                    double[][] bm = R.transpose().getArray();

                    // This is the real values dataset
                    double[][] test = mat.transpose().getArray();

                    //créer yp & x
                    double[][] yp = new double[dim+1][n];
                    double[][] x  = new double[m][n];
                    for (int j=0; j<n; j++) {
                        // Why are you adding it on the top and not on the bottom ? Doesn't this affect the rest?
                        yp[0][j] = 1;
                        for (int i=1; i<=dim; i++) {
                            double a = 1;
                            yp[i][j]  = a-2*a*Math.random();
                            /*
                            / Use these lines to use real values. Note that the scaling is necessary to not to get NaN in yt
                            yp[i][j]  = test[i-1][j];
                            if(yp[i][j] == 1 ||yp[i][j] == -1){
                                yp[i][j] *= 0.9;
                            }
                            */
                        }

                        float[] yt = new float[m];
                        for (int i=0; i<dim; i++) {
                            yt[i] = (float)yp[i+1][j];
                        }
                        // Following lines until 'yt = decoded;' replace the following line: yt = pca.decode(yt);
                        // Please verify this is correct.
                        float[] decoded = new float[m];
                        for (int i=0; i<m; i++) {
                            decoded[i] = 0;
                            for (int o=0; o<outputDepth; o++) {
                                decoded[i] += bm[o][i] * ((yt[o] > 0) ? -yt[o] / (yt[o] - 1) : yt[o] / (yt[o] + 1));
                            }
                            decoded[i] += means.getArray()[0][i];
                        }
                        yt = decoded;

                        for (int i=0; i<m; i++) {
                            x[i][j] = yt[i] / (1 + Math.abs(yt[i]));
                            //x[i][j] = yt[i];
                            //x[i][j] = (yt[i] > 0) ? -yt[i] / (yt[i] - 1) : yt[i] / (yt[i] + 1);
                        }
                    }

                    // Ici, on a yp une matrice de données aléatoires augmentée, et x, sa version décodée par pca

                    /* solveTranspose() is 'badly' implemented and return a transposed result, this is why your w2p
                     * is rotated. I suggest to use:
                     * Matrix m2p = new Matrix(yp).solveTranspose(new Matrix(x)).transpose();
                     * instead, however i did not change it to maintain the code as much as possible as yours was.
                     */
                    Matrix m2p = new Matrix(yp).solveTranspose(new Matrix(x));

                    System.out.println("m2p: "+m2p.getColumnDimension()+"x"+m2p.getRowDimension());

                    // I don't understand why would you try to make Y = W2+ * X ? It should be X = f(W2+ * Y+)
                    Matrix myr = m2p.times(new Matrix(x));
                    Matrix myp = new Matrix(yp);

                    float dd = 0;
                    for (int i=0; i<myr.getRowDimension(); i++) {
                        for (int j=0; j<myr.getColumnDimension(); j++) {
                            double d = myr.get(i, j) - myp.get(i, j);
                            dd += Math.abs(d);
                        }
                    }
                    System.out.println("dd: "+(dd/myr.getRowDimension())/myr.getColumnDimension());

                    // Note that you are taking line 0 (sounds coherent with putting Y+ lines on top, but
                    // I did not do the math to verify this is correct
                    float[] decoderBias = new float[m];
                    for (int i=0; i<m; i++) {
                        decoderBias[i] = (float)m2p.get(0, i);
                    }

                    float[][] decoderWeights = new float[dim][m];
                    for (int i=0; i<dim; i++) {
                        for (int j=0; j<m; j++) {
                            decoderWeights[i][j] = (float)m2p.get(i+1, j);
                        }
                    }

                    decoder.setBias(decoderBias);
                    decoder.setWeights(decoderWeights);
                }
            }

            // Uncomment these lines to force-use the transpose
            //decoder.setBias(copy(means.getArray())[0]);
            //decoder.setWeights(copy(R.transpose().getArray()));

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
     * Converts double[] to Matrix
     */
    private Matrix doubleToMatrix(double[] x) {
        double[][] tmp = new double[1][x.length];
        tmp[0] = x;
        return new Matrix(tmp);
    }

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
    public String getTypeName() {
        return "[PAE]";
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
        encoder.clearGradient();
        decoder.clearGradient();
    }

}

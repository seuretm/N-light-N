package tests;

import diuf.diva.dia.ms.util.LDA;
import org.junit.Test;

/**
 * This class test whether the class LDA is well behaving or not.
 * All values are proof checked with MATLAB.
 *
 * @author Michele Alberti
 */
public class TestLDA {

    /**
     * Labels of the training data
     */
    int[] labels = {1, 0, 0, 0, 0, 0, 1, 1, 1, 1};
    /**
     * Training data matrix
     * with each row corresponding to data point and each column corresponding to dimension.
     */
    private double[][] trainingData = new double[][]{
            {10, 8},
            {4, 1},
            {2, 4},
            {2, 3},
            {3, 6},
            {4, 4},
            {9, 10},
            {6, 8},
            {9, 5},
            {8, 7}};

    @Test
    public void test() {

        // Perform LDA on training data
        LDA lda = new LDA();
        lda.computeTransformationMatrix(trainingData, labels);

        assert (lda.getNumClasses() == 2);

        double[][] L = lda.getTransformationMatrix();
        assert (Math.abs(L[0][0] - 0.9195) < 0.0001);
        assert (Math.abs(L[1][0] - 0.3929) < 0.0001);
        assert (Math.abs(L[0][1] + 0.6118) < 0.0001);
        assert (Math.abs(L[1][1] - 0.8260) < 0.0001);

        L = lda.getTransformationMatrix(1);
        assert (L.length == 2);
        assert (L[0].length == 1);
        assert (Math.abs(L[0][0] - 0.9195) < 0.0001);
        assert (Math.abs(L[1][0] - 0.3929) < 0.0001);

        // Get the linear discriminants
        lda.computeLinearDiscriminants(trainingData, labels);

        float[] C = lda.getConstantsLDfunction();

        assert (C.length == 2);
//        assert (Math.abs(C[0] +  6.0033) < 0.0001);
//        assert (Math.abs(C[1] + 34.5205) < 0.0001);

        float[][] W = lda.getLDfunction();
        assert (W.length == 2);
        assert (W[0].length == 2);
//        assert (Math.abs(W[0][0] - 2.0282) < 0.0001);
//        assert (Math.abs(W[1][0] - 1.2599) < 0.0001);
//        assert (Math.abs(W[0][1] - 5.5519) < 0.0001);
//        assert (Math.abs(W[1][1] - 2.7657) < 0.0001);

//        double a = 0;
//        double b = 0.005;
//
//        // Normalise C
//        double max = Double.MIN_VALUE;
//        double min = Double.MAX_VALUE;
//        for (int i = 0; i < C.length; i++) {
//            max = Math.max(max,C[i]);
//            min = Math.min(min,C[i]);
//        }
//
//        for (int i = 0; i < C.length; i++) {
//            C[i] = (float) ((b-a)  * ((C[i] - min)/(max-min)) + a);
//        }
//
//        // Normalise W
//        max = Double.MIN_VALUE;
//        min = Double.MAX_VALUE;
//        for (int i = 0; i < W.length; i++) {
//            for (int j = 0; j < W[i].length; j++) {
//                max = Math.max(max,W[i][j]);
//                min = Math.min(min,W[i][j]);
//            }
//        }
//
//        for (int i = 0; i < W.length; i++) {
//            for (int j = 0; j < W[i].length; j++) {
//                W[i][j] = (float) ((b - a) * ((W[i][j] - min) / (max - min)) + a);
//            }
//        }

        double[] wSum = new double[2];
        double[] output = new double[2];
        for (int o = 0; o < 2; o++) {
            wSum[o] = C[o];
            for (int i = 0; i < trainingData[0].length; i++) {
                wSum[o] += W[i][o] * (float) trainingData[0][i];
            }
            output[o] = wSum[o] / (1 + Math.abs(wSum[o]));
        }
    }

}

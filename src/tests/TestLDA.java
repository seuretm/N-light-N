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

    /**
     * Labels of the training data
     */
    int[] labels = {2, 1, 1, 1, 1, 1, 2, 2, 2, 2};

    @Test
    public void test() {

        // Perform LDA on training data
        LDA lda = new LDA(trainingData, labels);

        assert (lda.getNumClasses() == 2);

        double[][] L = lda.getLinearDiscriminants();
        assert (Math.abs(L[0][0] - 0.9195) < 0.0001);
        assert (Math.abs(L[1][0] - 0.3929) < 0.0001);
        assert (Math.abs(L[0][1] + 0.6118) < 0.0001);
        assert (Math.abs(L[1][1] - 0.8260) < 0.0001);

        L = lda.getLinearDiscriminants(1);
        assert (L.length == 2);
        assert (L[0].length == 1);
        assert (Math.abs(L[0][0] - 0.9195) < 0.0001);
        assert (Math.abs(L[1][0] - 0.3929) < 0.0001);

    }

}

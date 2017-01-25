package tests;

import Jama.Matrix;
import diuf.diva.dia.ms.util.PCA;
import org.junit.Test;

/**
 * This class test whether the class PCA is well behaving or not.
 * All values are proof checked with MATLAB and R.
 *
 * @author Michele Alberti
 */
public class TestPCA {

    /**
     * Training data matrix
     * with each row corresponding to data point and each column corresponding to dimension.
     */
    private Matrix trainingData = new Matrix(new double[][]{
            {2.5, 2.4},
            {0.5, 0.7},
            {2.2, 2.9},
            {1.9, 2.2},
            {3.1, 3.0},
            {2.3, 2.7},
            {2.0, 1.6},
            {1.0, 1.1},
            {1.5, 1.6},
            {1.1, 0.9}});
    /**
     * Object responsible to compute the PCA
     */
    private PCA pca;
    /**
     * Transformation matrix W
     */
    private Matrix W;
    /**
     * Training data transformed into the subspace Y
     */
    private Matrix transformedData;
    /**
     * Training data recovered after being transformed into subspace Y
     */
    private Matrix recoveredData;

    @Test
    public void test() {

        // Center the data
        trainingData = PCA.centerMatrix(trainingData);

        // Create the PCA and thus compute it
        pca = new PCA(trainingData);

        // Get the matrix W
        W = pca.getW();

        assert (Math.abs(W.get(0, 0) - 0.6778) < 0.0001);
        assert (Math.abs(W.get(0, 1) + 0.7351) < 0.0001);
        assert (Math.abs(W.get(1, 0) - 0.7351) < 0.0001);
        assert (Math.abs(W.get(1, 1) - 0.6778) < 0.0001);

        transformedData = pca.transform(trainingData);
        assert (Math.abs(transformedData.get(0, 0) - 0.8280) < 0.0001);
        assert (Math.abs(transformedData.get(0, 1) + 0.1751) < 0.0001);
        assert (Math.abs(transformedData.get(1, 0) + 1.7776) < 0.0001);
        assert (Math.abs(transformedData.get(1, 1) - 0.1429) < 0.0001);
        assert (Math.abs(transformedData.get(2, 0) - 0.9922) < 0.0001);
        assert (Math.abs(transformedData.get(2, 1) - 0.3844) < 0.0001);
        assert (Math.abs(transformedData.get(3, 0) - 0.2742) < 0.0001);
        assert (Math.abs(transformedData.get(3, 1) - 0.1304) < 0.0001);
        assert (Math.abs(transformedData.get(4, 0) - 1.6758) < 0.0001);
        assert (Math.abs(transformedData.get(4, 1) + 0.2095) < 0.0001);
        assert (Math.abs(transformedData.get(5, 0) - 0.9129) < 0.0001);
        assert (Math.abs(transformedData.get(5, 1) - 0.1753) < 0.0001);
        assert (Math.abs(transformedData.get(6, 0) + 0.0991) < 0.0001);
        assert (Math.abs(transformedData.get(6, 1) + 0.3498) < 0.0001);
        assert (Math.abs(transformedData.get(7, 0) + 1.1446) < 0.0001);
        assert (Math.abs(transformedData.get(7, 1) - 0.0464) < 0.0001);
        assert (Math.abs(transformedData.get(8, 0) + 0.4380) < 0.0001);
        assert (Math.abs(transformedData.get(8, 1) - 0.0178) < 0.0001);
        assert (Math.abs(transformedData.get(9, 0) + 1.2238) < 0.0001);
        assert (Math.abs(transformedData.get(9, 1) + 0.1627) < 0.0001);

        recoveredData = pca.recover(transformedData);
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 2; j++) {
                assert (Math.abs(recoveredData.get(i, j) - trainingData.get(i, j)) < 1e-15);
            }
        }
    }

    @Test
    public void testCovarianceMatrix() {

        double[][] data = new double[][]{
                {4.0, 2.0, 0.60},
                {4.2, 2.1, 0.59},
                {3.9, 2.0, 0.58},
                {4.3, 2.1, 0.62},
                {4.1, 2.2, 0.63}
        };

        Matrix m = PCA.getCovarianceMatrix(new Matrix(data));

        assert (Math.abs(m.get(0, 0) - 0.025) < 0.001);
        assert (Math.abs(m.get(0, 1) - 0.0075) < 0.0001);
        assert (Math.abs(m.get(0, 2) - 0.00175) < 0.00001);
        assert (Math.abs(m.get(1, 0) - m.get(0, 1)) == 0);
        assert (Math.abs(m.get(1, 1) - 0.0070) < 0.0001);
        assert (Math.abs(m.get(1, 2) - 0.00135) < 0.00001);
        assert (Math.abs(m.get(2, 0) - m.get(0, 2)) == 0);
        assert (Math.abs(m.get(2, 1) - m.get(1, 2)) == 0);
        assert (Math.abs(m.get(2, 2) - 0.00043) < 0.00001);
    }

    private static void printMatrix(Matrix m) {
        for (int r = 0; r < m.getRowDimension(); r++) {
            for (int c = 0; c < m.getColumnDimension(); c++) {
                System.out.print(m.get(r, c));
                if (c == m.getColumnDimension() - 1) continue;
                System.out.print(", ");
            }
            System.out.println("");
        }
    }

}

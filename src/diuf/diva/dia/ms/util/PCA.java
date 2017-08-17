package diuf.diva.dia.ms.util;

import Jama.Matrix;
import com.mkobos.pca_transform.Assume;
import com.mkobos.pca_transform.covmatrixevd.CovarianceMatrixEVDCalculator;
import com.mkobos.pca_transform.covmatrixevd.EVDBased;
import com.mkobos.pca_transform.covmatrixevd.EVDResult;
import com.mkobos.pca_transform.covmatrixevd.SVDBased;
import diuf.diva.dia.ms.util.misc.EVDT;

/**
 * This is a class for doing PCAs, based on https://github.com/mkobos/pca_transform
 * @author Mateusz Kobos, Michele Alberti, Mathias Seuret
 */
public final class PCA {

    /**
     * The matrix W responsible of transform the data.
     * This is the core result of the PCA algorithm.
     * Old version name: pcaRotationTransformation
     */
    private final Matrix W;

    /**
     * Part of the original SVD vector that is responsible for transforming the
     * input data into a vector of zeros.
     */
    private final Matrix zerosRotationTransformation;

    private final double[] means;
    private final double threshold;

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Constructor
    ///////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * Create the PCA transformation. Use the popular SVD method for internal
     * calculations
     * @param data data matrix used to compute the PCA transformation.
     * Rows of the matrix are the instances/samples, columns are dimensions.
     */
    public PCA(Matrix data) {
        this(data, data.getColumnDimension());
    }

    /**
     * Create the PCA transformation. Use the popular SVD method for internal
     * calculations
     *
     * @param data data matrix used to compute the PCA transformation. Rows of
     * the matrix are the instances/samples, columns are dimensions.
     * @param nbComponents dimensionality of the transformation matrix (dimensions of the sub subspace)
     */
    public PCA(Matrix data, int nbComponents) {
        this(data, new SVDBased(), nbComponents);
    }

    /**
     * Create the PCA transformation
     *
     * @param data data matrix used to compute the PCA transformation.
     * Rows of the matrix are the instances/samples, columns are dimensions.
     * Data will be in any case centered.
     * @param evdCalc method of computing eigenvalue decomposition of data's covariance matrix
     * @param nbComponents dimensionality of the transformation matrix (dimensions of the sub subspace)
     */
    public PCA(Matrix data, CovarianceMatrixEVDCalculator evdCalc, int nbComponents) {
        if (data.getColumnDimension() < nbComponents) {
            throw new IllegalArgumentException(
                    "[ERROR][PCA] The data has not enough dimension(" + data.getColumnDimension() + ")"
            );
        }

        // Get the number of input dimensions.
        this.means = getColumnsMeans(data);

        // Center the data matrix columns about zero
        data = shiftColumns(data, means);

        EVDResult evd = evdCalc.run(data);
        EVDT evdT = new EVDT(evd);

        // A 3-sigma-like ad-hoc rule
        this.threshold = 3 * evdT.getThreshold();

        // Get only the values of the matrices that correspond to standard deviations above the threshold
        Matrix a = evdT.getVAboveThreshold();
        Matrix b = evdT.getVBelowThreshold();

        // Join a and b horizontally [ a , b ]
        Matrix x = new Matrix(a.getRowDimension(), a.getColumnDimension() + b.getColumnDimension());
        x.setMatrix(0, a.getRowDimension()-1, 0, a.getColumnDimension() - 1, a);
        x.setMatrix(0, a.getRowDimension()-1, a.getColumnDimension(), a.getColumnDimension() + b.getColumnDimension() - 1, b);

        this.W = getSubMatrix(x, x.getRowDimension(), nbComponents);
        this.zerosRotationTransformation = evdT.getVBelowThreshold();

    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Public
    ///////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * Execute selected transformation on given data.
     *
     * @param data data to transform.
     * Rows of the matrix are the instances/samples, columns are dimensions.
     * @return transformed data
     */
    public Matrix transform(Matrix data) {
        return data.times(W);
    }

    /**
     * Recovers the original data from a transformed given data.
     *
     * @param data data to recover. Rows of the matrix are the
     * instances/samples, columns are dimensions.
     * @return recovered data
     */
    public Matrix recover(Matrix data) {
        return data.times(W.transpose());
    }

    /**
     * Check if given point lies in PCA-generated subspace. If it does not, it
     * means that the point doesn't belong to the transformation domain i.e. it
     * is an outlier.
     *
     * @param pt point. If the original PCA data matrix was set to be centered,
     * this point will also be centered using the same parameters.
     * @return true iff the point lies on all principal axes
     */
    public boolean belongsToGeneratedSubspace(Matrix pt) {
        Assume.assume(pt.getRowDimension() == 1);
        pt = shiftColumns(pt, means);
        Matrix zerosTransformedPt = pt.times(zerosRotationTransformation);
        assert zerosTransformedPt.getRowDimension() == 1;
        /**
         * Check if all coordinates of the point were zeroed by the
         * transformation
         */
        for (int c = 0; c < zerosTransformedPt.getColumnDimension(); c++) {
            if (Math.abs(zerosTransformedPt.get(0, c)) > threshold) {
                return false;
            }
        }
        return true;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Getters & Setters
    ///////////////////////////////////////////////////////////////////////////////////////////////

    public double[] getMeans() {
        return means;
    }

    public Matrix getW() {
        return W;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Public static
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * Returns the covariance matrix of the data
     *
     * @param m data to compute the covariance matrix on
     * @return covariance matrix
     */
    public static Matrix getCovarianceMatrix(Matrix m) {
        return EVDBased.calculateCovarianceMatrixOfCenteredData(centerMatrix(m));
    }

    /**
     * Subtract a value to each column
     *
     * @param data   the matrix on which we want to perform the operation
     * @param shifts the values to subtract to each column
     * @return the data matrix minus the shifts
     */
    public static Matrix shiftColumns(Matrix data, double[] shifts) {
        Assume.assume(shifts.length == data.getColumnDimension());
        Matrix m = new Matrix(data.getRowDimension(), data.getColumnDimension());
        for (int c = 0; c < data.getColumnDimension(); c++) {
            for (int r = 0; r < data.getRowDimension(); r++) {
                m.set(r, c, data.get(r, c) - shifts[c]);
            }
        }
        return m;
    }

    /**
     * Compute the means column-wise
     *
     * @param m the data
     * @return the column-wise means
     */
    public static double[] getColumnsMeans(Matrix m) {
        double[] means = new double[m.getColumnDimension()];
        for (int c = 0; c < m.getColumnDimension(); c++) {
            double sum = 0;
            for (int r = 0; r < m.getRowDimension(); r++) {
                sum += m.get(r, c);
            }
            means[c] = sum / m.getRowDimension();
        }
        return means;
    }

    /**
     * Center the data subtracting to each columns their means
     *
     * @param data the matrix on which we want to perform the operation
     * @return the data matrix centered on zero
     */
    public static Matrix centerMatrix(Matrix data) {
        return shiftColumns(data, getColumnsMeans(data));
    }

    /**
     * Subsamples a matrix
     *
     * @param m      the original data
     * @param nbRows number of rows to be kept
     * @param nbCols number of cols to be kept
     * @return the matrix m subsampled to nbrows and nbCols rows and columns respectively
     */
    public static Matrix getSubMatrix(Matrix m, int nbRows, int nbCols) {
        Matrix res = new Matrix(nbRows, nbCols);
        for (int i=0; i<nbRows; i++) {
            for (int j=0; j<nbCols; j++) {
                res.set(i, j, m.get(i, j));
            }
        }
        return res;
    }

}


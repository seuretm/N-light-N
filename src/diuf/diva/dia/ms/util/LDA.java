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

package diuf.diva.dia.ms.util;

import Jama.EigenvalueDecomposition;
import Jama.Matrix;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

/**
 * This code computes linear discriminant analysis (LDA).
 * It is a direct adaptation of my LDA implementation in MATLAB. Unfortunately being not
 * present in JAVA a direct translation for mldivide() (which is a\b in MATLAB), I had
 * to use Matrix.inverse() which is not the fastest solution.
 *
 * @author Michele Alberti
 */

public class LDA {

    /**
     * Means of every class
     */
    private final double[][] mu;
    /**
     * Overall mean (global mean for whole dataset)
     */
    private final double[] omu;
    /**
     * Mean number of point per class
     */
    private final int nmu;
    /**
     * Unique class labels
     */
    private final ArrayList<Integer> classLabel;
    /**
     * Number of different classes
     */
    private final int numClasses;
    /**
     * Number of feature per sample
     */
    private final int numFeatures;
    /**
     * Number of samples in the training dataset
     */
    private final int numSamples;
    /**
     * Data split into separated classes
     */
    private final ArrayList<double[]>[] subset;
    /**
     * Within-class scatter matrix
     */
    private final double[][] sw;
    /**
     * Between-classes scatter matrix
     */
    private final double[][] sb;
    /**
     * The transformation projection matrix L
     */
    private final double[][] L;

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Constructor
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * Computes linear discriminant analysis
     *
     * @param data   data on which to compute LDA
     * @param labels class labels for the data provided*
     */
    public LDA(double[][] data, int[] labels) {

        // Data and labels vectors must have the same size
        assert (data.length == labels.length);

        // Determine number of different classes
        classLabel = new ArrayList<>();
        for (int label : labels) {
            if (!classLabel.contains(label)) {
                classLabel.add(label);
            }
        }
        Collections.sort(classLabel);

        // Init some support variable
        numClasses = classLabel.size();
        numFeatures = data[0].length;
        numSamples = data.length;

        // Split data into subsets for easier management
        subset = new ArrayList[numClasses];
        for (int c = 0; c < numClasses; c++) {
            subset[c] = new ArrayList<>();
            for (int i = 0; i < numSamples; i++) {
                if (labels[i] == classLabel.get(c)) {
                    subset[c].add(data[i]);
                }
            }
        }

        // Compute per class means
        mu = new double[numClasses][numFeatures];
        for (int c = 0; c < numClasses; c++) {
            for (int i = 0; i < subset[c].size(); i++) {
                for (int f = 0; f < numFeatures; f++) {
                    mu[c][f] += subset[c].get(i)[f];
                }
            }
            for (int f = 0; f < numFeatures; f++) {
                mu[c][f] /= subset[c].size();
            }
        }

        // Compute overall mean
        omu = new double[numFeatures];
        for (int i = 0; i < numSamples; i++) {
            for (int f = 0; f < numFeatures; f++) {
                omu[f] += data[i][f];
            }
        }
        for (int f = 0; f < numFeatures; f++) {
            omu[f] /= numSamples;
        }

        // Compute mean number of point per class
        nmu = numSamples / numClasses;

        // Compute within class scatter matrix
        double[][] tmp = new double[1][numFeatures];
        double[][][] psw = new double[numClasses][numFeatures][numFeatures];
        for (int c = 0; c < numClasses; c++) {
            for (int i = 0; i < subset[c].size(); i++) {
                tmp[0] = subtract(subset[c].get(i), mu[c]);
                psw[c] = add(psw[c], multiply(transpose(tmp), tmp));
            }
            // Balance class influence (ignore size of class in final SW)
            for (int i = 0; i < numFeatures; i++) {
                for (int j = 0; j < numFeatures; j++) {
                    psw[c][i][j] /= subset[c].size();
                }
            }
        }
        sw = new double[numFeatures][numFeatures];
        for (int i = 0; i < numFeatures; i++) {
            for (int j = 0; j < numFeatures; j++) {
                for (int c = 0; c < numClasses; c++) {
                    sw[i][j] += psw[c][i][j] * nmu;
                }
            }
        }

        // Compute the between-classes scatter matrix
        double[][][] psb = new double[numClasses][numFeatures][numFeatures];
        for (int c = 0; c < numClasses; c++) {
            tmp[0] = subtract(mu[c], omu);
            psb[c] = multiply(transpose(tmp), tmp);
        }

        sb = new double[numFeatures][numFeatures];
        for (int i = 0; i < numFeatures; i++) {
            for (int j = 0; j < numFeatures; j++) {
                for (int c = 0; c < numClasses; c++) {
                    sb[i][j] += psb[c][i][j] * nmu;
                }
            }
        }

        // Compute J
        Matrix J = new Matrix(sw).inverse().times(new Matrix(sb));

        // Compute eigenvectors & eigenvalues
        EigenvalueDecomposition eig = J.eig();
        double[][] V = new double[numFeatures + 1][numFeatures];
        double[] D = eig.getRealEigenvalues();
        tmp = eig.getV().getArray();
        for (int j = 0; j < numFeatures; j++) {
            for (int i = 0; i < numFeatures; i++) {
                V[i][j] = tmp[i][j];
            }
            V[numFeatures][j] = D[j];
        }

        // Init the index that we will use to sort the eigenvalues
        Integer[] index = new Integer[numFeatures];
        for (int i = 0; i < numFeatures; i++) {
            index[i] = i;
        }


        // Sort the index according to the eigenvalues
        Arrays.sort(index, (a, b) -> {
            if (V[numFeatures][a] > V[numFeatures][b]) {
                return -1;
            } else if (V[numFeatures][a] < V[numFeatures][b]) {
                return 1;
            }
            return 0;
        });

        // Compose L from the sorted eigenvectors
        L = new double[numFeatures][numFeatures];
        for (int j = 0; j < numFeatures; j++) {
            for (int i = 0; i < numFeatures; i++) {
                L[i][j] = V[i][index[j]];
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Public
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * Getter for the transformation matrix L
     *
     * @return return the transformation matrix L
     */
    public double[][] getLinearDiscriminants() {
        return getLinearDiscriminants(numFeatures);
    }

    /**
     * Getter for the transformation matrix L sub-sampled
     *
     * @param m number of cols to be kept
     * @return return the transformation matrix L sub-sampled to only m columns from left
     */
    public double[][] getLinearDiscriminants(int m) {
        double[][] rv = new double[numFeatures][m];
        for (int i = 0; i < numFeatures; i++) {
            System.arraycopy(L[i], 0, rv[i], 0, m);
        }
        return rv;
    }

    /**
     * Getter for the inverse of the transformation matrix L
     *
     * @return return the transformation matrix L
     */
    public double[][] getInverseLinearDiscriminants() {
        return getInverseLinearDiscriminants(numFeatures);
    }

    /**
     * Getter for the inverse of the transformation matrix L sub-sampled
     *
     * @param m number of cols to be kept
     * @return return the transformation matrix L sub-sampled to only m columns from left
     */
    public double[][] getInverseLinearDiscriminants(int m) {
        double[][] M = new Matrix(L).inverse().getArray();
        double[][] rv = new double[m][numFeatures];
        for (int i = 0; i < m; i++) {
            System.arraycopy(M[i], 0, rv[i], 0, numFeatures);
        }
        return rv;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Private
    ///////////////////////////////////////////////////////////////////////////////////////////////

    // Return c = a * b
    private double[][] multiply(double[][] a, double[][] b) {
        int m1 = a.length;
        int n1 = a[0].length;
        int m2 = b.length;
        int n2 = b[0].length;
        if (n1 != m2) throw new RuntimeException("Illegal matrix dimensions.");
        double[][] c = new double[m1][n2];
        for (int i = 0; i < m1; i++) {
            for (int j = 0; j < n2; j++) {
                for (int k = 0; k < n1; k++) {
                    c[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        return c;
    }

    // Return B = A^T
    private double[][] transpose(double[][] a) {
        int m = a.length;
        int n = a[0].length;
        double[][] b = new double[n][m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                b[j][i] = a[i][j];
            }
        }
        return b;
    }

    // Return c = a + b
    private double[][] add(double[][] a, double[][] b) {
        int m = a.length;
        int n = a[0].length;
        double[][] c = new double[m][n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                c[i][j] = a[i][j] + b[i][j];
        return c;
    }

    // Return c = a - b
    private double[] subtract(double[] a, double[] b) {
        int n = a.length;
        double[] c = new double[n];
        for (int i = 0; i < n; i++)
            c[i] = a[i] - b[i];
        return c;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Getters&Setters
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * Returns the number of classes
     */
    public int getNumClasses() {
        return numClasses;
    }

    /**
     * Returns the number of features
     */
    public int getNumFeatures() {
        return numFeatures;
    }
}

package diuf.diva.dia.ms.util.misc;


import Jama.Matrix;
import diuf.diva.dia.ms.util.PCA;

import java.util.List;

/**
 * Encodes a reconstruction score.
 */
public class ReconstructionScore {

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Distances
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * Computes the distance between the two input patch using the
     * Euclidean formula for each pixel. Therefore input must
     * me pixel-friendly (size multiple of 3 and have a meaning in that
     * sense).
     *
     * @param a first variable
     * @param b second variable
     * @return the euclidean distance
     */
    public static float euclideanDistance(float[] a, float[] b) {

        // Input must be dividable in pixels and same size ofc
        assert (a.length % 3 == 0);
        assert (a.length == b.length);

        // For each pixel
        float d = 0;
        for (int i = 0; i < a.length; i += 3) {
            float[] A = new float[]{a[i], a[i + 1], a[i + 2]};
            float[] B = new float[]{b[i], b[i + 1], b[i + 2]};
            d += euclidean(A, B);
            //System.out.println("d = " + d + "\t[" + A[0] + "," + A[1] + "," + A[2] + "]\t" +"[" + B[0] + "," + B[1] + "," + B[2] + "]");
        }

        return d / (a.length / 3);
    }

    /**
     * Computes the Scaled Offest Invariant distance.
     * This metric tries to match the shapes of the two samples minimizing their
     * distances by modifying scale and offset only (not the shape or relative position of points).
     *
     * @param a first variable
     * @param b second variable
     * @return the SOID
     */
    public static float scaleOffsetInvarDist(float[] a, float[] b) {
        assert (a.length == b.length);
        final int N = a.length;

        // Computing means
        float muA = 0;
        float muB = 0;
        for (int i = 0; i < N; i++) {
            muA += a[i];
            muB += b[i];
        }
        muA /= N;
        muB /= N;

        //Centering the data
        for (int i = 0; i < N; i++) {
            a[i] -= muA;
            b[i] -= muB;
        }

        // Computing eta
        float top = 0;
        float bot = 0;
        for (int i = 0; i < N; i++) {
            top += (b[i]*a[i]);
            bot += (a[i]*a[i]);
        }
        float eta = top / bot;

        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            float d = eta * a[i] - b[i];
            sum += d * d;
        }

        return sum / N;
        //  return (float) Math.sqrt(sum) / N;
    }

    /**
     * Computes the normalized correlation of two variables.
     * This makes use of the Pearson correlation coefficient to compute
     * the correlation between two samples.
     *
     * @param x first variable
     * @param y second variable
     * @return normalized correlation between 0 and 1.
     */
    public static float normalizedCorrelation(float[] x, float[] y) {
        assert (x.length == y.length);

        double meanX = 0;
        double meanY = 0;
        int n = x.length;

        // Compute means
        for (int i = 0; i < n; i++) {
            meanX += x[i];
            meanY += y[i];
        }
        meanX /= n;
        meanY /= n;

        // Compute upper and lower part of the Person formula
        double sxy = 0;
        double sx = 0;
        double sy = 0;
        for (int i = 0; i < n; i++) {
            sxy += (x[i] - meanX) * (y[i] - meanY);
            sx += Math.pow(x[i] - meanX, 2);
            sy += Math.pow(y[i] - meanY, 2);
        }
        sx = Math.sqrt(sx);
        sy = Math.sqrt(sy);

        // Return 1-|p|
        return (float) (1 - (Math.abs(sxy) / (1e-5f + sx * sy)));
    }

    /**
     * Computes the distance between the two input patch using the
     * delta94 formula for each pixel. Therefore input must
     * me pixel-friendly (size multiple of 3 and have a meaning in that
     * sense).
     *
     * @param a first variable
     * @param b second variable
     * @return the deltaE*94 distance
     */
    public static float delta94distance(float[] a, float[] b) {

        // Input must be dividable in pixels and same size ofc
        assert (a.length % 3 == 0);
        assert (a.length == b.length);

        // For each pixel
        float d = 0;
        for (int i = 0; i < a.length; i += 3) {

            float[] A = XYZtoLab(rgbToXYZ(new float[]{a[i], a[i + 1], a[i + 2]}));
            float[] B = XYZtoLab(rgbToXYZ(new float[]{b[i], b[i + 1], b[i + 2]}));
            d += deltaE94(A, B);
        }

        return d/ (a.length / 3);
    }

    /**
     * Computes the Mahalanobis Distance between the two input patch using the
     * formula for each pixel. Therefore input must
     * me pixel-friendly (size multiple of 3 and have a meaning in that
     * sense).
     *
     * @param a first variable
     * @param b second variable
     * @return the Mahalanobis distance
     */
    public static Float mahalanobisDistance(float[] a, float[] b) {

        // Input must be dividable in pixels and same size ofc
        assert (a.length % 3 == 0);
        assert (b.length % 3 == 0);
        assert (a.length == b.length);

        // Collect the whole population of pixels to compute the covariance matrix
        double[][] d = new double[2 * (a.length / 3)][3];
        for (int i = 0; i < a.length; i += 3) {
            d[i / 3] = new double[]{a[i], a[i + 1], a[i + 2]};
            d[(a.length / 3) + i / 3] = new double[]{b[i], b[i + 1], b[i + 2]};
        }

        Matrix data = new Matrix(d);

        // Get covariance matrix
        Matrix c = PCA.getCovarianceMatrix(data);

        if (!data.lu().isNonsingular() || !c.lu().isNonsingular()) {
            c = Matrix.identity(3, 3);
        }

        // Compute Mahalanobis distance
        float distance = 0;
        for (int i = 0; i < d.length / 2; i++) {

            Matrix x = data.getMatrix(i, i, 0, data.getColumnDimension() - 1);
            Matrix y = data.getMatrix(i + d.length / 2, i + d.length / 2, 0, data.getColumnDimension() - 1);
            Matrix tmp = x.minus(y);

            distance += Math.sqrt(tmp.times(c.inverse()).times(tmp.transpose()).get(0, 0));
        }

        return distance / (a.length / 3);
    }

    /**
     * @param l a list
     * @return mean value of the list
     */
    public static float getMean(List<Float> l) {
        float sum = 0;
        for (Float f : l) {
            sum += f;
        }
        return sum / l.size();
    }

    /**
     * @param l a list
     * @return variance of the list
     */
    public static float getVariance(List<Float> l) {
        float mean = getMean(l);
        float sum = 0;
        for (Float f : l) {
            float d = (f - mean);
            sum += d * d;
        }
        return sum / l.size();
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Utility
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * Convert a pixel in RGB values into XYZ space.
     * It is assumed illuminant D65 and view  2�
     *
     * @param rgb values of the pixel
     * @return the same pixel in XYZ
     */
    protected static float[] rgbToXYZ(float[] rgb) {

        assert (rgb.length == 3);

        double R = (rgb[0] / 255);        //R from 0 to 255
        double G = (rgb[1] / 255);        //G from 0 to 255
        double B = (rgb[2] / 255);        //B from 0 to 255

        R = (R > 0.04045) ? Math.pow((R + 0.055) / 1.055, 2.4) : R / 12.92;
        G = (G > 0.04045) ? Math.pow((G + 0.055) / 1.055, 2.4) : G / 12.92;
        B = (B > 0.04045) ? Math.pow((B + 0.055) / 1.055, 2.4) : B / 12.92;

        R *= 100;
        G *= 100;
        B *= 100;

        //Observer. = 2�, Illuminant = D65
        float X = (float) (R * 0.4124 + G * 0.3576 + B * 0.1805);
        float Y = (float) (R * 0.2126 + G * 0.7152 + B * 0.0722);
        float Z = (float) (R * 0.0193 + G * 0.1192 + B * 0.9505);

        return new float[]{X, Y, Z};
    }

    /**
     * Convert a pixel in XYZ values into Lab space.
     * It is assumed illuminant D65 and view  2�
     *
     * @param xyz values of the pixel
     * @return the same pixel in Lab
     */
    protected static float[] XYZtoLab(float[] xyz) {

        assert (xyz.length == 3);

        double X = xyz[0] / 95.047;          //ref_X =  95.047   Observer= 2�, Illuminant= D65
        double Y = xyz[1] / 100.000;         //ref_Y = 100.000
        double Z = xyz[2] / 108.883;         //ref_Z = 108.883

        X = (X > 0.008856) ? Math.pow(X, 1 / 3.0) : (7.787 * X) + (16 / 116.0);
        Y = (Y > 0.008856) ? Math.pow(Y, 1 / 3.0) : (7.787 * Y) + (16 / 116.0);
        Z = (Z > 0.008856) ? Math.pow(Z, 1 / 3.0) : (7.787 * Z) + (16 / 116.0);

        float L = (float) (116 * Y) - 16;
        float a = (float) (500 * (X - Y));
        float b = (float) (200 * (Y - Z));

        return new float[]{L, a, b};
    }

    /**
     * Computes the CIE76 distance.
     *
     * @param a first point in Lab colorspace
     * @param b second point in Lab colorspace
     * @return CIE76 distance
     */
    protected static float deltaE76(float[] a, float[] b) {
        assert (a.length == 3);
        assert (b.length == 3);

        return euclideanDistance(a, b);
    }

    /**
     * Computes the CIE94 distance.
     *
     * @param a first point in Lab colorspace
     * @param b second point in Lab colorspace
     * @return CIE94 distance
     */
    protected static float deltaE94(float[] a, float[] b) {
        assert (a.length == 3);
        assert (b.length == 3);

        // Delta L
        double deltaL = a[0] - b[0];

        // Delta C
        double C1 = Math.sqrt(a[1] * a[1] + a[2] * a[2]);
        double C2 = Math.sqrt(b[1] * b[1] + b[2] * b[2]);
        double deltaC = C1 - C2;

        // Delta a & b
        double deltaA = a[1] - b[1];
        double deltaB = a[2] - b[2];

        // Delta H
        double deltaH = deltaA * deltaA + deltaB * deltaB - deltaC * deltaC;
        deltaH = (deltaH < 0) ? 0 : Math.sqrt(deltaH);

        deltaC /= 1 + (0.045 * C1);
        deltaH /= 1 + (0.015 * C1);

        return (float) Math.sqrt(deltaL * deltaL + deltaC * deltaC + deltaH * deltaH);
    }

    /**
     * Computes the Euclidean distance.
     *
     * @param a first point
     * @param b second point
     * @return Euclidean distance
     */
    protected static float euclidean(float[] a, float[] b) {
        assert (a.length == b.length);
        float sum = 0.0f;
        for (int i = 0; i < a.length; i++) {
            float d = a[i] - b[i];
            sum += d * d;
        }
        return (float) Math.sqrt(sum);
    }

}
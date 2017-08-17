package diuf.diva.dia.ms.util;


public class Util {

    /**
     * @return the max |value| in the matrix m
     */
    public static float max(float[][] m) {
        // Find max
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < m.length; i++) {
            for (int j = 0; j < m[i].length; j++) {
                max = Math.max(max, Math.abs(m[i][j]));
            }
        }
        return max;
    }

    /**
     * @return the max |value| in the vector m
     */
    public static float max(float[] m) {
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < m.length; i++) {
            max = Math.max(max, Math.abs(m[i]));
        }
        return max;
    }

    /**
     * @return the matrix m multiplied element-wise by C
     */
    public static float[][] elementMultiply(float[][] m, float c) {
        for (int i = 0; i < m.length; i++) {
            for (int j = 0; j < m[i].length; j++) {
                m[i][j] *= c;
            }
        }
        return m;
    }

    /**
     * @return the vector m multiplied element-wise by C
     */
    public static float[] elementMultiply(float[] m, float c) {
        for (int i = 0; i < m.length; i++) {
            m[i] *= c;
        }
        return m;
    }

    public static float[][] normalise(float[][] m, float c) {

        // Find max
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < m.length; i++) {
            for (int j = 0; j < m[i].length; j++) {
                max = Math.max(max, Math.abs(m[i][j]));
            }
        }

        for (int i = 0; i < m.length; i++) {
            for (int j = 0; j < m[i].length; j++) {
                m[i][j] /= (max * c);
            }
        }

        return m;
    }

    public static float[] normalise(float[] m, float c) {
        float[][] tmp = new float[1][m.length];
        tmp[0] = m;
        return normalise(tmp, c)[0];
    }

}

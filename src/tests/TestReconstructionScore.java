package tests;

import diuf.diva.dia.ms.util.misc.ReconstructionScore;
import org.junit.Test;

/**
 * This class test whether the class ReconstructionScore have well working distances measurements.
 *
 * @author Michele Alberti
 */
public class TestReconstructionScore extends ReconstructionScore {

    @Test
    public void testEuclidean() {
        float d = ReconstructionScore.euclidean(new float[]{0, 0}, new float[]{1, 1});
        assert (Math.abs(d - Math.sqrt(2)) < 0.000001);
    }

    @Test
    public void testRGBtoXYZ() {

        float[] rgb = new float[]{255, 171, 32};
        float[] xyz = ReconstructionScore.rgbToXYZ(rgb);

        assert (Math.abs(xyz[0] - 56.064) < 0.001);
        assert (Math.abs(xyz[1] - 50.490) < 0.001);
        assert (Math.abs(xyz[2] - 8.157) < 0.001);
    }

    @Test
    public void testXYZtoLAB() {

        float[] rgb = new float[]{255, 171, 32};
        float[] xyz = ReconstructionScore.rgbToXYZ(rgb);
        float[] lab = ReconstructionScore.XYZtoLab(xyz);

        assert (Math.abs(lab[0] - 76.369) < 0.001);
        assert (Math.abs(lab[1] - 21.182) < 0.001);
        assert (Math.abs(lab[2] - 74.945) < 0.001);

        rgb = new float[]{0, 0, 0};
        xyz = ReconstructionScore.rgbToXYZ(rgb);
        lab = ReconstructionScore.XYZtoLab(xyz);

        assert (lab[0] == 0);
        assert (lab[1] == 0);
        assert (lab[2] == 0);
    }

    @Test
    public void testCIE76distance() {

        float[] a = ReconstructionScore.XYZtoLab(ReconstructionScore.rgbToXYZ(new float[]{255, 0, 0}));
        float[] b = ReconstructionScore.XYZtoLab(ReconstructionScore.rgbToXYZ(new float[]{0, 0, 0}));

        float d = ReconstructionScore.deltaE76(a, b);

        assert (Math.abs(d - 117.3447) < 0.0001);

        a = ReconstructionScore.XYZtoLab(ReconstructionScore.rgbToXYZ(new float[]{155, 155, 155}));
        b = ReconstructionScore.XYZtoLab(ReconstructionScore.rgbToXYZ(new float[]{0, 0, 0}));

        d = ReconstructionScore.deltaE76(a, b);

        assert (Math.abs(d - 63.9806) < 0.0001);
    }

    @Test
    public void testCIE94distance() {

        /* DISTANCE IS NOT SYMMETRIC!
         * deltaE94(a,b) != deltaE94(b,a)
         */
        float[] a = ReconstructionScore.XYZtoLab(ReconstructionScore.rgbToXYZ(new float[]{255, 0, 0}));
        float[] b = ReconstructionScore.XYZtoLab(ReconstructionScore.rgbToXYZ(new float[]{0, 0, 0}));

        assert (Math.abs(ReconstructionScore.deltaE94(a, b) - 56.2996) < 0.0001);
        assert (Math.abs(ReconstructionScore.deltaE94(b, a) - 117.3447) < 0.0001);
    }

    @Test
    public void testDelta94distance() {

        float[] a = new float[]{0, 0, 0, 255, 0, 0, 155, 155, 155};
        float[] b = new float[]{255, 0, 0, 0, 0, 0, 100, 100, 100};

        float d = ReconstructionScore.delta94distance(a, b);

        assert (Math.abs(d - (56.2996 + 117.3447 + 21.606)) < 0.0001);
    }

    @Test
    public void testMahalanobis() {

        float[] a = new float[]{-0.2750f, -0.6650f, -0.1290f, 0.8451f, 0.8002f, 0.8554f, -2.7712f, -2.5598f, -2.7899f, 0.9181f, 0.7397f, 1.1393f, -0.5193f, -0.9521f, -0.6167f};
        float[] b = new float[]{-1.2675f, -1.0757f, -0.8233f, 0.6211f, 0.6120f, 0.4452f, -1.7955f, -2.3093f, -2.1071f, -1.0578f, -0.7493f, -1.1578f, 0.1515f, 0.5380f, 0.4041f};

        assert (Math.abs(mahalanobisDistance(a, b) - 2.1948) < 0.0001);

    }

    @Test
    public void testScaledOffsetInvariant() {

        final int N = 15;

        float[] a = new float[N];
        float[] b = new float[N];
        float[] c = new float[N];
        float[] d = new float[N];
        float[] e = new float[N];

        for (int i = 0; i < N; i++) {
            a[i] = i;
            b[i] = i+2;
            c[i] = i*2;
            d[i] = (i+2)*-2;
            e[i] = (float) Math.pow(i,2);
        }

        assert (ReconstructionScore.scaleOffsetInvarDist(a,b) < 1e-5);
        assert (ReconstructionScore.scaleOffsetInvarDist(a,c) < 1e-5);
        assert (ReconstructionScore.scaleOffsetInvarDist(a,d) < 1e-5);
        assert (ReconstructionScore.scaleOffsetInvarDist(a,e)-275.02228 < 1e-5);


        System.out.println(ReconstructionScore.scaleOffsetInvarDist(a,e));
        System.out.println(ReconstructionScore.scaleOffsetInvarDist(e,a));


    }

}

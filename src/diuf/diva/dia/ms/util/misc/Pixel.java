package diuf.diva.dia.ms.util.misc;

/**
 * Support class for easier data structure modelling.
 * Currently used in trainClassifier, trainSCAE, ImageAnalysis.
 * @author Michele Alberti
 */
public class Pixel {
    /**
     * X coordinate of the pixel
     */
    public final int x;
    /**
     * Y coordinate of the pixel
     */
    public final int y;

    public Pixel(int x, int y) {
        this.x = x;
        this.y = y;
    }

    @Override
    public String toString() {
        StringBuilder s = new StringBuilder();
        s.append("p[");
        s.append(x);
        s.append(",");
        s.append(y);
        s.append("]");
        return s.toString();
    }
}
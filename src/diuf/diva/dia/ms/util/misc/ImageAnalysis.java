package diuf.diva.dia.ms.util.misc;

import diuf.diva.dia.ms.script.XMLScript;
import diuf.diva.dia.ms.util.DataBlock;

import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;

/**
 * This class is used for data balancing while training. It analyses a datablock
 * and stores references to different pixels of different classes such that is possible
 * to get them for training.
 * @author Michele Alberti
 */
public class ImageAnalysis {
    /**
     * For script-like printing only
     */
    private final SimpleDateFormat ft = new SimpleDateFormat("HH:mm:ss.SSS");
    /**
     * This hashMap stores one arrayList per each new class found. In the arrayList
     * are contained pixels belonging to that class
     */
    private final HashMap<Integer, ArrayList<Pixel>> data = new HashMap<>();
    /**
     * Final number of classes found in the picture
     */
    public final int nbClasses;
    /**
     * List which keeps track of next random representative to serve
     * The key is the class, the value the next representative to serve on the list: data
     */
    private final HashMap<Integer, Integer> next = new HashMap<>();

    /**
     * Creates a image analysis
     *
     * @param gt          the images that need to be analysed
     * @param inputWidth  size of input, need to handle borders
     * @param inputHeight size of input, need to handle borders
     */
    public ImageAnalysis(final DataBlock gt, final int inputWidth, final int inputHeight) {
        long startTime = System.currentTimeMillis();

        /* Population of the data keeping in consideration to skip borders of image (because if
         * we then center the input we want the input patch to be within the border of the image!
         */

        int index = gt.getDepth() - 1;
        for (int x = inputWidth / 2; x <= gt.getWidth() - inputWidth; x++) {
            for (int y = inputHeight / 2; y <= gt.getHeight() - inputHeight; y++) {
                int correctClass = Math.round((gt.getValues(x, y)[index] + 1) * 255 / 2.0f);
                if (!data.containsKey(correctClass)) {
                    next.put(correctClass, 0);
                    data.put(correctClass, new ArrayList<>());
                }
                data.get(correctClass).add(new Pixel(x, y));
            }
        }

        this.nbClasses = data.size();

        // Log creation
        System.out.println(ft.format(new Date()) + ": ImageAnalysis created in: " + (int) (System.currentTimeMillis() - startTime) / 1000.0 + " sec " + this.toString());
        /*
        This output is redundant with previous line and spams into the log with big datasets
        System.out.println(ft.format(new Date()) + "Number of elements:");
        for (int key : data.keySet()) {
            System.out.println(ft.format(new Date()) + "  class "+key+": "+data.get(key).size());
        }
        */


    }

    /**
     * Sub-sample (or super sample if there are too few!) the list to a specific total amount of pixels. This is done to prevent storing in memory
     * millions of pixels unnecessarily
     *
     * @param samples the TOTAL amount of sample that this object will store, equally divided among classes
     */
    public void subSample(int samples) {
        long startTime = System.currentTimeMillis();

        // For each class
        for (int c = 0; c < nbClasses; c++) {
            // If this class was not present skip
            if (data.get(c) == null) continue;

            // Init the new array list for storing the sub sampled points
            ArrayList<Pixel> ssp = new ArrayList<>();

            // Populate the sub sampled points with random representative until we have enough
            for (int i = 0; i < (int) Math.ceil(samples / (nbClasses * 1.0)); i++) {
                ssp.add(getRandomRepresentative(c));
            }

            // Remove old array list and set the new one
            data.remove(c);
            data.put(c, ssp);
        }
        System.gc();

        // Log sub sampling
        System.out.println(ft.format(new Date()) + ": ImageAnalysis sub-sampled in: " + (int) (System.currentTimeMillis() - startTime) / 1000.0 + " sec " + this.toString());

    }

    /**
     * Select, return the a random pixel on the list.
     *
     * @param c the class of which the representative will be chosen
     * @return the next pixel on the list
     */
    private Pixel getRandomRepresentative(int c) {
        return (data.get(c) != null) ? data.get(c).get((int) (XMLScript.getRandom().nextDouble() * data.get(c).size())) : null;
    }

    /**
     * Select, return the a random pixel on the list.
     *
     * @param c the class of which the representative will be chosen
     * @return the next pixel on the list
     */
    public Pixel getNextRepresentative(int c) {
        if (data.get(c) != null) {
            int n = next.get(c);
            if (n >= data.get(c).size()) {
                 /* This is life-saver in case of a mistake! If this methods is called too many times
                 * we print a warning but we prevent the crash of the program!
                 */
                //System.out.println("[WARNING][ImageAnalysis.java - getNextRepresentative()] You called too many times this methods for this class");
                n = (int) (XMLScript.getRandom().nextDouble() * data.get(c).size());
            }

            next.put(c, n + 1);
            return data.get(c).get(n);
        } else {
            return null;
        }


    }

    @Override
    public String toString() {
        StringBuilder s = new StringBuilder();
        s.append("[");
        for (int i = 0; i < nbClasses; i++) {
            s.append((data.get(i) != null) ? data.get(i).size() : 0);
            if (i < nbClasses - 1) {
                s.append(",");
            }

        }
        s.append("]");
        return s.toString();
    }
}

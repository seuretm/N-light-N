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

import diuf.diva.dia.ms.script.XMLScript;
import diuf.diva.dia.ms.util.misc.Pixel;

import java.awt.image.BufferedImage;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;

/**
 * This corresponds to a DataBlock with a 2 matrix for the GT.
 * It is mainly designed for work with images
 *
 * @author Michele Alberti
 */
public class GroundTruthDataBlock extends BiDataBlock {

    /**
     * Number of classes in the dataset
     */
    private final int NB_CLASSES = 4; // Hardcoded for speed purposes
    /**
     * The values of the gt corresponding to the pixels of the image
     */
    private int[][] gt;
    /**
     * This hashMap stores one arrayList per each new class found. In the arrayList
     * are contained pixels belonging to that class
     */
    private HashMap<Integer, ArrayList<Pixel>> data;

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Constructor
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * Creates a GroundTruthDataBlock
     *
     * @param db DataBlock with original data
     */
    public GroundTruthDataBlock(BiDataBlock db) {
        this(db, new int[db.width][db.height]);
    }

    /**
     * Creates a GroundTruthDataBlock
     *
     * @param db DataBlock with original data
     */
    public GroundTruthDataBlock(BiDataBlock db, int[][] gt) {
        super(db.bi);
        this.gt = gt;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Public
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * Analyses an image and create lists of pixels separates by classes.
     */
    public void analyse() {
        analyse(0, 0);
    }

    /**
     * Analyses an image and create lists of pixels separates by classes.
     *
     * @param inputWidth  size of input, needed to handle borders
     * @param inputHeight size of input, needed to handle borders
     */
    public void analyse(final int inputWidth, final int inputHeight) {
        long startTime = System.currentTimeMillis();

        // This is necessary otherwise old sampled data could be still selected
        data = new HashMap<>();

        /* Population of the data keeping in consideration to skip borders of image (because if
         * we then center the input we want the input patch to be within the border of the image!
         */
        for (int x = inputWidth / 2; x < width - inputWidth / 2; x++) {
            for (int y = inputHeight / 2; y < height - inputHeight / 2; y++) {
                boolean[] labels = getLabels(gt[x][y]);
                // Iterate all classes
                for (int c = 0; c < NB_CLASSES; c++) {
                    // If the label is on that pixel
                    if (labels[c]) {
                        // If not yet created, created the list for that class
                        if (!data.containsKey(c)) {
                            data.put(c, new ArrayList<>());
                        }

                        // Add the pixel to the list
                        data.get(c).add(new Pixel(x, y));
                    }
                }
            }
        }

        // Log creation
        System.out.println(new SimpleDateFormat("HH:mm:ss.SSS").format(new Date()) + ": ImageAnalysis created in: " + (int) (System.currentTimeMillis() - startTime) / 1000.0 + " sec " + this.toString());
    }

    /**
     * Retrieves the label as an array of boolean from the image passed as parameter
     *
     * @param gt the gt to convert
     * @return an array of boolean with TRUE in correspondence of the present labels at the selected location.
     * e.g: pixel value of 0x0110 will be converted in new boolean{false,true,true,false};
     */
    private boolean[] getLabels(int gt) {
        boolean[] labels = new boolean[NB_CLASSES];
        for (int c = 0; c < NB_CLASSES; c++) {
            labels[c] = ((gt >> c) & 0x1) == 1;
        }
        return labels;
    }

    /**
     * Sub-sample (or super sample if there are too few!) the list to a specific total amount of pixels.
     * This is done to prevent storing in memory millions of pixels unnecessarily
     *
     * @param samples the TOTAL amount of sample that this object will store, equally divided among classes
     */
    public void subSample(int samples) {
        long startTime = System.currentTimeMillis();

        // For each class
        for (int c = 0; c < NB_CLASSES; c++) {
            // If this class was not present skip
            if (data.get(c) == null) continue;

            // Init the new array list for storing the sub sampled points
            ArrayList<Pixel> ssp = new ArrayList<>();

            // Populate the sub sampled points with random representative until we have enough
            for (int i = 0; i < Math.ceil(samples / data.size()); i++) {
                ssp.add(getRandomRepresentative(c));
            }

            // Remove old array list and set the new one
            data.remove(c);
            data.put(c, ssp);
        }
        System.gc();

        // Log sub sampling
        System.out.println(new SimpleDateFormat("HH:mm:ss.SSS").format(new Date()) + ": ImageAnalysis sub-sampled in: " + (int) (System.currentTimeMillis() - startTime) / 1000.0 + " sec " + this.toString());

    }

    /**
     * Select, return the a random pixel on the list.
     *
     * @param c the class of which the representative will be chosen
     * @return the next pixel on the list
     */
    public Pixel getRandomRepresentative(int c) {
        return (data.get(c) != null) ? data.get(c).get(Random.nextInt(data.get(c).size())) : null;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Getters & Setters
    ///////////////////////////////////////////////////////////////////////////////////////////////

    public int getGt(int x, int y) {
        return gt[x][y];
    }

    public void setGt(int x, int y, int value) {
        gt[x][y] = value;
    }

    public int getNumberOfClasses() {
        return NB_CLASSES;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Borders handling
    ///////////////////////////////////////////////////////////////////////////////////////////////
    @Override
    public float getValue(int channel, int x, int y) {
        // Adapt borders
        while (x < 0 || x >= width) {
            if (x < 0) {
                x = -x;
            }
            if (x >= width) {
                x = width - (x - width) - 1;
            }
        }
        while (y < 0 || y >= height) {
            if (y < 0) {
                y = -y;
            }
            if (y >= height) {
                y = height - (y - height) - 1;
            }
        }

        return super.getValue(channel, x, y);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Utility
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * @return the matrix gt[][] in form of BufferedImage. Useful for working with SegmentationAnalysis
     */
    public BufferedImage getGtAsBufferedImage() {
        BufferedImage bufferedImage = new BufferedImage(bi.getWidth(), bi.getHeight(), bi.getType());
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                bufferedImage.setRGB(x, y, gt[x][y]);
            }
        }
        return bufferedImage;
    }

    @Override
    public String toString() {
        StringBuilder s = new StringBuilder();
        s.append("[");
        for (int i = 0; i < NB_CLASSES; i++) {
            s.append((data.get(i) != null) ? data.get(i).size() : 0);
            if (i < NB_CLASSES - 1) {
                s.append(",");
            }

        }
        s.append("]");
        return s.toString();
    }

}

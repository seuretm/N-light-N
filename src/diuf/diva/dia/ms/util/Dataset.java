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

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;

/**
 * This is representing a dataset
 * a SCAE.
 * @author Alberti Michele, Mathias Seuret
 */
public final class Dataset<D extends DataBlock> implements Iterable<D> {

    /**
     * Explicits the type of the dataset
     */
    public final TYPE type;
    /**
     * List of DataBlocks.
     */
    private ArrayList<D> data = new ArrayList<>();

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Constructor
    ///////////////////////////////////////////////////////////////////////////////////////////////
    public Dataset(TYPE type) {
        this.type = type;
    }

    /**
     * Populates a dataset with datablock read from images
     * @param path containing the images
     * @param sizeLimit maximum number of images to load
     * @param buffered loads the DataBlocks as BiDataBlock
     * @throws Exception if an image fails to load
     */
    public void loadDataBlocks(String path, int sizeLimit, boolean buffered) {
        // Verify the path
        File folder = new File(path);
        if (!folder.exists()) {
            throw new Error("The path " + path + " does not exist.");
        }
        if (!folder.isDirectory()) {
            throw new Error(path + " is not a directory.");
        }

        // Load the images
        int size = 0;
        String[] fList = folder.list();
        Arrays.sort(fList);
        for (String fName : fList) {
            if (fName.equals(".DS_Store")) {
                continue;
            }

            try {
                data.add(buffered ? (D) new BiDataBlock(path + "/" + fName) : (D) new DataBlock(new Image(path + "/" + fName)));
            } catch (IOException e) {
                e.printStackTrace();
            }
            if (sizeLimit!=0 && ++size>=sizeLimit) {
                break;
            }
        }
        System.gc();
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Public
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * Adds a datablock to the dataset.
     */
    public boolean add(D d) {
        return data.add(d);
    }

    /**
     * Tosses the dataset.
     */
    public void shuffle() {
        Collections.shuffle(data);
    }

    /**
     * @return an iterator over the datablocks
     */
    public Iterator<D> iterator() {
        return data.iterator();
    }

    /**
     * Return the selected element of the dataset
     * @param n index
     * @return the n-th datablock
     */
    public D get(int n) {
        return data.get(n);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Getter & Setters
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * Return the size of the data stored in the dataset
     * @return the number of datablocks
     */
    public int size() {
        return data.size();
    }

    /**
     * Types of dataset
     */
    public enum TYPE {
        NORMAL,
        GT // It is expected to be generified with GroundTruthDataBlocks
    }
}

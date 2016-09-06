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

import diuf.diva.dia.ms.util.Image.Colorspace;
import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Iterator;

/**
 * This is a set of datablocks which can be used for training
 * a SCAE.
 * @author Mathias Seuret
 */
public class Dataset implements Collection<DataBlock> {
    
    /**
     * List of datablocks.
     */
    protected ArrayList<DataBlock> data = new ArrayList<>();
    
    /**
     * Colorspace of the dataset.
     */
    protected Image.Colorspace colorspace;
    
    /**
     * Creates a dataset.
     * @param colorspace colorspace to use
     */
    public Dataset(Image.Colorspace colorspace) {
        this.colorspace = colorspace;
    }
    
    /**
     * Creates a dataset.
     * @param path containing the images
     * @param colorspace colorspace to use
     * @param sizeLimit maximum number of images to load
     * @throws Exception if an image fails to load
     */
    public Dataset(String path, Image.Colorspace colorspace, int sizeLimit) throws Exception {
        this(path, colorspace, sizeLimit, false);
    }
        
    /**
     * Creates a dataset.
     * @param path containing the images
     * @param colorspace colorspace to use
     * @param sizeLimit maximum number of images to load
     * @throws Exception if an image fails to load
     */
    public Dataset(String path, Image.Colorspace colorspace, int sizeLimit, boolean buffered) throws Exception {
        File fold = new File(path);
        if (!fold.exists()) {
            throw new Error("The path " + path + " does not exist.");
        }
        if (!fold.isDirectory()) {
            throw new Error(path + " is not a directory.");
        }
        this.colorspace = colorspace;
        int size = 0;
        String[] fList = fold.list();
        Arrays.sort(fList);
        for (String fName : fList) {
            if (fName.equals(".DS_Store")) {
                continue;
            }
            if (colorspace==Image.Colorspace.RGB && buffered) {
                BiDataBlock bid = new BiDataBlock(path + "/" + fName);
                data.add(bid);
            } else {
                Image img = new Image(path + "/" + fName);
                img.convertTo(colorspace);
                DataBlock db = new DataBlock(img);
                data.add(db);
            }
            if (sizeLimit!=0 && ++size>=sizeLimit) {
                break;
            }
        }
    }
    
    /**
     * Adds a datablock to the dataset.
     * @param db datablock
     */
    public boolean add(DataBlock db) {
        if (colorspace!=Colorspace.RGB && db instanceof BiDataBlock) {
            throw new Error("BiDataBlocks can only be added to datasets storing RGB data");
        }
        return data.add(db);
    }
    
    /**
     * @return a valid random index
     */
    public int getRandomIndex() {
        return (int)(data.size() * Math.random());
    }
    
    /**
     * Tosses the dataset.
     */
    public void randomPermutation() {
        for (int i=0; i<data.size(); i++) {
            int j = (int)(Math.random()*data.size());
            DataBlock k = data.get(i);
            data.set(i, data.get(j));
            data.set(j, k);
        }
    }

    /**
     * So that we can do a for-each.
     * @return an iterator
     */
    @Override
    public Iterator<DataBlock> iterator() {
        randomPermutation();
        return data.iterator();
    }
    
    /**
     * Return the selected element of the dataset
     * @param n index
     * @return the n-th datablock
     */
    public DataBlock get(int n) {
        return data.get(n);
    }
    
    /**
     * Return the size of the data stored in the dataset
     * @return the number of datablocks
     */
    public int size() {
        return data.size();
    }

    @Override
    public boolean isEmpty() {
        return data.isEmpty();
    }

    @Override
    public boolean contains(Object o) {
        return data.contains(o);
    }

    @Override
    public Object[] toArray() {
        return data.toArray();
    }

    @Override
    public <T> T[] toArray(T[] a) {
        return data.toArray(a);
    }

    @Override
    public boolean remove(Object o) {
        return data.remove(o);
    }

    @Override
    public boolean containsAll(Collection<?> c) {
        return data.containsAll(c);
    }

    @Override
    public boolean addAll(Collection<? extends DataBlock> c) {
        return data.addAll(c);
    }

    @Override
    public boolean removeAll(Collection<?> c) {
        return data.removeAll(c);
    }

    @Override
    public boolean retainAll(Collection<?> c) {
        return data.retainAll(c);
    }

    @Override
    public void clear() {
        data.clear();
    }
}

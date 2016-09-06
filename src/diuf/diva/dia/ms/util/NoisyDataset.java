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
import java.util.Arrays;

/**
 * Two datasets, one containing clean documents, the other one noisy documents.
 * @author Mathias Seuret
 */
public class NoisyDataset {
    /**
     * Dataset of clean images.
     */
    public final Dataset clean;
    
    /**
     * Dataset of noisy versions.
     */
    public final Dataset noisy;
    
    /**
     * Creates a NoisyDataset.
     * @param cleanPath folder containing clean images
     * @param noisyPath folder containing noisy versions of the images
     * @param colorspace desired colorspace
     * @param max maximum number of images to load
     * @throws IOException if one of the images could not be loaded
     */
    public NoisyDataset(String cleanPath, String noisyPath, Image.Colorspace colorspace, int max) throws IOException {
        clean = new Dataset(colorspace);
        noisy = new Dataset(colorspace);
        loadFiles(cleanPath, noisyPath, max);
    }

    private void loadFiles(String cp, String np, int max) throws IOException {
        checkFolder(cp);
        checkFolder(np);
        if (max==0) {
            max = Integer.MAX_VALUE;
        }
        
        File cFold = new File(cp);
        String[] cNames = cFold.list();
        Arrays.sort(cNames);
        File nFold = new File(np);
        String[] nNames = nFold.list();
        Arrays.sort(nNames);
        
        for (int i=0; i<cNames.length && max>0; i++, max--) {
            //System.out.println("Loading "+cFold+"/"+cNames[i]);
            Image img = new Image(cFold+"/"+cNames[i]);
            DataBlock cidb = new DataBlock(img);
            //System.out.println("Loading "+nFold+"/"+nNames[i]);
            img = new Image(nFold+"/"+nNames[i]);
            DataBlock nidb = new DataBlock(img);
            
            if (cidb.getWidth()!=nidb.getWidth() || cidb.getHeight()!=cidb.getHeight()) {
                throw new Error(
                        "clean and noisy document images do not have the same dimensions:\n"+
                        cidb.getWidth()+"x"+cidb.getHeight()+" for "+cNames[i]+"\n"+
                        nidb.getWidth()+"x"+nidb.getHeight()+" for "+nNames[i]
                );
            }
            
            clean.add(cidb);
            noisy.add(nidb);
        }
    }
    
    private void checkFolder(String folder) {
        File fold = new File(folder);
        if (!fold.exists()) {
            throw new Error("The folder "+folder+" does not exist.");
        }
        if (!fold.isDirectory()) {
            throw new Error(folder+" is not a directory.");
        }
    }
    
    /**
     * @return a valid random index
     */
    public int getRandomIndex() {
        return clean.getRandomIndex();
    }
    
    /**
     * @param n index
     * @return n-th clean image
     */
    public DataBlock getClean(int n) {
        return clean.get(n);
    }
    
    /**
     * @param n index
     * @return n-th noisy image
     */
    public DataBlock getNoisy(int n) {
        return noisy.get(n);
    }
    
    /**
     * @return number of images
     */
    public int size() {
        return noisy.size();
    }
}

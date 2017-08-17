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


package diuf.diva.dia.ms.ml.ae.frequency;

import diuf.diva.dia.ms.util.DataBlock;
import java.io.Serializable;

/**
 * Abstract class for managing different kind of spectral transforms which can
 * be used by the SpectralAutoEncoder.
 * @author Mathias Seuret
 */
public abstract class SpectralTransform implements Serializable {
    protected final int width;
    protected final int height;
    protected final int depth;
    
    protected final float[][][] arr3d;
    
    public SpectralTransform(int w, int h, int d) {
        width = w;
        height = h;
        depth = d;
        arr3d = new float[d][w][h];
    }
    
    /**
     * Applies the transform.
     * @param src source DataBlock
     * @param ox position x in the db
     * @param oy position y in the db
     * @param dst array for storing the result
     */
    public void forward(DataBlock src, int ox, int oy, float[] dst) {
        dataBlockToArray(src, ox, oy, arr3d);
        realForward(arr3d);
        chDim(arr3d, dst);
    }
    
    /**
     * Applies the transform.
     * @param src source data block
     * @param ox source position x
     * @param oy source position y
     * @param dst destination data block
     * @param dx destination x
     * @param dy destination y
     */
    public void forward(DataBlock src, int ox, int oy, DataBlock dst, int dx, int dy) {
        dataBlockToArray(src, ox, oy, arr3d);
        realForward(arr3d);
        arrayToDataBlock(arr3d, dst, dx, dy);
    }
    
    /**
     * Applies the transform
     * @param src source data
     * @param dst destination DB
     * @param ox position x in the DB
     * @param oy position y in the DB
     */
    public void forward(float[] src, DataBlock dst, int ox, int oy) {
        chDim(src, arr3d);
        realForward(arr3d);
        arrayToDataBlock(arr3d, dst, ox, oy);
    }
    
    /**
     * Applies the transform
     * @param src source array
     * @param dst destination array
     */
    public void forward(float[] src, float[] dst) {
        chDim(src, arr3d);
        realForward(arr3d);
        chDim(arr3d, dst);
    }
    
    /**
     * Applies the inverse transform.
     * @param src source DataBlock
     * @param ox position x in the data block
     * @param oy position y in the data block
     * @param dst destination array
     */
    public void inverse(DataBlock src, int ox, int oy, float[] dst) {
        dataBlockToArray(src, ox, oy, arr3d);
        realInverse(arr3d);
        chDim(arr3d, dst);
    }
    
    /**
     * Applies the inverse transform.
     * @param src source data block
     * @param ox source position x
     * @param oy source position y
     * @param dst destination data block
     * @param dx destination x
     * @param dy destination y
     */
    public void inverse(DataBlock src, int ox, int oy, DataBlock dst, int dx, int dy) {
        dataBlockToArray(src, ox, oy, arr3d);
        realInverse(arr3d);
        arrayToDataBlock(arr3d, dst, dx, dy);
    }
   
    /**
     * Applies the inverse transform
     * @param src source data
     * @param dst destination DB
     * @param ox position x in the DB
     * @param oy position y in the DB
     */
    public void inverse(float[] src, DataBlock dst, int ox, int oy) {
        chDim(src, arr3d);
        realInverse(arr3d);
        arrayToDataBlock(arr3d, dst, ox, oy);
    }
    
    /**
     * Applies the inverse transform
     * @param src source array
     * @param dst destination array
     */
    public void inverse(float[] src, float[] dst) {
        chDim(src, arr3d);
        realInverse(arr3d);
        chDim(arr3d, dst);
    }
    
    /**
     * Data order: a[z][x][y]
     * @param a array
     */
    protected abstract void realForward(float[][][] a);
    
    /**
     * Data order: a[z][x][y]
     * @param a array
     */
    protected abstract void realInverse(float[][][] a);
    
    /**
     *
     * @param db DataBlock
     * @param ox origin x
     * @param oy origin y
     * @param arr destination
     */
    protected static void dataBlockToArray(DataBlock db, int ox, int oy, float[][][] arr) {
        for (int x=0; x<db.getWidth(); x++) {
            for (int y=0; y<db.getHeight(); y++) {
                for (int z=0; z<db.getDepth(); z++) {
                    arr[z][x][y] = db.getValue(z, x+ox, y+oy);
                }
            }
        }
    }
    
    /**
     *
     * @param arr array
     * @param db DataBlock
     * @param ox origin x
     * @param oy origin y
     */
    protected static void arrayToDataBlock(float[][][] arr, DataBlock db, int ox, int oy) {
        for (int x=0; x<db.getWidth(); x++) {
            for (int y=0; y<db.getHeight(); y++) {
                for (int z=0; z<db.getDepth(); z++) {
                    db.setValue(z, x+ox, y+oy, arr[z][x][y]);
                }
            }
        }
    }
    
    /**
     * Puts the content of a 3D array into a 1D array.
     * @param src source array
     * @param dst destination array
     */
    protected void chDim(float[][][] src, float[] dst) {
        int p=0;
        for (int w=0; w<width; w++) {
            for (int h=0; h<height; h++) {
                for (int d=0; d<depth; d++) {
                    dst[p++] = src[d][w][h];
                }
            }
        }
    }
    
    /**
     * Puts the content of a 1D array into a 3D array.
     * @param src source array
     * @param dst destination array
     */
    protected void chDim(float[] src, float[][][] dst) {
        int p=0;
        for (int w=0; w<width; w++) {
            for (int h=0; h<height; h++) {
                for (int d=0; d<depth; d++) {
                    dst[d][w][h] = src[p++];
                }
            }
        }
    }
}

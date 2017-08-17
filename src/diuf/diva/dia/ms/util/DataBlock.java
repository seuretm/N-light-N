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

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;

/**
 * This corresponds to a 3-dimensions array, with some additional features.
 *
 * @author Mathias Seuret, Michele Alberti
 */
public class DataBlock implements Serializable, Cloneable {
    static public final long serialVersionUID = 1507544894050698666l;
    /**
     * Width of the array.
     */
    protected final int width;

    /**
     * Height of the array.
     */
    protected final int height;

    /**
     * Depth of the array.
     */
    protected final int depth;

    /**
     * The array.
     */
    protected final float[][][] value;

    /**
     * The weights.
     */
    protected final float[][] weight;

    /**
     * Colorspace of the image
     */
    protected Image.Colorspace type = null;

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC Constructor
    ///////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * Constructs a data field. A data field is basically a 3D array with
     * some accessors.
     * @param width of the array
     * @param height of the array
     * @param depth  of the array
     */
    public DataBlock(int width, int height, int depth) {
        this.width  = width;
        this.height = height;
        this.depth  = depth;
        value = new float[width][height][depth];
        weight = new float[width][height];
    }

    /**
     * Creates a DataBlock from a diuf.diva.dia.ms.util.Image.
     *
     * @param src source image
     */
    public DataBlock(Image src) {
        this(src.getWidth(), src.getHeight(), src.getDepth());
        for (int x = 0; x < src.getWidth(); x++) {
            for (int y = 0; y < src.getHeight(); y++) {
                for (int c = 0; c < src.getDepth(); c++) {
                    setValue(c, x, y, src.get(c, x, y));
                }
            }
        }
        type = src.getColorspace();
    }

    /**
     * Creates the datablock out of a buffered image.
     *
     * @param src source buffered image
     */
    public DataBlock(BufferedImage src) {
        this(src.getWidth(), src.getHeight(), 3);

        for (int x = 0; x < src.getWidth(); x++) {
            for (int y = 0; y < src.getHeight(); y++) {
                int r = (src.getRGB(x, y) >> 16) & 0xFF;
                int g = (src.getRGB(x, y) >> 8) & 0xFF;
                int b = src.getRGB(x, y) & 0xFF;
                setValue(0, x, y, 2 * (r / 255.0f) - 1);
                setValue(1, x, y, 2 * (g / 255.0f) - 1);
                setValue(2, x, y, 2 * (b / 255.0f) - 1);
            }
        }
    }

    /**
     * Loads the datablock from an image file.
     *
     * @param fName file name of the image
     * @throws IOException if the image cannot be loaded
     */
    public DataBlock(String fName) throws IOException {
        this(ImageIO.read(new File(fName)));
        this.setColorspace(Image.Colorspace.RGB);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // PROTECTED Constructor
    ///////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * Constructs a data block to be used as BiDatablock (child class).
     * Main difference is that the field value are not initialized to save space.
     */
    DataBlock(int width, int height) {
        this.width = width;
        this.height = height;
        this.depth = 3;
        value = null;
        weight = null;
    }

    /**
     * Creates a DataBlock from another DataBlock by cloning it.
     *
     * @param db datablock to clone
     */
    DataBlock(DataBlock db) {
        this.width = db.width;
        this.height = db.height;
        this.depth = db.depth;
        this.value = new float[width][height][depth];
        this.weight = new float[width][height];
        db.copyTo(this, 0, 0);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Getter & Setters
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * Loads a datablock from a stream.
     *
     * @param ois object input stream
     * @return the new datablock
     * @throws IOException            if the stream cannot be read
     * @throws ClassNotFoundException if the class of object in the stream does not match
     */
    public static DataBlock load(ObjectInputStream ois) throws IOException, ClassNotFoundException {
        return (DataBlock) ois.readObject();
    }

    /**
     * Loads a datablock from a file.
     *
     * @param fname file name
     * @return the new datablock
     * @throws IOException            if the file cannot be read
     * @throws ClassNotFoundException if the class in the stream does not match
     */
    public static DataBlock load(String fname) throws IOException, ClassNotFoundException {
        ObjectInputStream ois = new ObjectInputStream(new BufferedInputStream(new FileInputStream(fname)));
        DataBlock db = load(ois);
        ois.close();
        return db;
    }

    /**
     * @return the width of the block
     */
    public int getWidth() {
        return width;
    }

    /**
     * @return the height of the block
     */
    public int getHeight() {
        return height;
    }

    /**
     * @return the depth of the block
     */
    public int getDepth() {
        return depth;
    }

    /**
     * Increase the value, and add 1 to the weight.
     * @param z coordinate - channel in images
     * @param x coordinate
     * @param y coordinate
     * @param v a float
     */
    public void addValue(int z, int x, int y, float v) {
        value[x][y][z] += v;
        weight[x][y]+=1;
    }

    /**
     * Returns a stored value.
     * @param channel z coordinate, channel in images
     * @param x coordinate
     * @param y coordinate
     * @return a float
     */
    public float getValue(int channel, int x, int y) {
        return value[x][y][channel];
    }

    /**
     * Sets a value.
     * @param channel z coordinate, channel in images
     * @param x coordinate
     * @param y coordinate
     * @param v new value
     */
    public void setValue(int channel, int x, int y, float v) {
        value[x][y][channel] = v;
    }

    /**
     * Returns the values at (x,y,:) coordinates.
     * @param x coordinate
     * @param y coordinate
     * @return the array, not a copy
     */
    public float[] getValues(int x, int y) {
        return value[x][y];
    }

    /**
     * Sets the values at (x,y,:) coordinates.
     * @param x coordinate
     * @param y coordinate
     * @param z the array that's going to be set at (x,y,:)
     */
    public void setValues(int x, int y, float[] z) {
        assert (z.length == depth);

        value[x][y] = z;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Normalize
    ///////////////////////////////////////////////////////////////////////////////////////////////

    public Image.Colorspace getColorspace() {
        return type;
    }

    public void setColorspace(Image.Colorspace cs) {
        type = cs;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Utility
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * Normalizes the values such that they're in [0,1].
     */
    public void normalize() {
        float min = Float.MAX_VALUE;
        float max = Float.MIN_VALUE;

        for (int x = 0; x < getWidth(); x++) {
            for (int y = 0; y < getHeight(); y++) {
                for (int z = 0; z < getDepth(); z++) {
                    float v = getValue(z, x, y);
                    if (v > max) {
                        max = v;
                    }
                    if (v < min) {
                        min = v;
                    }
                }
            }
        }

        if (max == min) {
            return;
        }

        for (int x = 0; x < getWidth(); x++) {
            for (int y = 0; y < getHeight(); y++) {
                for (int z = 0; z < getDepth(); z++) {
                    setValue(z, x, y, 2 * (getValue(z, x, y) - min) / (max - min) - 1);
                }
            }
        }
    }

    /**
     * Divides the values by the weights, then reset the weights to 1.0f
     */
    public void normalizeWeights() {
        for (int x=0; x<width; x++) {
            for (int y=0; y<height; y++) {
                if (weight[x][y] == 0.0f || weight[x][y] == 1.0f) {
                    continue;
                }
                for (int i = 0; i < value[x][y].length; i++) {
                    value[x][y][i] /= weight[x][y];
                }
                weight[x][y] = 1.0f;
            }
        }
    }

    @Override
    public DataBlock clone() {
        DataBlock db = new DataBlock(width, height, depth);
        copyTo(db, 0, 0);
        return db;
    }

    /**
     * Copies the content of this data block into another block.
     *
     * @param dst  destination
     * @param posX x coordinate on the destination
     * @param posY y coordinate on the destination
     */
    public void copyTo(DataBlock dst, int posX, int posY) {
        assert (dst.getDepth() == getDepth());
        assert (dst.getWidth() >= getWidth() + posX);
        assert (dst.getHeight() >= getHeight() + posY);

        for (int x = 0; x < getWidth(); x++) {
            for (int y = 0; y < getHeight(); y++) {
                for (int z = 0; z < getDepth(); z++) {
                    dst.setValue(z, x + posX, y + posY, value[x][y][z]);
                }
                dst.weight[x][y] = weight[x][y];
            }
        }
    }

    /**
     * Erases the content of the block and sets the weights to 0.
     */
    public void clear() {
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                for (int z = 0; z < depth; z++) {
                    value[x][y][z] = 0;
                }
                weight[x][y] = 0;
            }
        }
    }

    /**
     * Pastes something in the array and uses a weighted sum so that it
     * can handle overlapping.
     *
     * @param source array
     * @param from   initial position in the source array
     * @param x      coordinate
     * @param y      coordinate
     */
    public void weightedPaste(float[] source, int from, int x, int y) {
        assert (source.length >= value[x][y].length + from);

        for (int i = 0; i < value[x][y].length; i++) {
            value[x][y][i] += source[from + i];
        }
        weight[x][y] += 1.0f;
    }

    /**
     * Puts the values from an array to a patch.
     *
     * @param arr    array
     * @param posX   position of the patch
     * @param posY   position of the patch
     * @param width  width of the patch
     * @param height height of the patch
     */
    public void weightedPatchPaste(float[] arr, int posX, int posY, int width, int height) {
        assert (arr.length == width * height * getDepth());

        int n = 0;
        for (int x = posX; x < posX + width; x++) {
            for (int y = posY; y < posY + height; y++) {
                for (int z = 0; z < getDepth(); z++) {
                    setValue(z, x, y, getValue(z, x, y) + arr[n++]);
                }
                weight[x][y] += 1.0f;
            }
        }
    }

    /**
     * Pastes a datablock at the giving position, asserting that it has the
     * correct depth.
     *
     * @param source data block
     * @param x      position x
     * @param y      position y
     */
    public void weightedPaste(DataBlock source, int x, int y) {
        assert (source.getDepth() == depth);
        assert (source.getWidth() + x < width);
        assert (source.getHeight() + y < height);

        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                weightedPaste(source.getValues(i, j), 0, x + i, y + j);
            }
        }
    }

    /**
     * Puts the values from a patch into an array.
     * @param arr target array
     * @param posX coordinate of the patch
     * @param posY coordinate of the patch
     * @param w width of the patch
     * @param h height of the patch
     */
    public void patchToArray(float[] arr, int posX, int posY, int w, int h) {
        assert (arr.length == w * h * depth);

        int i = 0;
        for (int x=posX; x<posX+w; x++) {
            for (int y=posY; y<posY+h; y++) {
                for (int z = 0; z < depth; z++) {
                    arr[i++] = getValue(z, x, y);
                }
            }
        }
    }

    /**
     * Puts the values from a patch into an array, and returns A COPY of it
     * @param posX coordinate of the patch
     * @param posY coordinate of the patch
     * @param w width of the patch
     * @param h height of the patch
     * @return the patch into array form
     */
    public float[] patchToArray(int posX, int posY, int w, int h) {
        float[] returnValue = new float[w * h * depth];
        int i = 0;
        for (int x = posX; x < posX + w; x++) {
            for (int y = posY; y < posY + h; y++) {
                for (int z = 0; z < depth; z++) {
                    returnValue[i++] = getValue(z, x, y);
                }
            }
        }
        return returnValue;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Save & load
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * Puts the values from a patch into an array, and returns A COPY of it after weighting it
     * @param posX coordinate of the patch
     * @param posY coordinate of the patch
     * @param w width of the patch
     * @param h height of the patch
     * @return the patch into array form
     */
    public float[] weightedPatchToArray(int posX, int posY, int w, int h) {
        float[] returnValue = new float[w * h * depth];
        int i = 0;
        for (int x = posX; x < posX + w; x++) {
            for (int y = posY; y < posY + h; y++) {
                for (int z = 0; z < depth; z++) {
                    returnValue[i++] = getValue(z, x, y)/weight[x][y];
                }
            }
        }
        return returnValue;
    }

    /**
     * Puts the values from an array to a patch.
     * @param arr array
     * @param posX position of the patch
     * @param posY position of the patch
     * @param width width of the patch
     * @param height height of the patch
     */
    public void arrayToPatch(float[] arr, int posX, int posY, int width, int height) {
        assert (arr.length==width*height*getDepth());

        int n = 0;

        for (int x=posX; x<posX+width; x++) {
            for (int y=posY; y<posY+height; y++) {
                for (int z=0; z<getDepth(); z++) {
                    setValue(z, x, y, arr[n++]);
                }
                weight[x][y] += 1.0f;
            }
        }
    }

    /**
     * Saves the datablock to a stream.
     *
     * @param oos object output stream
     * @throws IOException if the stream cannot accept more data I guess
     */
    public void save(ObjectOutputStream oos) throws IOException {
        oos.writeObject(this);
    }

    /**
     * Saves the datablock to a file.
     *
     * @param fname file name
     * @throws IOException if the file cannot be written to
     */
    public void save(String fname) throws IOException {
        ObjectOutputStream oos = new ObjectOutputStream(
                new DataOutputStream(
                        new BufferedOutputStream(
                                new FileOutputStream(fname)
                        )
                )
        );
        save(oos);
        oos.flush();
        oos.close();
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Image Datablock features
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /***
     * Turns the DataBlock into an image, assuming the colorspace of the DataBlock
     * has a colorspace (see setColorspace()).
     * @return an Image created from this datablock.
     */
    public Image getImage() {
        if (type==null) {
            switch (getDepth()) {
                case 1:
                    type = Image.Colorspace.GRAYSCALE;
                    break;
                case 3:
                    type = Image.Colorspace.RGB;
                    break;
                default:
                    throw new RuntimeException("The type for this datablock is not defined");
            }
        }
        if (type.depth != getDepth()) {
            throw new IllegalArgumentException(
                    type + " has " + type.depth + " channels, this data field has " + getDepth()
            );
        }
        Image img = new Image(getWidth(), getHeight(), type);

        for (int x = 0; x < getWidth(); x++) {
            for (int y = 0; y < getHeight(); y++) {
                for (int c = 0; c < getDepth(); c++) {
                    img.set(c, x, y, getValue(c, x, y));
                }
            }
        }

        return img;
    }
}

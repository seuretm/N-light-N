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
import java.awt.image.ColorModel;
import java.io.File;
import java.io.IOException;

/**
 * This is a datablock based on a buffered image storing RGB data.
 * @author Mathias Seuret, Michele Alberti
 */
public class BiDataBlock extends DataBlock {

    /**
     * Buffered image
     */
    private BufferedImage bi;

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Constructor
    ///////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * Loads a BiDataBlock from an image file.
     * @param fName file name
     * @throws IOException if the image cannot be read
     */
    public BiDataBlock(String fName) throws IOException {
        this(ImageIO.read(new File(fName)));
    }
    
    /**
     * Creates a BiDataBlock from a buffered image.
     * @param bi non-null reference to a BufferedImage
     */
    public BiDataBlock(BufferedImage bi) {
        super(bi.getWidth(), bi.getHeight());
        this.bi = bi;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Getter & Setters
    ///////////////////////////////////////////////////////////////////////////////////////////////

    @Override
    public float getValue(int channel, int x, int y) {
        int rgb = bi.getRGB(x, y);
        switch (channel) {
            case 0:
                return ((rgb >> 16) & 0xFF) / 255.0f * 2.0f - 1.0f;
            case 1:
                return ((rgb >> 8) & 0xFF) / 255.0f * 2.0f - 1.0f;
            case 2:
                return (rgb & 0xFF) / 255.0f * 2.0f - 1.0f;
        }
        return 0;
    }

    @Override
    public float[] getValues(int x, int y) {
        int rgb = bi.getRGB(x, y);
        float[] rv = new float[3];
        rv[0] = 2.0f * ((rgb >> 16) & 0x0000FF) / 255.0f - 1.0f;
        rv[1] = 2.0f * ((rgb >> 8) & 0x0000FF) / 255.0f - 1.0f;
        rv[2] = 2.0f * (rgb & 0x0000FF) / 255.0f - 1.0f;
        return rv;
    }

    @Override
    public void setValue(int channel, int x, int y, float v) {
        int rgb = bi.getRGB(x, y);
        int i = (int) ((v + 1) / 2.0f * 255.0f);
        switch (channel) {
            case 0:
                rgb = (rgb & 0x00FFFF) | (i<<16);
                break;
            case 1:
                rgb = (rgb & 0xFF00FF) | (i<<8);
                break;
            case 2:
                rgb = (rgb & 0xFFFF00) | i;
                break;
        }
        bi.setRGB(x, y, rgb);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Utility
    ///////////////////////////////////////////////////////////////////////////////////////////////

    @Override
    public DataBlock clone() {
        ColorModel cm = bi.getColorModel();
        return new BiDataBlock(new BufferedImage(cm, bi.copyData(null), cm.isAlphaPremultiplied(), null));
    }

    @Override
    public void clear() {
        for (int x = 0; x < getWidth(); x++) {
            for (int y = 0; y < getHeight(); y++) {
                bi.setRGB(x, y, 0);
            }
        }
    }

    @Override
    public void weightedPaste(float[] source, int from, int x, int y) {
        int r = (int) ((source[from] + 1) * 255.0f / 2.0f);
        int g = (int) ((source[from + 1] + 1) * 255.0f / 2.0f);
        int b = (int) ((source[from + 2] + 1) * 255.0f / 2.0f);
        int rgb = (r << 16) | (g << 8) | b;
        bi.setRGB(x, y, rgb);
    }

    @Override
    public void weightedPaste(DataBlock source, int px, int py) {
        assert (source.getDepth() == getDepth());
        assert (source.getWidth() + px < getWidth());
        assert (source.getHeight() + py < getHeight());

        for (int x=0; x<getWidth(); x++) {
            for (int y=0; y<getHeight(); y++) {
                weightedPaste(source.getValues(x, y), 0, px + x, py + y);
            }
        }
    }

}

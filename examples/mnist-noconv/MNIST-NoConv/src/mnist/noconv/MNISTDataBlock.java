/*****************************************************
  Training N-light-N on MNIST
  
  -------------------
  Author:
  2016 by Mathias Seuret <mathias.seuret@unifr.ch>
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
package mnist.noconv;

import diuf.diva.dia.ms.util.DataBlock;
import diuf.diva.dia.ms.util.Image;
import java.io.DataInputStream;
import java.io.IOException;

/**
 * This class extends DataBlock and is used for loading MNIST images from
 * their specific binary format.
 * @author Mathias Seuret
 */
public class MNISTDataBlock extends DataBlock {
    // Label, i.e., value of the digit between 0 and 9
    public final int label;
    
    /**
     * Loads a DataBlock from the MNIST dataset
     * @param dataIn data input stream corresponding to the pixel values
     * @param labelIn data input stream corresponding to the labels
     * @throws IOException if the streams cannot be read
     */
    public MNISTDataBlock(DataInputStream dataIn, DataInputStream labelIn) throws IOException {
        // Call of super constructor indicating width, height and depth of the data block
        super(28, 28, 1);
        
        // Reading the values
        for (int y=0; y<getWidth(); y++) {
            for (int x=0; x<getHeight(); x++) {
                int v = dataIn.readByte() & 0xFF;
                // Setting the values normalized in [-1, 1]
                setValue(0, x, y, 1 - 2.0f * v / 255.0f);
            }
        }
        
        // Reading the label
        label = labelIn.readByte() & 0xFF;
    }
}

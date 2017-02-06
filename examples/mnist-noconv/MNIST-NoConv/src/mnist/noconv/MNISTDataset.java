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

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;

/**
 *
 * @author Mathias Seuret
 */
public class MNISTDataset {
    
    public ArrayList<MNISTDataBlock> digits;
    
    public MNISTDataset(String dataFileName, String labelFileName) throws FileNotFoundException, IOException {
        this(dataFileName, labelFileName, Integer.MAX_VALUE);
    }
    
    public MNISTDataset(String dataFileName, String labelFileName, int maxSize) throws FileNotFoundException, IOException {
        // Preparing storage
        digits = new ArrayList<>();

        
        try (
                // Opening the files
                DataInputStream dataIn = new DataInputStream(new FileInputStream(dataFileName));
                DataInputStream labelIn = new DataInputStream(new FileInputStream(labelFileName))
            ) {
            
            // Skipping magic numbers
            dataIn.readInt();
            labelIn.readInt();
            
            // Reading number of images
            int nbImages = dataIn.readInt();
            if (labelIn.readInt()!=nbImages) {
                throw new Error("inconsistent image number in "+dataFileName+" and "+labelFileName);
            }
            nbImages = Math.min(nbImages, maxSize);
            
            // Skipping image dimension - known to be 28x28
            dataIn.readInt();
            dataIn.readInt();
            
            // Reading images
            System.out.println("Reading "+nbImages+" images...");
            for (int n=1; n<=nbImages; n++) {
                MNISTDataBlock db = new MNISTDataBlock(dataIn, labelIn);
                digits.add(db);
                if ((n%1000)==0) {
                    System.out.println(" "+n+"...");
                }
            }   System.out.println("Done");
        }
    }
}

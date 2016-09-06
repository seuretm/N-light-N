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

package diuf.diva.dia.ms.ml.ae;

/**
 *
 * @author Mathias Seuret
 */
public class ToRealUnit extends AutoEncoder {

    public ToRealUnit(int inW, int inH, int inD, int oD) {
        super(inW, inH, inD, oD);
        if (inW!=1 || inH!=1) {
            throw new Error(
                    "Real units cannot be added to convolved layers."
            );
        }
        if (oD!=inD) {
            throw new Error(
                    "Real units require same input and output lengths."
            );
        }
    }

    @Override
    public void encode() {
        float[] input = getInputArray();
        for (int i = 0; i < inputLength; i++) {
            getOutputArray()[i] = (input[i] > 0.5) ? 1 : -1;
        }
    }

    @Override
    public void decode() {
        for (int i = 0; i < inputLength; i++) {
            decoded[i] = getOutputArray()[i] > 0.5 ? 1 : -1;
        }
    }

    @Override
    public float train() {
        // nothing to do
        return 0;
    }


    @Override
    public boolean needsBinaryInput() {
        return true;
    }


    @Override
    public void deleteFeatures(int... number) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
    @Override
    public char getTypeChar() {
        return '?';
    }


    @Override
    public AutoEncoder clone() {
        throw new UnsupportedOperationException("Clone has not yet been implemented here");
    }

}

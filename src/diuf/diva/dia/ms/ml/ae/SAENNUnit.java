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

import java.io.Serializable;

/**
 * Autoencodign neural network unit.
 * @author Mathias Seuret
 */

public class SAENNUnit extends AutoEncoder implements Serializable {

    /**
     * Autoencoding neural network used by the unit.
     */
    SAENN nn;
    
    /**
     * Constructor of the class
     * @param inW input width
     * @param inH input height
     * @param inD input depth
     * @param oD  output depth
     */
    public SAENNUnit(int inW, int inH, int inD, int oD) {
        super(inW, inH, inD, oD);
        nn = new SAENN(inD*inH*inW, oD);
    }
    
    public SAENNUnit(int inW, int inH, int inD, int oD, float[][] eWeights, float[][] dWeights, float[] eBias, float[] dBias) {
        super(inW, inH, inD, oD);
        nn = new SAENN(inD*inH*inW, oD, eWeights, dWeights, eBias, dBias);
    }

    @Override
    public void encode() {
        nn.setInput(getInputArray());
        nn.encode();
        for (int o=0; o<outputDepth; o++) {
            getOutputArray()[o] = nn.getEncoded()[o];
        }
    }

    @Override
    public void decode() {
        for (int h=0; h<outputDepth; h++) {
            nn.getEncoded()[h] = getOutputArray()[h];
        }
        nn.decode();
        for (int v = 0; v < inputLength; v++) {
            decoded[v] = nn.getDecoded()[v];
        }
    }
    
    public float[] decode(float[] val) {
        for (int i=0; i<outputDepth; i++) {
            getOutputArray()[i] = val[i];
        }
        decode();
        return decoded;
    }
    
    public float[] encode(float[] val) {
        for (int i = 0; i < inputLength; i++) {
            inputArray[i] = val[i];
        }
        nn.setInput(inputArray);
        nn.encode();
        for (int o=0; o<outputDepth; o++) {
            getOutputArray()[o] = nn.getEncoded()[o];
        }
        return getOutputArray();
    }

    @Override
    public float train() {
        nn.setInput(getInputArray());
        return nn.train();
    }
    
    public float trainDecoder() {
        nn.setInput(getInputArray());
        return nn.trainDecoder();
    }

    @Override
    public boolean hasBinaryOutput() {
        return false;
    }

    @Override
    public boolean needsBinaryInput() {
        return false;
    }

    @Override
    public void trainingDone() {
        // Nothing to do.
    }
    
    public SAENN getAENN() {
        return nn;
    }

    @Override
    public char getTypeChar() {
        return 'n';
    }

    @Override
    public boolean isDenoising() {
        return false;
    }

    @Override
    public AutoEncoder clone() {
        throw new UnsupportedOperationException("Clone has not yet been implemented here");
    }
}

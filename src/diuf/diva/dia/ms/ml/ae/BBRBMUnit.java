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

import diuf.diva.dia.ms.ml.rbm.BasicBBRBM;
import java.io.Serializable;

/**
 * This is experimental and should be used only as base for future experiments
 * on RBMs.
 * @author Mathias Seuret
 */

public class BBRBMUnit extends AutoEncoder implements Serializable {
    BasicBBRBM rbm;
    protected boolean isTraining = false;
    
    /**
     * Constructs a binary-to-binary RBM.
     * @param inW input patch width
     * @param inH input patch height
     * @param inD input patch depth
     * @param oD number of hidden units (i.e., features)
     */
    public BBRBMUnit(int inW, int inH, int inD, int oD) {
        super(inW, inH, inD, oD);
        rbm = new BasicBBRBM(inW*inH*inD, oD);
    }

    @Override
    public void encode() {
        rbm.load(getInputArray());
        rbm.updateHidden();
        for (int h=0; h<outputDepth; h++) {
            getOutputArray()[h] = rbm.getHidden()[h];
        }
    }

    @Override
    public void decode() {
        for (int h=0; h<outputDepth; h++) {
            rbm.getHidden()[h] = (getOutputArray()[h] > 0.5f) ? 1 : 0;
        }
        rbm.decode();
        for (int v = 0; v < inputLength; v++) {
            decoded[v] = rbm.getVisible()[v];
        }
    }

    @Override
    public float train() {
        return rbm.train();
    }

    @Override
    public boolean hasBinaryOutput() {
        return true;
    }

    @Override
    public boolean needsBinaryInput() {
        return true;
    }
    
    
    @Override
    public String toString() {
        return rbm.toString();
    }
    
    @Override
    public void trainingDone() {
        // Nothing to do.
    }

    @Override
    public void deleteFeatures(int... number) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
    @Override
    public String getTypeName() {
        return "[BRBM]";
    }

    @Override
    public boolean isDenoising() {
        return false;
    }

    @Override
    public AutoEncoder clone() {
        throw new UnsupportedOperationException("Clone has not yet been implemented here");
    }

    @Override
    public void startTraining() {
        rbm.startTraining();
        isTraining = true;
    }

    @Override
    public void stopTraining() {
        rbm.stopTraining();
        isTraining = false;
    }

    @Override
    public boolean isTraining() {
        return isTraining;
    }

    @Override
    public void clearGradient() {
        throw new UnsupportedOperationException("Not supported yet.");
    }
}

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

package diuf.diva.dia.ms.ml.ae.ccnn;

import diuf.diva.dia.ms.ml.Trainable;
import diuf.diva.dia.ms.ml.ae.ffcnn.FFCNN;
import diuf.diva.dia.ms.ml.layer.NeuralLayer;
import diuf.diva.dia.ms.util.DataBlock;

import java.io.Serializable;
import java.util.ArrayList;

/**
 * Combined Convolution Neural Network. This class is a prototype and has not
 * been deeply tested so far. The idea is to combine two or several FFCNN having
 * potentially different input sources by adding one or several classification
 * layers on top. Fine-tuning is done through all FFCNNs.
 * @author Mathias Seuret
 */
public class CCNN implements Serializable, Trainable {
    
    protected ArrayList<FFCNN> base = new ArrayList();
    protected ArrayList<FFCNN> leaves = new ArrayList();
    protected ArrayList<CCNN> branches = new ArrayList();
    protected FFCNN top;
    
    protected int midSize = 0;
    protected int outSize;
    
    protected float[] input;
    protected float[] error;
    protected NeuralLayer[] layer;
    protected NeuralLayer topLayer;
    protected float[] output;
    protected boolean isTraining = false;
    
    /**
     * Constructs a Combined Convolutional Neural Network.
     * @param ccnn an array of CCNN to use as input
     * @param ffcnn an array of FFCNN to use as input
     * @param topNeurons the number of neurons to use in the classifier
     */
    public CCNN(CCNN[] ccnn, FFCNN[] ffcnn, int... topNeurons) {
        if (topNeurons==null) {
            throw new Error("It is mandatory to have at least one neural layer on top of the CCNN.");
        }
        
        if (ffcnn!=null) {
            for (FFCNN f : ffcnn) {
                leaves.add(f);
            }
        }
        if (ccnn!=null) {
            for (CCNN c : branches) {
                branches.add(c);
            }
        }
        
        for (FFCNN f : leaves) {
            midSize += f.getOutput().getDepth();
        }
        for (CCNN c : branches) {
            midSize += c.getOutputSize();
        }
        
        input = new float[midSize];
        error = new float[midSize];
        
        float[] pIn = input;
        float[] pEr = error;
        layer = new NeuralLayer[topNeurons.length];
        
        for (int i=0; i<topNeurons.length; i++) {
            layer[i] = new NeuralLayer(pIn, topNeurons[i]);
            layer[i].setPreviousError(pEr);
            pIn = layer[i].getOutputArray();
            pEr = layer[i].getError();
        }
        
        outSize = topNeurons[topNeurons.length-1];
        topLayer = layer[layer.length-1];
        output = topLayer.getOutputArray();
        
        base = getBase();
    }
    
    private ArrayList<FFCNN> getBase() {
        ArrayList<FFCNN> res = new ArrayList();
        
        for (CCNN c : branches) {
            res.addAll(c.getBase());
        }
        res.addAll(leaves);
        
        return res;
    }
    
    /**
     * @param n an ffcnn number
     * @return the n-th ffcnn
     */
    public FFCNN getFFCNN(int n) {
        return base.get(n);
    }
    
    /**
     * Sets the input of a given FFCNN composing the CCNN to the origin
     * of a given datablock (coordinates 0,0).
     * @param ffcnnNum number of the FFCNN
     * @param db data block
     */
    public void setInput(int ffcnnNum, DataBlock db) {
        setInput(ffcnnNum, db, 0, 0);
    }
    
    /**
     * Sets the input of an FFCNN on a given datablock.
     * @param ffcnnNum number of the FFCNN
     * @param db data block
     * @param posX positon X
     * @param posY position Y
     */
    public void setInput(int ffcnnNum, DataBlock db, int posX, int posY) {
        base.get(ffcnnNum).setInput(db, posX, posY);
    }
    
    /**
     * Centers the input of an FFCNN on the given coordinates.
     * @param ffcnnNum number of the FFCNN
     * @param db data block
     * @param posX center x
     * @param posY center y
     */
    public void centerInput(int ffcnnNum, DataBlock db, int posX, int posY) {
        base.get(ffcnnNum).centerInput(db, posX, posY);
    }
    
    /**
     * Computes the result, which can then be accessed with some getters.
     */
    public void compute() {
        int pos = 0;
        for (FFCNN f : leaves) {
            f.compute();
            System.arraycopy(f.getOutput().getValues(0, 0), 0, input, pos, f.getOutput().getValues(0, 0).length);
            pos += f.getOutput().getValues(0, 0).length;
        }
        for (CCNN c : branches) {
            c.compute();
            System.arraycopy(c.getOutput(), 0, input, pos, c.getOutputSize());
            pos += c.getOutputSize();
        }
        for (NeuralLayer l : layer) {
            l.compute();
        }
    }
    
    /**
     * Set the expected value.
     * @param outNum output number
     * @param val usually 0 or 1.
     */
    public void setExpected(int outNum, float val) {
        topLayer.setExpected(outNum, val);
    }
    
    /**
     * Adds error without knowing the expected value
     * @param outNum output number
     * @param err error
     */
    protected void addError(int outNum, float err) {
        topLayer.addError(outNum, err);
    }
    
    /**
     * Applies learning methods &amp; stuff. To use it properly,
     * set the inputs of the different FFCNN, call the compute()
     * method, set the expected values, and finally call
     * this method.
     */
    public void learn() {
        float err = 0;
        for (float f : topLayer.getError()) {
            err += Math.abs(f);
        }
        err /= topLayer.getError().length;
        
        for (int i=layer.length-1; i>=0; i--) {
            layer[i].backPropagate();
            layer[i].learn();
        }
        int pos = 0;
        for (FFCNN f : leaves) {
            for (int i=0; i<f.getOutputDepth(); i++) {
                f.addError(i, error[pos++]);
            }
        }
        for (CCNN c : branches) {
            for (int i=0; i<c.getOutputSize(); i++) {
                c.addError(i, error[pos++]);
            }
            c.learn();
        }
        for (int i=0; i<error.length; i++) {
            error[i] = 0.0f;
        }
    }
    
    /**
     * @return the number of outputs
     */
    public int getOutputSize() {
        return outSize;
    }
    
    /**
     * @return the outputs of the top layer
     */
    public float[] getOutput() {
        return output;
    }
    
    /**
     * @return the classification result
     */
    public int getClassNumber() {
        int opt = 0;
        for (int i=1; i<output.length; i++) {
            if (output[i]>output[opt]) {
                opt = i;
            }
        }
        return opt;
    }

    @Override
    public void startTraining() {
        for (FFCNN ffcnn : base) {
            ffcnn.startTraining();
        }
        for (FFCNN ffcnn : leaves) {
            ffcnn.startTraining();
        }
        for (CCNN ccnn : branches) {
            ccnn.startTraining();
        }
        isTraining = true;
    }

    @Override
    public void stopTraining() {
        for (FFCNN ffcnn : base) {
            ffcnn.stopTraining();
        }
        for (FFCNN ffcnn : leaves) {
            ffcnn.stopTraining();
        }
        for (CCNN ccnn : branches) {
            ccnn.stopTraining();
        }
        isTraining = false;
    }

    @Override
    public boolean isTraining() {
        return isTraining;
    }
}

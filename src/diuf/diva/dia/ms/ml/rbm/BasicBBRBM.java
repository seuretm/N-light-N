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

package diuf.diva.dia.ms.ml.rbm;
import java.io.Serializable;
import static java.lang.Math.*;

/**
 * Basic Binary-Binary RBM, based on the following web page:
 * http://blog.echen.me/2011/07/18/introduction-to-restricted-boltzmann-machines/
 * @author Mathias Seuret
 */
public class BasicBBRBM implements Serializable {
    /**
     * Number of visible units - of inputs.
     */
    int nbVisible;
    
    /**
     * Number of hidden units.
     */
    int nbHidden;
    
    /**
     * Storing visible values.
     */
    int[] visible;
    
    /**
     * Storing hidden values.
     */
    int[] hidden;
    
    /**
     * Weights.
     */
    double[][] w;
    
    /**
     * Positive values for CD, with some extra space
     * for the bias.
     */
    double[][] positive;
    
    /**
     * Negative values for CD, with some extra space
     * for the bias.
     */
    double[][] negative;
    
    /**
     * Bias weights for visible units.
     */
    double[] vb;
    
    /**
     * Bias weights for hidden units.
     */
    double[] hb;
    
    /**
     * Learning eps
     */
    double eps = 1e-3;
    
    /**
     * Creates an RBM.
     * @param nbVisible number of visible units
     * @param nbHidden number of hidden units
     */
    public BasicBBRBM(int nbVisible, int nbHidden) {
        this.nbVisible = nbVisible;
        this.nbHidden  = nbHidden;
        
        visible = new int[nbVisible];
        hidden  = new int[nbHidden];
        
        w = new double[nbVisible][nbHidden];
        for (int v=0; v<nbVisible; v++) {
            for (int h=0; h<nbHidden; h++) {
                w[v][h] = randomInitialWeight();
            }
        }
        
        positive = new double[nbVisible+1][nbHidden+1];
        negative = new double[nbVisible+1][nbHidden+1];
        
        vb = new double[nbVisible];
        hb = new double[nbHidden];
    }
    
    /**
     * Generates a random weight between -inf and +inf, with most of
     * the values between -0.04 and +0.04.
     * @return a float
     */
    private double randomInitialWeight() {
        return sqrt(-2*0.0001*log(random()))*signum(random()-0.5);
    }
    
    /**
     * @return the visible values
     */
    public int[] getVisible() {
        return visible;
    }
    
    /**
     * @return the hidden values
     */
    public int[] getHidden() {
        return hidden;
    }
    
    /**
     * Loads a sample.
     * @param sample array containing nbVisible values
     */
    public void load(int[] sample) {
        assert (sample.length==nbVisible);
        
        for (int i=0; i<nbVisible; i++) {
            visible[i] = sample[i];
        }
    }
    
    /**
     * Loads a sample, converting the values such that values lower than 0.5 are
     * considered as 0, while other values are considered as 1.
     * @param sample array containing nbVisible values
     */
    public void load(float[] sample) {
        assert (sample.length==nbVisible);
        
        for (int i=0; i<nbVisible; i++) {
            visible[i] = sample[i]<0.5 ? 0 : 1;
        }
    }
    
    /**
     * Trains the RBM on the following sample.
     * @param sample array containing nbVisible values
     */
    public void train(int[] sample) {
        load(sample);
        train();
    }
    
    /**
     * Trains the RBM.
     * @return the number of reconstruction differences divided by the number of visible units
     */
    public float train() {
        updateHidden();
        computePos();
        
        // Could iterate several times on these
        int diff = updateVisible();
        updateHidden();
        
        computeNeg();
        applyCD();
        
        return diff / (float)nbVisible;
    }
    
    /**
     * Computes the positive values used for the CD algo
     */
    public void computePos() {
        computePosNeg(positive);
    }
    
    /**
     * Computes the negative values used for teh CD algo
     */
    public void computeNeg() {
        computePosNeg(negative);
    }
    
    /**
     * Computes the visible units, with p=0.5.
     */
    public void decode() {
        for (int v=0; v<nbVisible; v++) {
            double sum = vb[v];
            for (int h=0; h<nbHidden; h++) {
                sum += hidden[h]*w[v][h];
            }
            double p = activation(sum);
            visible[v] = (p<0.5) ? 0 : 1;
        }
    }
    
    /**
     * Updates the hidden units from the visible units.
     */
    public void updateHidden() {
        for (int h=0; h<nbHidden; h++) {
            double sum = hb[h];
            for (int v=0; v<nbVisible; v++) {
                sum += visible[v]*w[v][h];
            }
            double p = activation(sum);
            hidden[h] = (random()<p) ? 1 : 0;
        }
    }
    
    /**
     * Updates the visible units from the hidden units.
     * @return the number of differences between the input and its reconstruction
     */
    public int updateVisible() {
        int diff = 0;
        for (int v=0; v<nbVisible; v++) {
            double sum = vb[v];
            for (int h=0; h<nbHidden; h++) {
                sum += hidden[h]*w[v][h];
            }
            double p = activation(sum);
            int n = (random()<p) ? 1 : 0;
            diff += (visible[v]!=n) ? 1 : 0;
            visible[v] = n;
        }
        return diff;
    }
    
    /**
     * Activation function.
     * @param x value
     * @return for now a sigmoid
     */
    public double activation(double x) {
        return 1 / (1+exp(-x));
    }
    
    /**
     * Computes the positive or negative array, depending on the current
     * state of the RBM.
     * @param dst destination array
     */
    private void computePosNeg(double[][] dst) {
        for (int v=0; v<nbVisible; v++) {
            dst[v][nbHidden] = visible[v];
            for (int h=0; h<nbHidden; h++) {
                dst[v][h] = visible[v]*hidden[h];
            }
        }
        for (int h=0; h<nbHidden; h++) {
            dst[nbVisible][h] = hidden[h];
        }
    }
    
    /**
     * Applies the CD using the positive and negative arrays.
     */
    private void applyCD() {
        for (int v=0; v<nbVisible; v++) {
            vb[v] += eps * (positive[v][nbHidden]-negative[v][nbHidden]);
            for (int h=0; h<nbHidden; h++) {
                w[v][h] += eps * (positive[v][h]-negative[v][h]);
            }
        }
        for (int h=0; h<nbHidden; h++) {
            hb[h] += eps * (positive[nbVisible][h]-negative[nbVisible][h]);
        }
    }
    
    /**
     * @return a string representation of the RBM
     */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        
        sb.append(String.format("        Bias"));
        for (int h=0; h<nbHidden; h++) {
            sb.append(String.format("      H-%d", h));
        }
        sb.append(String.format("\nBias            "));
        for (int h=0; h<nbHidden; h++) {
            sb.append(String.format(" %8.5f", hb[h]));
        }
        sb.append("\n");
        
        for (int v=0; v<nbVisible; v++) {
            sb.append(String.format("V-%d     %8.5f", v, vb[v]));
            for (int h=0; h<nbHidden; h++) {
                sb.append(String.format(" %8.5f", w[v][h]));
            }
            sb.append("\n");
        }
        
        return sb.toString();
    }
}
